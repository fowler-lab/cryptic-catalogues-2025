import pandas as pd
from protocols import utils
import os
import piezo
import json


def prep_phenotypes(drug, pheno_path, samples_path, version, validation=False):
    """Prepare phenotype data for a given drug and CRyPTIC dataset version.

    Imports phenotype and sample data, corrects known drug code issues,
    filters by dataset version, drug, and quality, and removes duplicates.

    Parameters
    ----------
    drug : str
        Drug code to filter phenotypes for (e.g., "INH", "RIF", "STM").
    pheno_path : str
        Path to the phenotype data file.
    samples_path : str
        Path to the sample metadata file.
    version : str
        CRyPTIC table version (e.g., "v1.0", "v3.3.0").
    validation : bool, optional
        If True, use validation dataset filtering (default is False).

    Returns
    -------
    pandas.DataFrame
        Filtered phenotype data containing relevant samples and columns."""

    # import phenotypes and samples and genoms
    phenotypes = utils.read_data(pheno_path).reset_index()
    samples = utils.read_data(samples_path).reset_index()

    # streptomycin has the wrong drug code in the phenotypes table
    if drug == "STM":
        phenotypes.replace({"DRUG": {"STR": "STM"}}, inplace=True)

    assert version in ["v1.0", "v2.0", "v3.0.0", "v3.3.0", "v3.4.0"]

    if version == "v1.0":

        matched = samples[
            (samples.in_final_tables) & (samples.dataset == "CRyPTIC-v1.0")
        ].UNIQUEID.unique()
        phenotypes = phenotypes[phenotypes.UNIQUEID.isin(matched)]

    elif version == "v2.0":

        matched = samples[
            (samples.in_final_tables)
            & (samples.dataset.isin(["CRyPTIC-v1.0", "CRyPTIC-v2.0"]))
        ].UNIQUEID.unique()
        phenotypes = phenotypes[phenotypes.UNIQUEID.isin(matched)]

    elif version in ["v3.0.0", "v3.3.0", "v3.4.0"]:

        if not validation:
            matched = samples[
                (samples.in_final_tables) & (samples.dataset == "CRyPTIC-v1.0")
            ].UNIQUEID.unique()
            phenotypes = phenotypes[phenotypes.UNIQUEID.isin(matched)]

        else:

            matched = samples[
                (samples.in_final_tables) & (samples.dataset != "CRyPTIC-v1.0")
            ].UNIQUEID.unique()
            phenotypes = phenotypes[phenotypes.UNIQUEID.isin(matched)]

    # filter for drug
    phenotypes = phenotypes[phenotypes.DRUG == drug]

    # discard low quality phenotypes
    phenotypes = phenotypes[phenotypes.QUALITY != "LOW"]
    phenotypes = phenotypes[phenotypes.PHENOTYPE.isin(["R", "S"])]

    # handle duplicates (keep R if R, otherwise first)
    phenotypes = (
        phenotypes.groupby("UNIQUEID", group_keys=False)
        .apply(utils.filter_multiple_phenos)
        .reset_index(drop=True)
    )

    # filter relevant columns for catomatic and rename id column
    phenotypes = phenotypes[["UNIQUEID", "DRUG", "PHENOTYPE", "METHOD_MIC", "METHOD_3"]]
    # phenotypes.set_index("UNIQUEID", inplace=True)

    return phenotypes


def prep_mutations(path, genes, mut_path, var_path, version="v3.1.0", train=True):
    """Prepare mutation data for selected genes and CRyPTIC dataset version.

    Loads mutation and variant data, computes Fraction Read Support,
    merges tables, removes low-quality or synonymous mutations, and
    filters relevant columns for downstream analysis.

    Parameters
    ----------
    path : str
        Base path for storing processed mutation and variant files.
    genes : list of str
        List of gene names to include.
    mut_path : str
        Path to the full mutation data file.
    var_path : str
        Path to the full variant data file.
    version : str, optional
        CRyPTIC table version (default is "v3.1.0").
    train : bool, optional
        If True, filters out synonymous mutations (default is True).

    Returns
    -------
    pandas.DataFrame
        Filtered mutation data with relevant columns."""

    # Uses most up to date mutations table, as this should contain all genomes from previous versions
    mut_dir = f"{path}{'_'.join(genes)}_MUTATIONS.csv"
    var_dir = f"{path}{'_'.join(genes)}_VARIANTS.csv"

    if not os.path.exists(mut_dir):
        mutations = utils.read_data(mut_path).reset_index()
        variants = utils.read_data(var_path).reset_index()
        mutations[mutations.GENE.isin(genes)].to_csv(mut_dir)
        variants[variants.GENE.isin(genes)].to_csv(var_dir)

    mutations = pd.read_csv(mut_dir, low_memory=False)
    variants = pd.read_csv(var_dir, low_memory=False)

    if version == "v1":

        variants["FRS"] = variants.apply(
            lambda row: (
                1 if row["IS_FILTER_PASS"] is True else row["COVERAGE"] / row["DP"]
            ),
            axis=1,
        )
        mutations = pd.merge(
            mutations,
            variants[["UNIQUEID", "GENE", "POSITION", "FRS"]],
            on=["UNIQUEID", "GENE", "POSITION"],
            how="left",
        )
        mutations = mutations[~mutations.IS_NULL]

    elif version in ["v3.0.0", "v3.1.0", "v3.3.0", "v3.4.0"]:
        variants["FRS"] = variants.apply(
            lambda row: (
                row["MINOR_READS"] / row["COVERAGE"] if row["MINOR_READS"] > 0 else 1
            ),
            axis=1,
        )
        if "FRS" in mutations.columns:
            mutations = mutations.drop(columns=["FRS"])

        mutations = pd.merge(
            mutations,
            variants[["UNIQUEID", "GENE", "GENE_POSITION", "FRS"]],
            on=["UNIQUEID", "GENE", "GENE_POSITION"],
            how="left",
        )
        mutations = mutations[~mutations.IS_NULL]
        mutations["MUTATION"] = mutations.apply(
            lambda x: f"{x['GENE']}@{x['MINOR_MUTATION'] if x['IS_MINOR'] else x['MUTATION']}",
            axis=1,
        )

    # flag synonymous mutations
    mutations["IS_SYNONYMOUS"] = mutations["MUTATION"].apply(
        lambda x: x.split("@")[-1][0] == x.split("@")[-1][-1]
    )

    # filter out synonymous mutations
    if train:
        # if validating, would keep synonymous mutations
        mutations = mutations[~mutations.IS_SYNONYMOUS]

    # filter our remaining Z's
    mutations["aa"] = [i[-1] for i in mutations.MUTATION.values]
    mutations = mutations[~mutations.aa.isin(["Z", "z"])]

    # drop duplicate entries
    mutations = mutations.drop_duplicates(["UNIQUEID", "MUTATION", "FRS"], keep="first")

    # filter relevant columns for catomatic and rename id column
    mutations = mutations[
        ["UNIQUEID", "MUTATION", "FRS", "IS_MINOR", "MINOR_MUTATION", "IS_NULL"]
    ].rename(columns={"ENA_RUN": "UNIQUEID"})
    # mutations.set_index("UNIQUEID", inplace=True)

    return mutations


def piezo_predict(
    iso_df,
    catalogue_file,
    drug,
    U_to_R=False,
    U_to_S=False,
    Print=False,
    log_false_predictions=False,
    return_predictions=False,
):
    """
    Predicts drug resistance based on genetic mutations using a resistance catalogue.

    Parameters:
    iso_df (pd.DataFrame): DataFrame containing isolate data with UNIQUEID, PHENOTYPE, and GENE_MUT columns.
    catalogue_file (str): Path to the resistance catalogue file.
    drug (str): The drug for which resistance predictions are to be made.
    U_to_R (bool, optional): If True, treat 'U' predictions as 'R'. Defaults to False.
    U_to_S (bool, optional): If True, treat 'U' predictions as 'S'. Defaults to False.
    Print (bool, optional): If True, prints the confusion matrix, DPR, sensitivity, and specificity. Defaults to True.
    log_false_predictions (bool, optional): If True, also return the ids for False Positive and Negative samples for discrepany analysis

    Returns:
    list: Confusion matrix, isolate DPR, sensitivity, specificity, and false negative IDs.
    """
    # Load and parse the catalogue with piezo
    catalogue = piezo.ResistanceCatalogue(catalogue_file)

    # Ensure the UNIQUEID and PHENOTYPE columns are used correctly
    ids = iso_df["UNIQUEID"].unique().tolist()
    labels = iso_df.groupby("UNIQUEID")["PHENOTYPE"].first().reindex(ids).tolist()
    predictions = []

    for id_ in ids:

        # For each sample
        df = iso_df[iso_df["UNIQUEID"] == id_]

        # Predict phenotypes for each mutation via lookup
        mut_predictions = []

        for var in df["MUTATION"]:
            if pd.isna(var):
                predict = "S"
            else:
                try:
                    predict = catalogue.predict(var)
                except ValueError:
                    predict = "U"
            if isinstance(predict, dict):
                if drug in predict.keys():
                    mut_predictions.append(predict[drug])
            else:
                mut_predictions.append(predict)

        # Make sample-level prediction from mutation-level predictions. R > U > S
        if "R" in mut_predictions:
            predictions.append("R")
        elif "U" in mut_predictions:
            if U_to_R:
                predictions.append("R")
            elif U_to_S:
                predictions.append("S")
            else:
                predictions.append("U")
        else:
            predictions.append("S")

    # Log false negative samples
    if log_false_predictions:
        FN_id = [
            id_
            for id_, label, pred in zip(ids, labels, predictions)
            if pred == "S" and label == "R"
        ]

        FP_id = [
            id_
            for id_, label, pred in zip(ids, labels, predictions)
            if pred == "R" and label == "S"
        ]

    # Generate confusion matrix for performance analysis
    cm = utils.confusion_matrix(labels, predictions, classes=["R", "S", "U"])

    TP = cm[0, 0]
    FN = cm[0, 1]
    TN = cm[1, 1]
    FP = cm[1, 0]
    UP = cm[0, 2]
    UN = cm[1, 2]

    if Print:
        print(cm)

    # Calculate ternary performance metrics
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP)
    dpr = (len(labels) - predictions.count("U")) / len(labels)

    # Calculate binary performance metrics
    sensitivity2 = TP / (TP + FN + UP)
    specificity2 = (TN + UN) / (TN + UN + FP)

    if Print:
        print("Catalogue coverage of isolates:", dpr)
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)

    if log_false_predictions:
        return [
            cm,
            dpr,
            sensitivity,
            specificity,
            sensitivity2,
            specificity2,
            FN_id,
            FP_id,
        ]
    elif not return_predictions:
        return [cm, dpr, sensitivity, specificity, sensitivity2, specificity2]
    elif return_predictions:
        return [ids, labels, predictions]


def weighted_score(df, weights=(0.5, 0.3, 0.2)):
    w1, w2, w3 = weights
    df["Score"] = w1 * df["SENSITIVITY"] + w2 * df["SPECIFICITY"] + w3 * df["DPR"]
    return df.sort_values(by="Score", ascending=False)


def expand_catalogue_pair(cat1, cat2, drugs, model, cat_names, who_cat=None):
    """
    Expand and compare two resistance catalogues by applying rule-based matching.

    For each drug, expands both catalogues to include mutations from the other
    that fall under defined rules, allowing effective content comparison between
    catalogues rather than just direct row matches. Supports optional WHOv1
    expert rule mapping.

    Parameters
    ----------
    cat1 : pandas.DataFrame
        First catalogue containing mutation rules and predictions.
    cat2 : pandas.DataFrame
        Second catalogue containing mutation rules and predictions.
    drugs : list of str
        List of drugs to process.
    model : list
        List of possible prediction values (e.g., ["R", "S", "U"]).
    cat_names : list of str
        Names or identifiers for the two catalogues (used in output keys).
    who_cat : str or None, optional
        WHO catalogue version ("WHOv1", "WHOv2", or None). Used to flag
        expert-intervention rules for WHO catalogues.

    Returns
    -------
    dict
        Dictionary mapping each drug to a set of expanded catalogues:
        {
            "cat1_name": <expanded DataFrame>,
            "cat2_name": <expanded DataFrame>,
            "merged": <combined DataFrame>
        }.
    """

    cat1_no_rules = cat1[~cat1["MUTATION"].str.contains(r"[*?=]", regex=True)]
    cat1_rules_only = cat1[cat1["MUTATION"].str.contains(r"[*?=]", regex=True)]
    # cat2_no_rules = cat2[cat2["EVIDENCE"].apply(lambda x: bool(x))]
    # cat2_rules_only = cat2[~cat2["EVIDENCE"].apply(lambda x: bool(x))]

    if who_cat == "WHOv1":

        w_garc_path = "catalogues/whov1/WHO-UCN-GTB-PCI-2021.7-eng_w_GARC.xlsx"
        # flag mutations that had expert rules applied to WHOv1
        if not os.path.exists(w_garc_path):
            utils.generate_garc_mutations(
                "catalogues/whov1/WHO-UCN-GTB-PCI-2021.7-eng.xlsx",
                "data/NC_000962.3.gbk",
            ).to_excel(w_garc_path)

        who_original = pd.read_excel(w_garc_path)

        filtered = who_original[~who_original["Additional grading criteria"].isna()]

        prediction_map = {
            "Assoc w R": "R",
            "Assoc w R - Interim": "R",
            "Uncertain significance": "U",
            "Not assoc w R": "S",
            "Not assoc w R - Interim": "S",
        }
        filtered["PREDICTION"] = filtered["INITIAL CONFIDENCE GRADING"].map(
            prediction_map
        )
        mapping = dict(zip(filtered["GARC_MUTATION"], filtered["PREDICTION"]))
        normalized_order = {
            "&".join(sorted(key.split("&"))): val for key, val in mapping.items()
        }

        def normalize_order(mutation_str):
            return "&".join(sorted(mutation_str.split("&")))

        cat2_no_rules = cat2[
            (~cat2["MUTATION"].str.contains(r"[*?=]", regex=True))
            & (~cat2["MUTATION"].str.contains("indel"))
        ]
        cat2_no_rules["PREDICTION"] = cat2_no_rules.apply(
            lambda row: normalized_order.get(
                normalize_order(row["MUTATION"]), row["PREDICTION"]
            ),
            axis=1,
        )
        cat2_rules_only = cat2[
            (cat2["MUTATION"].str.contains(r"[*?=]", regex=True))
            | (cat2["MUTATION"].isin(mapping.keys()))
        ]

    elif who_cat == "WHOv2":
        print("WHOv2 rule comparisons not yet supported, still need to implement")
    else:
        cat2_no_rules = cat2[~cat2["MUTATION"].str.contains(r"[*?=]", regex=True)]
        cat2_rules_only = cat2[cat2["MUTATION"].str.contains(r"[*?=]", regex=True)]

    expanded_catalogues = {}

    row = {
        "GENBANK_REFERENCE": "NC00962.3",
        "CATALOGUE_NAME": "-",
        "CATALOGUE_VERSION": 0,
        "CATALOGUE_GRAMMAR": "GARC1",
        "PREDICTION_VALUES": model,
        "DRUG": None,
        "MUTATION": None,
        "PREDICTION": None,
        "SOURCE": {},
        "EVIDENCE": {},
        "OTHER": {},
    }

    for drug in drugs:

        # catalogues filtered by drug
        cat1_drug = cat1[cat1.DRUG == drug]
        cat2_drug = cat2[cat2.DRUG == drug]
        # catalogues with rules removed, filtered for drug
        cat1_no_rules_drug = cat1_no_rules[cat1_no_rules.DRUG == drug]
        cat2_no_rules_drug = cat2_no_rules[cat2_no_rules.DRUG == drug]
        # catalogue expert rules (not defaults) filtered for drug
        cat2_rules_drug = cat2_rules_only[cat2_rules_only.DRUG == drug]
        cat1_rules_drug = cat1_rules_only[cat1_rules_only.DRUG == drug]
        # add placeholder rules to rule catalogues so avoid piezo error
        for i in model:
            if i != "U":
                row["PREDICTION"] = i
                row["MUTATION"] = "placeholder@A1A"
                row["DRUG"] = drug
                cat2_rules_drug = pd.concat(
                    [cat2_rules_drug, pd.DataFrame([row])], ignore_index=True
                )
                cat1_rules_drug = pd.concat(
                    [cat1_rules_drug, pd.DataFrame([row])], ignore_index=True
                )

        genes = set(
            cat2_drug["MUTATION"].apply(lambda x: x.split("@")[0]).tolist()
            + cat1_drug["MUTATION"].apply(lambda x: x.split("@")[0]).tolist()
        )
        # add a default wildcard U rule to rule catalogues so that a U is thrown if a mutation is not shared
        for gene in genes:
            for mut in [f"{gene}@*?", f"{gene}@-*?"]:
                row["PREDICTION"] = "U"
                row["MUTATION"] = mut
                row["DRUG"] = drug
                cat2_rules_drug = pd.concat(
                    [cat2_rules_drug, pd.DataFrame([row])], ignore_index=True
                )
                cat1_rules_drug = pd.concat(
                    [cat1_rules_drug, pd.DataFrame([row])], ignore_index=True
                )

        # write out rule catalogues so piezo can read them in and scan the other non-rule catalogue
        cat1_rules_drug["EVIDENCE"] = cat1_rules_drug["EVIDENCE"].to_json()
        cat1_rules_drug.to_csv(f"./catalogues/temp/cat1_rules_only.csv")
        cat1_rules_piezo = piezo.ResistanceCatalogue(
            f"./catalogues/temp/cat1_rules_only.csv"
        )

        cat2_rules_drug["EVIDENCE"] = cat2_rules_drug["EVIDENCE"].to_json()
        cat2_rules_drug.to_csv(f"./catalogues/temp/cat2_rules_only.csv")
        cat2_rules_piezo = piezo.ResistanceCatalogue(
            f"./catalogues/temp/cat2_rules_only.csv"
        )

        # use cat2 catalogue to scan cat1 non-rule catalogue to find variants that fall under that rule
        vars = []
        for var in cat1_no_rules_drug.MUTATION:
            try:
                prediction = cat2_rules_piezo.predict(var)
                if prediction[drug] in ["R", "S"]:
                    vars.append((var, prediction[drug]))
            except ValueError:
                continue
        # add variants from cat1 catalogue that fall under cat2 rules to cat2 non-rule catalogue
        for m, p in vars:
            row["MUTATION"] = m
            row["PREDICTION"] = p
            row["EVIDENCE"] = {"expanded_rule"}
            cat2_no_rules_drug = pd.concat(
                [cat2_no_rules_drug, pd.DataFrame([row])], ignore_index=True
            )
        # use cat1rule catalogue to scan cat2 non-rule catalogue to find variants that fall under that rule
        vars = []
        for var in cat2_no_rules_drug.MUTATION:
            try:
                prediction = cat1_rules_piezo.predict(var)
                if prediction[drug] in ["R", "S"]:
                    vars.append((var, prediction[drug]))
            except ValueError:
                continue
        # add variants from cat2 catalogue that fall under cat1 rules to cat1 non-rule catalogue
        for m, p in vars:
            row["MUTATION"] = m
            row["PREDICTION"] = p
            row["EVIDENCE"] = {"expanded_rule"}
            cat1_no_rules_drug = pd.concat(
                [cat1_no_rules_drug, pd.DataFrame([row])], ignore_index=True
            )

        if who_cat == "WHOv1":
            # remap final predictions back to merged df (slightly hacky)
            prediction_map = dict(zip(cat2_drug["MUTATION"], cat2_drug["PREDICTION"]))
            cat2_no_rules_drug_for_merged = cat2_no_rules_drug.copy()
            cat2_no_rules_drug_for_merged["PREDICTION"] = (
                cat2_no_rules_drug_for_merged.apply(
                    lambda row: (
                        prediction_map[row["MUTATION"]]
                        if row["MUTATION"] in prediction_map
                        else row["PREDICTION"]
                    ),
                    axis=1,
                )
            )
            expanded_catalogues[drug] = {
                cat_names[0]: cat1_no_rules_drug.drop_duplicates("MUTATION"),
                cat_names[1]: cat2_no_rules_drug.drop_duplicates("MUTATION"),
                "merged": pd.merge(
                    cat1_no_rules_drug,
                    cat2_no_rules_drug_for_merged,
                    on="MUTATION",
                    how="outer",
                    suffixes=(f"_{cat_names[0]}", f"_{cat_names[1]}"),
                ),
            }
        else:
            expanded_catalogues[drug] = {
                cat_names[0]: cat1_no_rules_drug.drop_duplicates("MUTATION"),
                cat_names[1]: cat2_no_rules_drug.drop_duplicates("MUTATION"),
                "merged": pd.merge(
                    cat1_no_rules_drug,
                    cat2_no_rules_drug,
                    on="MUTATION",
                    how="outer",
                    suffixes=(f"_{cat_names[0]}", f"_{cat_names[1]}"),
                ),
            }

    return expanded_catalogues


def expand_pair_coverages(cat1, cat2, mutations, phenotypes, drug, model, who_cat=None):
    """
    Expand and compare catalogue coverage for a single drug using mutation and phenotype data.

    Applies rule-based catalogues to mutation–phenotype pairs to determine
    which samples receive R/S predictions from each catalogue, enabling
    coverage comparison between catomatic and WHO catalogues.

    Parameters
    ----------
    cat1 : pandas.DataFrame
        First catalogue (e.g., catomatic) containing mutation rules and predictions.
    cat2 : pandas.DataFrame
        Second catalogue (e.g., WHO) containing mutation rules and predictions.
    mutations : pandas.DataFrame
        Mutation data filtered for the specified drug.
    phenotypes : pandas.DataFrame
        Phenotype data for the same set of samples.
    drug : str
        Drug name to analyze.
    model : list
        List of possible prediction values (e.g., ["R", "S", "U"]).
    who_cat : str or None, optional
        WHO catalogue version ("WHOv1", "WHOv2", or None). Enables special
        handling of WHO expert rule mappings.

    Returns
    -------
    dict
        Dictionary mapping catalogue names ("cat1_no_rules", "cat1_rules",
        "cat2_no_rules", "cat2_rules", "none") to lists of sample IDs with
        R/S predictions.
    """

    # note - dump synonymous mutations before calling

    cat1_no_rules = cat1[~cat1["MUTATION"].str.contains(r"[*?=]", regex=True)]
    cat1_rules_only = cat1[cat1["MUTATION"].str.contains(r"[*?=]", regex=True)]

    if who_cat == "WHOv1":

        w_garc_path = "catalogues/whov1/WHO-UCN-GTB-PCI-2021.7-eng_w_GARC.xlsx"
        # flag mutations that had expert rules applied to WHOv1
        if not os.path.exists(w_garc_path):
            utils.generate_garc_mutations(
                "catalogues/whov1/WHO-UCN-GTB-PCI-2021.7-eng.xlsx",
                "data/NC_000962.3.gbk",
            ).to_excel(w_garc_path)

        who_original = pd.read_excel(w_garc_path)

        filtered = who_original[~who_original["Additional grading criteria"].isna()]

        prediction_map = {
            "Assoc w R": "R",
            "Assoc w R - Interim": "R",
            "Uncertain significance": "U",
            "Not assoc w R": "S",
            "Not assoc w R - Interim": "S",
        }
        filtered["PREDICTION"] = filtered["INITIAL CONFIDENCE GRADING"].map(
            prediction_map
        )
        mapping = dict(zip(filtered["GARC_MUTATION"], filtered["PREDICTION"]))
        normalized_order = {
            "&".join(sorted(key.split("&"))): val for key, val in mapping.items()
        }

        def normalize_order(mutation_str):
            return "&".join(sorted(mutation_str.split("&")))

        cat2_no_rules = cat2[
            (~cat2["MUTATION"].str.contains(r"[*?=]", regex=True))
            & (~cat2["MUTATION"].str.contains("indel"))
        ]
        cat2_no_rules["PREDICTION"] = cat2_no_rules.apply(
            lambda row: normalized_order.get(
                normalize_order(row["MUTATION"]), row["PREDICTION"]
            ),
            axis=1,
        )

        cat2_rules_only = cat2[
            (cat2["MUTATION"].str.contains(r"[*?=]", regex=True))
            | (cat2["MUTATION"].isin(mapping.keys()))
        ]

    elif who_cat == "WHOv2":
        print("WHOv2 rule comparisons not yet supported, still need to implement")
    else:
        cat2_no_rules = cat2[~cat2["MUTATION"].str.contains(r"[*?=]", regex=True)]
        cat2_rules_only = cat2[cat2["MUTATION"].str.contains(r"[*?=]", regex=True)]

    row = {
        "GENBANK_REFERENCE": "NC00962.3",
        "CATALOGUE_NAME": "-",
        "CATALOGUE_VERSION": 0,
        "CATALOGUE_GRAMMAR": "GARC1",
        "PREDICTION_VALUES": model,
        "DRUG": None,
        "MUTATION": None,
        "PREDICTION": None,
        "SOURCE": {},
        "EVIDENCE": {},
        "OTHER": {},
    }

    expanded_coverage = {
        name: []
        for name in [
            "cat1_no_rules",
            "cat1_rules",
            "cat2_no_rules",
            "cat2_rules",
            "none",
        ]
    }

    # catalogues filtered by drug
    cat1_drug = cat1[(cat1.DRUG == drug)]
    cat2_drug = cat2[(cat2.DRUG == drug)]
    # catalogues with rules removed, filtered for drug
    cat1_no_rules_drug = cat1_no_rules[
        (cat1_no_rules.DRUG == drug) & (cat1_no_rules.PREDICTION.isin(["R", "S", "U"]))
    ]
    cat2_no_rules_drug = cat2_no_rules[
        (cat2_no_rules.DRUG == drug) & (cat2_no_rules.PREDICTION.isin(["R", "S", "U"]))
    ]
    # catalogue expert rules (not defaults) filtered for drug
    cat1_rules_drug = cat1_rules_only[
        (cat1_rules_only.DRUG == drug)
        & (cat1_rules_only.PREDICTION.isin(["R", "S", "U"]))
    ]
    cat2_rules_drug = cat2_rules_only[
        (cat2_rules_only.DRUG == drug)
        & (cat2_rules_only.PREDICTION.isin(["R", "S", "U"]))
    ]

    # add placeholder rules to catalogues so avoid piezo error
    for i in model:
        row["PREDICTION"] = i
        row["MUTATION"] = "placeholder@A1A"
        row["DRUG"] = drug
        cat2_rules_drug = pd.concat(
            [cat2_rules_drug, pd.DataFrame([row])], ignore_index=True
        )
        cat1_rules_drug = pd.concat(
            [cat1_rules_drug, pd.DataFrame([row])], ignore_index=True
        )
        cat2_no_rules_drug = pd.concat(
            [cat2_no_rules_drug, pd.DataFrame([row])], ignore_index=True
        )
        cat1_no_rules_drug = pd.concat(
            [cat1_no_rules_drug, pd.DataFrame([row])], ignore_index=True
        )

    genes_cat_1 = set(cat1_drug["MUTATION"].apply(lambda x: x.split("@")[0]).tolist())
    genes_cat_2 = set(cat2_drug["MUTATION"].apply(lambda x: x.split("@")[0]).tolist())

    genes_source = genes_cat_1  # catomatic genes - as these are the ones that are tier 1 with at least 1 RAV

    # add a default wildcard U rule to rule catalogues (as were previously dumped)
    for gene in genes_source:
        for mut in [f"{gene}@*?", f"{gene}@-*?"]:
            row["PREDICTION"] = "U"
            row["MUTATION"] = mut
            row["DRUG"] = drug
            cat1_rules_drug = pd.concat(
                [cat1_rules_drug, pd.DataFrame([row])], ignore_index=True
            )
            cat1_no_rules_drug = pd.concat(
                [cat1_no_rules_drug, pd.DataFrame([row])], ignore_index=True
            )

    for gene in genes_source:
        for mut in [f"{gene}@*?", f"{gene}@-*?"]:
            row["PREDICTION"] = "U"
            row["MUTATION"] = mut
            row["DRUG"] = drug
            cat2_rules_drug = pd.concat(
                [cat2_rules_drug, pd.DataFrame([row.copy()])], ignore_index=True
            )
            cat2_no_rules_drug = pd.concat(
                [cat2_no_rules_drug, pd.DataFrame([row.copy()])], ignore_index=True
            )

    # write out rule catalogues so piezo can read them in and scan the other non-rule catalogue
    cat1_rules_drug["EVIDENCE"] = cat1_rules_drug["EVIDENCE"].apply(json.dumps)
    cat1_rules_drug.to_csv(f"./catalogues/temp/cat1_rules_only.csv")
    cat1_rules_piezo = piezo.ResistanceCatalogue(
        f"./catalogues/temp/cat1_rules_only.csv"
    )
    cat2_rules_drug["EVIDENCE"] = cat2_rules_drug["EVIDENCE"].apply(json.dumps)
    cat2_rules_drug.to_csv(f"./catalogues/temp/cat2_rules_only.csv")
    cat2_rules_piezo = piezo.ResistanceCatalogue(
        f"./catalogues/temp/cat2_rules_only.csv"
    )

    # write out non-rule catalogues so piezo can read them in and scan the other non-rule catalogue
    cat1_no_rules_drug["EVIDENCE"] = cat1_no_rules_drug["EVIDENCE"].apply(json.dumps)
    cat1_no_rules_drug.to_csv(f"./catalogues/temp/cat1_no_rules_only.csv")
    cat1_no_rules_piezo = piezo.ResistanceCatalogue(
        f"./catalogues/temp/cat1_no_rules_only.csv"
    )
    cat2_no_rules_drug["EVIDENCE"] = cat2_no_rules_drug["EVIDENCE"].apply(json.dumps)
    cat2_no_rules_drug.to_csv(f"./catalogues/temp/cat2_no_rules_only.csv")
    cat2_no_rules_piezo = piezo.ResistanceCatalogue(
        f"./catalogues/temp/cat2_no_rules_only.csv"
    )

    phenotypes.set_index("UNIQUEID", inplace=True)
    mutations.set_index("UNIQUEID", inplace=True)
    all_data = phenotypes.join(mutations, how="inner")
    all_data.reset_index(inplace=True)

    catalogues = {
        "cat1_no_rules": cat1_no_rules_piezo,
        "cat1_rules": cat1_rules_piezo,
        "cat2_no_rules": cat2_no_rules_piezo,
        "cat2_rules": cat2_rules_piezo,
    }

    for id_ in all_data.UNIQUEID.unique():
        df = all_data[all_data.UNIQUEID == id_]

        made_prediction = False  # track if any catalogue gave R or S

        for name, catalogue in catalogues.items():
            mut_predictions = []

            for var in df["MUTATION"]:
                if pd.isna(var):
                    predict = "S"
                else:
                    try:
                        predict = catalogue.predict(var)
                    except ValueError:
                        predict = "U"

                if isinstance(predict, dict):
                    if drug in predict:
                        mut_predictions.append(predict[drug])
                else:
                    mut_predictions.append(predict)

            # Collapse to sample-level: R > U > S
            if "R" in mut_predictions:
                final_pred = "R"
            elif "U" in mut_predictions:
                final_pred = "U"
            else:
                final_pred = "S"

            if final_pred in ("R", "S"):
                expanded_coverage[name].append(id_)
                made_prediction = True

        # if no catalogue could make an R or S call, add to "none"
        if not made_prediction:
            expanded_coverage["none"].append(id_)

    return expanded_coverage
