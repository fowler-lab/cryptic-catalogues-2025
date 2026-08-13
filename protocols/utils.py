import json
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.proportion import proportions_ztest
import warnings
import tempfile
import gumpy

warnings.simplefilter(action="ignore", category=FutureWarning)
from scipy.stats import norm
import multiprocessing as mp


def read_data(file_path):
    """
    Reads .pkl, .pkl.gz, .csv, .csv.gz, or .parquet files
    """
    file_path = Path(file_path)
    ext = "".join(file_path.suffixes).lower()

    read_funcs = {
        ".pkl": pd.read_pickle,
        ".pkl.gz": pd.read_pickle,
        ".csv": lambda f: pd.read_csv(f, low_memory=False),
        ".csv.gz": lambda f: pd.read_csv(f, low_memory=False),
        ".parquet": pd.read_parquet,
    }

    if ext in read_funcs:
        return read_funcs[ext](file_path)

    raise ValueError(f"Unsupported file type: {ext}")


def confusion_matrix(labels, predictions, classes):
    """
    Creates a confusion matrix for given labels and predictions with specified classes.

    Parameters:
    labels (list): Actual labels.
    predictions (list): Predicted labels.
    classes (list): List of all classes.

    Returns:
    np.ndarray: Confusion matrix.
    """
    cm = np.zeros((len(classes) - 1, len(classes)), dtype=int)
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    for label, prediction in zip(labels, predictions):
        if label in class_to_index and prediction in class_to_index:
            cm[class_to_index[label], class_to_index[prediction]] += 1

    return cm


def filter_multiple_phenos(phenotypes):
    if phenotypes.empty:
        return phenotypes

    priority = {"R": 0, "S": 1, "U": 2}

    out = phenotypes.copy()
    out["_prio"] = out["PHENOTYPE"].map(priority).fillna(99)

    key = ["UNIQUEID", "DRUG"]

    # keep rows with the best phenotype within each UNIQUEID+DRUG
    best_prio = out.groupby(key)["_prio"].transform("min")
    out = out[out["_prio"].eq(best_prio)]

    # within that phenotype, prefer METHOD_MIC present
    out["_has_mic"] = out["METHOD_MIC"].notna().astype(int)
    out = out.sort_values(
        by=key + ["_has_mic"],
        ascending=[True, True, False],
        kind="mergesort",
    )

    out = out.drop_duplicates(subset=key, keep="first")

    return out.drop(columns=["_prio", "_has_mic"])


def flatten_grid_results(grid):
    """Flattens parameter search grid results into a DataFrame"""
    return pd.DataFrame(
        [
            {
                "DRUG": drug,
                "BACKGROUND_RATE": background_rate,
                "conf_level": conf_level,
                "SENSITIVITY": metrics.get("sens"),
                "SPECIFICITY": metrics.get("spec"),
                "DPR": metrics.get("dpr"),
                "SENSITIVITY2": metrics.get("sens2"),
                "SPECIFICITY2": metrics.get("spec2"),
            }
            for (drug, background_rate, conf_level), metrics in grid.items()
        ]
    )


def str_to_dict(val):
    if isinstance(val, str):  # Only convert if it's a valid string
        return json.loads(val)
    return val  # Keep as is (e.g., NaN values)


def load_catomatic_catalogue(drug, background, p, frs, dir):
    """loads catomatic atalogue, coverts evidence to dict, and removes default rows"""
    cat = pd.read_csv(
        f"{dir}{drug.lower()}/bg_{background}_p_{p}_FRS_{frs}.csv", index_col=0
    )
    cat["CATALOGUE_VERSION"], cat["CATALOGUE_NAME"] = 0, "-"
    cat["EVIDENCE"] = cat["EVIDENCE"].apply(str_to_dict)
    return cat[
        ~cat["EVIDENCE"].apply(lambda x: isinstance(x, dict) and "default_rule" in x)
    ]


def extract_value(value, *keys):
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, IndexError, TypeError):
        return None


def abs_err_to_rel(values, errors):
    """Convert absolute errors to relative errors, ensuring valid shapes and bounds."""
    errors = [np.ravel(e) for e in errors]  # Ensures everything is 1D
    # Stack into a 2D array
    errors = np.array(errors).T
    if errors.shape[0] != 2:
        errors = errors.reshape(2, -1)
    return (
        np.vstack(
            [
                np.maximum(0, values - errors[0, :]),  # Lower error
                np.minimum(1, errors[1, :] - values),  # Upper error
            ]
        )
        if values.size > 0 and errors.size > 0
        else np.array([])
    )


def wilson_ci(row, R_col, S_col):
    """
    Calculates the Wilson confidence interval for a given row based on the specified column names
    for the counts of successes (R) and failures (S).

    Parameters:
        row (pd.Series): The row of the DataFrame.
        R_col (str): The name of the column for successes (R).
        S_col (str): The name of the column for failures (S).

    Returns:
        list: A list containing the lower and upper bounds of the Wilson confidence interval.
    """
    z = norm.ppf(0.975)  # 95% confidence level (z = 1.96)

    # Extract the values for R and S based on the provided column names
    R = row[R_col]
    S = row[S_col]
    if np.isnan(R) or np.isnan(S) or (R + S == 0):
        return [np.nan, np.nan]
    n = R + S
    p_hat = R / n if n > 0 else 0

    # Wilson confidence interval calculation
    denominator = 1 + z**2 / n
    center_adjusted = p_hat + z**2 / (2 * n)

    margin_of_error = z * np.sqrt((p_hat * (1 - p_hat) / n) + z**2 / (4 * n**2))

    lower_bound = (center_adjusted - margin_of_error) / denominator
    upper_bound = (center_adjusted + margin_of_error) / denominator

    return [abs(lower_bound), abs(upper_bound)]


def compute_metric_differences(results_df, pvals_df):
    """
    Compute metric differences between pairs of catalogues for each drug.

    Calculates the difference in key performance metrics between two
    catalogues specified in pvals_df for every drug in results_df.

    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame containing performance metrics by drug and catalogue.
    pvals_df : pandas.DataFrame
        DataFrame specifying catalogue comparisons with columns 'DRUG' and 'Comparison'.

    Returns
    -------
    pandas.DataFrame
        DataFrame with metric differences for each drug and comparison.
    """
    metrics = ["SENSITIVITY", "SPECIFICITY", "COVERAGE", "SENSITIVITY2", "SPECIFICITY2"]
    diffs = []

    for _, row in pvals_df.iterrows():
        drug = row["DRUG"]
        cat1, cat2 = row["Comparison"].split(" > ")

        r1 = results_df[
            (results_df["DRUG"] == drug) & (results_df["catalogue"] == cat1)
        ].iloc[0]
        r2 = results_df[
            (results_df["DRUG"] == drug) & (results_df["catalogue"] == cat2)
        ].iloc[0]

        diff_row = {"DRUG": drug, "Comparison": row["Comparison"]}
        for metric in metrics:
            diff_row[f"{metric}_diff"] = r1[metric] - r2[metric]

        diffs.append(diff_row)

    return pd.DataFrame(diffs)


def generate_pvals(df, comparisons):
    """
    Compute p-values for performance metric differences between catalogue pairs.

    Performs two-sided proportion Z-tests for sensitivity, specificity, and related
    metrics across specified catalogue comparisons for each drug.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing counts (TP, TN, FP, FN, UP, UN) by drug and catalogue.
    comparisons : list of tuple
        List of catalogue pairs to compare, e.g.
        [('catomatic_v3.4.0', 'WHOv1'), ('WHOv1', 'WHOv2')].

    Returns
    -------
    pandas.DataFrame
        DataFrame of p-values for each metric and drug comparison.
    """

    assert comparisons is not None, "comparison keys must be spcified"

    metric_funcs = {
        "sens_p_value": lambda r: (r["TP"], r["TP"] + r["FN"]),
        "spec_p_value": lambda r: (r["TN"], r["TN"] + r["FP"]),
        "dpr_p_value": lambda r: (
            r["TP"] + r["TN"] + r["FP"] + r["FN"],
            r["TP"] + r["TN"] + r["FP"] + r["FN"] + r["UP"] + r["UN"],
        ),
        "sens2_p_value": lambda r: (r["TP"], r["TP"] + r["FN"] + r["UP"]),
        "spec2_p_value": lambda r: (r["TN"] + r["UN"], r["TN"] + r["FP"] + r["UN"]),
    }
    results = []
    # Loop over each drug and comparison
    for drug in df["DRUG"].unique():
        df_drug = df[df["DRUG"] == drug]
        for cat1, cat2 in comparisons:
            row = {"DRUG": drug, "Comparison": f"{cat1} > {cat2}"}
            g1 = df_drug[df_drug["catalogue"] == cat1].iloc[0]
            g2 = df_drug[df_drug["catalogue"] == cat2].iloc[0]
            for col_name, fn in metric_funcs.items():
                c1, n1 = fn(g1)
                c2, n2 = fn(g2)
                stat, pval = proportions_ztest(
                    [c1, c2], [n1, n2], alternative="two-sided"
                )
                row[col_name] = pval
            results.append(row)

    return pd.DataFrame(results)


# Globals to be initialized in each process
global_reference = None
global_ref_genes = None
global_genome_indices = None


def initializer(genome_path, genome_indices_df):
    """
    Initialize global genome reference objects for parallel mutation parsing.

    Loads the genome into memory and builds gene-level reference objects
    for use by worker processes.

    Parameters
    ----------
    genome_path : str or Path
        Path to the genome reference file (e.g., GenBank format).
    genome_indices_df : pandas.DataFrame
        DataFrame of genome indices used to map variant positions.
    """
    global global_reference, global_ref_genes, global_genome_indices
    global_reference = gumpy.Genome(genome_path)
    ref_gene_dict = list(global_reference.genes)
    global_ref_genes = {
        gene: global_reference.build_gene(gene) for gene in ref_gene_dict
    }
    global_genome_indices = genome_indices_df


def get_normalized_garc(gene_name: str, pos: int, ref: str, alt: str):
    """
    Generate a normalized GARC mutation string for a single variant.

    Builds a temporary VCF for the variant, compares it to the reference
    genome, and returns mutation(s) in standardized GARC format.

    Parameters
    ----------
    gene_name : str
        Gene name associated with the mutation.
    pos : int
        Genomic position of the variant.
    ref : str
        Reference base(s).
    alt : str
        Alternate base(s).

    Returns
    -------
    str
        Normalized GARC mutation string (e.g., "katG@S315T" or multiple joined by '&').
    """
    vcf_str = (
        "##fileformat=VCFv4.2\n"
        "##source=garcattempt\n"
        "##contig=<ID=NC_000962.3>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample\n"
        f"NC_000962.3\t{pos}\t.\t{ref.upper()}\t{alt.upper()}\t.\tPASS\t.\tGT\t1/1\n"
    )

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".vcf") as f:
        f.write(vcf_str)
        temp_vcf_path = f.name

    vcf = gumpy.VCFFile(temp_vcf_path)
    # vcf = gumpy.VCFFile(io.StringIO(vcf_str))
    sample = global_reference + vcf

    ref_gene = global_ref_genes[gene_name]
    alt_gene = sample.build_gene(gene_name)

    diff = ref_gene - alt_gene
    mutations = [gene_name + "@" + mut for mut in diff.mutations.tolist()]

    return "&".join(mutations)


def parse_mutation_worker(args):
    """
    Worker function to normalize a mutation entry for parallel processing.

    Parses mutation information, retrieves genome position, and converts
    insertion/deletion variants to normalized GARC format. Returns an
    error string if normalization fails.

    Parameters
    ----------
    args : tuple
        Tuple containing (index, mutation, variant_name).

    Returns
    -------
    tuple
        (index, normalized_mutation) or (index, "ERROR@...") if failed.
    """
    index, mutation, variant_name = args
    if ("ins" in mutation or "del" in mutation or "-" in mutation) and (
        "&" not in mutation
    ):
        try:
            gene = mutation.split("@")[0]
            pos = int(
                global_genome_indices[
                    global_genome_indices.variant == variant_name
                ].genome_index.values[0]
            )
            ref = mutation.split("@")[1].split("_")[-2]
            alt = mutation.split("@")[1].split("_")[-1]
            normalised = get_normalized_garc(gene_name=gene, pos=pos, ref=ref, alt=alt)
            return index, normalised
        except Exception as e:
            return index, f"ERROR@{str(e)}"
    else:
        return index, mutation


def generate_garc_mutations(catalogue_path, genome_path):
    """
    Generate GARC-formatted mutation identifiers from a WHO mutation catalogue.

    Loads mutation and genome index data, processes mutation names into
    standardized GARC format, and uses parallel processing to translate
    mutations based on genomic context.

    Parameters
    ----------
    catalogue_path : str or Path
        Path to the WHO mutation catalogue Excel file.
    genome_path : str or Path
        Path to the genome reference file used for mutation translation.

    Returns
    -------
    pandas.DataFrame
        DataFrame of mutations with an added 'GARC_MUTATION' column.
    """
    # Load data
    who_original = pd.read_excel(
        catalogue_path, sheet_name="Mutation_catalogue"
    ).reset_index()
    genome_indices = pd.read_excel(
        catalogue_path, sheet_name="Genome_indices"
    ).reset_index()

    who_original["variant (common_name)"] = who_original[
        "variant (common_name)"
    ].str.replace(r"\s*\(.*?\)", "", regex=True)

    # Preprocess mutations
    who_original["Mutation"] = (
        who_original["variant (common_name)"]
        .str.replace("_", "@", n=1)
        .str.replace(" ", "&")
    )
    who_original.dropna(subset=["drug"], inplace=True)

    # Prepare data for workers
    args_list = [
        (idx, row["Mutation"], row["variant (common_name)"])
        for idx, row in who_original.iterrows()
    ]

    # Parallel processing
    ctx = mp.get_context("fork")  # "fork" preferred on Unix for memory efficiency
    with ctx.Pool(
        mp.cpu_count(), initializer, (genome_path, genome_indices)
    ) as pool:  # dont use max cores, avoid thrashing
        results = pool.map(parse_mutation_worker, args_list)

    # Reintegrate results
    translated_dict = dict(results)
    translated_vars = [
        translated_dict.get(i + 1, None) for i in range(len(who_original))
    ]
    who_original["GARC_MUTATION"] = translated_vars

    return who_original


def _counts_for(drug, expanded_catalogues, use_filtered):
    """
    Compute counts of unique, shared, and exclusive mutations for a given drug.

    Returns the number of mutations found only in the catomatic catalogue,
    only in the WHO catalogue, and in both, optionally excluding rule-based
    or uncertain entries.

    Parameters
    ----------
    drug : str
        Drug name to analyze.
    expanded_catalogues : dict
        Dictionary of expanded catalogues for each drug, containing
        'cat', 'who', and 'merged' DataFrames.
    use_filtered : bool
        If True, exclude rule-based, uncertain (U), and indel mutations.

    Returns
    -------
    tuple of int
        (only_cat, only_who, shared) mutation counts.
    """
    if use_filtered:
        # RULES EXCLUDED
        cat_filtered = expanded_catalogues[drug]["cat"]
        cat_filtered = cat_filtered[
            (~cat_filtered.MUTATION.str.contains(r"[?*=]", regex=True, na=False))
            & (cat_filtered.EVIDENCE != {"expanded_rule"})
        ]
        cat_filtered = cat_filtered[
            (cat_filtered.PREDICTION != "U")
            & (~cat_filtered.MUTATION.str.contains("indel"))
        ]
        mutations_cat = set(cat_filtered["MUTATION"])

        who_filtered = expanded_catalogues[drug]["who"]
        who_filtered = who_filtered[
            (~who_filtered.MUTATION.str.contains(r"[?*=]", regex=True, na=False))
            & (who_filtered.EVIDENCE != {"expanded_rule"})
        ]
        who_filtered = who_filtered[
            (who_filtered.PREDICTION != "U")
            & (~who_filtered.MUTATION.str.contains("indel"))
        ]
        mutations_who = set(who_filtered["MUTATION"])

        only_cat = len(mutations_cat - mutations_who)
        only_who = len(mutations_who - mutations_cat)
        shared = len(mutations_cat & mutations_who)
    else:
        # RULES APPLIED
        merged = expanded_catalogues[drug]["merged"]
        shared = merged[
            (~merged.PREDICTION_who.isna()) & (~merged.PREDICTION_cat.isna())
        ].MUTATION.nunique()
        only_who = merged[
            (~merged.PREDICTION_who.isna()) & (merged.PREDICTION_cat.isna())
        ].MUTATION.nunique()
        only_cat = merged[
            (merged.PREDICTION_who.isna()) & (~merged.PREDICTION_cat.isna())
        ].MUTATION.nunique()

    return only_cat, only_who, shared


def _counts_for_ids(drug, pair_coverages):
    """
    Compute sample-level overlap counts between CatoMatic and WHO catalogues for a given drug.

    Returns the number of samples uniquely or jointly classified by each
    catalogue type (rules or algorithmic) and those with no predictions.

    Parameters
    ----------
    drug : str
        Drug name to analyze.
    pair_coverages : dict
        Dictionary of sample-level coverage results per drug, containing
        keys like 'cat1_no_rules', 'cat2_rules', 'cat2_no_rules', and 'none'.

    Returns
    -------
    tuple of int
        (cat_only, who_only_rules, who_only_excluded, both_rules,
         both_excluded, none)
    """
    if drug not in pair_coverages:
        return 0, 0, 0, 0, 0, 0

    d = pair_coverages[drug]

    # Ignore cat1_rules (always empty)
    cat = set(d.get("cat1_no_rules", []))

    # WHO separated into rules vs excl
    who_rules = set(d.get("cat2_rules", []))
    who_excl = set(d.get("cat2_no_rules", []))

    none = set(d.get("none", []))

    # intersections - ids can exist in both who_rules and who_excl buckets, but excl rules wins
    # as we are tyring to ask 'what do the rules actually add in practise'
    both_excluded = len(cat & who_excl)
    both_rules = len(cat & who_rules - (cat & who_excl))

    # unique parts
    cat_only = len(cat - (who_rules | who_excl))
    who_only_rules = len(who_rules - who_excl - cat)
    who_only_excl = len(who_excl - who_rules - cat)

    return cat_only, who_only_rules, who_only_excl, both_rules, both_excluded, len(none)


def classify_predictions(row, suffixes=("cat1", "cat2")):

    if not pd.isna(row[f"PREDICTION_{suffixes[1]}"]) and not pd.isna(
        row[f"PREDICTION_{suffixes[0]}"]
    ):
        return f"{row[f'PREDICTION_{suffixes[0]}']}.{row[f'PREDICTION_{suffixes[1]}']}"
    elif pd.isna(row[f"PREDICTION_{suffixes[1]}"]) and not pd.isna(
        row[f"PREDICTION_{suffixes[0]}"]
    ):
        return f"{row[f'PREDICTION_{suffixes[0]}']}.U"
    elif not pd.isna(row[f"PREDICTION_{suffixes[1]}"]) and pd.isna(
        row[f"PREDICTION_{suffixes[0]}"]
    ):
        return f"U.{row[f'PREDICTION_{suffixes[1]}']}"
    else:
        return "U.U"  # Default case if both are NaN


def _counts_cat_vs_cats_all(drug, expanded_catalogues):
    """
    Compute mutation overlap counts between catomatic-1 and catomatic-2 catalogues.

    Calculates the number of mutations unique to each catalogue and shared
    between them for a given drug, using merged prediction data.

    Parameters
    ----------
    drug : str
        Drug name to analyze.
    expanded_catalogues : dict
        Dictionary of expanded catalogues for each drug, containing
        a 'merged' DataFrame with 'PREDICTION_cat' and 'PREDICTION_cats_all' columns.

    Returns
    -------
    tuple of int
        (only_cat, only_cats_all, shared) mutation counts.
    """
    merged = expanded_catalogues[drug]["merged"]

    shared = merged[
        (~merged.PREDICTION_cat.isna()) & (~merged.PREDICTION_cats_all.isna())
    ].MUTATION.nunique()

    only_cat = merged[
        (~merged.PREDICTION_cat.isna()) & (merged.PREDICTION_cats_all.isna())
    ].MUTATION.nunique()

    only_cats_all = merged[
        (merged.PREDICTION_cat.isna()) & (~merged.PREDICTION_cats_all.isna())
    ].MUTATION.nunique()

    return only_cat, only_cats_all, shared
