import pandas as pd
from protocols import utils
import os


def prep_phenotypes(
    drug, pheno_path, genomes_path, samples_path, version, validation=False
):

    # import phenotypes and samples and genoms

    phenotypes = utils.read_data(pheno_path).reset_index()
    genomes = utils.read_data(genomes_path).reset_index()
    samples = utils.read_data(samples_path).reset_index()

    # streptomycin has the wrong drug code in the phenotypes table
    if drug == "STM":
        phenotypes.replace({"DRUG": {"STR": "STM"}}, inplace=True)

    assert version in ["v1.0", "v2.0", "v3.0"]

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

    elif version == "v3.0":

        if validation:

            matched = samples[
                (samples.in_final_tables) & (samples.dataset == "CRyPTIC-v3.0")
            ].UNIQUEID.unique()
            phenotypes = phenotypes[phenotypes.UNIQUEID.isin(matched)]

        else:

            matched = samples[(samples.in_final_tables)].UNIQUEID.unique()
            phenotypes = phenotypes[phenotypes.UNIQUEID.isin(matched)]

    # filter for drug
    phenotypes = phenotypes[phenotypes.DRUG == drug]

    # discard low quality phenotypes
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

    # Uses most up to date mutations table, as this should contain all genomes from previous versions
    mut_dir = f"{path}{'_'.join(genes)}_MUTATIONS.csv"
    var_dir = f"{path}{'_'.join(genes)}_VARIANTS.csv"

    if not os.path.exists(mut_dir):
        print("not exist")
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

    elif version == "v3.0.0":

        variants["FRS"] = variants.apply(
            lambda row: (
                row["MINOR_READS"] / row["COVERAGE"] if row["MINOR_READS"] > 0 else 1
            ),
            axis=1,
        )
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

    elif version == "v3.1.0":

        # recalcaulte FRS from variants table to be 100% sure
        variants["FRS"] = variants.apply(
            lambda row: (
                row["MINOR_READS"] / row["COVERAGE"] if row["MINOR_READS"] > 0 else 1
            ),
            axis=1,
        )
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

    # drop duplicate entries
    mutations = mutations.drop_duplicates(["UNIQUEID", "MUTATION", "FRS"], keep="first")

    # filter relevant columns for catomatic and rename id column
    mutations = mutations[["UNIQUEID", "MUTATION", "FRS"]].rename(
        columns={"ENA_RUN": "UNIQUEID"}
    )
    # mutations.set_index("UNIQUEID", inplace=True)

    return mutations
