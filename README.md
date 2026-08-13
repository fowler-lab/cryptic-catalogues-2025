# Rapidly and reproducibly building a comprehensive catalogue of resistance-associated variants for M. tuberculosis

This repository downloads and reproduces the data in https://doi.org/10.1101/2025.10.02.679941

## Environment setup

To run the code in this repository, first create and activate the Conda environment defined in `env.yml`:

```bash
# Create the environment from the YAML file
conda env create -f env.yml

# Activate the environment
conda activate cryptic-catalogues-25
```

## Data setup

This project uses version 3.4.0 of the CRyPTIC project, which contains large `.parquet` files hosted on Zenodo.

Run the following once to download the necessary files:

```bash
python data_setup.py
```

This will take a few minutes.

## Notebooks

Results are generated in Jupyter notebooks, numbered 01 - 05.

They can either be run from scratch (which will take a few hours) or they can read in pre-generated data (instant) to produces the plots. Just set the flag at the start of each notebook. Using the data above, either option will give the same results.
