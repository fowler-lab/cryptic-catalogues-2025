import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from protocols import utils
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.collections import PolyCollection
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.path import Path


plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = 7
plt.rcParams["figure.figsize"] = (6.69, 5.02)

colours = {
    "WHO1": ("#47d2f5", "#47d2f5", "#47d2f5"),
    "WHOv1": ("#3381B5", "#3381B5", "#3381B5"),
    "WHOv1_no_rules": ("#c872c5", "#c872c5", "#c872c5"),
    "WHOv2": ("#2ba74e", "#2ba74e", "#2ba74e"),
    "MTBC-CRyPTICv1.1.1-2025.8": ("#bd2f3f", "#bd2f3f", "#bd2f3f"),
    "MTBC-CRyPTICv3.4.0-2025.8": ("#fd8c53", "#fd8c53", "#fd8c53"),
}


plot_order = [
    "RIF",
    "INH",
    "EMB",
    "PZA",
    "LEV",
    "MXF",
    "BDQ",
    "CFZ",
    "LZD",
    "DLM",
    "AMI",
    "STM",
    "ETH",
    "KAN",
    "CAP",
]


def plot_pheno_counts(phenotypes, title, savefig):
    """
    Plot counts of resistant and susceptible phenotypes for each drug.

    Groups phenotype data by drug and phenotype, creates a bar plot of
    unique sample counts, and saves the figure.

    Parameters
    ----------
    phenotypes : pandas.DataFrame
        Phenotype data containing 'DRUG', 'PHENOTYPE', and 'UNIQUEID' columns.
    title : str
        Title used in console output.
    savefig : str or Path
        Path to save the resulting figure (e.g., 'plots/pheno_counts.png').
    """

    # Compute the count for each (DRUG, PHENOTYPE)
    barplot = (
        phenotypes.groupby(["DRUG", "PHENOTYPE"])["UNIQUEID"]
        .nunique()
        .reset_index()
        .rename(columns={"UNIQUEID": "count"})
    )

    # Compute total count per DRUG (sum of R, S, U)
    total_counts = barplot.groupby("DRUG")["count"].sum().reset_index()

    # Order DRUGs by total count descending
    # plot_order = total_counts.sort_values("count", ascending=False)["DRUG"].tolist()

    # Create the bar plot
    plt.figure(figsize=(6.69, 2.5))
    axis = sns.barplot(
        data=barplot,
        x="DRUG",
        y="count",
        hue="PHENOTYPE",
        hue_order=["S", "R"],
        order=[d for d in plot_order if d in barplot.DRUG.unique()],
        dodge=True,
        palette=["gainsboro", "dimgrey"],  # palette=["#034e7b", "#990000"],
        alpha=0.95,
    )

    axis.set_ylabel("")
    axis.set_xlabel("")
    # axis.set_ylim([0, 29000])
    axis.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )

    # Customize the plot
    plt.xticks(rotation=0, fontsize=7)
    plt.yticks(fontsize=7)
    # plt.ylabel("# unique samples", fontsize=7)
    # plt.xlabel("Drug", fontsize=7)
    print(f"{title}: {phenotypes.UNIQUEID.nunique()} samples")
    plt.legend(title="Phenotype")
    sns.set_theme(style="whitegrid")

    # Annotate bars with values
    for p in plt.gca().patches:
        if p.get_height() > 0:  # Only label non-zero bars
            plt.text(
                p.get_x() + p.get_width() / 2,
                p.get_height() + 0.5,
                f"{int(p.get_height()):,d}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    axis.legend().set_visible(False)
    # plt.legend(frameon=False, fontsize=7)
    plt.grid(False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(savefig, transparent=True, bbox_inches="tight")
    # Show the plot
    plt.show()


def plot_truthtables(truth_table, U_to_S=False, fontsize=10, colors=None, save=None):
    """
    Plots a truth table as a confusion matrix to denote each cell with perfect squares or proportional rectangles.

    Parameters:
    truth_table (pd.DataFrame): DataFrame containing the truth table values.
                                The DataFrame should have the following structure:
                                - Rows: True labels ("R" and "S")
                                - Columns: Predicted labels ("R", "S", and optionally "U")
    U_to_S (bool): Whether to separate the "U" values from the "S" column. If True,
                   an additional column for "U" values will be used.
    fontsize (int): Font size for the text in the plot.
    colors (list): List of four colors for the squares.
                   Defaults to red and green for the diagonal, pink and green for the off-diagonal.

    Returns:
    None
    """

    # Default colors if none provided
    if colors is None:
        if U_to_S:
            colors = ["#e41a1c", "#4daf4a", "#fc9272", "#4daf4a"]
        else:
            colors = ["#e41a1c", "#4daf4a", "#fc9272", "#4daf4a", "#4daf4a", "#4daf4a"]

    # Determine the number of columns for U_to_S condition
    num_columns = 3 if not U_to_S else 2
    num_rows = 2

    # Adjust the figure size to ensure square cells
    figsize = (
        (num_columns / 1.8, num_rows / 1.8)
        if num_columns == 2
        else (num_columns * 1.5 / 1.8, num_rows / 1.8)
    )

    fig = plt.figure(figsize=figsize)
    axes = plt.gca()

    if not U_to_S:
        assert (
            len(colors) == 6
        ), "The length of supplied colors must be 6, one for each cell"
        axes.add_patch(Rectangle((2, 0), 1, 1, fc=colors[4], alpha=0.5))
        axes.add_patch(Rectangle((2, 1), 1, 1, fc=colors[5], alpha=0.5))

        axes.set_xlim([0, 3])
        axes.set_xticks([0.5, 1.5, 2.5])
        axes.set_xticklabels(["S", "R", "U"], fontsize=9)
    else:
        assert (
            len(colors) == 4
        ), "The length of supplied colors must be 4, one for each cell"
        axes.set_xlim([0, 2])
        axes.set_xticks([0.5, 1.5])
        axes.set_xticklabels(["S+U", "R"], fontsize=9)

    # Apply provided colors for the squares
    axes.add_patch(Rectangle((0, 0), 1, 1, fc=colors[0], alpha=0.8))
    axes.add_patch(Rectangle((1, 0), 1, 1, fc=colors[1], alpha=0.8))
    axes.add_patch(Rectangle((1, 1), 1, 1, fc=colors[2], alpha=0.8))
    axes.add_patch(Rectangle((0, 1), 1, 1, fc=colors[3], alpha=0.8))

    axes.set_ylim([0, 2])
    axes.set_yticks([0.5, 1.5])
    axes.set_yticklabels(["R", "S"], fontsize=9)

    # Add text to the plot
    axes.text(
        1.5,
        0.5,
        int(truth_table["R"]["R"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    axes.text(
        1.5,
        1.5,
        int(truth_table["R"]["S"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    axes.text(
        0.5,
        1.5,
        int(truth_table["S"]["S"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    axes.text(
        0.5,
        0.5,
        int(truth_table["S"]["R"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )

    if not U_to_S:
        axes.text(
            2.5,
            0.5,
            int(truth_table["U"]["R"]),
            ha="center",
            va="center",
            fontsize=fontsize,
        )
        axes.text(
            2.5,
            1.5,
            int(truth_table["U"]["S"]),
            ha="center",
            va="center",
            fontsize=fontsize,
        )

    axes.set_aspect("equal")  # Ensure squares remain squares

    if save != None:
        plt.savefig(save, format="pdf", bbox_inches="tight", transparent=True)

    plt.show()


def grid_search_plots(df):
    """
    Plot grid search performance metrics for each drug.

    Creates line plots of sensitivity, specificity, and DPR
    across background rates and p-value thresholds, saving one
    figure per drug.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing columns 'DRUG', 'Metric', 'conf_level',
        'BACKGROUND_RATE', and 'value'.
    """
    metric_colors = {
        "SENSITIVITY": "#8B0000",  # dark red
        "SPECIFICITY": "#4682B4",  # steel blue
        "DPR": "#2E8B57",  # sea green
    }
    for drug in df["DRUG"].unique():
        drug_data = df[df["DRUG"] == drug]

        plt.figure(figsize=(2, 1.5))

        for metric in drug_data["Metric"].unique():
            for p in drug_data["conf_level"].unique():
                subset = drug_data[
                    (drug_data["Metric"] == metric) & (drug_data["conf_level"] == p)
                ]
                sns.lineplot(
                    data=subset,
                    x="BACKGROUND_RATE",
                    y="value",
                    color=metric_colors[metric],
                    alpha=(
                        0.5 if p == 0.90 else 1.0
                    ),  # Example: faded for 0.90, solid for 0.95
                    marker="o",
                    linestyle="--" if p == 0.90 else "-",  # Dashed for 0.90
                    markersize=3,
                    linewidth=1.7 if p == 0.9 else 1.2,
                )

        plt.title(drug, fontsize=7)
        plt.xlabel("")
        plt.ylabel("")  # No y-label
        plt.ylim(0, 105)
        plt.tick_params(labelsize=7)
        sns.despine(top=True, right=True)
        plt.xticks(np.arange(0.05, 0.26, 0.05))

        plt.tight_layout()
        plt.savefig(f"figs/grid_search/{drug}.pdf", transparent=True)

        plt.show()
        plt.close()


def plot_frs_sens(df, figsize=(4.3, 2.2)):
    """
    Plot sensitivity versus build FRS for different test FRS values.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'Build_FRS', 'Test_FRS', and 'Sensitivity' columns.
    drug : str, optional
        Drug name used for labeling or selection (default is "AMI").
    figsize : tuple, optional
        Figure size in inches (default is (4.3, 2.2)).
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Get all unique test FRS values, sorted for consistent plotting
    test_vals = sorted(df["Test_FRS"].unique())

    # Generate alphas that fade from 1.0 → 0.5
    alphas = np.linspace(1.0, 0.4, len(test_vals))

    for test_val, alpha in zip(test_vals, alphas):
        group = df[df["Test_FRS"] == test_val].sort_values("Build_FRS")

        # Plot the curve
        ax.plot(
            group["Build_FRS"],
            group["Sensitivity"],
            marker="o",
            markersize=3,
            color="#990000",
            alpha=alpha,
        )

        # Annotate last point with Test_FRS value (rounded)
        x_last = group["Build_FRS"].iloc[-1]
        y_last = group["Sensitivity"].iloc[-1]
        offset = (df["Build_FRS"].max() - df["Build_FRS"].min()) * 0.04  # 2% of x-range
        ax.text(
            x_last + offset,
            y_last,
            f"{test_val:.1f}",
            fontsize=6,
            va="center",
            color="#990000",
            alpha=alpha,
        )

    # Axis styling
    ax.set_xlabel("Build FRS", fontsize=8)
    ax.set_ylabel("Sensitivity (%)", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.ylim(80, 89)

    plt.tight_layout()
    plt.show()


def plot_catalogue_bar_charts(perf_df):
    """
    Plot horizontal bar charts of catalogued R and S counts by Build FRS for each drug.

    Parameters
    ----------
    perf_df : pandas.DataFrame
        DataFrame containing 'Drug', 'Build_FRS', 'catalogued_R', and 'catalogued_S' columns.
    """

    metrics = ["catalogued_R", "catalogued_S"]
    titles = ["R rows", "S rows"]

    for drug in perf_df["Drug"].unique():
        drug_df = perf_df[perf_df["Drug"] == drug].copy()

        # Round Build_FRS to 2 decimals to avoid float noise
        drug_df["Build_FRS"] = drug_df["Build_FRS"].round(2)

        drug_df = drug_df.drop_duplicates(subset=["Build_FRS"]).sort_values("Build_FRS")

        fig, axes = plt.subplots(1, 2, figsize=(1.2, 2.4), sharey=True)

        combined_array = np.concatenate(
            [drug_df["catalogued_R"].to_numpy(), drug_df["catalogued_S"].to_numpy()]
        )
        min_val = min(combined_array)
        max_val = max(combined_array)

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            ax.barh(
                y=drug_df["Build_FRS"],
                width=drug_df[metric],
                color="white",
                edgecolor="black",
                linewidth=0.5,
                height=0.1,
                alpha=0.7,
            )

            ax.set_xlim(0, max_val + 1.5)

            # Hide x-axis completely
            ax.set_xlabel("")
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)

            # Add count numbers at end of bars
            for idx, (yval, width) in enumerate(
                zip(drug_df["Build_FRS"], drug_df[metric])
            ):
                ax.text(
                    width + 0.2,
                    yval,
                    f"{int(width)}",
                    va="center",
                    ha="left",
                    fontsize=6,
                    alpha=0.7,
                )
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_color((0, 0, 0, 0.5))
            ax.set_title(title, fontsize=7)
            ax.tick_params(labelsize=7)
            sns.despine(ax=ax, left=False, bottom=True)

            if i != 0:
                ax.set_yticks([])

        fig.suptitle(drug, fontsize=9)
        plt.tight_layout()
        plt.savefig(
            f"figs/frs/{drug}_catalogue_counts.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.show()
        plt.close()


def plot_perf_heatmaps(performance_df, draw_axes=True):
    """
    Plot performance heatmaps for each drug across Build FRS and Test FRS values.

    Parameters
    ----------
    performance_df : pandas.DataFrame
        DataFrame containing 'Drug', 'Build_FRS', 'Test_FRS',
        'Sensitivity', 'Specificity', and 'DPR' columns.
    draw_axes : bool, optional
        If True, display FRS axis labels and ticks (default is True).
    """

    # Define custom colormaps
    red_gray_cmap = mcolors.LinearSegmentedColormap.from_list(
        "red_gray", ["#D3D3D3", "#8B0000"]
    )  # Dark red → Light gray
    blue_gray_cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_gray", ["#D3D3D3", "#4682B4"]
    )  # Dark blue → Light gray
    green_gray_cmap = mcolors.LinearSegmentedColormap.from_list(
        "green_gray", ["#D3D3D3", "#2E8B57"]
    )  # Dark green → Light gray

    metrics = ["Sensitivity", "Specificity", "DPR"]
    colormaps = [red_gray_cmap, blue_gray_cmap, green_gray_cmap]
    colormaps = ["Reds", "Blues", "Greens"]
    for drug in performance_df["Drug"].unique():
        fig, axes = plt.subplots(1, 3, figsize=(6.69, 2))

        for i, (metric, cmap) in enumerate(zip(metrics, colormaps)):
            subset_df = performance_df[performance_df["Drug"] == drug].pivot(
                index="Build_FRS", columns="Test_FRS", values=metric
            )

            # Format annotations to remove scientific notation
            annot_values = subset_df.map(lambda x: f"{x:.0f}")

            ax = sns.heatmap(
                subset_df,
                annot=annot_values,
                fmt="",
                cmap=cmap,
                ax=axes[i],
                vmin=70,
                vmax=90,
                square=True,
                cbar=False,
            )

            ax.set_xlabel("")
            ax.set_ylabel("")
            # ax.set_ylabel("Build min FRS", fontsize=6)
            ax.invert_yaxis()
            ax.set_title(drug)

            if draw_axes:
                # Force normal notation on axis tick labels
                ax.xaxis.set_major_formatter(
                    ticker.FuncFormatter(
                        lambda x, _: (
                            f"{subset_df.columns[int(x)]:.2f}"
                            if int(x) < len(subset_df.columns)
                            else ""
                        )
                    )
                )
                ax.yaxis.set_major_formatter(
                    ticker.FuncFormatter(
                        lambda y, _: (
                            f"{subset_df.index[int(y)]:.2f}"
                            if int(y) < len(subset_df.index)
                            else ""
                        )
                    )
                )
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])

        plt.savefig(f"figs/frs/{drug}.pdf", bbox_inches="tight", transparent=True)
        plt.tight_layout()
        plt.show()


def plot_mutation_error_bars(
    frs_prop_data, color_map={}, min_err=1, label_cutoff=15, figpath=None
):
    """
    Plot error bars showing resistant proportions vs FRS for each mutation and drug.

    Parameters
    ----------
    frs_prop_data : dict
        Nested dictionary of mutation data by drug, containing FRS, proportion,
        error, background, and count values.
    color_map : dict, optional
        Optional mapping of mutation colors per drug (default is empty dict).
    min_err : float, optional
        Minimum allowed error difference for filtering (default is 1).
    label_cutoff : int, optional
        Maximum mutation label length before truncation (default is 15).
    figpath : str or Path, optional
        Directory path to save figures (default is None).
    """
    np.random.seed(2)

    for drug, mutations_dict in frs_prop_data.items():
        plt.figure(figsize=(6.6, 2.5))

        unique_mutations = list(mutations_dict.keys())
        num_mutations = len(unique_mutations)
        if len(color_map) == 0:
            mutation_colors = dict(
                zip(unique_mutations, sns.color_palette("tab20", num_mutations))
            )  # More distinct colors
        else:
            mutation_colors = color_map[drug]
        truncated_labels = {
            mutation: (
                mutation[:label_cutoff] + "..."
                if len(mutation) > label_cutoff
                else mutation
            )
            for mutation in unique_mutations
        }

        mutation_jitter = {
            mutation: np.random.uniform(-0.045, 0.045) for mutation in unique_mutations
        }

        for mutation, data in mutations_dict.items():

            x_values = np.array(data["frs"])  # FRS values
            y_values = np.array(data["y"])  # Proportion values
            y_errors = data["error"]  # Error bars
            jitter = mutation_jitter[
                mutation
            ]  # Use the fixed jitter value per mutation
            background = data["background"]
            y_errors = utils.abs_err_to_rel(y_values, y_errors)
            filt = np.abs(np.array(y_errors[0]) - np.array(y_errors[1])) <= min_err
            x_values = x_values[filt]
            y_values = y_values[filt]
            lower = np.maximum(np.array(y_errors[0])[filt], 0)
            upper = np.minimum(np.array(y_errors[1])[filt], 1)
            y_errors = (lower, upper)

            is_significant = np.any(np.array(data["count"])[filt] > 3)
            color = mutation_colors[mutation] if is_significant else "grey"
            alpha = 1 if is_significant else 0.2

            plt.errorbar(
                x=x_values + jitter,
                y=y_values,
                yerr=y_errors,
                fmt="o",
                capsize=0,
                linewidth=1.25,
                markersize=2.5,
                alpha=alpha,
                label=truncated_labels[mutation] if is_significant else None,
                color=color,
            )
        for i, start in enumerate(np.arange(0.05, 1.0, 0.1)):  # Iterate over x-ranges
            if i % 2 == 0:
                plt.axvspan(start, start + 0.1, color="lightgrey", alpha=0.3)
        plt.axhline(background, linewidth=1)
        plt.xlabel("min FRS (Binned, with Jitter)", fontsize=7)
        plt.ylabel("Proportion Resistant", fontsize=7)
        plt.title(f"{drug}", fontsize=7)
        plt.xticks(np.arange(0.1, 1.05, 0.1))  # Minor ticks at every 0.05
        plt.ylim(-0.05, 1.05)  # Keep proportions in range
        plt.grid(True, linestyle="--", alpha=0.5)
        num_cols = min(num_mutations, 6)
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            frameon=False,
            fontsize=6,
            ncol=num_cols,
        )
        plt.tight_layout()
        sns.despine()
        plt.grid(False)
        if figpath is not None:
            plt.savefig(f"{figpath}{drug}_frs_vs_prop.pdf")
        plt.show()


def split_FRS_essential_violins(data):
    """
    Plot violin plots of read support (FRS) for essential and non-essential genes.

    Compares read support distributions between essential, essential excluding embB,
    and non-essential genes, scaling violin widths by relative sample counts.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing 'Gene', 'Gene Category', and 'Read Support' columns.
    """
    outlier_gene = "embB"

    # Filter FRS < 0.9
    base = data[data["Read Support"] < 0.9].copy()

    # Three groups
    essential_all = base[base["Gene Category"] == "Essential"].copy()
    essential_all["Category3"] = "Essential"

    essential_no_embB = base[
        (base["Gene Category"] == "Essential") & (base["Gene"] != outlier_gene)
    ].copy()
    essential_no_embB["Category3"] = "Essential excl. embB"

    non_essential = base[base["Gene Category"] == "Non-Essential"].copy()
    non_essential["Category3"] = "Non-Essential"

    # Combine
    plot_data = pd.concat([essential_all, essential_no_embB, non_essential], axis=0)

    order = ["Essential", "Essential excl. embB", "Non-Essential"]

    # Totals for scaling + annotation
    total_counts = {
        "Essential": len(data[data["Gene Category"] == "Essential"]),
        "Essential excl. embB": len(
            data[
                (data["Gene Category"] == "Essential") & (data["Gene"] != outlier_gene)
            ]
        ),
        "Non-Essential": len(data[data["Gene Category"] == "Non-Essential"]),
    }
    shown_counts = plot_data["Category3"].value_counts().to_dict()
    scaling_factors = {
        k: shown_counts.get(k, 0) / total_counts[k]
        for k in total_counts
        if total_counts[k] > 0
    }
    max_scale = max(scaling_factors.values()) if scaling_factors else 1

    # === Plot ===
    fig, ax = plt.subplots(figsize=(4.25, 2))

    sns.violinplot(
        x="Category3",
        y="Read Support",
        data=plot_data,
        order=order,
        inner=None,
        cut=0,
        bw=0.2,
        scale="area",
        palette={
            "Essential": "#CFAFEF",
            "Essential excl. embB": "#AF7AC5",
            "Non-Essential": "#FFD1DC",
        },
        ax=ax,
    )

    # Scale violin widths
    category_positions = dict(zip(order, range(len(order))))
    for artist in ax.findobj(match=PolyCollection):
        path = artist.get_paths()[0]
        verts = path.vertices
        x_mean = np.mean(verts[:, 0])
        closest_cat = min(
            category_positions, key=lambda k: abs(category_positions[k] - x_mean)
        )
        scale = scaling_factors.get(closest_cat, 1) / max_scale
        verts[:, 0] = (verts[:, 0] - x_mean) * scale + x_mean
        artist.set_edgecolor("black")
        artist.set_linewidth(0.8)
        artist.set_alpha(0.8)

    # === Labels ===
    ax.set_xlabel("")
    ax.set_ylabel("Fraction Read Support", fontsize=7)
    ax.tick_params(axis="both", labelsize=7)
    sns.despine(ax=ax, top=True, right=True)

    plt.tight_layout()
    plt.show()


def frs_gene_violins(all_mutations):
    """
    Plot FRS distributions for each gene (FRS < 0.9) with violin widths
    proportional to the fraction of mutations shown. Annotates each gene
    with total and plotted counts.

    Parameters
    ----------
    all_mutations : pandas.DataFrame
        DataFrame containing 'GENE', 'FRS', and 'Category' columns.
    """

    x_axis_order = [
        "rpoB",
        "gyrA",
        "gyrB",
        "rplC",
        "dprE1",
        "atpE",
        "rpsL",
        "rrs",
        "embA",
        "embB",
        "eis",
        "tlyA",
        "pepQ",
        "inhA",
        "ethA",
        "ahpC",
        "katG",
        "ddn",
        "pncA",
        "Rv0678",
        "gid",
        "fabG1",
    ]

    # === STEP 1: Filter for FRS < 0.9 only ===
    plot_data = all_mutations[all_mutations["FRS"] < 0.9].copy()

    # === STEP 2: Compute fractions ===
    total_counts = all_mutations.groupby("GENE").size()
    shown_counts = plot_data.groupby("GENE").size()
    fractions = (shown_counts / total_counts).fillna(0)
    max_fraction = fractions.max()  # normalization reference

    plot_data["GENE"] = pd.Categorical(
        plot_data["GENE"], categories=x_axis_order, ordered=True
    )

    # === STEP 3: Build palette ===
    category_map = plot_data.set_index("GENE")["Category"].to_dict()
    full_category_map = {
        gene: category_map.get(gene, "Non-Essential") for gene in x_axis_order
    }
    palette = {
        gene: {"Essential": "#CFAFEF", "Non-Essential": "#FFD1DC"}.get(
            category, "#FFFFFF"
        )
        for gene, category in full_category_map.items()
    }

    # === STEP 4: Plot ===
    fig, ax = plt.subplots(figsize=(6.69, 2))

    sns.violinplot(
        x="GENE",
        y="FRS",
        data=plot_data,
        inner=None,
        cut=0,
        bw=0.2,
        scale="count",  # initial scaling, we'll override
        order=x_axis_order,
        ax=ax,
        palette=palette,
    )

    ax.set_ylabel("Fraction Read Support", fontsize=7)
    ax.set_xlabel("")
    ax.tick_params(axis="both", labelsize=7)
    sns.despine(ax=ax, top=True, right=True)

    # === STEP 5: Rescale violin areas ===
    category_positions = dict(
        zip(
            [t.get_text() for t in ax.get_xticklabels()],
            range(len(ax.get_xticklabels())),
        )
    )

    shrink_factor = 0.48  # try values like 0.6–0.8 until it looks good
    for artist in ax.findobj(match=PolyCollection):
        verts = artist.get_paths()[0].vertices
        path = Path(verts)

        # Compute polygon area (shoelace formula)
        poly = path.to_polygons()
        if not poly:  # sometimes empty
            continue
        poly = poly[0]  # take outer polygon
        current_area = 0.5 * np.abs(
            np.dot(poly[:, 0], np.roll(poly[:, 1], 1))
            - np.dot(poly[:, 1], np.roll(poly[:, 0], 1))
        )

        # Which gene this violin belongs to
        x_mean = np.mean(verts[:, 0])
        closest_gene = min(
            category_positions, key=lambda g: abs(category_positions[g] - x_mean)
        )

        frac = fractions.get(closest_gene, 0)
        if frac <= 0 or current_area == 0:
            continue

        # Desired area (relative to max_fraction)
        target_area = frac / max_fraction

        # Scale factor for width only
        scale_factor = np.sqrt(target_area / current_area)

        # Apply rescale around violin center
        verts[:, 0] = (verts[:, 0] - x_mean) * scale_factor * shrink_factor + x_mean

        artist.set_edgecolor("black")
        artist.set_linewidth(0.8)
        artist.set_alpha(0.8)

    # === STEP 6: Annotate ===
    for gene, i in category_positions.items():
        total = total_counts.get(gene, 0)
        plotted = shown_counts.get(gene, 0)
        frac = (plotted / total * 100) if total > 0 else 0

        label = f"{plotted}/{total}\n{frac:.0f}%"

        ax.text(
            i, 1.1 if i % 2 == 0 else 1.0, label, ha="center", va="center", fontsize=5.5
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_scatter_who(
    df,
    who_results,
    outfile,
    who_drugs,
    sensitivity="SENSITIVITY",
    specificity="SPECIFICITY",
    figsize=(6.5, 4.8),
    show_graph=False,
):
    """
    Plot scatter comparison of WHOv1 results between this study and external WHO data.

    Creates side-by-side scatter plots of sensitivity and specificity for selected drugs,
    comparing values from this study to published WHOv1 results.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing this study's performance metrics by drug and catalogue.
    who_results : pandas.DataFrame
        DataFrame containing external WHO results for comparison.
    outfile : str or Path
        Path to save the generated figure.
    who_drugs : list of str
        List of drugs to include in the plot.
    sensitivity : str, optional
        Column name for sensitivity metric (default is "SENSITIVITY").
    specificity : str, optional
        Column name for specificity metric (default is "SPECIFICITY").
    figsize : tuple, optional
        Figure size in inches (default is (6.5, 4.8)).
    show_graph : bool, optional
        If True, display the plot after saving (default is False).
    """

    # Drugs to plot in specified order
    ordered_drugs = [drug for drug in plot_order if drug in who_drugs]

    # Subset WHOv1 catalogues
    df_who1 = df[df.catalogue == "WHOv1"].copy()
    who1_results = who_results[who_results.catalogue == "WHO1"].copy()

    # Reindex for consistent order
    df_who1 = df_who1.set_index("DRUG").reindex(ordered_drugs).reset_index()
    who1_results = who1_results.set_index("drug").reindex(ordered_drugs).reset_index()

    # Set up 2 stacked subplots (Sensitivity, Specificity)
    fig, axes = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=figsize, constrained_layout=True
    )

    metrics = [(sensitivity, "Sensitivity (%)", 0), (specificity, "Specificity (%)", 1)]

    x_positions = np.arange(len(ordered_drugs))
    offsets = np.linspace(-0.2, 0.2, 2)  # consistent with catomatic
    marker_style = "_"

    for ax, (metric, ylabel, colour_idx) in zip(axes, metrics):
        # WHOv1 (external WHO report)
        ax.scatter(
            x_positions + offsets[0],
            (
                who1_results[metric.lower()]
                if metric.lower() in who1_results.columns
                else who1_results[metric]
            ),
            marker=marker_style,
            color=colours["WHO1"][colour_idx],
            s=120,
            linewidth=3,
            label="WHOv1",
        )

        # WHOv1 (this study)
        ax.scatter(
            x_positions + offsets[1],
            df_who1[metric] * 100,
            marker=marker_style,
            color=colours["WHOv1"][colour_idx],
            s=120,
            linewidth=3,
            label="WHOv1 (this study)",
        )

        # Formatting
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, 105)
        ax.tick_params(axis="both", labelsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

        # Faint vertical separators
        for xpos in x_positions[:-1]:
            ax.axvline(
                x=xpos + 0.5, color="lightgrey", linestyle="-", linewidth=0.3, alpha=0.5
            )

    # Remove xticks completely
    axes[-1].set_xticks([])
    axes[-1].set_xticklabels([])

    # Add drug labels manually under bottom panel
    for xpos, drug in enumerate(ordered_drugs):
        axes[-1].text(xpos, -8, drug, ha="center", va="top", fontsize=9)

    # Add legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="upper center", ncol=2)

    # Fix x-axis range
    for ax in axes:
        ax.set_xlim(-0.5, len(ordered_drugs) - 0.5)

    # Save
    fig.savefig(outfile, bbox_inches="tight", dpi=600, transparent=True)
    if show_graph:
        plt.show()
    plt.close()


def plot_scatter_catomatic(
    df,
    outfile,
    who_drugs,
    catalogues="all",
    sensitivity="SENSITIVITY",
    specificity="SPECIFICITY",
    dpr="COVERAGE",
    figsize=(6.5, 6.5),
    show_graph=False,
):
    """ "
    Plot scatter comparisons of CatoMatic catalogue performance metrics by drug.

    Creates stacked scatter plots showing DPR, sensitivity, and specificity
    across multiple catalogues for selected drugs, including optional confidence
    intervals when available.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing performance metrics by drug and catalogue.
    outfile : str or Path
        Path to save the generated figure.
    who_drugs : list of str
        List of drugs to include in the plot.
    catalogues : list or str, optional
        Catalogues to include; use "all" to plot all available (default "all").
    sensitivity : str, optional
        Column name for sensitivity metric (default "SENSITIVITY").
    specificity : str, optional
        Column name for specificity metric (default "SPECIFICITY").
    dpr : str, optional
        Column name for coverage/DPR metric (default "COVERAGE").
    figsize : tuple, optional
        Figure size in inches (default (6.5, 6.5)).
    show_graph : bool, optional
        If True, display the plot after saving (default False).
    """
    # Subset drugs in specified order
    ordered_drugs = [drug for drug in plot_order if drug in who_drugs]

    if catalogues == "all":
        catalogues = df.catalogue.unique()

    # Subset dataframe to only selected catalogues
    df = df[df.catalogue.isin(catalogues)].copy()

    # Create mapping catalogue -> dataframe reindexed to drug order
    df_by_cat = {
        cat: df[df.catalogue == cat]
        .set_index("DRUG")
        .reindex(ordered_drugs)
        .reset_index()
        for cat in catalogues
    }

    # Set up 3 stacked subplots (DPR, Sensitivity, Specificity)
    fig, axes = plt.subplots(
        nrows=3, ncols=1, sharex=True, figsize=figsize, constrained_layout=True
    )

    metrics = [
        (dpr, "DPR (%)", 2),
        (sensitivity, "Sensitivity (%)", 0),
        (specificity, "Specificity (%)", 1),
    ]

    marker_style = "_"
    offsets = np.linspace(-0.2, 0.2, len(catalogues))
    x_positions = np.arange(len(ordered_drugs))

    for ax, (metric, ylabel, colour_idx) in zip(axes, metrics):
        for i, cat in enumerate(catalogues):
            df_cat = df_by_cat[cat]
            color = colours.get(cat, ("grey", "grey"))[colour_idx]

            # Central points
            ax.scatter(
                x_positions + offsets[i],
                df_cat[metric] * 100,
                marker=marker_style,
                color=color,
                s=110,
                linewidth=3.5,
                label=cat,
            )

            # Add confidence intervals if available
            low_col = f"{metric}_LOW"
            high_col = f"{metric}_HIGH"
            if low_col in df_cat.columns and high_col in df_cat.columns:
                yvals = df_cat[metric] * 100
                yerr = np.array(
                    [
                        np.maximum(
                            yvals - df_cat[low_col] * 100, 0
                        ),  # lower error (no negatives)
                        np.maximum(
                            df_cat[high_col] * 100 - yvals, 0
                        ),  # upper error (no negatives)
                    ]
                )

                ax.errorbar(
                    x_positions + offsets[i],
                    yvals,
                    yerr=yerr,
                    fmt="none",  # don’t add extra markers
                    ecolor=color,
                    elinewidth=1,
                    capsize=2,
                    alpha=0.8,
                )

        # Formatting
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, 105)
        ax.tick_params(axis="both", labelsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

        # Add faint vertical separators between drug bins
        for xpos in x_positions[:-1]:
            ax.axvline(
                x=xpos + 0.5, color="lightgrey", linestyle="-", linewidth=0.3, alpha=0.5
            )

    # Remove xticks completely
    axes[-1].set_xticks([])
    axes[-1].set_xticklabels([])

    # Add drug labels manually, centered under each bin
    for xpos, drug in enumerate(ordered_drugs):
        axes[-1].text(
            xpos,
            -8,
            drug,  # place just below the axis
            ha="center",
            va="top",
            fontsize=9,
        )

    # Legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(
    #    handles, labels, frameon=False, fontsize=7,
    #    loc="upper center", ncol=len(catalogues)
    # )

    # Fix x-axis limits to remove awkward padding
    for ax in axes:
        ax.set_xlim(-0.5, len(ordered_drugs) - 0.5)
    fig.savefig(outfile, bbox_inches="tight", dpi=600, transparent=True)

    if show_graph:
        plt.show()
    plt.close()


def plot_floating_bars(
    expanded_catalogues,
    valid_drugs,
    output_path=None,
    figsize=(6.69, 8),
    legend=False,
    fontsize=7,
):
    """
    Floating stacked bar plot with custom order:
      - Below the line: WHOv1 rules only, In both with WHOv1 rules
      - Above the line: WHOv1 only excl. rules, In both excl. WHOv1 rules, CAT-only
      - Centering point = between In both with WHOv1 rules (below) and WHOv1 only excl. rules (above)

    For PZA and ETH, bars are scaled down (e.g. ÷10) to fit the axis,
    but annotations show the true values.
    """

    drug_order = [
        "PZA",
        "ETH",
        "STM",
        "INH",
        "EMB",
        "RIF",
        "CFZ",
        "CAP",
        "BDQ",
        "MXF",
        "LEV",
        "DLM",
        "AMI",
        "KAN",
        "LZD",
    ]

    drugs, segs_all = [], []
    for drug in drug_order:
        if drug not in valid_drugs:
            continue

        # Counts
        cat_only_applied, who_only_applied, shared_applied = utils._counts_for(
            drug, expanded_catalogues, use_filtered=False
        )
        _, who_only_excluded, shared_excluded = utils._counts_for(
            drug, expanded_catalogues, use_filtered=True
        )

        segs_all.append(
            {
                "cat_only_applied": cat_only_applied,
                "both_excluded": shared_excluded,
                "both_rules": max(shared_applied - shared_excluded, 0),
                "who_only_rules": max(who_only_applied - who_only_excluded, 0),
                "who_only_excluded": who_only_excluded,
            }
        )
        drugs.append(drug)

    # Colors + labels
    colors = {
        "cat_only_applied": mcolors.to_rgba("#bb3e4d", 1.0),
        "both_rules": mcolors.to_rgba("#9c969c", 0.6),
        "both_excluded": mcolors.to_rgba("#9c969c", 1),
        "who_only_rules": mcolors.to_rgba("#3582B6", 0.6),
        "who_only_excluded": mcolors.to_rgba("#3582B6", 1),
    }
    labels = {
        "cat_only_applied": "catomatic-1",
        "both_excluded": "catomatic-1; WHOv1 algorithm",
        "both_rules": "catomatic-1; WHOv1 rules",
        "who_only_rules": "WHOv1 rules",
        "who_only_excluded": "WHOv1 algorithm",
    }
    legend_order = [
        "cat_only_applied",
        "both_rules",
        "both_excluded",
        "who_only_excluded",
        "who_only_rules",
    ]

    # Scaling factors for big bars
    scale_down = {"PZA": 2.4, "ETH": 2.4}  # adjust divisors as needed

    fig, ax = plt.subplots(figsize=figsize)

    for i, drug in enumerate(drugs):
        segs = segs_all[i]
        factor = scale_down.get(drug, 1)  # only shrink PZA/ETH

        # --- BELOW LINE ---
        bottom = 0
        for key in ["who_only_excluded", "who_only_rules"]:
            h = segs[key] / factor
            if h > 0:
                ax.bar(
                    i,
                    h,
                    bottom=bottom - h,
                    color=colors[key],
                    edgecolor="white",
                    linewidth=1,
                )
                # annotate with *true* count
                ax.text(
                    i,
                    bottom - h / 2,
                    str(segs[key]),
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                )
                bottom -= h

        # --- ABOVE LINE ---
        top = 0
        for key in ["both_excluded", "both_rules", "cat_only_applied"]:
            h = segs[key] / factor
            if h > 0:
                ax.bar(
                    i, h, bottom=top, color=colors[key], edgecolor="white", linewidth=1
                )
                ax.text(
                    i,
                    top + h / 2,
                    str(segs[key]),
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                )
                top += h

        # Add small "÷factor" note above scaled drugs
        if factor > 1:
            ax.text(
                i,
                top + (0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                f"÷ {factor}",
                ha="center",
                va="bottom",
                fontsize=6,
                color="gray",
            )

    # Cosmetics
    ax.axhline(0, color="0.2", linewidth=0.6, alpha=0.6, zorder=-1)
    ax.set_xticks(range(len(drugs)))
    ax.set_xticklabels(drugs, fontsize=fontsize)
    ax.yaxis.set_visible(False)
    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)

    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.01
    ax.set_ylim(ymin - pad, ymax + pad)

    # Legend
    patches = [mpatches.Patch(color=colors[k], label=labels[k]) for k in legend_order]
    if legend:
        ax.legend(
            handles=patches,
            fontsize=7,
            frameon=False,
            loc="upper center",
            labelspacing=0.85,
        )

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path)
    plt.show()
    plt.close(fig)


def plot_bar_charts(
    expanded_catalogues,
    valid_drugs,
    output_path=None,
    figsize=(6.69, 8),
    legend=False,
    fontsize=8,
):
    """
    Plots two normal stacked bar charts (one above the other):
      - Top: Including WHOv1 rules
      - Bottom: Rules removed (excluded)
    Each has y-axis labels and counts inside bars.
    """

    drug_order = [
        "PZA", "ETH", "STM", "INH", "EMB", "RIF", "CFZ", "CAP",
        "BDQ", "MXF", "LEV", "DLM", "AMI", "KAN", "LZD",
    ]

    # Colors (darker red, grey, blue)
    colors = {
        "who_only": mcolors.to_rgba("#3582B6", 1.0),  # blue
        "both": mcolors.to_rgba("#9c969c", 1.0),      # grey
        "cat_only": mcolors.to_rgba("#bb3e4d", 1.0),  # red
    }

    labels = {
        "who_only": "WHOv1 only",

        "cat_only": "catomatic-1 only",
        "both": "catomatic-1 & WHOv1",
    }

    scale_down = {"PZA": 2, "ETH": 2}

    # Collect values
    drugs, data_rules, data_excl = [], [], []
    for drug in drug_order:
        if drug not in valid_drugs:
            continue

        cat_only_applied, who_only_applied, shared_applied = utils._counts_for(
            drug, expanded_catalogues, use_filtered=False
        )
        cat_only_excluded, who_only_excluded, shared_excluded = utils._counts_for(
            drug, expanded_catalogues, use_filtered=True
        )

        factor = scale_down.get(drug, 1)
        data_rules.append({
            "cat_only": cat_only_applied / factor,
            "both": shared_applied / factor,
            "who_only": who_only_applied / factor,
            "factor": factor,
        })
        data_excl.append({
            "cat_only": cat_only_excluded / factor,  # no cat-only exclusion equivalent
            "both": shared_excluded / factor,
            "who_only": who_only_excluded / factor,
            "factor": factor,
        })
        drugs.append(drug)

    # --- Create figure with 2 subplots ---
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    def plot_bars(ax, data, title):
        bottoms = [0] * len(drugs)
        for key in ["cat_only", "both", "who_only"]:
            heights = [d[key] for d in data]
            bars = ax.bar(drugs, heights, bottom=bottoms,
                          color=colors[key], edgecolor="white", linewidth=1)
            # Add counts
            for bar, val, d in zip(bars, heights, data):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + val / 2,
                        str(int(val * d["factor"])),
                        ha="center", va="center", fontsize=fontsize
                    )
            bottoms = [b + h for b, h in zip(bottoms, heights)]
        #ax.set_title(title, fontsize=fontsize + 1)
        ax.set_ylabel("Number of Variants", fontsize=fontsize)
        ax.yaxis.set_visible(True)
        ax.set_ylim(0, 250)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_xlim(-0.5, len(drugs) - 0.5)
        ax.tick_params(axis='y', labelsize=fontsize)


    plot_bars(axes[0], data_excl, "Rules removed")

    plot_bars(axes[1], data_rules, "Including WHOv1 rules")

    if legend:
        patches = [mpatches.Patch(color=colors[k], label=labels[k]) for k in colors]
        axes[0].legend(handles=patches, fontsize=fontsize, frameon=False, loc="upper right")

    plt.xticks(fontsize=fontsize)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    plt.show()
    plt.close(fig)


def plot_stacked_percentage_bars(
    expanded_catalogues,
    pair_coverages,
    valid_drugs,
    output_path=None,
    figsize=(6.69, 10),
    legend=True,
):
    """
    Plot stacked percentage bar charts showing catalogue overlaps by drug.

    Creates two stacked bar plots per drug:
      - Top: mutation-level overlaps from expanded catalogues.
      - Bottom: sample-level overlaps from pair coverages.
    Each bar segment represents the proportion of variants or samples
    uniquely or jointly captured by catomatic and WHO catalogues.

    Parameters
    ----------
    expanded_catalogues : dict
        Dictionary of expanded catalogues per drug (from expand_catalogue_pair).
    pair_coverages : dict
        Dictionary of sample-level coverage results per drug.
    valid_drugs : list of str
        List of drugs to include in the plots.
    output_path : str or Path, optional
        File path to save the figure (default is None, i.e., only display).
    figsize : tuple, optional
        Figure size in inches (default is (6.69, 10)).
    legend : bool, optional
        If True, display a shared legend (default is True).
    """

    # Fixed plotting order
    drug_order = [
        "BDQ",
        "CFZ",
        "DLM",
        "LZD",
        "AMI",
        "KAN",
        "PZA",
        "EMB",
        "MXF",
        "LEV",
        "CAP",
        "ETH",
        "STM",
        "RIF",
        "INH",
    ]

    # Colors
    colors = {
        "cat_only": mcolors.to_rgba("#bb3e4d", 1.0),  # red
        "both_rules": mcolors.to_rgba("#9c969c", 1),  # grey full
        "both_excluded": mcolors.to_rgba("#9c969c", 0.5),  # grey transparent
        "who_only_rules": mcolors.to_rgba("#3582B6", 0.93),  # blue full
        "who_only_excluded": mcolors.to_rgba("#3582B6", 0.5),  # blue transparent
    }

    labels = {
        "cat_only": "catomatic-1",
        "both_rules": "catomatic-1; WHOv1 rules",
        "both_excluded": "catomatic-1; WHOv1 algorithm",
        "who_only_excluded": "WHOv1 algorithm",
        "who_only_rules": "WHOv1 rules",
    }

    legend_order = [
        "cat_only",
        "both_rules",
        "both_excluded",
        "who_only_rules",
        "who_only_excluded",
    ]
    legend_order = [
        "who_only_rules",
        "who_only_excluded",
        "both_excluded",
        "both_rules",
        "cat_only",
    ]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    drugs = []
    segs_all = []

    # --- collect mutation counts ---
    for drug in drug_order:
        if drug not in valid_drugs:
            continue

        cat_only_applied, who_only_applied, shared_applied = utils._counts_for(
            drug, expanded_catalogues, use_filtered=False
        )
        _, who_only_excluded, shared_excluded = utils._counts_for(
            drug, expanded_catalogues, use_filtered=True
        )

        segs = {
            "cat_only": cat_only_applied,
            "both_rules": max(shared_applied - shared_excluded, 0),
            "both_excluded": shared_excluded,
            "who_only_rules": max(who_only_applied - who_only_excluded, 0),
            "who_only_excluded": who_only_excluded,
        }

        total = sum(segs.values())
        if total == 0:
            continue

        for k in list(segs.keys()):
            segs[k + "_pct"] = 100 * segs[k] / total

        segs["total"] = total
        segs_all.append(segs)
        drugs.append(drug)

    # --- PLOT 1: MUTATIONS ---
    for i, drug in enumerate(drugs):
        segs = segs_all[i]
        bottom = 0
        for key in [
            "cat_only",
            "both_rules",
            "both_excluded",
            "who_only_excluded",
            "who_only_rules",
        ]:
            h = segs[key + "_pct"]
            if h > 0:
                ax1.bar(
                    i,
                    h,
                    bottom=bottom,
                    width=0.75,
                    color=colors[key],
                    edgecolor="white",
                    linewidth=1,
                )
                if h > 4:
                    ax1.text(
                        i,
                        bottom + h / 2,
                        f"{segs[key]}\n{h:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
                else:
                    ax1.text(
                        i,
                        bottom + h / 2,
                        f"{segs[key]}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )

                bottom += h
        # ax1.text(i, 101, f"{segs['total']}", ha="center", va="bottom",
        #        fontsize=8, fontweight="bold")

    # --- PLOT 2: SAMPLES ---
    for i, drug in enumerate(drugs):
        cat_only, who_only_rules, who_only_excl, both_rules, both_excluded, none = (
            utils._counts_for_ids(drug, pair_coverages)
        )
        segs_ids = {
            "cat_only": cat_only,
            "both_rules": both_rules,
            "both_excluded": both_excluded,
            "who_only_rules": who_only_rules,
            "who_only_excluded": who_only_excl,
        }
        total_ids = sum(segs_ids.values()) + none

        bottom = 0
        for key in [
            "cat_only",
            "both_rules",
            "both_excluded",
            "who_only_excluded",
            "who_only_rules",
        ]:
            h = 100 * segs_ids[key] / total_ids if total_ids > 0 else 0
            if h > 0:
                ax2.bar(
                    i,
                    h,
                    bottom=bottom,
                    width=0.75,
                    color=colors[key],
                    edgecolor="white",
                    linewidth=1,
                )
                if h > 4:
                    ax2.text(
                        i,
                        bottom + h / 2,
                        f"{segs_ids[key]}\n{h:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
                elif h > 1.4:
                    ax2.text(
                        i,
                        bottom + h / 2,
                        f"{segs_ids[key]}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )

                bottom += h
        # ax2.text(i, 101, f"{total_ids}", ha="center", va="bottom",
        #         fontsize=8, fontweight="bold")

    # Cosmetics
    for ax in (ax1, ax2):
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.5, len(drugs) - 0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticks(range(0, 101, 20))
        ax.tick_params(axis="y", labelsize=8)

    ax1.set_ylabel("Proportion of variants captured", fontsize=8)
    ax2.set_ylabel(
        "Proportion of samples captured (with at least 1 mutation)", fontsize=8
    )
    ax2.set_xticks(range(len(drugs)))
    ax2.set_xticklabels(drugs, fontsize=8)

    # Legend (shared)
    if legend:
        patches = [
            mpatches.Patch(color=colors[k], label=labels[k]) for k in legend_order
        ]
        fig.legend(
            handles=patches,
            fontsize=7,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.00),
            ncol=3,
        )

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_floating_bars_cat_vs_cats_all(
    expanded_catalogues,
    valid_drugs,
    output_path=None,
    figsize=(6.69, 4),
    legend=False,
    fontsize=7,
):
    """
    Plot floating stacked bar charts comparing catomatic-1 vs catomatic-2 catalogues.

    Displays mutation-level overlaps for each drug:
      - Red bars (below axis): catomatic-1 only
      - Grey bars (centered): shared between both
      - Orange bars (above axis): catomatic-2 only
    Scales certain drugs (e.g., PZA, ETH) for visual clarity.

    Parameters
    ----------
    expanded_catalogues : dict
        Dictionary of expanded catalogues for each drug (from expand_catalogue_pair).
    valid_drugs : list of str
        List of drugs to include in the plot.
    output_path : str or Path, optional
        File path to save the figure (default is None, display only).
    figsize : tuple, optional
        Figure size in inches (default is (6.69, 4)).
    legend : bool, optional
        If True, display a legend (default is False).
    fontsize : int, optional
        Font size for labels (default is 7).
    """

    drug_order = [
        "BDQ",
        "PZA",
        "ETH",
        "INH",
        "STM",
        "CAP",
        "EMB",
        "RIF",
        "DLM",
        "CFZ",
        "MXF",
        "LEV",
        "AMI",
        "KAN",
        "LZD",
    ]

    drugs, segs_all = [], []
    for drug in drug_order:
        if drug not in valid_drugs:
            continue

        cat_only, cats_all_only, both = utils._counts_cat_vs_cats_all(
            drug, expanded_catalogues
        )
        segs_all.append(
            {"cat_only": cat_only, "both": both, "cats_all_only": cats_all_only}
        )
        drugs.append(drug)

    # colors
    colors = {
        "cat_only": "#db5363",  # red
        "both": "#b1b0b3",  # purple
        "cats_all_only": "#fd8c53",  # blue
    }
    labels = {
        "cat_only": "In catomatic-1",
        "both": "In both",
        "cats_all_only": "In catomatic-2",
    }
    legend_order = ["cat_only", "both", "cats_all_only"]

    # scaling factors for PZA & ETH
    scale_down = {"PZA": 1, "ETH": 1}

    fig, ax = plt.subplots(figsize=figsize)

    for i, drug in enumerate(drugs):
        segs = segs_all[i]
        factor = scale_down.get(drug, 1)

        cat_h = segs["cat_only"] / factor
        both_h = segs["both"] / factor
        cats_all_h = segs["cats_all_only"] / factor

        # --- BLUE (cats_all_only, above axis) ---
        if cats_all_h > 0:
            ax.bar(
                i,
                cats_all_h,
                bottom=0,
                color=colors["cats_all_only"],
                edgecolor="white",
                linewidth=1,
            )
            ax.text(
                i,
                cats_all_h / 2,
                str(segs["cats_all_only"]),
                ha="center",
                va="center",
                fontsize=fontsize,
            )

        # --- PURPLE (both, below axis) ---
        if both_h > 0:
            ax.bar(
                i,
                both_h,
                bottom=-both_h,
                color=colors["both"],
                edgecolor="white",
                linewidth=1,
            )
            ax.text(
                i,
                -both_h / 2,
                str(segs["both"]),
                ha="center",
                va="center",
                fontsize=fontsize,
            )

        # --- RED (cat_only, stacked further below purple) ---
        if cat_h > 0:
            ax.bar(
                i,
                cat_h,
                bottom=-both_h - cat_h,
                color=colors["cat_only"],
                edgecolor="white",
                linewidth=1,
            )
            ax.text(
                i,
                -both_h - cat_h / 2,
                str(segs["cat_only"]),
                ha="center",
                va="center",
                fontsize=fontsize,
            )

        # Add "÷ factor" note if scaled
        if factor > 1:
            ax.text(
                i,
                ax.get_ylim()[1] * 0.02,
                f"÷ {factor}",
                ha="center",
                va="bottom",
                fontsize=6,
                color="gray",
            )

    # horizontal grey line at y=0
    ax.axhline(0, color="0.2", linewidth=0.6, alpha=0.35, zorder=-1)

    # cosmetics
    ax.set_xticks(range(len(drugs)))
    ax.set_xticklabels(drugs, fontsize=fontsize)

    ax.yaxis.set_visible(False)
    for side in ["left", "top", "right"]:
        ax.spines[side].set_visible(False)

    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.01
    ax.set_ylim(ymin - pad, ymax + pad)

    if legend:
        patches = [
            mpatches.Patch(color=colors[k], label=labels[k]) for k in legend_order[::-1]
        ]
        ax.legend(handles=patches, fontsize=7, frameon=False, loc="lower right")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    plt.show()
    plt.close(fig)
