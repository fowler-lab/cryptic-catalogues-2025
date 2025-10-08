import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from protocols import utils
import seaborn as sns
from matplotlib_venn import venn2
from scipy.stats import norm
import matplotlib.cm as cm
import multiprocessing as mp
from matplotlib.collections import PolyCollection
import matplotlib.patches as mpatches
import os
import matplotlib.colors as mcolors




plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = 7
plt.rcParams["figure.figsize"] = (6.69, 5.02)

plot_order = [
    "RIF",  # Rifampicin
    "INH",  # Isoniazid
    "EMB",  # Ethambutol
    "PZA",  # Pyrazinamide
    "LEV", "MXF",  # Levofloxacin, Moxifloxacin
    "BDQ", "CFZ",   # Bedaquiline, Clofazimine
    "LZD",          # Linezolid
    "DLM",    # Delamanid, Pretomanid
    "AMI",          # Amikacin
    "STM",          # Streptomycin
    "ETH", "PTO",
    "KAN", 'CAP'   # Ethionamide, Prothionamide
]


colours = {
    "WHO1": ("#47d2f5", "#47d2f5", "#47d2f5"),
    "WHOv1": ("#3381B5", "#3381B5", "#3381B5"),
    "WHOv1_no_rules": ("#c872c5", "#c872c5", "#c872c5"),
    "WHOv2": ("#2ba74e", "#2ba74e", "#2ba74e"),
    "MTBC-CRyPTICv1.1.1-2025.8": ("#bd2f3f", "#bd2f3f", "#bd2f3f"),
    "MTBC-CRyPTICv3.4.0-2025.8": ("#fd8c53", "#fd8c53", "#fd8c53"),
}

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
        nrows=2, ncols=1, sharex=True,
        figsize=figsize, constrained_layout=True
    )

    metrics = [
        (sensitivity, "Sensitivity (%)", 0),
        (specificity, "Specificity (%)", 1)
    ]

    x_positions = np.arange(len(ordered_drugs))
    offsets = np.linspace(-0.2, 0.2, 2)  # consistent with catomatic
    marker_style = "_"

    for ax, (metric, ylabel, colour_idx) in zip(axes, metrics):
        # WHOv1 (external WHO report)
        ax.scatter(
            x_positions + offsets[0],
            who1_results[metric.lower()] if metric.lower() in who1_results.columns else who1_results[metric],
            marker=marker_style,
            color=colours["WHO1"][colour_idx],
            s=120,
            linewidth=3,
            label="WHOv1"
        )

        # WHOv1 (this study)
        ax.scatter(
            x_positions + offsets[1],
            df_who1[metric] * 100,
            marker=marker_style,
            color=colours["WHOv1"][colour_idx],
            s=120,
            linewidth=3,
            label="WHOv1 (this study)"
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
            ax.axvline(x=xpos + 0.5, color="lightgrey", linestyle="-", linewidth=0.3, alpha=0.5)

    # Remove xticks completely
    axes[-1].set_xticks([])
    axes[-1].set_xticklabels([])

    # Add drug labels manually under bottom panel
    for xpos, drug in enumerate(ordered_drugs):
        axes[-1].text(
            xpos, -8, drug,
            ha="center", va="top", fontsize=9
        )

    # Add legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    #fig.legend(
    #    handles, labels, frameon=False, fontsize=8,
    #    loc="upper center", ncol=2
    #)

    # Fix x-axis range
    for ax in axes:
        ax.set_xlim(-0.5, len(ordered_drugs) - 0.5)

    # Save
    fig.savefig(outfile, bbox_inches="tight", dpi=600, transparent=True)
    if show_graph:
        plt.show()
    plt.close()


def plot_bar_who(df, who_results, outfile, who_drugs, show_graph=False):
    # Drugs to plot, in specified order
    ordered_drugs = [drug for drug in plot_order if drug in who_drugs][::-1]
    y = np.arange(len(ordered_drugs))

    # Prepare the two WHO catalogues
    df_who1 = df[df.catalogue == "WHOv1"].copy()
    df_who2 = df[df.catalogue == "WHOv2"].copy()
    who1_results = who_results[who_results.catalogue == "WHO1"].copy()
    who2_results = who_results[who_results.catalogue == "WHO2"].copy()

    # Reindex all dataframes to match ordered drug list
    df_who1 = df_who1.set_index("DRUG").reindex(ordered_drugs).reset_index()
    df_who2 = df_who2.set_index("DRUG").reindex(ordered_drugs).reset_index()
    who1_results = who1_results.set_index("drug").reindex(ordered_drugs).reset_index()
    who2_results = who2_results.set_index("drug").reindex(ordered_drugs).reset_index()

    # Create the plot
    fig, (axis1, axis2) = plt.subplots(
        ncols=2, sharey=True, figsize=(5.5, 7.5), gridspec_kw={"width_ratios": [1, 1]}
    )

    # Helper function to draw one set of bars
    def plot_bar(axis, y_shift, values, color, text_align="right", hollow=False):

        axis.barh(
            y + y_shift,
            values,
            0.15,
            facecolor="none" if hollow else color,
            edgecolor=color,
            linewidth=1,
            alpha=1.0,
        )
        for i, val in enumerate(values):
            if pd.notnull(val) and val > 0:
                axis.text(
                    val + 2,
                    y[i] + y_shift,
                    "%.1f" % (val),
                    ha=text_align,
                    va="center",
                    fontweight="heavy",
                    color=color,
                    fontsize=5.5,
                )

    # First panel: SENSITIVITY
    plot_bar(axis1, 0.3, who1_results["sensitivity"], colours["WHOv1"][0], hollow=True)
    if "SENSITIVITY2" in df_who1.columns:
        plot_bar(axis1, 0.1, df_who1["SENSITIVITY2"] * 100, colours["WHOv1"][0])
    plot_bar(axis1, -0.1, who2_results["sensitivity"], colours["WHOv2"][0], hollow=True)
    if "SENSITIVITY2" in df_who2.columns:
        plot_bar(axis1, -0.3, df_who2["SENSITIVITY2"] * 100, colours["WHOv2"][0])

    axis1.invert_xaxis()
    axis1.set_yticks([])
    axis1.spines["top"].set_visible(False)
    axis1.spines["left"].set_visible(False)
    axis1.set_yticklabels([])
    axis1.set_ylim(-0.5, len(ordered_drugs) - 0.5)
    axis1.set_xlim(100, 0)

    # Second panel: SPECIFICITY
    plot_bar(axis2, 0.3, who1_results["specificity"], colours["WHOv1"][1], text_align="left", hollow=True)
    if "SPECIFICITY2" in df_who1.columns:
        plot_bar(axis2, 0.1, df_who1["SPECIFICITY2"] * 100, colours["WHOv1"][1], text_align="left")
    plot_bar(axis2, -0.1, who2_results["specificity"], colours["WHOv2"][1], text_align="left", hollow=True)
    if "SPECIFICITY2" in df_who2.columns:
        plot_bar(axis2, -0.3, df_who2["SPECIFICITY2"] * 100, colours["WHOv2"][1], text_align="left")

    axis2.spines["top"].set_visible(False)
    axis2.spines["right"].set_visible(False)
    axis2.set_yticks([])
    axis2.set_ylim(-0.5, len(ordered_drugs) - 0.5)
    axis2.set_xlim(0, 100)



    # Add drug names in the center
    for i, drug in zip(y, ordered_drugs):
        axis2.text(-11, i, drug, ha="center", va="center", fontsize=7)

    # Save the figure
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
    # Subset drugs in specified order
    ordered_drugs = [drug for drug in plot_order if drug in who_drugs]

    if catalogues == "all":
        catalogues = df.catalogue.unique()

    # Subset dataframe to only selected catalogues
    df = df[df.catalogue.isin(catalogues)].copy()

    # Create mapping catalogue -> dataframe reindexed to drug order
    df_by_cat = {
        cat: df[df.catalogue == cat].set_index("DRUG").reindex(ordered_drugs).reset_index()
        for cat in catalogues
    }

    # Set up 3 stacked subplots (DPR, Sensitivity, Specificity)
    fig, axes = plt.subplots(
        nrows=3, ncols=1, sharex=True,
        figsize=figsize, constrained_layout=True
    )

    metrics = [
        (dpr, "DPR (%)", 2),
        (sensitivity, "Sensitivity (%)", 0),
        (specificity, "Specificity (%)", 1)
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
                label=cat
            )

            # Add confidence intervals if available
            low_col = f"{metric}_LOW"
            high_col = f"{metric}_HIGH"
            if low_col in df_cat.columns and high_col in df_cat.columns:
                yvals = df_cat[metric] * 100
                yerr = np.array([
                    yvals - df_cat[low_col] * 100,   # lower error
                    df_cat[high_col] * 100 - yvals  # upper error
                ])
                ax.errorbar(
                    x_positions + offsets[i],
                    yvals,
                    yerr=yerr,
                    fmt="none",       # don’t add extra markers
                    ecolor=color,
                    elinewidth=1,
                    capsize=2,
                    alpha=0.8
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
            ax.axvline(x=xpos + 0.5, color="lightgrey", linestyle="-", linewidth=0.3, alpha=0.5)

    # Remove xticks completely
    axes[-1].set_xticks([])
    axes[-1].set_xticklabels([])

    # Add drug labels manually, centered under each bin
    for xpos, drug in enumerate(ordered_drugs):
        axes[-1].text(
            xpos, -8, drug,  # place just below the axis
            ha="center", va="top", fontsize=9
        )

    # Legend at top
    handles, labels = axes[0].get_legend_handles_labels()
    #fig.legend(
    #    handles, labels, frameon=False, fontsize=7,
    #    loc="upper center", ncol=len(catalogues)
    #)

    # Fix x-axis limits to remove awkward padding
    for ax in axes:
        ax.set_xlim(-0.5, len(ordered_drugs) - 0.5)
    fig.savefig(outfile, bbox_inches="tight", dpi=600, transparent=True)

    if show_graph:
        plt.show()
    plt.close()



def plot_bar_catomatic(
    df,
    outfile,
    who_drugs,
    catalogues='all',
    sensitivity="SENSITIVITY",
    specificity="SPECIFICITY",
    barwidth=0.285,
    figsize=(4.4, 8),
    show_graph=False,
):
    # Subset and reverse the plot order
    ordered_drugs = [drug for drug in plot_order if drug in who_drugs][::-1]
    y = np.arange(len(ordered_drugs))

    if catalogues=='all':
        catalogues = df.catalogue.unique()

    # Subset dataframe to only the specified catalogues
    df = df[df.catalogue.isin(catalogues)].copy()

    # Create mapping of catalogue to dataframe and assign color positions
    df_by_cat = {
        cat: df[df.catalogue == cat].set_index("DRUG").reindex(ordered_drugs).reset_index()
        for cat in catalogues
    }

    n_cats = len(catalogues)
    shifts = np.linspace(barwidth * (n_cats - 1) / 2, -barwidth * (n_cats - 1) / 2, n_cats)

    # Create the plot
    fig, (axis1, axis2) = plt.subplots(
        ncols=2, sharey=True, figsize=figsize, gridspec_kw={"width_ratios": [1, 1]}
    )

    # Helper to draw bars
    def plot_bar(axis, y_shift, values, color, text_align="right", hollow=False):
        axis.barh(
            y + y_shift,
            values,
            barwidth,
            facecolor="none" if hollow else color,
            edgecolor=color,
            linewidth=0,
            alpha=1.0,
        )
        for i, val in enumerate(values):
            if pd.notnull(val) and val > 0:
                axis.text(
                    val + 2,
                    y[i] + y_shift,
                    "%.1f" % val,
                    ha=text_align,
                    va="center",
                    fontweight="heavy",
                    color=color,
                    fontsize=6,
                )

    # Plot each catalogue
    for i, cat in enumerate(catalogues):
        shift = shifts[i]
        color = colours.get(cat, ("grey", "grey"))  # default grey if colour not defined
        df_cat = df_by_cat[cat]

        # Sensitivity
        plot_bar(axis1, shift, df_cat[sensitivity] * 100, color[0])

        # Specificity
        plot_bar(axis2, shift, df_cat[specificity] * 100, color[1], text_align="left")

    # Format axes
    axis1.invert_xaxis()
    axis1.set_yticks([])
    axis1.set_yticklabels([])
    axis1.set_ylim(-0.5, len(ordered_drugs) - 0.5)
    axis1.spines["top"].set_visible(False)
    axis1.spines["left"].set_visible(False)

    axis2.set_yticks([])
    axis2.set_ylim(-0.5, len(ordered_drugs) - 0.5)
    axis2.spines["top"].set_visible(False)
    axis2.spines["right"].set_visible(False)

    # Add drug names
    label_offset = -0.08  # tweak this value as needed
    for i, drug in zip(y, ordered_drugs):
        axis2.text(-11, i + label_offset, drug, ha="center", va="center", fontsize=7)

    # Save the figure
    fig.savefig(outfile, bbox_inches="tight", dpi=600, transparent=True)

    if show_graph:
        plt.show()
    plt.close()


def plot_bar_catomatic_coverage(
    df,
    outfile,
    who_drugs,
    catalogues='all',
    figsize=(2.2, 8),
    barwidth=0.285,
    show_graph=False,
):
    # Subset and reverse drug order
    ordered_drugs = [drug for drug in plot_order if drug in who_drugs][::-1]
    y = np.arange(len(ordered_drugs))

    if catalogues=='all':
        catalogues = df.catalogue.unique()

    # Subset df to selected catalogues
    df = df[df.catalogue.isin(catalogues)].copy()

    # Create catalogue-specific dataframes reindexed by drug order
    df_by_cat = {
        cat: df[df.catalogue == cat].set_index("DRUG").reindex(ordered_drugs).reset_index()
        for cat in catalogues
    }

    # Compute bar shifts based on number of catalogues
    n_cats = len(catalogues)
    shifts = np.linspace(barwidth * (n_cats - 1) / 2, -barwidth * (n_cats - 1) / 2, n_cats)

    # Plot
    fig, axis1 = plt.subplots(ncols=1, sharey=True, figsize=figsize)

    axis1.invert_xaxis()

    # Helper to draw bars
    def plot_bar(axis, y_shift, values, color, text_align="right", hollow=False):
        axis.barh(
            y + y_shift,
            values,
            barwidth,
            facecolor="none" if hollow else color,
            edgecolor=color,
            linewidth=0,
            alpha=1.0,
        )
        for i, val in enumerate(values):
            if pd.notnull(val) and val > 0:
                axis.text(
                    val + 2,
                    y[i] + y_shift,
                    "%.1f" % val,
                    ha=text_align,
                    va="center",
                    fontweight="heavy",
                    color=color,
                    fontsize=6,
                )

    # Plot each catalogue's coverage
    for i, cat in enumerate(catalogues):
        shift = shifts[i]
        color = colours[cat][2]  # Assume correct key in colours dict
        df_cat = df_by_cat[cat]
        plot_bar(axis1, shift, df_cat["COVERAGE"] * 100, color)

    # Axis styling
    axis1.set_ylabel("")
    axis1.set_yticks([])
    axis1.set_ylim(-0.5, len(ordered_drugs) - 0.5)
    axis1.spines["top"].set_visible(False)
    axis1.spines["left"].set_visible(False)

    # Save figure
    fig.savefig(outfile, bbox_inches="tight", dpi=600, transparent=True)

    if show_graph:
        plt.show()
    plt.close()



# Function to plot error bars with jittered x-values
def plot_mutation_error_bars(
    frs_prop_data, color_map={}, min_err=1, label_cutoff=15, figpath=None
):
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
            if i % 2 == 0:  # Shade every other section
                plt.axvspan(start, start + 0.1, color="lightgrey", alpha=0.3)
        plt.axhline(background, linewidth=1)
        plt.xlabel("min FRS (Binned, with Jitter)", fontsize=7)
        plt.ylabel("Proportion Resistant", fontsize=7)
        plt.title(f"{drug}", fontsize=7)
        plt.xticks(np.arange(0.1, 1.05, 0.1))  # Minor ticks at every 0.05
        plt.ylim(-0.05, 1.05)  # Keep proportions in range
        plt.grid(True, linestyle="--", alpha=0.5)
        num_cols = min(num_mutations, 6)  # Adjust '5' as needed for your plot width
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


def plot_venn_catomatic_per_drug(expanded_catalogues, valid_drugs, output_dir):
    """Plot Venn diagrams per drug comparing rules-included vs. all-samples catalogues."""

    for drug in valid_drugs:
        fig, ax = plt.subplots(1, 1, figsize=(2.2, 1.5), constrained_layout=True)

        merged = expanded_catalogues[drug]['merged']

        shared = merged[
            (~merged.PREDICTION_cat.isna()) & (~merged.PREDICTION_cats_all.isna())
        ].MUTATION.nunique()

        only_cat = merged[
            (~merged.PREDICTION_cat.isna()) & (merged.PREDICTION_cats_all.isna())
        ].MUTATION.nunique()

        only_cats_all = merged[
            (merged.PREDICTION_cat.isna()) & (~merged.PREDICTION_cats_all.isna())
        ].MUTATION.nunique()

        venn_diagram = venn2(
            subsets=(only_cat, only_cats_all, shared),
            set_labels=('', ''),
            ax=ax,
            set_colors=(colours["MTBC-CRyPTICv1.1.1-2025.8"][2], colours["MTBC-CRyPTICv3.4.0-2025.8"][2])
        )

        # Set patch transparencies
        for patch_id, alpha in zip(['10', '01', '11'], [1, 1, 1.0]):
            patch = venn_diagram.get_patch_by_id(patch_id)
            if patch:
                patch.set_alpha(alpha)

        # Set label styles
        if venn_diagram.get_label_by_id('10'):
            venn_diagram.get_label_by_id('10').set_color('white')
        for label in venn_diagram.subset_labels:
            if label:
                label.set_fontsize(10)
        for label in venn_diagram.set_labels:
            if label:
                label.set_fontsize(7)

        # Save plot
        print(drug)

        plt.show()
        fig.savefig(f'{output_dir}/{drug}_venn_catomatic.pdf')
        plt.close(fig)


def plot_venn_pair_per_drug(expanded_catalogues, valid_drugs, output_dir):
    for drug in valid_drugs:
        fig, axes = plt.subplots(1, 2, figsize=(2.2, 1.5), constrained_layout=True)

        for ax, use_filtered, title_suffix in zip(
            axes,
            [False, True],
            ['Rules Applied', 'Rules Excluded']
        ):
            if use_filtered:
                cat_filtered = expanded_catalogues[drug]['cat']
                #should already be filtered out, but just in case
                cat_filtered = cat_filtered[(~cat_filtered.MUTATION.str.contains(r"[?*=]", regex=True, na=False))&(cat_filtered.EVIDENCE!={'expanded_rule'})]
                cat_filtered = cat_filtered[(cat_filtered.PREDICTION!='U')&(~cat_filtered.MUTATION.str.contains('indel'))]
                mutations_cat = set(cat_filtered['MUTATION'])

                who_filtered = expanded_catalogues[drug]['who']
                #should already be filtered out, but just in case
                who_filtered = who_filtered[(~who_filtered.MUTATION.str.contains(r"[?*=]", regex=True, na=False))&(who_filtered.EVIDENCE!={'expanded_rule'})]
                who_filtered = who_filtered[(who_filtered.PREDICTION!='U')&(~who_filtered.MUTATION.str.contains('indel'))]
                mutations_who = set(who_filtered['MUTATION'])

                only_cat = len(mutations_cat - mutations_who)
                only_who = len(mutations_who - mutations_cat)
                shared = len(mutations_cat & mutations_who)
            else:
                merged = expanded_catalogues[drug]['merged']
                shared = merged[(~merged.PREDICTION_who.isna()) & (~merged.PREDICTION_cat.isna())].MUTATION.nunique()
                only_who = merged[(~merged.PREDICTION_who.isna()) & (merged.PREDICTION_cat.isna())].MUTATION.nunique()
                only_cat = merged[(merged.PREDICTION_who.isna()) & (~merged.PREDICTION_cat.isna())].MUTATION.nunique()

            venn_diagram = venn2(
                subsets=(only_cat, only_who, shared),
                set_labels=('', ''),
                ax=ax,
                set_colors=(colours["MTBC-CRyPTICv1.1.1-2025.8"][2], colours["WHOv1"][2])
            )

            if venn_diagram.get_patch_by_id('10'):  # Left circle
                venn_diagram.get_patch_by_id('10').set_alpha(0.9)

            if venn_diagram.get_patch_by_id('01'):  # Right circle
                venn_diagram.get_patch_by_id('01').set_alpha(0.7)

            if venn_diagram.get_patch_by_id('11'):  # Overlap
                venn_diagram.get_patch_by_id('11').set_alpha(1)

            # Make left circle label (subset '10') white
            if venn_diagram.get_label_by_id('10'):
                venn_diagram.get_label_by_id('10').set_color('white')


            #ax.set_title(title_suffix, fontsize=7)
            for label in venn_diagram.set_labels:
                if label:
                    label.set_fontsize(7)
            for label in venn_diagram.subset_labels:
                if label:
                    label.set_fontsize(10)

        #fig.suptitle(f'{drug}', fontsize=3)
        fig.savefig(f'{output_dir}/{drug}_venn_pair.pdf')
        print (drug)
        plt.show()
        plt.close(fig)


def _counts_for(drug, expanded_catalogues, use_filtered):
    """Returns (only_cat, only_who, shared) for a given drug."""
    if use_filtered:
        # RULES EXCLUDED
        cat_filtered = expanded_catalogues[drug]['cat']
        cat_filtered = cat_filtered[
            (~cat_filtered.MUTATION.str.contains(r"[?*=]", regex=True, na=False)) &
            (cat_filtered.EVIDENCE != {'expanded_rule'})
        ]
        cat_filtered = cat_filtered[
            (cat_filtered.PREDICTION != 'U') &
            (~cat_filtered.MUTATION.str.contains('indel'))
        ]
        mutations_cat = set(cat_filtered['MUTATION'])

        who_filtered = expanded_catalogues[drug]['who']
        who_filtered = who_filtered[
            (~who_filtered.MUTATION.str.contains(r"[?*=]", regex=True, na=False)) &
            (who_filtered.EVIDENCE != {'expanded_rule'})
        ]
        who_filtered = who_filtered[
            (who_filtered.PREDICTION != 'U') &
            (~who_filtered.MUTATION.str.contains('indel'))
        ]
        mutations_who = set(who_filtered['MUTATION'])

        only_cat = len(mutations_cat - mutations_who)
        only_who = len(mutations_who - mutations_cat)
        shared   = len(mutations_cat & mutations_who)
    else:
        # RULES APPLIED
        merged = expanded_catalogues[drug]['merged']
        shared   = merged[(~merged.PREDICTION_who.isna()) & (~merged.PREDICTION_cat.isna())].MUTATION.nunique()
        only_who = merged[(~merged.PREDICTION_who.isna()) & ( merged.PREDICTION_cat.isna())].MUTATION.nunique()
        only_cat = merged[( merged.PREDICTION_who.isna()) & (~merged.PREDICTION_cat.isna())].MUTATION.nunique()

    return only_cat, only_who, shared

def _counts_for_ids(drug, pair_coverages):
    """
    Returns (cat_only, who_only_rules, who_only_excluded, both_rules, both_excluded, none)
    for sample IDs for a given drug.
    """
    if drug not in pair_coverages:
        return 0, 0, 0, 0, 0, 0

    d = pair_coverages[drug]

    # Ignore cat1_rules (always empty)
    cat = set(d.get("cat1_no_rules", []))

    # WHO separated into rules vs excl
    who_rules = set(d.get("cat2_rules", []))
    who_excl  = set(d.get("cat2_no_rules", []))

    none      = set(d.get("none", []))

    # intersections - ids can exist in both who_rules and who_excl buckets, but excl rules wins 
    # as we are tyring to ask 'what do the rules actually add in practise'
    both_excluded = len(cat & who_excl)
    both_rules    = len(cat & who_rules - (cat & who_excl))

    # unique parts
    cat_only       = len(cat - (who_rules | who_excl))
    who_only_rules = len(who_rules - who_excl - cat)
    who_only_excl  = len(who_excl - who_rules - cat)

    return cat_only, who_only_rules, who_only_excl, both_rules, both_excluded, len(none)


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

def plot_floating_bars(expanded_catalogues, valid_drugs, output_path=None,
                       figsize=(6.69, 8), legend=False, fontsize=7):
    """
    Floating stacked bar plot with custom order:
      - Below the line: WHOv1 rules only, In both with WHOv1 rules
      - Above the line: WHOv1 only excl. rules, In both excl. WHOv1 rules, CAT-only
      - Centering point = between In both with WHOv1 rules (below) and WHOv1 only excl. rules (above)

    For PZA and ETH, bars are scaled down (e.g. ÷10) to fit the axis,
    but annotations show the true values.
    """

    drug_order = ['PZA', 'ETH', 'STM', 'INH', 'EMB', 'RIF', 'CFZ', 'CAP', 'BDQ',
                  'MXF', 'LEV', 'DLM', 'AMI', 'KAN', 'LZD']

    drugs, segs_all = [], []
    for drug in drug_order:
        if drug not in valid_drugs:
            continue

        # Counts
        cat_only_applied, who_only_applied, shared_applied = _counts_for(drug, expanded_catalogues, use_filtered=False)
        _, who_only_excluded, shared_excluded = _counts_for(drug, expanded_catalogues, use_filtered=True)

        segs_all.append({
            "cat_only_applied": cat_only_applied,
            "both_excluded": shared_excluded,
            "both_rules": max(shared_applied - shared_excluded, 0),
            "who_only_rules": max(who_only_applied - who_only_excluded, 0),
            "who_only_excluded": who_only_excluded
        })
        drugs.append(drug)

    # Colors + labels
    colors = {
        "cat_only_applied": mcolors.to_rgba("#bb3e4d", 1.0),
        "both_rules": mcolors.to_rgba("#9c969c", 0.6),
        "both_excluded": mcolors.to_rgba("#9c969c", 1),
        "who_only_rules": mcolors.to_rgba("#3582B6", 0.6),
        "who_only_excluded": mcolors.to_rgba("#3582B6", 1)
    }
    labels = {
        "cat_only_applied": "catomatic-1",
        "both_excluded": "catomatic-1; WHOv1 algorithm",
        "both_rules": "catomatic-1; WHOv1 rules",
        "who_only_rules": "WHOv1 rules",
        "who_only_excluded": "WHOv1 algorithm"
    }
    legend_order = ["cat_only_applied", "both_rules", "both_excluded", "who_only_excluded", "who_only_rules"]

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
                ax.bar(i, h, bottom=bottom - h,
                       color=colors[key], edgecolor="white", linewidth=1)
                # annotate with *true* count
                ax.text(i, bottom - h/2, str(segs[key]),
                        ha="center", va="center", fontsize=fontsize)
                bottom -= h

        # --- ABOVE LINE ---
        top = 0
        for key in ["both_excluded", "both_rules", "cat_only_applied"]:
            h = segs[key] / factor
            if h > 0:
                ax.bar(i, h, bottom=top,
                       color=colors[key], edgecolor="white", linewidth=1)
                ax.text(i, top + h/2, str(segs[key]),
                        ha="center", va="center", fontsize=fontsize)
                top += h

        # Add small "÷factor" note above scaled drugs
        if factor > 1:
            ax.text(i, top + (0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])),
                    f"÷ {factor}", ha="center", va="bottom",
                    fontsize=6, color="gray")

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
        ax.legend(handles=patches,
                  fontsize=7, frameon=False,
                  loc="upper center", labelspacing=0.85)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path)
    plt.show()
    plt.close(fig)



def plot_rules_vs_excluded(expanded_catalogues, valid_drugs, output_path=None, figsize=(8, 10), legend=False):
    """
    Horizontal stacked bar plot comparing with vs without WHOv1 rules.
    For each drug:
      - Top bar = without WHOv1 rules
      - Bottom bar = with WHOv1 rules
    Each bar has 3 layers: WHO-only, In both, CAT-only.
    
    Bars are ordered by CAT-only (without rules), descending.
    """
    drug_order = ["RIF", "INH", "EMB", "PZA", "LEV", "MXF", "BDQ", "CFZ",
                  "LZD", "DLM", "AMI", "STM", "ETH", "KAN", "CAP"]

    # keep order but only for valid drugs
    drugs = [d for d in drug_order if d in valid_drugs]

    segs_all = []
    for drug in drugs:
        # counts with rules
        cat_only_applied, who_only_applied, both_applied = _counts_for(drug, expanded_catalogues, use_filtered=False)
        # counts without rules
        cat_only_excl, who_only_excl, both_excl = _counts_for(drug, expanded_catalogues, use_filtered=True)

        segs_all.append({
            "drug": drug,
            "with": {"who": who_only_applied, "both": both_applied, "cat": cat_only_applied},
            "without": {"who": who_only_excl, "both": both_excl, "cat": cat_only_excl}
        })

    # --- reorder by CAT-only without rules ---
    segs_all.sort(key=lambda s: s["without"]["cat"], reverse=True)

    colors = {"who": "#3582B6", "both": "#9c969c", "cat": "#bb3e4d"}
    labels = {"who": "WHO-only", "both": "In both", "cat": "CAT-only"}

    fig, ax = plt.subplots(figsize=figsize)

    bar_height = 0.35
    spacing = 0.9  # reduce spacing between pairs
    for i, segs in enumerate(segs_all):
        base_y = i * spacing

        # --- without rules (top bar of pair) ---
        left = 0
        for key in ["who", "both", "cat"]:
            h = segs["without"][key]
            if h > 0:
                ax.barh(base_y + bar_height/2, h, left=left, height=bar_height,
                        color=colors[key], edgecolor="white", linewidth=1)
                ax.text(left + h/2, base_y + bar_height/2, str(h),
                        ha="center", va="center", fontsize=7, color="white")
                left += h
        ax.text(-0.5, base_y + bar_height/2, "without rules", 
                va="center", ha="right", fontsize=7, color="black")

        # --- with rules (bottom bar of pair) ---
        left = 0
        for key in ["who", "both", "cat"]:
            h = segs["with"][key]
            if h > 0:
                ax.barh(base_y - bar_height/2, h, left=left, height=bar_height,
                        color=colors[key], edgecolor="white", linewidth=1)
                ax.text(left + h/2, base_y - bar_height/2, str(h),
                        ha="center", va="center", fontsize=7, color="white")
                left += h
        ax.text(-0.5, base_y - bar_height/2, "with rules", 
                va="center", ha="right", fontsize=7, color="black")

        # --- one drug label centered on the pair ---
        ax.text(-2.0, base_y, segs["drug"], 
                va="center", ha="right", fontsize=8, fontweight="bold")

    # cosmetics
    ax.set_yticks([])  # hide default ticks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # keep only the y-axis line
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(0.8)

    # legend
    patches = [mpatches.Patch(color=colors[k], label=labels[k]) for k in ["who", "both", "cat"]]
    if legend:
        ax.legend(handles=patches, fontsize=7, frameon=False,
                  loc="upper right", ncol=3)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)
    plt.show()
    plt.close(fig)


import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_stacked_5layer(expanded_catalogues, valid_drugs, output_path=None, figsize=(8, 6), legend=False):
    """
    Vertical stacked bar plot (5 layers per drug) with custom colors & hatching:
      - CAT-only = solid red
      - In both, with WHOv1 rules = red with grey diagonal hatch
      - In both, excl. WHOv1 rules = solid grey
      - WHOv1 only, excl. rules = solid blue
      - WHOv1 rules only = blue with lightblue diagonal hatch

    Bars are ordered by (CAT-only + In both with WHOv1 rules), descending.
    """
    drug_order = ["RIF", "INH", "EMB", "PZA", "LEV", "MXF", "BDQ", "CFZ",
                  "LZD", "DLM", "AMI", "STM", "ETH", "KAN", "CAP"]

    drugs, segs_all = [], []

    # --- build segment counts ---
    for drug in drug_order:
        if drug not in valid_drugs:
            continue

        # counts with rules
        cat_only_applied, who_only_applied, shared_applied = _counts_for(drug, expanded_catalogues, use_filtered=False)
        # counts without rules
        _, who_only_excluded, shared_excluded = _counts_for(drug, expanded_catalogues, use_filtered=True)

        segs = {
            "who_rules": max(who_only_applied - who_only_excluded, 0),
            "both_rules": max(shared_applied - shared_excluded, 0),
            "who_excl": who_only_excluded,
            "both_excl": shared_excluded,
            "cat_only": cat_only_applied
        }
        segs_all.append(segs)
        drugs.append(drug)

    # --- reorder by (cat_only + both_rules) ---
    order_metric = [s["cat_only"] + s["both_rules"] for s in segs_all]
    sort_idx = sorted(range(len(drugs)), key=lambda i: order_metric[i], reverse=True)

    drugs    = [drugs[i] for i in sort_idx]
    segs_all = [segs_all[i] for i in sort_idx]

    # base colors
    red       = "#bb3e4d"
    blue      = "#3582B6"
    lightblue = "#3582B650" 
    grey      = "#9c969c"

    # style definitions (facecolor, hatch, edgecolor)
    styles = {
        "cat_only":   {"facecolor": red,  "hatch": None,   "edgecolor": "white"},
        "both_rules": {"facecolor": red,  "hatch": "||||", "edgecolor": grey},    
        "both_excl":  {"facecolor": grey, "hatch": None,   "edgecolor": "white"},
        "who_excl":   {"facecolor": blue, "hatch": None,   "edgecolor": "white"},
        "who_rules":  {"facecolor": lightblue, "hatch": None, "edgecolor": 'lightgrey'}
    }

    labels = {
        "who_rules": "WHOv1 rules only",
        "both_rules": "In both, with WHOv1 rules",
        "who_excl": "WHOv1 only, excl. rules",
        "both_excl": "In both, excl. WHOv1 rules",
        "cat_only": "CAT-only"
    }

    # bottom → top stacking order
    stack_order = ["cat_only", "both_rules", "both_excl", "who_excl", "who_rules"]

    fig, ax = plt.subplots(figsize=figsize)

    for i, segs in enumerate(segs_all):
        bottom = 0
        for key in stack_order:
            h = segs[key]
            if h > 0:
                style = styles[key]
                # draw fill + hatch
                ax.bar(
                    i, h, bottom=bottom,
                    color=style["facecolor"],
                    hatch=style["hatch"],
                    edgecolor=style["edgecolor"],  # hatch stroke color
                    linewidth=1.5
                )
                # overlay white border
                ax.bar(
                    i, h, bottom=bottom,
                    color="none",
                    edgecolor="white",
                    linewidth=1.5
                )
                # inside label
                ax.text(i, bottom + h/2, str(h),
                        ha="center", va="center", fontsize=7, color="black")
                bottom += h

        # optional: total label above bar (combined CAT + both_rules)
        ax.text(i, bottom + 0.5, str(order_metric[sort_idx[i]]),
                ha="center", va="bottom", fontsize=7, color="black")

    # cosmetics
    ax.set_xticks(range(len(drugs)))
    ax.set_xticklabels(drugs, fontsize=8)
    ax.yaxis.set_visible(False)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)

    # legend
    patches = []
    for k in stack_order:
        style = styles[k]
        patches.append(mpatches.Patch(
            facecolor=style["facecolor"],
            edgecolor=style["edgecolor"],
            hatch=style["hatch"] if style["hatch"] else "",
            label=labels[k]
        ))
    if legend:
        ax.legend(handles=patches, fontsize=7, frameon=False,
                  loc="upper center", ncol=2)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path)
    plt.show()
    plt.close(fig)



import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

def plot_stacked_percentage_bars(expanded_catalogues, pair_coverages, valid_drugs,
                                 output_path=None, figsize=(6.69, 10), legend=True):
    """
    Stacked percentage bar plots (two axes):
      Top axis    = mutation overlaps (expanded_catalogues)
      Bottom axis = sample ID overlaps (pair_coverages)
    """

    # Fixed plotting order
    drug_order = ["BDQ", "CFZ", "DLM", "LZD", "AMI", "KAN", "PZA", "EMB",
                  "MXF", "LEV", "CAP", "ETH", "STM", "RIF", "INH"]

    # Colors
    colors = {
        "cat_only": mcolors.to_rgba("#bb3e4d", 1.0),       # red
        "both_rules": mcolors.to_rgba("#9c969c", 1),       # grey full
        "both_excluded": mcolors.to_rgba("#9c969c", 0.5),  # grey transparent
        "who_only_rules": mcolors.to_rgba("#3582B6", 0.93),   # blue full
        "who_only_excluded": mcolors.to_rgba("#3582B6", 0.5) # blue transparent
    }

    labels = {
        "cat_only": "catomatic-1",
        "both_rules": "catomatic-1; WHOv1 rules",
        "both_excluded": "catomatic-1; WHOv1 algorithm",
        "who_only_excluded": "WHOv1 algorithm",
        "who_only_rules": "WHOv1 rules",
    }

    legend_order = ["cat_only", "both_rules", "both_excluded", "who_only_rules", "who_only_excluded"]
    legend_order = ["who_only_rules", "who_only_excluded", "both_excluded", "both_rules", "cat_only"]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    drugs = []
    segs_all = []

    # --- collect mutation counts ---
    for drug in drug_order:
        if drug not in valid_drugs:
            continue

        cat_only_applied, who_only_applied, shared_applied = _counts_for(drug, expanded_catalogues, use_filtered=False)
        _, who_only_excluded, shared_excluded = _counts_for(drug, expanded_catalogues, use_filtered=True)

        segs = {
            "cat_only": cat_only_applied,
            "both_rules": max(shared_applied - shared_excluded, 0),
            "both_excluded": shared_excluded,
            "who_only_rules": max(who_only_applied - who_only_excluded, 0),
            "who_only_excluded": who_only_excluded
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
        for key in ["cat_only", "both_rules", "both_excluded", "who_only_excluded", "who_only_rules"]:
            h = segs[key + "_pct"]
            if h > 0:
                ax1.bar(i, h, bottom=bottom, width=0.75,
                        color=colors[key], edgecolor="white", linewidth=1)
                if h>4:
                    ax1.text(i, bottom + h/2,
                            f"{segs[key]}\n{h:.1f}%",
                            ha="center", va="center", fontsize=7)
                else:
                    ax1.text(i, bottom + h/2,
                            f"{segs[key]}",
                            ha="center", va="center", fontsize=7)

                bottom += h
        #ax1.text(i, 101, f"{segs['total']}", ha="center", va="bottom",
         #        fontsize=8, fontweight="bold")

    # --- PLOT 2: SAMPLES ---
    for i, drug in enumerate(drugs):
        cat_only, who_only_rules, who_only_excl, both_rules, both_excluded, none = _counts_for_ids(drug, pair_coverages)
        segs_ids = {
            "cat_only": cat_only,
            "both_rules": both_rules,
            "both_excluded": both_excluded,
            "who_only_rules": who_only_rules,
            "who_only_excluded": who_only_excl
        }
        total_ids = sum(segs_ids.values()) + none

        bottom = 0
        for key in ["cat_only", "both_rules", "both_excluded", "who_only_excluded", "who_only_rules"]:
            h = 100 * segs_ids[key] / total_ids if total_ids > 0 else 0
            if h > 0:
                ax2.bar(i, h, bottom=bottom, width=0.75,
                        color=colors[key], edgecolor="white", linewidth=1)
                if h > 4:
                    ax2.text(i, bottom + h/2,
                            f"{segs_ids[key]}\n{h:.1f}%",
                            ha="center", va="center", fontsize=7)
                elif h > 1.4:
                    ax2.text(i, bottom + h/2,
                            f"{segs_ids[key]}",
                            ha="center", va="center", fontsize=7)

                bottom += h
        #ax2.text(i, 101, f"{total_ids}", ha="center", va="bottom",
        #         fontsize=8, fontweight="bold")

    # Cosmetics
    for ax in (ax1, ax2):
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.5, len(drugs)-0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks(range(0, 101, 20))
        ax.tick_params(axis="y", labelsize=8)

    ax1.set_ylabel("Proportion of variants captured", fontsize=8)
    ax2.set_ylabel("Proportion of samples captured (with at least 1 mutation)", fontsize=8)
    ax2.set_xticks(range(len(drugs)))
    ax2.set_xticklabels(drugs, fontsize=8)

    # Legend (shared)
    if legend:
        patches = [mpatches.Patch(color=colors[k], label=labels[k]) for k in legend_order]
        fig.legend(handles=patches,
                   fontsize=7, frameon=False,
                   loc="upper center",
                   bbox_to_anchor=(0.5, -0.00),
                   ncol=3)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)





def plot_pheno_counts(phenotypes, title, savefig):
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
    #plot_order = total_counts.sort_values("count", ascending=False)["DRUG"].tolist()

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

def plot_catalogue_bar_charts(perf_df):
    """Plots 3 hollow horizontal bar charts (catalogued_R, catalogued_S, catalogued_RS) for each drug based on Build_FRS."""

    metrics = ["catalogued_R", "catalogued_S"]
    titles = ["R rows", "S rows"]

    for drug in perf_df["Drug"].unique():
        drug_df = perf_df[perf_df["Drug"] == drug].copy()

        # Round Build_FRS to 2 decimals to avoid float noise
        drug_df["Build_FRS"] = drug_df["Build_FRS"].round(2)

        drug_df = (
            drug_df
            .drop_duplicates(subset=["Build_FRS"])
            .sort_values("Build_FRS"))

        fig, axes = plt.subplots(1, 2, figsize=(1.2, 2.4), sharey=True)

        #min_val = np.max([0, np.min([drug_df['catalogued_R'].min(), drug_df['catalogued_S'].min(), drug_df['catalogued_RS'].min()])])
        #max_val = np.max([drug_df['catalogued_R'].max(), drug_df['catalogued_S'].max(), drug_df['catalogued_RS'].max()])

        combined_array = np.concatenate([drug_df["catalogued_R"].to_numpy(), drug_df["catalogued_S"].to_numpy()])
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
                alpha=0.7
            )
            

            ax.set_xlim(0, max_val + 1.5)

            # Hide x-axis completely
            ax.set_xlabel("")
            ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)

            # Add count numbers at end of bars
            for idx, (yval, width) in enumerate(zip(drug_df["Build_FRS"], drug_df[metric])):
                ax.text(
                    width + 0.2,  # small margin to right
                    yval,
                    f"{int(width)}",            # text to show
                    va='center',
                    ha='left',
                    fontsize=6,
                    alpha=0.7
                )
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_color((0, 0, 0, 0.5))  # (R, G, B, Alpha)
            ax.set_title(title, fontsize=7)
            ax.tick_params(labelsize=7)
            sns.despine(ax=ax, left=False, bottom=True)

            if i != 0:
                ax.set_yticks([])


        fig.suptitle(drug, fontsize=9)
        plt.tight_layout()
        plt.savefig(f"figs/frs/{drug}_catalogue_counts.pdf", bbox_inches="tight", transparent=True)
        plt.show()
        plt.close()

def plot_fitted_distribution(effects, df, log_ecoff, min_count, x_min, x_max):

    for _, row in effects.iterrows():
        mutation_name = row['Mutation']
        log2_mic = row['effect_size']  # Assuming log2(MIC) is stored in 'effect_size'
        mic = row['MIC']  # Actual MIC value in the 'MIC' column

        # Filter the DataFrame directly for the current mutation
        mutation_df = df[df['MUTATION'] == mutation_name]

        if len(mutation_df) > min_count:
            # Extract the intervals directly from the DataFrame
            mutation_intervals = list(zip(mutation_df['y_low_log'], mutation_df['y_high_log']))

            # Handle np.inf by replacing high values with an arbitrarily large width
            processed_intervals = []
            for low, high in mutation_intervals:
                if high == np.inf:
                    processed_intervals.append((low, x_max))  # Cap the high value at the plot limit
                else:
                    processed_intervals.append((low, high))

            # Get unique intervals for the current mutation
            unique_intervals = sorted(set(processed_intervals))

            # Calculate counts for each unique interval
            mutation_mic_counts = [processed_intervals.count(interval) for interval in unique_intervals]

            # Extract the midpoints and widths for plotting the bars
            interval_midpoints = [
                (low + (high if high != x_max else x_max)) / 2
                for low, high in unique_intervals
            ]
            interval_widths = [
                (high - low if high != x_max else x_max - low)
                for low, high in unique_intervals
            ]

            plt.figure(figsize=(2.5, 1.5))  # Create a new figure for each mutation

            # Step 1: Plot the histogram of calculated MIC intervals for this mutation
            plt.bar(interval_midpoints, height=mutation_mic_counts, width=interval_widths,
                    align='center', edgecolor='black', color='skyblue', label='True MIC Distribution')

            plt.axvline(x=log_ecoff, linestyle='--', color='orange')

            # Step 2: Overlay the fitted normal distribution for the current mutation
            x_values = np.linspace(x_min, x_max, 100)
            

            # Generate the normal distribution using log2(MIC) (effect size) and std
            y_values = norm.pdf(x_values, loc=log2_mic, scale=row['effect_std'])

            # Scale the normal distribution to match the height of the histogram
            y_values *= max(mutation_mic_counts) / max(y_values)

            # Plot the fitted curve
            plt.plot(x_values, y_values, label=f'Fitted Curve for {mutation_name}', linestyle='-', color='red')

            # Add text annotation for log2(MIC) and MIC
            #annotation_text = f"log2(MIC): {log2_mic:.2f} \\nMIC: {mic:.2f}"
            annotation_text = f"MIC: {mic:.2f}"
            plt.text(x_min + 0.5, max(mutation_mic_counts) * 0.8, annotation_text, fontsize=7, color='black',
                    bbox=dict(facecolor='white', edgecolor='white', alpha=0.7))

            # Customize the plot
            plt.xlabel('log2(MIC)', fontsize=7)
            plt.ylabel('Counts', fontsize=7)
            print (mutation_name)
            #plt.title(f'{mutation_name}', fontsize=9)  # Smaller font size
            plt.xlim([x_min, x_max])  # Set the consistent x-axis range

            # Set xticks to log2(MIC) values but label them with MIC values (2**x)
            ax = plt.gca()
            xticks = ax.get_xticks()
            mic_labels = [f'{2**x:.0f}' if 2**x >= 1 else f'{2**x:.2f}' for x in xticks]
            ax.set_xticklabels(mic_labels, fontsize=7)
            ax.set_xlabel('MIC (μg/mL)', fontsize=7)

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Show the plot for this mutation
            plt.show()


def grid_search_plots(df):
    metric_colors = {
        "SENSITIVITY": "#8B0000",  # dark red
        "SPECIFICITY": "#4682B4",  # steel blue
        "DPR": "#2E8B57"           # sea green
    }
    for drug in df["DRUG"].unique():
        drug_data = df[df["DRUG"] == drug]

        plt.figure(figsize=(2, 1.5))

        for metric in drug_data["Metric"].unique():
            for p in drug_data["p_value"].unique():
                subset = drug_data[
                    (drug_data["Metric"] == metric) &
                    (drug_data["p_value"] == p)
                ]
                sns.lineplot(
                    data=subset,
                    x="BACKGROUND_RATE",
                    y="value",
                    color=metric_colors[metric],
                    alpha=0.5 if p == 0.90 else 1.0,  # Example: faded for 0.90, solid for 0.95
                    marker="o",
                    linestyle="--" if p == 0.90 else "-",  # Dashed for 0.90
                    markersize=3,
                    linewidth=1.7 if p == 0.9 else 1.2
                )

        plt.title(drug, fontsize=7)
        plt.xlabel("")
        plt.ylabel("")  # No y-label
        plt.ylim(0, 105)
        plt.tick_params(labelsize=7)
        sns.despine(top=True, right=True)
        plt.xticks(np.arange(0.05, 0.26, 0.05))  

        plt.tight_layout()
        plt.savefig(f'figs/grid_search/{drug}.pdf', transparent=True)

        plt.show()
        plt.close()



def grid_search_radars(df):
    labels = ['SPECIFICITY', 'DPR', 'SENSITIVITY']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Unique values
    drugs = df['DRUG'].unique()
    p_values = df['p_value'].unique()

    # Assign colormaps per p_value
    pval_colormaps = {
        0.95: cm.Blues,
        0.90: cm.Oranges
    }

    # Angle shift for symmetric offset (±10 degrees)
    angle_shift = np.deg2rad(10)

    # Plot one radar chart per drug
    for j, drug in enumerate(drugs):
        fig, ax = plt.subplots(figsize=(2, 2), subplot_kw=dict(polar=True))
        subset = df[df['DRUG'] == drug]

        for p in p_values:
            sub_p = subset[subset['p_value'] == p]
            if sub_p.empty:
                continue

            background_rates = sorted(sub_p['BACKGROUND_RATE'].unique())
            color_map = pval_colormaps[p](np.linspace(0.4, 1, len(background_rates)))

            for i, rate in enumerate(background_rates):
                row = sub_p[sub_p['BACKGROUND_RATE'] == rate]
                if row.empty:
                    continue

                values = [row[label].values[0] for label in labels]
                values += values[:1]

                # Symmetric angle offset for p-values
                if p == 0.95:
                    angle_set = [(a - angle_shift) % (2 * np.pi) for a in angles]
                elif p == 0.90:
                    angle_set = [(a + angle_shift) % (2 * np.pi) for a in angles]
                else:
                    angle_set = angles

                ax.plot(angle_set, values,
                        label=f"p={p}, bg={rate}",
                        color=color_map[i], linewidth=0.65)
                ax.fill(angle_set, values, color=color_map[i], alpha=0.05)

        ax.set_theta_offset(-np.pi / 2)  # rotate 90° counterclockwise
        ax.set_xticks(angles[:-1])
        ax.tick_params(axis='y', labelsize=6, labelcolor='grey')
        if j == len(drugs):
            ax.set_xticklabels(['Specificity', 'DPR', 'Sensitivity'], fontsize=7)
        else:
            ax.set_xticklabels([''] * len(labels))
        ax.set_ylim(0, 1)
        ax.set_title(f"{drug}", fontsize=7)
        plt.tight_layout()
        plt.show()


    #plot seperate legend:
    # Prepare sorted unique values
    bg_rates = sorted(df['BACKGROUND_RATE'].unique())
    pvals = sorted(df['p_value'].unique())

    # Create a small figure for the legend
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.axis("off")  # turn off the axis

    # Column headers (p-values)
    for j, p in enumerate(pvals):
        ax.text(0.35 + j * 0.25, 0.9, f"p={p}", ha='center', va='bottom', fontsize=7)

    # Row headers (background rates) and legend boxes
    for i, bg in enumerate(bg_rates):
        y = 0.8 - i * 0.13
        ax.text(0.1, y, f"bg={bg:.2f}", ha='right', va='center', fontsize=7)
        for j, p in enumerate(pvals):
            # Get color for this bg/p combination
            cmap = pval_colormaps[p]
            color_index = bg_rates.index(bg)
            color = cmap(np.linspace(0.4, 1, len(bg_rates)))[color_index]
            
            # Draw a colored line as legend sample
            ax.plot([0.35 + j * 0.25 - 0.03, 0.35 + j * 0.25 + 0.03],
                    [y, y], color=color, linewidth=4)

    plt.tight_layout()
    plt.show()


def FRS_essential_violins(data):
    order = ["Essential", "Non-Essential"]

    # Define outlier gene to remove
    outlier_gene = "embB"

    # First dataset: all data < 0.9 FRS
    plot_data = data[data["Read Support"] < 0.9].copy()

    # Second dataset: outlier gene removed
    plot_data_no_outlier = plot_data[plot_data["Gene"] != outlier_gene]

    # Count totals for scaling
    total_counts = data["Gene Category"].value_counts()
    shown_counts = plot_data["Gene Category"].value_counts()
    scaling_factors = (shown_counts / total_counts).to_dict()

    fig, ax = plt.subplots(figsize=(2.2, 2))

    # === Plot full violins first ===
    sns.violinplot(
        x="Gene Category",
        y="Read Support",
        data=plot_data,
        inner=None,
        cut=0,
        bw=0.2,
        scale="area",
        palette={"Essential": "#CFAFEF", "Non-Essential": "#FFD1DC"},
        ax=ax,
        order=order
    )

    # Save number of violins before inset
    n_base_violins = len(ax.findobj(match=PolyCollection))

    # === Plot second set (inset violins) ===
    sns.violinplot(
        x="Gene Category",
        y="Read Support",
        data=plot_data_no_outlier,
        inner=None,
        cut=0,
        bw=0.2,
        scale="area",
        palette={"Essential": "#AF7AC5", "Non-Essential": "#F5B7B1"},
        ax=ax,
        order=order
    )

    # === Scale both sets ===
    category_positions = dict(zip(order, range(len(order))))
    max_scale = max(scaling_factors.values())

    # Process all violins
    all_artists = ax.findobj(match=PolyCollection)

    # Get full Essential stats (with embB)
    essential_total = data[data["Gene Category"] == "Essential"]
    essential_total_count = len(essential_total)
    essential_minor_count = len(essential_total[essential_total["Read Support"] < 0.9])

    # Get Essential stats without embB
    essential_no_embB = essential_total[essential_total["Gene"] != outlier_gene]
    essential_no_embB_count = len(essential_no_embB)
    essential_no_embB_minor_count = len(essential_no_embB[essential_no_embB["Read Support"] < 0.9])

    # Compute fractions
    minor_frac_with_embB = essential_minor_count / essential_total_count
    minor_frac_no_embB = essential_no_embB_minor_count / essential_no_embB_count

    # Compute dynamic scaling for inset violin
    inset_scaling_factor = np.round(1 - (minor_frac_no_embB / minor_frac_with_embB), 2)


    for i, artist in enumerate(all_artists):
        print (artist)
        path = artist.get_paths()[0]
        verts = path.vertices
        x_mean = np.mean(verts[:, 0])

        # Match category
        closest_cat = min(category_positions, key=lambda k: abs(category_positions[k] - x_mean))
        scale = scaling_factors[closest_cat] / max_scale

        # For inset violins, apply extra scaling to visually reduce width (~70%)
        if i == n_base_violins:
            scale *= inset_scaling_factor  # dynamic scaling now!


        verts[:, 0] = (verts[:, 0] - x_mean) * scale + x_mean

        # Shift if category is "Essential"

        verts[:, 0] += 0.2  # or whatever offset looks good

        artist.set_alpha(0.6)
        artist.set_edgecolor("black")
        artist.set_linewidth(0.8)

    # === Labels & annotations ===
    ax.set_xlabel("")
    ax.set_ylabel("Fraction Read Support", fontsize=6.9)
    ax.tick_params(axis='both', labelsize=7)

    # Manual shift of tick positions
    base_ticks = ax.get_xticks()
    new_ticks = list(base_ticks)

    # Shift "Essential" tick (index 0) right by +0.2
    new_ticks[0] += 0.2
    new_ticks[1] += 0.2

    # Apply new tick positions and same labels
    ax.set_xticks(new_ticks)
    ax.set_xticklabels(order)

    # Optional annotation block as in your original script
    for i, cat in enumerate(order):
        print (i, cat)
        total = total_counts[cat]
        plotted = shown_counts.get(cat, 0)
        frac = plotted / total * 100

        label = f"{plotted} / {total}\n{frac:.2f}%"
        #label = f"\n{frac:.2f}%"

        ax.text(
            i + 0.2,
            1,
            label,
            ha="center",
            va="center",
            fontsize=5,
            fontweight="normal",
            color="black",
        )

    # Filter essential data excluding embB
    essential_data_no_embB = data[
        (data["Gene Category"] == "Essential") & (data["Gene"] != outlier_gene)
    ]

    # Filter minor alleles only (FRS < 0.9) for that same set
    essential_minors_no_embB = essential_data_no_embB[
        essential_data_no_embB["Read Support"] < 0.9
    ]

    # Get counts
    essential_inset_total = len(essential_data_no_embB)
    essential_inset_minors = len(essential_minors_no_embB)
    essential_inset_frac = 100 * essential_inset_minors / essential_inset_total

    # Annotate near inset violin
    ax.text(
        new_ticks[0] + 0.5,
        0.7,  # adjust as needed
        f"excl. embB:\n{essential_inset_minors} / {essential_inset_total}\n{essential_inset_frac:.2f}%",
        #f"excl. embB:\n{essential_inset_frac:.2f}%",

        ha="center",
        va="center",
        fontsize=5,
        color="black"
    )

    sns.despine(ax=ax, top=True, right=True)
    plt.xlim(-0.3, 1.5)
    plt.tight_layout()
    plt.savefig('figs/frs/essential_frs_violins.png')
    plt.show()


def frs_gene_violins(all_mutations):

    x_axis_order = [
        "rpoB", "gyrA", "gyrB", "rplC", "dprE1", "atpE",  "rpsL", 'rrs',
        "embA", "embB","eis", "tlyA", "pepQ", "inhA", "ethA", "ahpC",
        "katG", "ddn", "pncA", "Rv0678", "gid", "fabG1"
    ]

    # === STEP 1: Filter for FRS < 0.9 only ===
    plot_data = all_mutations[all_mutations['FRS'] < 0.9].copy()

    # === STEP 2: Compute fractions ===
    total_counts = all_mutations.groupby('GENE').size()
    shown_counts = plot_data.groupby('GENE').size()
    fractions = (shown_counts / total_counts).fillna(0)
    max_fraction = fractions.max()  # normalization reference

    plot_data['GENE'] = pd.Categorical(plot_data['GENE'], categories=x_axis_order, ordered=True)

    # === STEP 3: Build palette ===
    category_map = plot_data.set_index('GENE')['Category'].to_dict()
    full_category_map = {gene: category_map.get(gene, 'Non-Essential') for gene in x_axis_order}
    palette = {
        gene: {'Essential': '#CFAFEF', 'Non-Essential': '#FFD1DC'}.get(category, '#FFFFFF')
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
        scale="count",   # initial scaling, we'll override
        order=x_axis_order,
        ax=ax,
        palette=palette
    )

    ax.set_ylabel("Fraction Read Support", fontsize=7)
    ax.set_xlabel("")
    ax.tick_params(axis='both', labelsize=7)
    sns.despine(ax=ax, top=True, right=True)

    # === STEP 5: Rescale violin areas ===
    category_positions = dict(zip(
        [t.get_text() for t in ax.get_xticklabels()],
        range(len(ax.get_xticklabels()))
    ))

    from matplotlib.path import Path

    shrink_factor = 0.48   # try values like 0.6–0.8 until it looks good
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
        print(f"{gene}: {plotted}/{total} = {frac:.2f}%")

        ax.text(
            i,
            1.1 if i % 2 == 0 else 1.0,
            label,
            ha="center", va="center",
            fontsize=5.5
        )

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def split_FRS_essential_violins(data):
    outlier_gene = "embB"

    # Filter FRS < 0.9
    base = data[data["Read Support"] < 0.9].copy()

    # Three groups
    essential_all = base[base["Gene Category"] == "Essential"].copy()
    essential_all["Category3"] = "Essential"

    essential_no_embB = base[(base["Gene Category"] == "Essential") &
                             (base["Gene"] != outlier_gene)].copy()
    essential_no_embB["Category3"] = "Essential excl. embB"

    non_essential = base[base["Gene Category"] == "Non-Essential"].copy()
    non_essential["Category3"] = "Non-Essential"

    # Combine
    plot_data = pd.concat([essential_all, essential_no_embB, non_essential], axis=0)

    order = ["Essential", "Essential excl. embB", "Non-Essential"]

    # Totals for scaling + annotation
    total_counts = {
        "Essential": len(data[data["Gene Category"] == "Essential"]),
        "Essential excl. embB": len(data[(data["Gene Category"] == "Essential") & (data["Gene"] != outlier_gene)]),
        "Non-Essential": len(data[data["Gene Category"] == "Non-Essential"])
    }
    shown_counts = plot_data["Category3"].value_counts().to_dict()
    scaling_factors = {k: shown_counts.get(k, 0)/total_counts[k] for k in total_counts if total_counts[k] > 0}
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
            "Non-Essential": "#FFD1DC"
        },
        ax=ax
    )

    # Scale violin widths
    category_positions = dict(zip(order, range(len(order))))
    for artist in ax.findobj(match=PolyCollection):
        path = artist.get_paths()[0]
        verts = path.vertices
        x_mean = np.mean(verts[:, 0])
        closest_cat = min(category_positions, key=lambda k: abs(category_positions[k] - x_mean))
        scale = scaling_factors.get(closest_cat, 1) / max_scale
        verts[:, 0] = (verts[:, 0] - x_mean) * scale + x_mean
        artist.set_edgecolor("black")
        artist.set_linewidth(0.8)
        artist.set_alpha(0.8)

    # === Labels ===
    ax.set_xlabel("")
    ax.set_ylabel("Fraction Read Support", fontsize=7)
    ax.tick_params(axis='both', labelsize=7)
    sns.despine(ax=ax, top=True, right=True)

    # === Annotations ===

  #  for cat, xpos in category_positions.items():
 #       total = total_counts[cat]
 #       plotted = shown_counts.get(cat, 0)
 #       frac = (plotted / total * 100) if total > 0 else 0

  #      label = f"{plotted} / {total}\n{frac:.1f}%"
  #      ax.text(
  #          xpos,
  #          0.95,   # just above violin
  #          label,
  #          ha="center", va="bottom",
  #          fontsize=6
  #      )

    plt.tight_layout()
    plt.show()



def plot_frs_sens(df, drug="AMI", figsize=(4.3, 2.2)):
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
            alpha=alpha
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
            alpha=alpha
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



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def _counts_cat_vs_cats_all(drug, expanded_catalogues):
    """
    Returns (only_cat, only_cats_all, shared) for a given drug,
    using the merged table with PREDICTION_cat and PREDICTION_cats_all.
    """
    merged = expanded_catalogues[drug]['merged']

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


def plot_floating_bars_cat_vs_cats_all(expanded_catalogues, valid_drugs,
                                       output_path=None, figsize=(6.69, 4),
                                       legend=False, fontsize=7):
    """
    Floating stacked bar plot for cat vs cats_all (no rules):
      - Above line: CAT only
      - On the line (centered at y=0): In both
      - Below the line: CATS_ALL only
    With scaling applied to PZA and ETH (÷ factor for bar height, text shows true count).
    """

    drug_order = ["BDQ", 'PZA', 'ETH', "INH", "STM", 'CAP', 'EMB', "RIF", "DLM", "CFZ", "MXF",
                  "LEV", "AMI", "KAN", "LZD"]

    drugs, segs_all = [], []
    for drug in drug_order:
        if drug not in valid_drugs:
            continue

        cat_only, cats_all_only, both = _counts_cat_vs_cats_all(drug, expanded_catalogues)
        segs_all.append({
            "cat_only": cat_only,
            "both": both,
            "cats_all_only": cats_all_only
        })
        drugs.append(drug)

    # colors
    colors = {
        "cat_only": "#db5363",     # red
        "both": "#b1b0b3",         # purple
        "cats_all_only": "#fd8c53" # blue
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
            ax.bar(i, cats_all_h, bottom=0,
                color=colors["cats_all_only"], edgecolor="white", linewidth=1)
            ax.text(i, cats_all_h/2, str(segs["cats_all_only"]),
                    ha="center", va="center", fontsize=fontsize)

        # --- PURPLE (both, below axis) ---
        if both_h > 0:
            ax.bar(i, both_h, bottom=-both_h,
                color=colors["both"], edgecolor="white", linewidth=1)
            ax.text(i, -both_h/2, str(segs["both"]),
                    ha="center", va="center", fontsize=fontsize)

        # --- RED (cat_only, stacked further below purple) ---
        if cat_h > 0:
            ax.bar(i, cat_h, bottom=-both_h - cat_h,
                color=colors["cat_only"], edgecolor="white", linewidth=1)
            ax.text(i, -both_h - cat_h/2, str(segs["cat_only"]),
                    ha="center", va="center", fontsize=fontsize)

        # Add "÷ factor" note if scaled
        if factor > 1:
            ax.text(i, ax.get_ylim()[1] * 0.02,
                    f"÷ {factor}", ha="center", va="bottom",
                    fontsize=6, color="gray")

    # horizontal grey line at y=0
    ax.axhline(0, color="0.2", linewidth=0.6, alpha=0.35, zorder=-1)

    # cosmetics
    ax.set_xticks(range(len(drugs)))
    ax.set_xticklabels(drugs, fontsize=fontsize)

    ax.yaxis.set_visible(False)
    for side in ['left', 'top', 'right']:
        ax.spines[side].set_visible(False)

    ymin, ymax = ax.get_ylim()
    pad = (ymax - ymin) * 0.01
    ax.set_ylim(ymin - pad, ymax + pad)

    if legend:
        patches = [mpatches.Patch(color=colors[k], label=labels[k]) for k in legend_order[::-1]]
        ax.legend(handles=patches, fontsize=7, frameon=False, loc="lower right")

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    plt.show()
    plt.close(fig)
