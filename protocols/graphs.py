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
    "WHOv1": ("#fdbb84", "#a6bddb", "#99d8c9"),
    "WHOv1_no_rules": ("#c994c7", "#807dba", "#02818a"),
    "WHOv2": ("#ef6548", "#3690c0", "#41ae76"),
    "catomatic_v3.4.0": ("#990000", "#034e7b", "#005824"),
    "catomatic_v3.4.0_all_samples": ("#e31a1c", "#6a51a3", "#238b45"),

}


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
    figsize=(2.2, 7.5),
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
            (~merged.PREDICTION_cat.isna()) & (~merged.PREDICTION_cat_all.isna())
        ].MUTATION.nunique()

        only_cat = merged[
            (~merged.PREDICTION_cat.isna()) & (merged.PREDICTION_cat_all.isna())
        ].MUTATION.nunique()

        only_cat_all = merged[
            (merged.PREDICTION_cat.isna()) & (~merged.PREDICTION_cat_all.isna())
        ].MUTATION.nunique()

        venn_diagram = venn2(
            subsets=(only_cat, only_cat_all, shared),
            set_labels=('', ''),
            ax=ax,
            set_colors=(colours["catomatic_v3.4.0"][2], colours["catomatic_v3.4.0_all_samples"][2])
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
                set_colors=(colours["catomatic_v3.4.0"][2], colours["WHOv1"][2])
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


    x_axis_order = ["rpoB", "gyrA", "gyrB", "rplC", "dprE1", "atpE",  "rpsL", 'rrs', "embA", "embB","eis", "tlyA", "pepQ", "inhA", "ethA", "ahpC", "katG", "ddn", "pncA", "Rv0678", "gid", "fabG1"]

    # Filter for FRS < 0.9 only
    plot_data = all_mutations[all_mutations['FRS'] < 0.9].copy()

    # === STEP 2: Compute scaling factors ===
    total_counts = all_mutations.groupby('GENE').size()
    shown_counts = plot_data.groupby('GENE').size()
    scaling_factors = (shown_counts / total_counts).clip(upper=0.3).fillna(0).to_dict()

    plot_data['GENE'] = pd.Categorical(plot_data['GENE'], categories=x_axis_order, ordered=True)


    # Map categories for all genes in gene_list, defaulting to 'Non-Essential' (or whatever makes sense)
    category_map = plot_data.set_index('GENE')['Category'].to_dict()
    full_category_map = {gene: category_map.get(gene, 'Non-Essential') for gene in x_axis_order}

    # Build full palette using that
    palette = {
        gene: {'Essential': '#CFAFEF', 'Non-Essential': '#FFD1DC'}.get(category, '#FFFFFF')
        for gene, category in full_category_map.items()
    }

    # === STEP 3: Plot ===
    fig, ax = plt.subplots(figsize=(6.69, 2))

    violin = sns.violinplot(
        x="GENE",
        y="FRS",
        data=plot_data,
        inner=None,
        cut=0,
        bw=0.2,
        scale="area",
        order=x_axis_order,
        ax=ax,
        palette=palette)


    ax.set_ylabel("Fraction Read Support", fontsize=7)
    ax.set_xlabel("")
    ax.tick_params(axis='both', labelsize=7)
    sns.despine(ax=ax, top=True, right=True)

    # === STEP 4: Scale violin widths ===
    max_scale = max(scaling_factors.values())
    category_positions = dict(zip(
        [t.get_text() for t in ax.get_xticklabels()],
        range(len(ax.get_xticklabels()))
    ))

    # Scale violins
    for artist in ax.findobj(match=PolyCollection):
        path = artist.get_paths()[0]
        verts = path.vertices
        x_mean = np.mean(verts[:, 0])
        closest_gene = min(category_positions, key=lambda g: abs(category_positions[g] - x_mean))
        scale = scaling_factors.get(closest_gene, 0) / max_scale
        verts[:, 0] = (verts[:, 0] - x_mean) * scale + x_mean
        artist.set_edgecolor("black")
        artist.set_linewidth(0.8)
        artist.set_alpha(0.8)

    # === STEP 5: Annotate ===
    for gene, i in category_positions.items():
        total = total_counts.get(gene, 0)
        plotted = shown_counts.get(gene, 0)
        frac = (plotted / total * 100) if total > 0 else 0

        # Compute violin area
        area = None
        for artist in ax.findobj(match=PolyCollection):
            path = artist.get_paths()[0]
            verts = path.vertices
            x_mean = np.mean(verts[:, 0])
            matched = min(category_positions, key=lambda g: abs(category_positions[g] - x_mean))
            if matched == gene:
                x = verts[:, 0]
                y = verts[:, 1]
                sort_idx = np.argsort(y)
                x_sorted = x[sort_idx]
                y_sorted = y[sort_idx]
                side_mask = x_sorted < x_mean
                area = np.trapezoid(x_sorted[side_mask] - x_mean, y_sorted[side_mask]) * 2
                break

        #label = f"{plotted}/{total}\n{frac:.0f}%"
        label = f"{frac:.0f}%"

        print (f"{gene}: {plotted}/{total} = {frac:.2f}%")

        ax.text(
            i, 
            1.0, #1.1 if i % 2 == 0 else 1.0,  # even indices slightly higher
            label,
            ha="center", va="center",
            fontsize=6
        )

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()