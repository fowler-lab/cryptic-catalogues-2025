import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = 7
plt.rcParams["figure.figsize"] = (6.69, 5.02)


def plot_bar_who(df, who_results, outfile, who_drugs, show_graph=False):

    df_who1 = df[df.catalogue == "WHOv1"]
    df_who2 = df[df.catalogue == "WHOv2"]
    who1_results = who_results[who_results.catalogue == "WHO1"]
    who2_results = who_results[who_results.catalogue == "WHO2"]

    fig, (axis1, axis2) = plt.subplots(
        ncols=2, sharey=True, figsize=(7, 10), gridspec_kw={"width_ratios": [1, 1]}
    )

    y = np.arange(len(who_drugs))
    axis1.barh(
        y + 0.3,
        who1_results["sensitivity"],
        0.2,
        label=who1_results["sensitivity"],
        alpha=0.5,
        color="#cccccc",
    )
    subset = who1_results[["sensitivity"]]
    subset.columns = ["x"]
    iy = 0
    for idx, row in subset.iterrows():
        if row.x > 0:
            axis1.text(
                row.x + 2,
                iy + 0.3,
                "%.1f" % row.x,
                ha="right",
                va="center",
                color="#cccccc",
                fontweight="light",
            )
        iy += 1

    axis1.barh(
        y + 0.1,
        df_who1["SENSITIVITY2"] * 100,
        0.2,
        label=df_who1["SENSITIVITY2"],
        alpha=0.5,
        color="#ef6548",
    )
    subset = df_who1[["SENSITIVITY2"]] * 100
    subset.columns = ["x"]
    iy = 0
    for idx, row in subset.iterrows():
        if row.x > 0:
            axis1.text(
                row.x + 2,
                iy + 0.1,
                "%.1f" % row.x,
                ha="right",
                va="center",
                color="#ef6548",
                fontweight="light",
            )
        iy += 1

    axis1.barh(
        y - 0.1,
        who2_results["sensitivity"],
        0.2,
        label=who2_results["sensitivity"],
        alpha=0.5,
        color="#aaaaaa",
    )
    subset = who2_results[["sensitivity"]]
    subset.columns = ["x"]
    iy = 0
    for idx, row in subset.iterrows():
        if row.x > 0:
            axis1.text(
                row.x + 2,
                iy - 0.1,
                "%.1f" % row.x,
                ha="right",
                va="center",
                color="#aaaaaa",
                fontweight="light",
            )
        iy += 1

    axis1.barh(
        y - 0.3,
        df_who2["SENSITIVITY2"] * 100,
        0.2,
        label=df_who2["SENSITIVITY2"],
        alpha=0.5,
        color="#990000",
    )
    subset = df_who2[["SENSITIVITY2"]] * 100
    subset.columns = ["x"]
    iy = 0
    for idx, row in subset.iterrows():
        if row.x > 0:
            axis1.text(
                row.x + 2,
                iy - 0.3,
                "%.1f" % row.x,
                ha="right",
                va="center",
                color="#990000",
                fontweight="light",
            )
        iy += 1

    axis2.barh(
        y + 0.3,
        who1_results["specificity"],
        0.2,
        label=who1_results["specificity"],
        alpha=0.5,
        color="#cccccc",
    )
    axis2.set_yticks(y, who1_results["drug"])
    subset = who1_results[["specificity"]]
    subset.columns = ["x"]
    iy = 0
    for idx, row in subset.iterrows():
        if row.x > 0:
            axis2.text(
                row.x + 2,
                iy + 0.3,
                "%.1f" % row.x,
                ha="left",
                va="center",
                color="#cccccc",
                fontweight="light",
            )
        iy += 1

    axis2.barh(
        y + 0.1,
        df_who1["SPECIFICITY2"] * 100,
        0.2,
        label=df_who1["SPECIFICITY2"],
        alpha=0.5,
        color="#74a9cf",
    )
    subset = df_who1[["SPECIFICITY2"]] * 100
    subset.columns = ["x"]
    iy = 0
    for idx, row in subset.iterrows():
        if row.x > 0:
            axis2.text(
                row.x + 2,
                iy + 0.1,
                "%.1f" % row.x,
                ha="left",
                va="center",
                color="#74a9cf",
                fontweight="light",
            )
        iy += 1

    axis2.barh(
        y - 0.1,
        who2_results["specificity"],
        0.2,
        label=who2_results["specificity"],
        alpha=0.5,
        color="#aaaaaa",
    )
    subset = who2_results[["specificity"]]
    subset.columns = ["x"]
    iy = 0
    for idx, row in subset.iterrows():
        if row.x > 0:
            axis2.text(
                row.x + 2,
                iy - 0.1,
                "%.1f" % row.x,
                ha="left",
                va="center",
                color="#aaaaaa",
                fontweight="light",
            )
        iy += 1

    axis2.barh(
        y - 0.3,
        df_who2["SPECIFICITY2"] * 100,
        0.2,
        label=df_who2["SPECIFICITY2"],
        alpha=0.5,
        color="#034e7b",
    )
    subset = df_who2[["SPECIFICITY2"]] * 100
    subset.columns = ["x"]
    iy = 0
    for idx, row in subset.iterrows():
        if row.x > 0:
            axis2.text(
                row.x + 2,
                iy - 0.3,
                "%.1f" % row.x,
                ha="left",
                va="center",
                color="#034e7b",
                fontweight="light",
            )
        iy += 1

    axis1.invert_xaxis()  # Flip the x-axis for back-to-back effect
    # axis1.legend_.remove()  # Remove duplicate legend
    axis1.set_ylabel("")  # Remove the left-side y-axis label
    axis1.spines["top"].set_visible(False)
    axis1.set_yticks([])
    axis1.spines["left"].set_visible(False)
    axis1.set_yticklabels([])  # Remove y-tick labels

    axis2.spines["top"].set_visible(False)
    axis2.set_yticks([])
    axis2.spines["right"].set_visible(False)
    # axis2.set_yticks(y,who1_results['drug'])
    # axis2.set_yticklabels(who1_results['drug'])
    axis2.set_yticklabels([])  # Remove y-tick labels

    for i, drug in zip(y, who_drugs[::-1]):
        axis2.text(-11, i, drug, ha="center", va="center", fontsize=7)

    fig.savefig(outfile, bbox_inches="tight", dpi=600, transparent=True)

    if show_graph:
        plt.show()
    plt.close()


def plot_bar_catomatic(df, outfile, who_drugs, show_graph=False):

    df_who1 = df[df.catalogue == "WHOv1"]
    df_who2 = df[df.catalogue == "WHOv2"]
    df_cat1 = df[df.catalogue == "catomatic_v1"]

    fig, (axis1, axis2) = plt.subplots(
        ncols=2, sharey=True, figsize=(7, 10), gridspec_kw={"width_ratios": [1, 1]}
    )

    y = np.arange(len(who_drugs))

    axis1.barh(
        y + 0.2,
        df_who1["SENSITIVITY"] * 100,
        0.2,
        label=df_who1["SENSITIVITY"],
        alpha=0.2,
        color="#990000",
    )
    subset = df_who1[["SENSITIVITY"]] * 100
    subset.columns = ["x"]
    iy = 0
    for idx, row in subset.iterrows():
        if row.x > 0:
            axis1.text(
                row.x + 2,
                iy + 0.2,
                "%.1f" % row.x,
                ha="right",
                va="center",
                color="#ef6548",
                fontweight="light",
            )
        iy += 1

    axis1.barh(
        y,
        df_who2["SENSITIVITY"] * 100,
        0.2,
        label=df_who2["SENSITIVITY"],
        alpha=0.6,
        color="#990000",
    )
    subset = df_who2[["SENSITIVITY"]] * 100
    subset.columns = ["x"]
    iy = 0
    for idx, row in subset.iterrows():
        if row.x > 0:
            axis1.text(
                row.x + 2,
                iy,
                "%.1f" % row.x,
                ha="right",
                va="center",
                alpha=0.6,
                color="#990000",
                fontweight="light",
            )
        iy += 1

    axis1.barh(
        y - 0.2,
        df_cat1["SENSITIVITY"] * 100,
        0.2,
        label=df_cat1["SENSITIVITY"],
        alpha=1,
        color="#990000",
    )
    subset = df_cat1[["SENSITIVITY"]] * 100
    subset.columns = ["x"]
    iy = 0
    for idx, row in subset.iterrows():
        if row.x > 0:
            axis1.text(
                row.x + 2,
                iy - 0.2,
                "%.1f" % row.x,
                ha="right",
                va="center",
                color="#990000",
                fontweight="light",
            )
        iy += 1

    axis2.barh(
        y + 0.2,
        df_who1["SPECIFICITY"] * 100,
        0.2,
        label=df_who1["SPECIFICITY"],
        alpha=0.5,
        color="#74a9cf",
    )
    subset = df_who1[["SPECIFICITY"]] * 100
    subset.columns = ["x"]
    iy = 0
    for idx, row in subset.iterrows():
        if row.x > 0:
            axis2.text(
                row.x + 2,
                iy + 0.2,
                "%.1f" % row.x,
                ha="left",
                va="center",
                color="#74a9cf",
                fontweight="light",
            )
        iy += 1

    axis2.barh(
        y,
        df_who2["SPECIFICITY"] * 100,
        0.2,
        label=df_who2["SPECIFICITY"],
        alpha=0.5,
        color="#034e7b",
    )
    subset = df_who2[["SPECIFICITY"]] * 100
    subset.columns = ["x"]
    iy = 0
    for idx, row in subset.iterrows():
        if row.x > 0:
            axis2.text(
                row.x + 2,
                iy,
                "%.1f" % row.x,
                ha="left",
                va="center",
                color="#034e7b",
                fontweight="light",
            )
        iy += 1

    axis2.barh(
        y - 0.2,
        df_who2["SPECIFICITY"] * 100,
        0.2,
        label=df_who2["SPECIFICITY"],
        alpha=1,
        color="#034e7b",
    )
    subset = df_who2[["SPECIFICITY"]] * 100
    subset.columns = ["x"]
    iy = 0
    for idx, row in subset.iterrows():
        if row.x > 0:
            axis2.text(
                row.x + 2,
                iy - 0.2,
                "%.1f" % row.x,
                ha="left",
                va="center",
                color="#034e7b",
                fontweight="light",
            )
        iy += 1

    axis1.invert_xaxis()  # Flip the x-axis for back-to-back effect
    # axis1.legend_.remove()  # Remove duplicate legend
    axis1.set_ylabel("")  # Remove the left-side y-axis label
    axis1.spines["top"].set_visible(False)
    axis1.set_yticks([])
    axis1.spines["left"].set_visible(False)
    axis1.set_yticklabels([])  # Remove y-tick labels

    axis2.spines["top"].set_visible(False)
    axis2.set_yticks([])
    axis2.spines["right"].set_visible(False)
    # axis2.set_yticks(y,who1_results['drug'])
    # axis2.set_yticklabels(who1_results['drug'])
    axis2.set_yticklabels([])  # Remove y-tick labels

    for i, drug in zip(y, who_drugs[::-1]):
        axis2.text(-11, i, drug, ha="center", va="center", fontsize=7)

    fig.savefig(outfile, bbox_inches="tight", dpi=600, transparent=True)
    if show_graph:
        plt.show()
    plt.close()


def plot_bar_catomatic_coverage(df, outfile, who_drugs, show_graph=False):

    df_who1 = df[df.catalogue == "WHOv1"]
    df_who2 = df[df.catalogue == "WHOv2"]
    df_cat1 = df[df.catalogue == "catomatic_v1"]

    fig, (axis1) = plt.subplots(ncols=1, sharey=True, figsize=(2, 10))

    y = np.arange(len(who_drugs))

    axis1.barh(
        y + 0.2,
        df_who1["COVERAGE"] * 100,
        0.2,
        label=df_who1["COVERAGE"],
        alpha=0.33,
        color="#333333",
    )
    subset = df_who1[["COVERAGE"]] * 100
    subset.columns = ["x"]
    iy = 0
    for idx, row in subset.iterrows():
        if row.x > 0:
            axis1.text(
                row.x + 2,
                iy + 0.2,
                "%.1f" % row.x,
                ha="right",
                va="center",
                color="#333333",
                alpha=0.33,
                fontweight="light",
            )
        iy += 1

    axis1.barh(
        y,
        df_who2["COVERAGE"] * 100,
        0.2,
        label=df_who2["COVERAGE"],
        alpha=0.66,
        color="#333333",
    )
    subset = df_who2[["COVERAGE"]] * 100
    subset.columns = ["x"]
    iy = 0
    for idx, row in subset.iterrows():
        if row.x > 0:
            axis1.text(
                row.x + 2,
                iy,
                "%.1f" % row.x,
                ha="right",
                va="center",
                color="#333333",
                alpha=0.66,
                fontweight="light",
            )
        iy += 1

    axis1.barh(
        y - 0.2,
        df_cat1["COVERAGE"] * 100,
        0.2,
        label=df_cat1["COVERAGE"],
        alpha=1,
        color="#333333",
    )
    subset = df_cat1[["COVERAGE"]] * 100
    subset.columns = ["x"]
    iy = 0
    for idx, row in subset.iterrows():
        if row.x > 0:
            axis1.text(
                row.x + 2,
                iy - 0.2,
                "%.1f" % row.x,
                ha="right",
                va="center",
                color="#333333",
                alpha=1,
                fontweight="light",
            )
        iy += 1

    axis1.invert_xaxis()  # Flip the x-axis for back-to-back effect
    # axis1.legend_.remove()  # Remove duplicate legend
    axis1.set_ylabel("")  # Remove the left-side y-axis label
    axis1.spines["top"].set_visible(False)
    axis1.set_yticks([])
    axis1.spines["left"].set_visible(False)
    axis1.set_yticklabels([])  # Remove y-tick labels

    # axis2.spines["top"].set_visible(False)
    # axis2.set_yticks([])
    # axis2.spines["right"].set_visible(False)
    # # axis2.set_yticks(y,who1_results['drug'])
    # # axis2.set_yticklabels(who1_results['drug'])
    # axis2.set_yticklabels([])  # Remove y-tick labels

    # for i,drug in zip(y, who_drugs[::-1]):
    #     axis1.text(-11, i, drug, ha='center', va='center', fontsize=7)

    fig.savefig(outfile, bbox_inches="tight", dpi=600, transparent=True)

    if show_graph:
        plt.show()
    plt.close()
