from typing import Dict, List, Optional

import numpy as np
from matplotlib import pyplot as plt


def joe_hdg_error_std(error: Dict[str, float], save: Optional[str] = "") -> None:
    """
    Plots the heading error in a bar plot with a predefined structure for
    the IEEE JOE 2024 manuscript. The results are plot in two different colors
    which differentiate the self and the cross validations.

    Args:
        error (Dict[str, float]): key-value pairs dictionary with the cross
            validation result of each method as values and the methods' name as
            key.
        save (str, optional): Path where to save the image. If an empty string is
            passed, then the plot is just displayed in screen. By default, "".
    """
    # Compute max error
    max_error = max([v for v in error.values() if v is not None])

    # Sort the results to show the grouped as batch methods an real-time methods
    keys = ["RAW", "magyc_ls", "magyc_nls", "magyc_bfg", "twostep", "ellipsoid_fit", "magfactor3", "magyc_ifg"]
    error_f = {}
    for k in error.keys():
        if k in keys:
            error_f[k] = error[k]
    error = {k: error_f[k] for k in sorted(error_f, key=lambda k: keys.index(k))}

    methods_tag = [
        "RAW",
        r"$\mathbf{MAGYC\!-\!LS}$",
        r"$\mathbf{MAGYC\!-\!NLS}$",
        r"$\mathbf{MAGYC\!-\!BFG}$",
        "TWOSTEP",
        "Ellipsoid Fit",
        "MagFactor3",
        r"$\mathbf{MAGYC\!-\!IFG}$",
    ]

    # Get color palette
    colors = _get_color_palette()
    raw_color = "gray"
    magyc_color = colors["Alloy Orange"]
    benchmark_color = colors["Midnight Green"]
    fail_color = colors["Rufous"]
    bar_colors = [raw_color, magyc_color, benchmark_color, fail_color]

    # Plot Parameters: xticks and bars color and width.
    x = np.arange(len(keys))
    colors = bar_colors[:3]
    bar_width = 0.45

    # Create the figure
    _, ax = plt.subplots(figsize=(12, 9))

    # Create a bar plot for self and cross validation results.
    for i, k in enumerate(keys):
        v = error[k]
        k = methods_tag[i]
        # Plot the bars for the methods that did not fail.
        if v is not None:
            ax.bar(x[i], v, bar_width, color=colors[0] if k in ["RAW"] else colors[1] if "MAGYC" in k else colors[2])
            ax.text(x[i], v + 0.1, f"{v:.2f}", ha="center", fontsize=16)

        # Otherwise, use as placeholder a red dashed line.
        else:
            y_min, y_max = ax.get_yticks()[[0, -1]]
            y_center = 0.5 * (y_min + y_max)
            ax.axvline(x[i], 0, 0.51, color=fail_color, linestyle='--', linewidth=1.4)
            ax.axvline(x[i], 0.61, 1, color=fail_color, linestyle='--', linewidth=1.4)
            ax.text(x[i], y_center, r"$\mathbf{FAIL}$", ha="center", fontsize=16, rotation="vertical",
                    color=fail_color)

    # Add x- and y-axis labels.
    ax.set_xticks(x)
    ax.set_xticklabels(methods_tag, fontsize=16)
    ax.set_yticks(np.arange(int(max_error))[1:])
    ax.set_yticklabels([])

    # Group the labels in batch and real-time.
    # Batch methods.
    start_batch, end_batch = 1, 6
    ax.text((start_batch + (end_batch - start_batch)*0.5) / len(keys),
            -0.08,
            "Batch",
            transform=ax.transAxes,
            ha="center",
            fontsize=16
            )
    ax.annotate('', xy=(start_batch / len(keys), -0.05), xytext=(end_batch/len(keys), -0.05), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='<->', color='k', lw=1.2))
    # Real-time methods.
    start_real, end_real = 6, 8
    ax.text((start_real + (end_real - start_real)*0.5) / len(keys),
            -0.08,
            "Real Time",
            transform=ax.transAxes,
            ha="center",
            fontsize=16
            )
    ax.annotate('', xy=(start_real/len(keys), -0.05), xytext=(end_real/len(keys), -0.05), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='<->', color='k', lw=1.2))

    # Set axes parameters.
    ax.set_xlim(left=x[0]-0.2, right=x[-1]+3*0.2)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Create horizontal grid.
    ax.grid(True, linestyle='--', color='gray', linewidth=0.5, axis='y', alpha=0.2)
    ax.set_axisbelow(True)  # ensure grid is below other plot elements

    # Remove axes.
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add y-axis label
    ax.text(-0.01, 1.015, "Heading Error [deg]", transform=ax.transAxes, ha="left", fontsize=16)

    # Show the plot or save the plot given the save parameter.
    plt.tight_layout()
    if save:
        plt.savefig(save, format="pdf")
        plt.savefig(".".join([save.split(".")[0], "png"]))
    else:
        plt.show()
    plt.close()


def joe_hdg_error_violin(error: Dict[str, float], save: Optional[str] = "") -> None:
    """
    Plots the heading error in a bar plot with a predefined structure for
    the IEEE JOE 2024 manuscript. The results are plot in two different colors
    which differentiate the self and the cross validations.

    Args:
        error (Dict[str, float]): key-value pairs dictionary with the cross
            validation result of each method as values and the methods' name as
            key.
        save (str, optional): Path where to save the image. If an empty string is
            passed, then the plot is just displayed in screen. By default, "".
    """
    # Compute max error
    max_p75 = np.ceil(max([np.percentile(v, 75) for v in error.values() if v is not None]))
    min_p25 = np.floor(min([np.percentile(v, 25) for v in error.values() if v is not None]))

    # Sort the results to show the grouped as batch methods an real-time methods
    keys = ["RAW", "magyc_bfg", "twostep", "ellipsoid_fit", "magfactor3", "magyc_ifg"]
    error_f = {}
    for k in error.keys():
        if k in keys:
            error_f[k] = error[k]
    error = {k: error_f[k] for k in sorted(error_f, key=lambda k: keys.index(k))}
    samples = max([len(v) for v in error.values() if v is not None])
    error_array = [v if v is not None else np.ones((samples, )) * np.nan for v in error.values()]
    error_array = [np.clip(v, min_p25, max_p75) for v in error_array]

    methods_tag = [
        "RAW",
        r"$\mathbf{MAGYC\!-\!BFG}$",
        "TWOSTEP",
        "Ellipsoid Fit",
        "MagFactor3",
        r"$\mathbf{MAGYC\!-\!IFG}$",
    ]

    # Get color palette
    colors = _get_color_palette()
    raw_color = "gray"
    magyc_color = colors["Alloy Orange"]
    benchmark_color = colors["Midnight Green"]
    fail_color = colors["Rufous"]

    # Plot Parameters: xticks and bars color and width.
    x = np.arange(len(keys))
    colors = [raw_color] + [magyc_color] + 3 * [benchmark_color] + [magyc_color]

    # Create the figure
    _, ax = plt.subplots(figsize=(9, 7.5))

    plots = ax.violinplot(error_array, positions=x, showmeans=False, showmedians=True, showextrema=False)

    # Set the color of the body
    for pc, color in zip(plots['bodies'], colors):
        pc.set_facecolor(color)

    # Set the color of the lines
    plots['cmedians'].set_colors(colors)

    # Add failed methods
    for i, v in enumerate(error.values()):
        if v is not None:
            continue
        ax.axvline(x[i], 0.0, 0.48, color=fail_color, linestyle='--', linewidth=1.4)
        ax.axvline(x[i], 0.6, 1.0, color=fail_color, linestyle='--', linewidth=1.4)
        ax.text(x[i], min_p25 + 0.5*(max_p75 - min_p25), r"$\mathbf{FAIL}$", ha="center", fontsize=16,
                rotation="vertical", color=fail_color)

    # # Add x- and y-axis labels.
    ax.set_xticks(x)
    ax.set_xticklabels(methods_tag, fontsize=12)
    ax.set_yticks(np.arange(min_p25 - 1, max_p75 + 2)[::2])
    ax.set_yticklabels(np.arange(min_p25 - 1, max_p75 + 2)[::2], fontsize=12)

    # Group the labels in batch and real-time.
    # Batch methods.
    start_batch, end_batch = 1, 4
    ax.text((start_batch + (end_batch - start_batch)*0.5) / len(keys),
            -0.12,
            "Batch",
            transform=ax.transAxes,
            ha="center",
            fontsize=16
            )
    ax.annotate('', xy=(start_batch / len(keys), -0.09), xytext=(end_batch/len(keys), -0.09), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='<->', color='k', lw=1.2))
    # Real-time methods.
    start_real, end_real = 4, 6
    ax.text((start_real + (end_real - start_real)*0.5) / len(keys),
            -0.12,
            "Real Time",
            transform=ax.transAxes,
            ha="center",
            fontsize=16
            )
    ax.annotate('', xy=(start_real/len(keys), -0.09), xytext=(end_real/len(keys), -0.09), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='<->', color='k', lw=1.2))

    # Set axes parameters.
    ax.set_xlim(left=x[0]-0.4, right=x[-1]+3*0.2)
    ax.tick_params(axis='x', length=0)
    ax.tick_params(axis='y', length=0)
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Create horizontal grid.
    ax.grid(True, linestyle='--', color='gray', linewidth=0.5, axis='y', alpha=0.7)
    ax.set_axisbelow(True)  # ensure grid is below other plot elements

    # Remove axes.
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # # Add y-axis label
    ax.text(-0.01, 1.1, "Heading Error [deg]", transform=ax.transAxes, ha="left", fontsize=16)

    # Show the plot or save the plot given the save parameter.
    plt.tight_layout()
    if save:
        plt.savefig(save, format="pdf")
        plt.savefig(".".join([save.split(".")[0], "png"]))
    else:
        plt.show()
    plt.close()


def joe_pos_error(error_norm_xy, time, save=""):
    # Color selection
    palette_normalized = _get_color_palette()
    magyc_ls_color = palette_normalized["Auburn"]
    magyc_nls_color = palette_normalized["Mint"]
    magyc_bfg_color = palette_normalized["Alloy Orange"]
    magyc_ifg_color = palette_normalized["Midnight Green"]
    ellipsoid_fit_color = palette_normalized["Gamboge"]
    magfactor3_color = palette_normalized["Dark Cyan"]
    raw_color = palette_normalized["Rich Black"]

    line_colors = np.array([
        magyc_ls_color,
        magyc_nls_color,
        magyc_bfg_color,
        magyc_ifg_color,
        ellipsoid_fit_color,
        magfactor3_color,
        raw_color,
    ])

    # Define methods from dictionary
    keys = ["RAW", "magyc_ls", "magyc_nls", "magyc_bfg", "ellipsoid_fit", "magfactor3", "magyc_ifg"]
    error_f = {}
    for k in error_norm_xy.keys():
        if k in keys:
            error_f[k] = error_norm_xy[k]
    error_norm_xy = {k: error_f[k] for k in sorted(error_f, key=lambda k: keys.index(k))}

    # Create key-value pairs for the plot colors.
    colors = {k: line_colors[i, :] for i, k in enumerate(keys)}

    methods_tag = [
        "RAW",
        r"$\mathbf{MAGYC\!-\!LS}$",
        r"$\mathbf{MAGYC\!-\!NLS}$",
        r"$\mathbf{MAGYC\!-\!BFG}$",
        "Ellipsoid Fit",
        "MagFactor3",
        r"$\mathbf{MAGYC\!-\!IFG}$",
    ]

    # Create figure and grid for table with results summary.
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[0.90, 0.1])

    # Plot xy error norm over time.
    ax = fig.add_subplot(gs[0, :])

    # Remove Axes.
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set x and y ticks position.
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Set axes limits.
    ax.set_xlim(0, 135)
    ax.set_ylim(0.1, 140)

    # Set y-axis scale
    ax.set_yscale("log")

    # Remove ticks.
    ax.tick_params(length=0, axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off",
                   right="off", labelleft="on")

    # Set x and y ticks values.
    ax.set_yticks([10**j for j in [-1, 0, 1, 2]])
    ax.set_yticklabels([f"$10^{{{j}}}$" for j in [-1, 0, 1, 2]], fontsize=14)
    ax.set_xticks(range(0, 135, 20), [str(x) for x in range(0, 135, 20)], fontsize=14)

    # Plot horizontal lines.
    for y in [i * (10**j) for i in range(1, 10) for j in [-1, 0, 1, 2]]:
        ax.plot(range(0, 130), [y] * len(range(0, 130)), "--", lw=0.2, color="black", alpha=0.4)

    # Plot results.
    for i, (k, v) in enumerate(error_norm_xy.items()):
        # Manually offset label
        y_pos = v[-1] - 0.5
        if k == "magyc_ls":
            y_pos -= 0.0
        elif k == "magyc_nls":
            y_pos -= 0.5
        elif k == "magyc_bfg2":
            y_pos -= 0.0
        elif k == "magyc_ifg2":
            y_pos += 0.5
        elif k == "ellipsoid_fit":
            y_pos += 0.0
        elif k == "magfacto3":
            y_pos -= 0.0
        elif k == "RAW":
            y_pos -= 10.

        ax.plot((time - time[0]) / 60, v + 1e-09, lw=3, color=colors[k])

        # Write label for each line.
        ax.text(135.5, y_pos, methods_tag[i], fontsize=18, color=colors[k])

    # y-axis title with measurements unit.
    ax.text(-7, 150, 'Position Error [m]', fontsize=18, ha="left")

    # x-axis label.
    ax.set_xlabel('Time [min]', fontsize=18)

    # Table with summary of xy norm error at the last measurement.
    ax = fig.add_subplot(gs[1, :])

    # Add a table under the plot
    cell_text = [[f"{error_norm_xy[k][-1]:.2f} m" for k in keys]]
    colors = {k: v for k, v in zip(methods_tag, colors.values())}
    row_labels = ['']

    # Write values in the table
    table = plt.table(cellText=cell_text, colLabels=methods_tag, rowLabels=row_labels, cellLoc="center", loc='top left',
                      bbox=[-0.06, 0.0, 1.25, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1, 1)

    # Set color to text
    for j in range(len(keys)):
        for i in range(2):
            table[i, j].set_text_props(color=colors[table[0, j].get_text().get_text()], weight="bold")

    # Set the y axis to invisible
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Add subgroups for methods
    ax.text((0.12 + 0.835) * 0.5, -0.6, "Batch", transform=ax.transAxes, ha="center", fontsize=16)
    ax.annotate('', xy=(0.12, -0.2), xytext=(0.835, -0.2), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='<->', color='k', lw=1.2))
    ax.text((0.835 + 1.195) * 0.5, -0.6, "Real Time", transform=ax.transAxes, ha="center", fontsize=16)
    ax.annotate('', xy=(0.835, -0.2), xytext=(1.195, -0.2), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='<->', color='k', lw=1.2))

    # Show the plot
    plt.tight_layout()
    if save:
        plt.savefig(save, format="pdf")
        plt.savefig(".".join([save.split(".")[0], "png"]))
    else:
        plt.show()
    plt.close()


def _get_color_palette() -> Dict[str, List[int]]:
    # Palette in RGB [0, 255]
    palette_rgb = {
        "Rich Black": [0,  18,  25],
        "Midnight Green": [0,  95, 115],
        "Dark Cyan": [10, 147, 150],
        "Tiffany Blue": [148, 210, 189],
        "Vanilla": [233, 216, 166],
        "Gamboge": [238, 155,   0],
        "Alloy Orange": [202, 103,   2],
        "Rust": [187,  62,   3],
        "Rufous": [174, 32,  18],
        "Auburn": [155,  34,  38],
        "Mint": [82, 183, 136]
    }

    # Palette in normalized RGB [0, 1]
    palette_normalized = {k: [i/255.0 for i in v] for k, v in palette_rgb.items()}

    return palette_normalized
