# general
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# spatial
import xarray as xa
from rasterio import enums

# plotting
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import colors as mcolors
import seaborn as sns

# custom
from reeftruth import utils, resampling


def generate_geo_axis(
    figsize: tuple[float, float] = (10, 10), map_proj=ccrs.PlateCarree(), dpi=300
):
    return plt.figure(figsize=figsize, dpi=dpi), plt.axes(projection=map_proj)


def plot_spatial(
    xa_da: xa.DataArray,
    fax: Axes = None,
    title: str = "default",
    figsize: tuple[float, float] = (10, 10),
    val_lims: tuple[float, float] = None,
    presentation_format: bool = False,
    labels: list[str] = ["l", "b"],
    cbar_dict: dict = None,
    cartopy_dict: dict = None,
    label_style_dict: dict = None,
    map_proj=ccrs.PlateCarree(),
    alpha: float = 1,
    extent: list[float] = None,
) -> tuple[Figure, Axes]:
    """
    Plot a spatial plot with colorbar, coastlines, landmasses, and gridlines.

    Parameters
    ----------
    xa_da (xa.DataArray): The input xarray DataArray representing the spatial data.
    title (str, optional): The title of the plot.
    cbar_name (str, optional): The name of the DataArray.
    val_lims (tuple[float, float], optional): The limits of the colorbar range.
    cmap_type (str, optional): The type of colormap to use.
    symmetric (bool, optional): Whether to make the colorbar symmetric around zero.
    edgecolor (str, optional): The edge color of the landmasses.
    orientation (str, optional): The orientation of the colorbar ('vertical' or 'horizontal').
    labels (list[str], optional): Which gridlines to include, as strings e.g. ["t","r","b","l"]
    map_proj (str, optional): The projection of the map.
    extent (list[float], optional): The extent of the plot as [min_lon, max_lon, min_lat, max_lat].

    Returns
    -------
    tuple: The figure and axes objects.
    TODO: saving option and tidy up presentation formatting
    """
    # may need to change this
    # for some reason fig not including axis ticks. Universal for other plotting
    if not fax:
        if extent == "global":
            fig, ax = generate_geo_axis(figsize=figsize, map_proj=ccrs.Robinson())
            ax.set_global()
        else:
            fig, ax = generate_geo_axis(figsize=figsize, map_proj=map_proj)

    else:
        fig, ax = fax[0], fax[1]

    if isinstance(extent, list):
        ax.set_extent(extent, crs=map_proj)

    default_cbar_dict = {
        "cbar_name": None,
        "cbar": True,
        "orientation": "vertical",
        "cbar_pad": 0.1,
        "cbar_frac": 0.025,
        "cmap_type": "seq",
    }

    if cbar_dict:
        for k, v in cbar_dict.items():
            default_cbar_dict[k] = v
        if val_lims:
            default_cbar_dict["extend"] = "both"

    # if not cbarn_name specified, make name of variable
    cbar_name = default_cbar_dict["cbar_name"]
    if isinstance(xa_da, xa.DataArray) and not cbar_name:
        cbar_name = xa_da.name

    # if title not specified, make title of variable at resolution
    if title:
        if title == "default":
            resolution_d = np.mean(utils.calculate_spatial_resolution(xa_da))
            resolution_m = np.mean(utils.degrees_to_distances(resolution_d))
            title = (
                f"{cbar_name} at {resolution_d:.4f}° (~{resolution_m:.0f} m) resolution"
            )

    # if colorbar limits not specified, set to be maximum of array
    if not val_lims:  # TODO: allow dynamic specification of only one of min/max
        vmin, vmax = np.nanmin(xa_da.values), np.nanmax(xa_da.values)
    else:
        vmin, vmax = min(val_lims), max(val_lims)

    if (
        default_cbar_dict["cmap_type"] == "div"
        or default_cbar_dict["cmap_type"] == "res"
    ):
        if vmax < 0:
            vmax = 0.01
        cmap, norm = ColourMapGenerator().get_cmap(
            default_cbar_dict["cmap_type"], vmin, vmax
        )
    else:
        cmap = ColourMapGenerator().get_cmap(default_cbar_dict["cmap_type"])

    im = xa_da.plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,  # for further formatting later
        transform=ccrs.PlateCarree(),
        alpha=alpha,
        norm=(
            norm
            if (
                default_cbar_dict["cmap_type"] == "div"
                or default_cbar_dict["cmap_type"] == "res"
            )
            else None
        ),
    )

    if presentation_format:
        fig, ax = customize_plot_colors(fig, ax)
        # ax.tick_params(axis="both", which="both", length=0)

    # nicely format spatial plot
    format_spatial_plot(
        image=im,
        fig=fig,
        ax=ax,
        title=title,
        # cbar_name=cbar_name,
        # cbar=default_cbar_dict["cbar"],
        # orientation=default_cbar_dict["orientation"],
        # cbar_pad=default_cbar_dict["cbar_pad"],
        # cbar_frac=default_cbar_dict["cbar_frac"],
        cartopy_dict=cartopy_dict,
        presentation_format=presentation_format,
        labels=labels,
        cbar_dict=default_cbar_dict,
        label_style_dict=label_style_dict,
    )

    return fig, ax, im


def format_cbar(image, fig, ax, cbar_dict, labels: list[str] = ["l", "b"]):

    if cbar_dict["orientation"] == "vertical":
        cbar_rect = [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
        labels = [el if el != "b" else "t" for el in labels or []]
    else:
        cbar_rect = [
            ax.get_position().x0,
            ax.get_position().y0 - 0.04,
            ax.get_position().width,
            0.02,
        ]
        labels = [el if el != "b" else "t" for el in labels or []]
    cax = fig.add_axes(cbar_rect)

    cb = plt.colorbar(
        image,
        orientation=cbar_dict["orientation"],
        label=cbar_dict["cbar_name"],
        cax=cax,
        extend=cbar_dict["extend"] if "extend" in cbar_dict else "neither",
    )
    if cbar_dict["orientation"] == "horizontal":
        cbar_ticks = cb.ax.get_xticklabels()
    else:
        cbar_ticks = cb.ax.get_yticklabels()

    return cb, cbar_ticks, labels


def format_cartopy_display(ax, cartopy_dict: dict = None):

    default_cartopy_dict = {
        "category": "physical",
        "name": "land",
        "scale": "10m",
        "edgecolor": "black",
        "facecolor": (0, 0, 0, 0),  # "none"
        "linewidth": 0.5,
        "alpha": 0.3,
    }

    if cartopy_dict:
        for k, v in cartopy_dict.items():
            default_cartopy_dict[k] = v

    ax.add_feature(
        cfeature.NaturalEarthFeature(
            default_cartopy_dict["category"],
            default_cartopy_dict["name"],
            default_cartopy_dict["scale"],
            edgecolor=default_cartopy_dict["edgecolor"],
            facecolor=default_cartopy_dict["facecolor"],
            linewidth=default_cartopy_dict["linewidth"],
            alpha=default_cartopy_dict["alpha"],
        )
    )

    return ax


def format_spatial_plot(
    image: xa.DataArray,
    fig: Figure,
    ax: Axes,
    title: str = None,
    # cbar: bool = True,
    # cmap_type: str = "seq",
    presentation_format: bool = False,
    labels: list[str] = ["l", "b"],
    cbar_dict: dict = None,
    cartopy_dict: dict = None,
    label_style_dict: dict = None,
) -> tuple[Figure, Axes]:
    """Format a spatial plot with a colorbar, title, coastlines and landmasses, and gridlines.

    Parameters
    ----------
        image (xa.DataArray): image data to plot.
        fig (Figure): figure object to plot onto.
        ax (Axes): axes object to plot onto.
        title (str): title of the plot.
        cbar_name (str): label of colorbar.
        cbar (bool): whether to include a colorbar.
        orientation (str): orientation of colorbar.
        cbar_pad (float): padding of colorbar.
        edgecolor (str): color of landmass edges.
        presentation_format (bool): whether to format for presentation.
        labels (list[str]): which gridlines to include, as strings e.g. ["t","r","b","l"]
        label_style_dict (dict): dictionary of label styles.

    Returns
    -------
        Figure, Axes
    """
    if cbar_dict and cbar_dict["cbar"]:
        cb, cbar_ticks, labels = format_cbar(image, fig, ax, cbar_dict, labels)

    ax = format_cartopy_display(ax, cartopy_dict)
    ax.set_title(title)

    # format ticks, gridlines, and colours
    ax.tick_params(axis="both", which="major")
    default_label_style_dict = {"fontsize": 12, "color": "black", "rotation": 45}

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        # x_inline=False, y_inline=False
    )
    gl.xlines = False
    gl.ylines = False

    if label_style_dict:
        for k, v in label_style_dict.items():
            default_label_style_dict[k] = v
    if presentation_format:
        default_label_style_dict["color"] = "white"
        if cbar_dict and cbar_dict["cbar"]:
            plt.setp(cbar_ticks, color="white")
            cb.set_label(cbar_dict["cbar_name"], color="white")

    gl.xlabel_style = default_label_style_dict
    gl.ylabel_style = default_label_style_dict

    if (
        not labels
    ):  # if no labels specified, set up something to iterate through returning nothing
        labels = [" "]
    if labels:
        # convert labels to relevant boolean: ["t","r","b","l"]
        gl.top_labels = "t" in labels
        gl.bottom_labels = "b" in labels
        gl.left_labels = "l" in labels
        gl.right_labels = "r" in labels

    return fig, ax


def get_n_colors_from_hexes(
    num: int,
    hex_list: list[str] = ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#d83c04"],
) -> list[str]:
    """
    from Wes Anderson: https://github.com/karthik/wesanderson/blob/master/R/colors.R
    Get a list of n colors from a list of hex codes.

    Args:
        num (int): The number of colors to return.
        hex_list (list[str]): The list of hex codes from which to create spectrum for sampling.

    Returns:
        list[str]: A list of n hex codes.
    """
    cmap = get_continuous_cmap(hex_list)
    colors = [cmap(i / num) for i in range(num)]
    hex_codes = [mcolors.to_hex(color) for color in colors]
    return hex_codes


class ColourMapGenerator:
    """
    Get a colormap for colorbar based on the specified type.

    Parameters
    ----------
    cbar_type (str, optional): The type of colormap to retrieve. Options are 'seq' for sequential colormap and 'div'
        for diverging colormap.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap: The colormap object.
    """

    def __init__(self):
        self.sequential_hexes = ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#d83c04"]
        self.diverging_hexes = ["#3B9AB2", "#78B7C5", "#FFFFFF", "#E1AF00", "#d83c04"]
        self.cyclical_hexes = [
            "#3B9AB2",
            "#78B7C5",
            "#EBCC2A",
            "#E1AF00",
            "#d83c04",
            "#E1AF00",
            "#EBCC2A",
            "#78B7C5",
            "#3B9AB2",
        ]
        self.conf_mat_hexes = ["#EEEEEE", "#3B9AB2", "#cae7ed", "#d83c04", "#E1AF00"]
        self.residual_hexes = ["#3B9AB2", "#78B7C5", "#fafbfc", "#E1AF00", "#d83c04"]
        self.lim_red_hexes = ["#EBCC2A", "#E1AF00", "#d83c04"]
        self.lim_blue_hexes = ["#3B9AB2", "#78B7C5", "#FFFFFF"]

    def get_cmap(self, cbar_type, vmin=None, vmax=None):
        if cbar_type == "seq":
            return get_continuous_cmap(self.sequential_hexes)
        if cbar_type == "inc":
            return get_continuous_cmap(self.sequential_hexes[2:])
        elif cbar_type == "div":
            if not (vmin and vmax):
                raise ValueError(
                    "Minimum and maximum values needed for divergent colorbar"
                )
            cmap = get_continuous_cmap(self.diverging_hexes)
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            return cmap, norm
            # return get_continuous_cmap(self.diverging_hexes)
        elif cbar_type == "res":
            if not (vmin and vmax):
                raise ValueError(
                    "Minimum and maximum values needed for divergent colorbar"
                )
            cmap = get_continuous_cmap(self.residual_hexes)
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            return cmap, norm
        elif cbar_type == "cyc":
            return get_continuous_cmap(self.cyclical_hexes)
        elif cbar_type == "lim_blue":
            return get_continuous_cmap(self.lim_blue_hexes)
        elif cbar_type == "lim_red":
            return get_continuous_cmap(self.lim_red_hexes)
        elif cbar_type == "spatial_conf_matrix":
            return mcolors.ListedColormap(self.conf_mat_hexes)
        else:
            raise ValueError(f"{cbar_type} not recognised.")


def hex_to_rgb(value):
    """
    Convert a hexadecimal color code to RGB values.

    Parameters
    ----------
    value (str): The hexadecimal color code as a string of 6 characters.

    Returns
    -------
    tuple: A tuple of three RGB values.
    """
    value = value.strip("#")  # removes hash symbol if present
    hex_el = len(value)
    return tuple(
        int(value[i : i + hex_el // 3], 16)  # noqa
        for i in range(0, hex_el, hex_el // 3)
    )


def rgb_to_dec(value):
    """
    Convert RGB color values to decimal values (each value divided by 256).

    Parameters
    ----------
    value (list): A list of three RGB values.

    Returns
    -------
    list: A list of three decimal values.
    """
    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    """
    Create and return a color map that can be used in heat map figures.

    Parameters
    ----------
    hex_list (list of str): List of hex code strings representing colors.
    float_list (list of float, optional): List of floats between 0 and 1, same length as hex_list. Must start with 0
        and end with 1.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap: The created color map.
    """
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]]
            for i in range(len(float_list))
        ]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap("my_cmp", segmentdata=cdict, N=256)
    return cmp


def customize_plot_colors(fig, ax, background_color="#212121", text_color="white"):
    # Set figure background color
    fig.patch.set_facecolor(background_color)

    # Set axis background color (if needed)
    ax.set_facecolor(background_color)

    # Set text color for all elements in the plot
    for text in fig.texts:
        text.set_color(text_color)
    for text in ax.texts:
        text.set_color(text_color)
    for text in ax.xaxis.get_ticklabels():
        text.set_color(text_color)
    for text in ax.yaxis.get_ticklabels():
        text.set_color(text_color)
    ax.title.set_color(text_color)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)

    # Set legend text color
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color(text_color)
    # # set cbar labels
    # cbar = ax.collections[0].colorbar
    # cbar.set_label(color=text_color)
    # cbar.ax.yaxis.label.set_color(text_color)
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    return fig, ax


def plot_heatmap_from_dict_pairs(
    dictionary: dict,
    statistic: str,
    key1: str = "key 1",
    key2: str = "key 2",
    cbar_label: str = None,
):
    """
    Plot a heatmap from a dictionary of pairs.

    Args:
        dictionary (dict): A dictionary containing the data.
        statistic (str): The statistic to be plotted.
        key1 (str, optional): The label for the first key. Defaults to "key 1".
        key2 (str, optional): The label for the second key. Defaults to "key 2".
        cbar_label (str, optional): The label for the colorbar. If not provided, the statistic name will be used.

    Returns:
        None
    """
    av_iteration_times = {
        k1: np.mean([dictionary[k1][k2][statistic] for k2 in dictionary[k1].keys()])
        for k1 in dictionary.keys()
    }
    sorted_data = sorted(av_iteration_times, key=av_iteration_times.get)[::-1]

    heatmap_data = []
    for k1 in sorted_data:
        row = []
        for k2 in dictionary[k1].keys():
            row.append(dictionary[k1][k2][statistic])
        heatmap_data.append(row)

    # slightly hacky way to get rasterio resampling information
    if isinstance(sorted_data[0], enums.Resampling):
        sorted_data_names = [resampling.name for resampling in sorted_data]
    else:
        sorted_data_names = sorted_data

    heatmap_df = pd.DataFrame(
        heatmap_data, index=sorted_data_names, columns=dictionary[sorted_data[0]].keys()
    )

    plt.figure(figsize=(10, 8))

    if not cbar_label:
        cbar_label = statistic

    sns.heatmap(
        heatmap_df,
        annot=True,
        cmap=ColourMapGenerator().get_cmap("seq"),
        cbar_kws={"label": cbar_label},
    )
    plt.title(f"Heatmap of {statistic} for {key1} and {key2}")
    plt.xlabel(key2)
    plt.ylabel(key1)
    plt.show()


def plot_performance_against_key(timings_dict, k1_name, title: str = None):
    """
    Plot the average iteration duration against a key value.

    Parameters
    ----------
    timings_dict : dict
        A dictionary containing the timing data.
    k1_name : str
        The label for the key value.
    title : str, optional
        The title of the plot, by default None.

    Returns
    -------
    None
    """
    ks = [k for k in timings_dict.keys()]
    av_iter_times = {
        k1: np.mean(
            [timings_dict[k1][k2]["iteration_time"] for k2 in timings_dict[k1].keys()]
        )
        for k1 in timings_dict.keys()
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    # Sort values in ascending order of iteration time
    sorted_av_iter = sorted(av_iter_times.items(), key=lambda x: x[1])
    sorted_av_iter_vals = [sorted_av_iter[val][1] for val in range(len(sorted_av_iter))]

    sorted_ks = [k for k, v in sorted_av_iter]
    xs = np.arange(len(ks))  # Label locations

    # slightly hacky way to get rasterio resampling information
    if isinstance(ks[0], enums.Resampling):
        ks = [resampling.name for resampling in ks]

    # Generate colors for the bar plot
    colors = sns.color_palette(
        get_n_colors_from_hexes(len(sorted_ks), ColourMapGenerator().sequential_hexes)
    )

    # Plotting the total duration
    ax.set_xlabel(k1_name)
    ax.set_ylabel("Average iteration duration (s)")
    sns.barplot(sorted_av_iter_vals, palette=colors, ax=ax)
    ax.grid(axis="y")
    ax.set_xticks(xs)
    ax.set_xticklabels(ks)
    fig.tight_layout()  # Adjust layout to make room for both y-axes
    plt.title(title)


def plot_comparative_histograms_visuals(
    arrays,
    labels,
    val_lims: tuple[float, float] = [-100, 100],
    figsize: tuple[float, float] = None,
    cbar_dict={"cmap_type": "div", "orientation": "horizontal"},
    hist_dict={"bins": 100, "density": False, "alpha": 0.5, "yscale": "log"},
    n_hist_bins: int = 100,
    # combined: bool = False,   # TODO: allow plotting on single figure
):
    for a_i, array in tqdm(enumerate(arrays), total=len(arrays)):            
        fig = plt.figure(figsize=figsize if figsize else (15, 5))
        gs = fig.add_gridspec(1, 2, height_ratios=[1])
        ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax_hist = fig.add_subplot(gs[0, 1])

        # Create spatial plot with PlateCarree projection
        plot_spatial(
            array,
            fax=(fig, ax_map),
            cbar_dict=cbar_dict,
            val_lims=val_lims,
        )

        # Create histogram plot
        ax_hist.hist(
            array.values.flatten(),
            bins=hist_dict["bins"],
            alpha=hist_dict["alpha"],
            label=labels[a_i],
            density=hist_dict["density"],
        )
        ax_hist.set_title(labels[a_i])
        if hist_dict["yscale"] == "log":
            ax_hist.set_yscale("log")

        # Add labels
        ax_hist.set_xlabel("Value")
        if hist_dict["density"]:
            ax_hist.set_ylabel("Density")
        else:
            ax_hist.set_ylabel("Frequency")


def plot_two_methods_comparative_histograms_visuals(
    arrays1,
    arrays2,
    ax_labels,
    arrays1_label=None,
    arrays2_label=None,
    cbar_dict: dict = None,
    val_lims: tuple[float, float] = [-100, 100],
    hist_range: tuple[float, float] = [-100, 100],
    map_extents: list[float] = [140, 145, -15, -10],
):
    if len(arrays1) != len(arrays2):
        raise ValueError("Array of arrays must be the same length")
    for a_i in tqdm(range(len(arrays1)), total=len(arrays1)):
        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(1, 3, height_ratios=[1])
        ax_map1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax_hist = fig.add_subplot(gs[0, 1])
        ax_map2 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())

        # Create spatial plot with PlateCarree projection
        plot_spatial(
            arrays1[a_i],
            fax=(fig, ax_map1),
            cbar_dict=cbar_dict if cbar_dict else {"orientation": "horizontal"},
            val_lims=val_lims,
            title=arrays1_label,
        )
        plot_spatial(
            arrays2[a_i],
            fax=(fig, ax_map2),
            cbar_dict=cbar_dict if cbar_dict else {"orientation": "horizontal"},
            val_lims=val_lims,
            title=arrays2_label,
        )
        if map_extents:
            [
                ax_map.set_extent(map_extents, crs=ccrs.PlateCarree())
                for ax_map in [ax_map1, ax_map2]
            ]

        # Create histogram plot
        ax_hist.hist(
            arrays1[a_i].values.flatten(),
            bins=100,
            alpha=0.5,
            label="first_method" if not arrays1_label else arrays1_label,
            density=True,
            color="#d83c04",
        )
        ax_hist.hist(
            arrays2[a_i].values.flatten(),
            bins=100,
            alpha=0.3,
            label="second_method" if not arrays2_label else arrays2_label,
            density=True,
            color="#3B9AB2",
        )
        ax_hist.set_title(ax_labels[a_i])
        ax_hist.set_yscale("log")
        if hist_range:
            ax_hist.set_xlim(hist_range)

        # Add labels
        ax_hist.set_xlabel("Value")
        ax_hist.set_ylabel("Density")
        ax_hist.legend()


def grid_subplots(total, wrap=None, **kwargs):
    if wrap is not None:
        cols = min(total, wrap)
        rows = 1 + (total - 1) // wrap
    else:
        cols = total
        rows = 1
    fig, ax = plt.subplots(rows, cols, **kwargs)
    return fig, ax
