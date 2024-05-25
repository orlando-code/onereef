# general
import numpy as np

# spatial
import xarray as xa

# plotting
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import colors as mcolors

# custom
import utils


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
    if not val_lims:
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
        cbar=default_cbar_dict["cbar"],
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
    cbar: bool = True,
    cmap_type: str = "seq",
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

    if label_style_dict:
        for k, v in label_style_dict.items():
            default_label_style_dict[k] = v
    if presentation_format:
        default_label_style_dict["color"] = "white"
        if cbar:
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
