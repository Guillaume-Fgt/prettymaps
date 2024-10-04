import warnings
from dataclasses import dataclass
from typing import Iterable, Optional

import geopandas as gp
import matplotlib.axes
import matplotlib.figure
import numpy as np
import osmnx as ox
import shapely.affinity
import shapely.ops
from matplotlib import pyplot as plt
from matplotlib.patches import Path, PathPatch
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.geometry.base import BaseGeometry

from .fetch import get_gdfs


@dataclass
class Plot:
    """
    Dataclass implementing a prettymaps Plot object. Attributes:
    - geodataframes: A dictionary of GeoDataFrames (one for each plot layer)
    - fig: A matplotlib figure
    - ax: A matplotlib axis object
    - background: Background layer (shapely object)
    """

    geodataframes: dict[str, gp.GeoDataFrame]
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    background: BaseGeometry



def transform_gdfs(
    gdfs: dict[str, gp.GeoDataFrame],
    x: float = 0,
    y: float = 0,
    scale_x: float = 1,
    scale_y: float = 1,
    rotation: float = 0,
) -> dict[str, gp.GeoDataFrame]:
    """
    Apply geometric transformations to dictionary of GeoDataFrames

    Args:
        gdfs (Dict[str, gp.GeoDataFrame]): Dictionary of GeoDataFrames
        x (float, optional): x-axis translation. Defaults to 0.
        y (float, optional): y-axis translation. Defaults to 0.
        scale_x (float, optional): x-axis scale. Defaults to 1.
        scale_y (float, optional): y-axis scale. Defaults to 1.
        rotation (float, optional): rotation angle (in radians). Defaults to 0.

    Returns:
        Dict[str, gp.GeoDataFrame]: dictionary of transformed GeoDataFrames
    """
    # Project geometries
    gdfs = {
        name: ox.project_gdf(gdf) if len(gdf) > 0 else gdf for name, gdf in gdfs.items()
    }
    # Create geometry collection from gdfs' geometries
    collection = GeometryCollection(
        [GeometryCollection(list(gdf.geometry)) for gdf in gdfs.values()],
    )
    # Translation, scale & rotation
    collection = shapely.affinity.translate(collection, x, y)
    collection = shapely.affinity.scale(collection, scale_x, scale_y)
    collection = shapely.affinity.rotate(collection, rotation)
    # Update geometries
    for i, layer in enumerate(gdfs):
        gdfs[layer].geometry = list(collection.geoms[i].geoms)
        # Reproject
        if len(gdfs[layer]) > 0:
            gdfs[layer] = ox.project_gdf(gdfs[layer], to_crs="EPSG:4326")

    return gdfs


def PolygonPatch(shape: BaseGeometry, **kwargs) -> PathPatch:
    """_summary_

    Args:
        shape (BaseGeometry): Shapely geometry
        kwargs: parameters for matplotlib's PathPatch constructor

    Returns:
        PathPatch: matplotlib PatchPatch created from input shapely geometry
    """
    # Init vertices and codes lists
    vertices, codes = [], []
    for geom in shape.geoms if hasattr(shape, "geoms") else [shape]:
        for poly in geom.geoms if hasattr(geom, "geoms") else [geom]:
            if type(poly) != Polygon:
                continue
            # Get polygon's exterior and interiors
            exterior = np.array(poly.exterior.xy)
            interiors = [np.array(interior.xy) for interior in poly.interiors]
            # Append to vertices and codes lists
            vertices += [exterior] + interiors
            codes += list(
                map(
                    # Ring coding
                    lambda p: [Path.MOVETO]
                    + [Path.LINETO] * (p.shape[1] - 2)
                    + [Path.CLOSEPOLY],
                    [exterior] + interiors,
                ),
            )
    # Generate PathPatch
    return PathPatch(
        Path(np.concatenate(vertices, 1).T, np.concatenate(codes)),
        **kwargs,
    )


def plot_gdf(
    layer: str,
    gdf: gp.GeoDataFrame,
    ax: matplotlib.axes.Axes,
    palette: Optional[list[str]] = None,
    width: Optional[dict[str, float] | float] = None,
    union: bool = False,
    dilate_points: Optional[float] = None,
    dilate_lines: Optional[float] = None,
    **kwargs,
) -> None:
    """
    Plot a layer

    Args:
        layer (str): layer name
        gdf (gp.GeoDataFrame): GeoDataFrame
        ax (matplotlib.axes.Axes): matplotlib axis object
        mode (str): drawing mode. Options: 'matplotlib'. Defaults to 'matplotlib'
        palette (Optional[List[str]], optional): Color palette. Defaults to None.
        width (Optional[Union[dict, float]], optional): Street widths. Either a dictionary or a float. Defaults to None.
        union (bool, optional): Whether to join geometries. Defaults to False.
        dilate_points (Optional[float], optional): Amount of dilation to be applied to point (1D) geometries. Defaults to None.
        dilate_lines (Optional[float], optional): Amount of dilation to be applied to line (2D) geometries. Defaults to None.

    Raises:
        Exception: _description_
    """

    # Get hatch and hatch_c parameter
    hatch_c = kwargs.pop("hatch_c") if "hatch_c" in kwargs else None

    # Convert GDF to shapely geometries
    geometries = gdf_to_shapely(
        layer,
        gdf,
        width,
        point_size=dilate_points,
        line_width=dilate_lines,
    )

    # Unite geometries
    if union:
        geometries = shapely.ops.unary_union(GeometryCollection([geometries]))

    if (palette is None) and ("fc" in kwargs) and (type(kwargs["fc"]) != str):
        palette = kwargs.pop("fc")

    for shape in geometries.geoms if hasattr(geometries, "geoms") else [geometries]:
        if type(shape) in [Polygon, MultiPolygon]:
            # Plot main shape (without silhouette)
            ax.add_patch(
                PolygonPatch(
                    shape,
                    lw=0,
                    ec=(
                        hatch_c if hatch_c else kwargs["ec"] if "ec" in kwargs else None
                    ),
                    fc=(
                        kwargs["fc"]
                        if "fc" in kwargs
                        else np.random.choice(palette)
                        if palette
                        else None
                    ),
                    **{k: v for k, v in kwargs.items() if k not in ["lw", "ec", "fc"]},
                ),
            )
            # Plot just silhouette
            ax.add_patch(
                PolygonPatch(
                    shape,
                    fill=False,
                    **{k: v for k, v in kwargs.items() if k not in ["hatch", "fill"]},
                ),
            )
        elif type(shape) == LineString:
            ax.plot(
                *shape.xy,
                c=kwargs["ec"] if "ec" in kwargs else None,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k in ["lw", "ls", "dashes", "zorder"]
                },
            )
        elif type(shape) == MultiLineString:
            for c in shape.geoms:
                ax.plot(
                    *c.xy,
                    c=kwargs["ec"] if "ec" in kwargs else None,
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k in ["lw", "lt", "dashes", "zorder"]
                    },
                )



##########


def plot_legends(gdf, ax):

    for _, row in gdf.iterrows():
        name = row.name
        x, y = np.concatenate(row.geometry.centroid.xy)
        ax.text(x, y, name)


##########


def graph_to_shapely(gdf: gp.GeoDataFrame, width: float = 1.0) -> BaseGeometry:
    """
    Given a GeoDataFrame containing a graph (street newtork),
    convert them to shapely geometries by applying dilation given by 'width'

    Args:
        gdf (gp.GeoDataFrame): input GeoDataFrame containing graph (street network) geometries
        width (float, optional): Line geometries will be dilated by this amount. Defaults to 1..

    Returns:
        BaseGeometry: Shapely
    """

    def highway_to_width(highway):
        if (type(highway) == str) and (highway in width):
            return width[highway]
        if isinstance(highway, Iterable):
            for h in highway:
                if h in width:
                    return width[h]
            return np.nan
        return np.nan

    # Annotate GeoDataFrame with the width for each highway type
    gdf["width"] = gdf.highway.map(highway_to_width) if type(width) == dict else width

    # Remove rows with inexistent width
    gdf.drop(gdf[gdf.width.isna()].index, inplace=True)

    with warnings.catch_warnings():
        # Supress shapely.errors.ShapelyDeprecationWarning
        warnings.simplefilter("ignore", shapely.errors.ShapelyDeprecationWarning)
        if not all(gdf.width.isna()):
            # Dilate geometries based on their width
            gdf.geometry = gdf.apply(
                lambda row: row["geometry"].buffer(row.width),
                axis=1,
            )

    return shapely.ops.unary_union(gdf.geometry)


def geometries_to_shapely(
    gdf: gp.GeoDataFrame,
    point_size: Optional[float] = None,
    line_width: Optional[float] = None,
) -> GeometryCollection:
    """
    Convert geometries in GeoDataFrame to shapely format

    Args:
        gdf (gp.GeoDataFrame): Input GeoDataFrame
        point_size (Optional[float], optional): Point geometries (1D) will be dilated by this amount. Defaults to None.
        line_width (Optional[float], optional): Line geometries (2D) will be dilated by this amount. Defaults to None.

    Returns:
        GeometryCollection: Shapely geometries computed from GeoDataFrame geometries
    """

    geoms = gdf.geometry.tolist()
    collections = [x for x in geoms if type(x) == GeometryCollection]
    points = [x for x in geoms if type(x) == Point] + [
        y for x in collections for y in x.geoms if type(y) == Point
    ]
    lines = [x for x in geoms if type(x) in [LineString, MultiLineString]] + [
        y
        for x in collections
        for y in x.geoms
        if type(y) in [LineString, MultiLineString]
    ]
    polys = [x for x in geoms if type(x) in [Polygon, MultiPolygon]] + [
        y for x in collections for y in x.geoms if type(y) in [Polygon, MultiPolygon]
    ]

    # Convert points into circles with radius "point_size"
    if point_size:
        points = [x.buffer(point_size) for x in points] if point_size > 0 else []
    if line_width:
        lines = [x.buffer(line_width) for x in lines] if line_width > 0 else []

    return GeometryCollection(list(points) + list(lines) + list(polys))


def gdf_to_shapely(
    layer: str,
    gdf: gp.GeoDataFrame,
    width: Optional[dict | float] = None,
    point_size: Optional[float] = None,
    line_width: Optional[float] = None,
    **kwargs,
) -> GeometryCollection:
    """
    Convert a dict of GeoDataFrames to a dict of shapely geometries

    Args:
        layer (str): Layer name
        gdf (gp.GeoDataFrame): Input GeoDataFrame
        width (Optional[Union[dict, float]], optional): Street network width. Can be either a dictionary or a float. Defaults to None.
        point_size (Optional[float], optional): Point geometries (1D) will be dilated by this amount. Defaults to None.
        line_width (Optional[float], optional): Line geometries (2D) will be dilated by this amount. Defaults to None.

    Returns:
        GeometryCollection: Output GeoDataFrame
    """

    # Project gdf
    try:
        gdf = ox.project_gdf(gdf)
    except:
        pass

    if layer in ["streets", "railway", "waterway"]:
        geometries = graph_to_shapely(gdf, width)
    else:
        geometries = geometries_to_shapely(
            gdf,
            point_size=point_size,
            line_width=line_width,
        )

    return geometries


def override_args(
    layers: dict,
    circle: Optional[bool],
    dilate: Optional[float | bool],
) -> dict:
    """
    Override arguments in layers' kwargs

    Args:
        layers (dict): prettymaps.plot() Layers parameters dict
        circle (Optional[bool]): prettymaps.plot() 'Circle' parameter
        dilate (Optional[Union[float, bool]]): prettymaps.plot() 'dilate' parameter

    Returns:
        dict: output dict
    """
    override_args = ["circle", "dilate"]
    for layer in layers:
        for arg in override_args:
            if arg not in layers[layer]:
                layers[layer][arg] = locals()[arg]
    return layers





def create_background(
    gdfs: dict[str, gp.GeoDataFrame],
    style: dict[str, dict],
) -> BaseGeometry:
    """
    Create a background layer given a collection of GeoDataFrames

    Args:
        gdfs (Dict[str, gp.GeoDataFrame]): Dictionary of GeoDataFrames
        style (Dict[str, dict]): Dictionary of matplotlib style parameters

    Returns:
        Tuple[BaseGeometry]: background geometry, bounds, width and height
    """

    # Create background
    background_pad = 1.1
    if "background" in style and "pad" in style["background"]:
        background_pad = style["background"].pop("pad")

    background = shapely.affinity.scale(
        box(
            *shapely.ops.unary_union(ox.project_gdf(gdfs["perimeter"]).geometry).bounds,
        ),
        background_pad,
        background_pad,
    )

    if "background" in style and "dilate" in style["background"]:
        background = background.buffer(style["background"].pop("dilate"))

    return background







# Plot
def plot(
    # Your query. Example:
    # - "Porto Alegre"
    # - (-30.0324999, -51.2303767) (lat/long coordinates)
    # - You can also provide a custom GeoDataFrame boundary as input
    query: str | tuple[float, float] | gp.GeoDataFrame,
    layers={},
    # Matplotlib params for drawing each layer
    style={},
    # Custom postprocessing function on layers
    postprocessing=None,
    # Circular boundary? Default: square
    circle=None,
    # Radius for circular or square boundary
    radius=None,
    # Dilate boundary by this much
    dilate=None,
    # Whether to save result
    save_as=None,
    # Figure parameters
    fig=None,
    ax=None,
    figsize=(12, 12),
    # Whether to display matplotlib
    show=True,
    # Transform (translation, scale, rotation) parameters
    x=0,
    y=0,
    scale_x=1,
    scale_y=1,
    rotation=0,
):
    """

    Draw a map from OpenStreetMap data.

    Parameters
    ----------
    query : string
        The address to geocode and use as the central point around which to get the geometries
    postprocessing: function
        (Optional) Apply a postprocessing step to the 'layers' dict
    radius
        (Optional) If not None, draw the map centered around the address with this radius (in meters)
    layers: dict
        Specify the name of each layer and the OpenStreetMap tags to fetch
    style: dict
        Drawing params for each layer (matplotlib params such as 'fc', 'ec', 'fill', etc.)
    osm_credit: dict
        OSM Caption parameters
    figsize: Tuple
        (Optional) Width and Height (in inches) for the Matplotlib figure. Defaults to (10, 10)
    ax: axes
        Matplotlib axes
    title: String
        (Optional) Title for the Matplotlib figure
    x: float
        (Optional) Horizontal displacement
    y: float
        (Optional) Vertical displacement
    scale_x: float
        (Optional) Horizontal scale factor
    scale_y: float
        (Optional) Vertical scale factor
    rotation: float
        (Optional) Rotation in angles (0-360)

    Returns
    -------
    layers: dict
        Dictionary of layers (each layer is a Shapely MultiPolygon)

    Notes
    -----

    """

    # 2. Init matplotlib figure and ax
    if fig is None:
        fig = plt.figure(figsize=figsize, dpi=300)
    if ax is None:
        ax = plt.subplot(111, aspect="equal")

    # 3. Override arguments in layers' kwargs dict
    layers = override_args(layers, circle, dilate)

    # 4. Fetch geodataframes
    gdfs = get_gdfs(query, layers, radius, dilate, -rotation)

    # 5. Apply transformations to GeoDataFrames (translation, scale, rotation)
    gdfs = transform_gdfs(gdfs, x, y, scale_x, scale_y, rotation)

    # 6. Apply a postprocessing function to the GeoDataFrames, if provided
    if postprocessing:
        gdfs = postprocessing(gdfs)

    # 7. Create background GeoDataFrame and get (x,y) bounds
    background = create_background(gdfs, style)

    # 8. Draw layers
    for layer in gdfs:
        if (layer in layers) or (layer in style):
            plot_gdf(
                layer,
                gdfs[layer],
                ax,
                width=(
                    layers[layer]["width"]
                    if (layer in layers) and ("width" in layers[layer])
                    else None
                ),
                **(style[layer] if layer in style else {}),
            )


    # 9. Draw background
    if "background" in style:
        zorder = (
            style["background"].pop("zorder") if "zorder" in style["background"] else -1
        )
        ax.add_patch(
            PolygonPatch(
                background,
                **{k: v for k, v in style["background"].items() if k != "dilate"},
                zorder=zorder,
            ),
        )



    # 11. Ajust figure and create PIL Image

    # Adjust axis
    ax.axis("off")
    ax.axis("equal")
    ax.autoscale()
    # Adjust padding
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    # Save result
    if save_as:
        plt.savefig(save_as)
    if not show:
        plt.close()

    # Generate plot
    plot = Plot(gdfs, fig, ax, background)

    return plot


def multiplot(*subplots, figsize=None, credit={}, **kwargs) -> None:

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111, aspect="equal")

    mode = "plotter" if kwargs.get("plotter") else "matplotlib"

    subplots_results = [
        plot(
            subplot.query,
            ax=ax,
            multiplot=True,
            **override_params(
                subplot.kwargs,
                {
                    k: v
                    for k, v in kwargs.items()
                    if k != "load_preset" or "load_preset" not in subplot.kwargs
                },
            ),
        )
        for subplot in subplots
    ]

    if mode == "matplotlib":
        ax.axis("off")
        ax.axis("equal")
        ax.autoscale()
