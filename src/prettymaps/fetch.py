

import re
from copy import deepcopy
from enum import StrEnum

import numpy as np
import osmnx as ox
from attr import dataclass
from geopandas import GeoDataFrame
from shapely.affinity import rotate, scale
from shapely.geometry import (
    Point,
    Polygon,
    box,
)
from shapely.ops import unary_union

from .taginfo_api import retrieve_tags


def calculate_gdf_area(gdf: GeoDataFrame) -> float:
    """Calculate the area of a GeoDataFrame."""
    projected_gdf = ox.projection.project_gdf(gdf)
    return projected_gdf.geometry.area.sum()


def define_area_by_osmid(osmid: str) -> GeoDataFrame:
    """Define the area where OSM features will be fetched using an OSM ID.

    An OSM ID can be of three types: node(N), way(W) or relation(R).
    examples: Manhattan R8398124, Brooklyn Bridge W375157262, US Post Office N4886770821

    For the area to be valid, it should not be null. So don't use node(N) .
    """
    area = ox.geocoder.geocode_to_gdf(osmid, by_osmid=True)
    if calculate_gdf_area(area) == 0:
        msg = "The area of this OSM ID is null. Check you didn't enter a node OSM ID(Nxxxxxx)"
        raise ValueError(
            msg,
        )
    return area


def retrieve_features_from_area(
    area: GeoDataFrame,
    tags: dict[str, bool | str | list[str]],
) -> GeoDataFrame:
    """Fetch the features of tags for a given area."""
    return ox.features.features_from_polygon(*area.geometry, tags=tags)

# Parse query (by coordinates, OSMId or name)
def parse_query(query):
    if isinstance(query, GeoDataFrame):
        return "polygon"
    if isinstance(query, tuple):
        return "coordinates"
    if re.match("""[A-Z][0-9]+""", query):
        return "osmid"
    return "address"



# Get circular or square boundary around point
def get_boundary(query, radius, circle=False, rotation=0):

    # Get point from query
    point = query if parse_query(query) == "coordinates" else ox.geocode(query)
    # Create GeoDataFrame from point
    boundary = ox.projection.project_gdf(
        GeoDataFrame(geometry=[Point(point[::-1])], crs="EPSG:4326"),
    )

    if circle:  # Circular shape
        # use .buffer() to expand point into circle
        boundary.geometry = boundary.geometry.buffer(radius)
    else:  # Square shape
        x, y = np.concatenate(boundary.geometry[0].xy)
        r = radius
        boundary = GeoDataFrame(
            geometry=[
                rotate(
                    Polygon(
                        [
                            (x - r, y - r),
                            (x + r, y - r),
                            (x + r, y + r),
                            (x - r, y + r),
                        ],
                    ),
                    rotation,
                ),
            ],
            crs=boundary.crs,
        )

    # Unproject
    boundary = boundary.to_crs(4326)

    return boundary


# Get perimeter from query
def get_perimeter(
    query,
    radius=None,
    by_osmid=False,
    circle=False,
    dilate=None,
    rotation=0,
    aspect_ratio=1,
    **kwargs,
):

    if radius:
        # Perimeter is a circular or square shape
        perimeter = get_boundary(query, radius, circle=circle, rotation=rotation)
    elif parse_query(query) == "polygon":
        # Perimeter was already provided
        perimeter = query
    else:
        # Fetch perimeter from OSM
        perimeter = ox.geocoder.geocode_to_gdf(
            query,
            by_osmid=by_osmid,
            **kwargs,
        )

    # Scale according to aspect ratio
    perimeter = ox.projection.project_gdf(perimeter)
    perimeter.loc[0, "geometry"] = scale(perimeter.loc[0, "geometry"], aspect_ratio, 1)
    perimeter = perimeter.to_crs(4326)

    # Apply dilation
    if dilate is not None:
        perimeter = ox.projection.project_gdf(perimeter)
        perimeter.geometry = perimeter.geometry.buffer(dilate)
        perimeter = perimeter.to_crs(4326)

    return perimeter


# Get a GeoDataFrame
def get_gdf(
    layer,
    perimeter,
    perimeter_tolerance=0,
    tags=None,
    osmid=None,
    custom_filter=None,
    **kwargs,  # weird mechanism to give values to tags or custom_filter arguments depending on layers
):

    # Apply tolerance to the perimeter
    perimeter_with_tolerance = (
        ox.projection.project_gdf(perimeter).buffer(perimeter_tolerance).to_crs(4326)
    )
    perimeter_with_tolerance = unary_union(perimeter_with_tolerance.geometry).buffer(0)

    # Fetch from perimeter's bounding box, to avoid missing some geometries
    bbox = box(*perimeter_with_tolerance.bounds)

    try:
        if layer in ["streets", "railway", "waterway"]:
            graph = ox.graph.graph_from_polygon(
                bbox,
                custom_filter=custom_filter,
                truncate_by_edge=True,
            )
            gdf = ox.convert.graph_to_gdfs(graph, nodes=False)
        elif layer == "coastline" or osmid is None:
            # Fetch geometries from OSM
            gdf = ox.features.features_from_polygon(
                bbox,
                tags={tags: True} if type(tags) is str else tags,
            )
        else:
            gdf = ox.geocoder.geocode_to_gdf(osmid, by_osmid=True)
    except:
        gdf = GeoDataFrame(geometry=[])

    # Intersect with perimeter
    gdf.geometry = gdf.geometry.intersection(perimeter_with_tolerance)
    # gdf = gdf[~gdf.geometry.is_empty]
    gdf.drop(gdf[gdf.geometry.is_empty].index, inplace=True)

    return gdf


# Fetch GeoDataFrames given query and a dictionary of layers
def get_gdfs(query, layers_dict, radius, dilate, rotation=0) -> dict:

    perimeter_kwargs = {}
    if "perimeter" in layers_dict:
        perimeter_kwargs = deepcopy(layers_dict["perimeter"])
        perimeter_kwargs.pop("dilate")  # remove dilate from perimeter dict

    # Get perimeter
    perimeter = get_perimeter(
        query,
        radius=radius,
        rotation=rotation,
        dilate=dilate,
        **perimeter_kwargs,
    )

    # Get other layers as GeoDataFrames
    gdfs = {"perimeter": perimeter}
    gdfs.update(
        {
            layer: get_gdf(layer, perimeter, **kwargs)
            for layer, kwargs in layers_dict.items()
            if layer != "perimeter"
        },
    )

    return gdfs
