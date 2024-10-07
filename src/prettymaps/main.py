import osmnx
from shapely.geometry import Point


def construct_prettymap(plane_name: str) -> None:
    x, y = osmnx.geocoder.geocode(plane_name)
    center_point = Point(x, y)
