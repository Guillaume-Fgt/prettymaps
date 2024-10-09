import re

import pytest
from osmnx._errors import InsufficientResponseError

from prettymaps.fetch import define_area_by_osmid


def test_define_area_by_osmid_invalid_id() -> None:
    with pytest.raises(InsufficientResponseError):
        define_area_by_osmid(osmid="")


def test_define_area_by_osmid_node_id() -> None:
    msg = re.escape(
        "The area of this OSM ID is null. Check you didn't enter a node OSM ID(Nxxxxxx)",
    )
    with pytest.raises(ValueError, match=msg):
        define_area_by_osmid(osmid="N4886770821")


def test_define_area_by_osmid_valid_id() -> None:
    area = define_area_by_osmid(osmid="R8398124")
    assert area["name"][0] == "Manhattan"
