"""Using the taginfo api to retrieve all sorts of data about osm tags and values.

https://taginfo.openstreetmap.org/taginfo/apidoc.
"""

import requests


def retrieve_tags(min_count: int = 1_000_000) -> list[str]:
    """Retrieve the tags used in OSM if they are employed more than *min_count* times.

    filters used:
        * only show tags that appear in the wiki
        * only show tags with latin lowercase letters (a to z) or underscore (_), first and last characters must be letters.
    """
    url = "https://taginfo.openstreetmap.org/api/4/keys/all?&filter=in_wiki,characters_plain&sortname=count_all&sortorder=desc"
    r = requests.get(url, timeout=10)
    return [tag["key"] for tag in r.json()["data"] if tag["count_all"] >= min_count]


def get_prevalent_values(session: requests.Session, tag: str) -> list[str]:
    """Get most prevalent values used by a given tag."""
    url = f"https://taginfo.openstreetmap.org/api/4/key/prevalent_values?key={tag}"
    r = session.get(url, timeout=10)
    print(r.json())
    return [value["value"] for value in r.json()["data"]]


print(retrieve_tags())
