import os
import pathlib
from dataclasses import dataclass
from typing import Optional, Union


def presets_directory():
    return os.path.join(pathlib.Path(__file__).resolve().parent, "presets")


def create_preset(
    name: str,
    layers: Optional[dict[str, dict]] = None,
    style: Optional[dict[str, dict]] = None,
    circle: Optional[bool] = None,
    radius: Optional[Union[float, bool]] = None,
    dilate: Optional[Union[float, bool]] = None,
) -> None:
    """
    Create a preset file and save it on the presets folder (prettymaps/presets/) under name 'name.json'

    Args:
        name (str): Preset name
        layers (Dict[str, dict], optional): prettymaps.plot() 'layers' parameter dict. Defaults to None.
        style (Dict[str, dict], optional): prettymaps.plot() 'style' parameter dict. Defaults to None.
        circle (Optional[bool], optional): prettymaps.plot() 'circle' parameter. Defaults to None.
        radius (Optional[Union[float, bool]], optional): prettymaps.plot() 'radius' parameter. Defaults to None.
        dilate (Optional[Union[float, bool]], optional): prettymaps.plot() 'dilate' parameter. Defaults to None.
    """

    # if not os.path.isdir('presets'):
    #    os.makedirs('presets')

    path = os.path.join(presets_directory(), f"{name}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "layers": layers,
                "style": style,
                "circle": circle,
                "radius": radius,
                "dilate": dilate,
            },
            f,
            ensure_ascii=False,
        )


def read_preset(name: str) -> dict[str, dict]:
    """
    Read a preset from the presets folder (prettymaps/presets/)

    Args:
        name (str): Preset name

    Returns:
        (Dict[str,dict]): parameters dictionary
    """
    path = os.path.join(presets_directory(), f"{name}.json")
    with open(path) as f:
        # Load params from JSON file
        params = json.load(f)
    return params


def delete_preset(name: str) -> None:
    """
    Delete a preset from the presets folder (prettymaps/presets/)

    Args:
        name (str): Preset name
    """

    path = os.path.join(presets_directory(), f"{name}.json")
    if os.path.exists(path):
        os.remove(path)






def presets():
    presets = [
        file.split(".")[0]
        for file in os.listdir(presets_directory())
        if file.endswith(".json")
    ]
    presets = sorted(presets)
    presets = pd.DataFrame(
        {"preset": presets, "params": list(map(read_preset, presets))},
    )

    # print('Available presets:')
    # for i, preset in enumerate(presets):
    #    print(f'{i+1}. {preset}')

    return pd.DataFrame(presets)


def preset(name):
    with open(os.path.join(presets_directory(), f"{name}.json")) as f:
        # Load params from JSON file
        params = json.load(f)
        return Preset(params)


@dataclass
class Preset:
    """
    Dataclass implementing a prettymaps Preset object. Attributes:
    - params: dictionary of prettymaps.plot() parameters
    """

    params: dict

    '''
    def _ipython_display_(self):
        """
        Implements the _ipython_display_() function for the Preset class.
        'params' will be displayed as a Markdown table with annotated hex colors
        """

        def light_color(hexstring):
            rgb = np.array(hex2color(hexstring))
            return rgb.mean() > .5

        def annotate_colors(text):
            matches = re.findall(
                '#(?:\\d|[a-f]|[A-F]){6}|#(?:\\d|[a-f]|[A-F]){4}|#(?:\\d|[a-f]|[A-F]){3}', text)
            for match in matches:
                text = text.replace(
                    match,
                    f'<span style="background-color:{match}; color:{"#000" if light_color(match) else "#fff"}">{match}</span>'
                )
            return text

        params = pd.DataFrame(self.params)
        params = params.applymap(lambda x: annotate_colors(
            yaml.dump(x, default_flow_style=False).replace('\n', '<br>')))
        params.iloc[1:, 2:] = ''

        IPython.display.display(IPython.display.Markdown(params.to_markdown()))
    '''


def override_params(default_dict: dict, new_dict: dict) -> dict:
    """
    Override parameters in 'default_dict' with additional parameters from 'new_dict'

    Args:
        default_dict (dict): Default dict to be overriden with 'new_dict' parameters
        new_dict (dict): New dict to override 'default_dict' parameters

    Returns:
        dict: default_dict overriden with new_dict parameters
    """

    final_dict = deepcopy(default_dict)

    for key in new_dict:
        if type(new_dict[key]) == dict:
            if key in final_dict:
                final_dict[key] = override_params(final_dict[key], new_dict[key])
            else:
                final_dict[key] = new_dict[key]
        else:
            final_dict[key] = new_dict[key]

    return final_dict
