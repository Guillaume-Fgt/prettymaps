from prettymaps.draw import override_args


def test_override_args_circle_true_dilate_true_not_present_previously() -> None:
    layers = {
        "perimeter": {"width": 12},
    }
    override_args(layers, circle=True, dilate=True)
    assert layers == {
        "perimeter": {"width": 12, "circle": True, "dilate": True},
    }


def test_override_args_circle_true_dilate_true_present_previously() -> None:
    layers = {
        "perimeter": {"width": 12, "circle": False, "dilate": False},
    }
    override_args(layers, circle=True, dilate=True)
    assert layers == {
        "perimeter": {"width": 12, "circle": False, "dilate": False},
    }


def test_override_args_circle_true_dilate_true_present_previously_unchanged() -> None:
    layers = {
        "perimeter": {"width": 12, "circle": False, "dilate": False},
    }
    override_args(layers, circle=False, dilate=False)
    assert layers == {
        "perimeter": {"width": 12, "circle": False, "dilate": False},
    }
