import numpy as np
import pytest
from test_binding import rig  # noqa: F401  (rig is a fixture)

from prismo.protocols import imaging


def tile(rig, file, **kwargs):  # noqa: F811
    kw = dict(channel="gfp", top_left=[0, 0], bot_right=[10, 20])
    return imaging.tile_image(rig.ctrl, str(file), **{**kw, **kwargs})


def test_tiles_once_in_the_chosen_channel_on_the_kinetix(rig, tmp_path):  # noqa: F811
    session = tile(rig, tmp_path / "GFP" / "post_expression", channel="cy5", exposure=200)
    imgs = session.arrays["cy5_200ms"]

    assert rig.driver.written == set()  # never touches a valve
    assert rig.runs[0]["camera"] == "kinetix" and rig.dev("lightpath").state == "kinetix"
    assert sorted(np.unique(imgs.values)) == [1, 2, 3, 4]  # every tile once
    assert ("filter", "cy5") in rig.events and rig.ctrl.exposure == 200
    for i, e in enumerate(rig.events):
        if e[0] == "snap":
            assert rig.events[i - 1] == ("shutter", "open")
            assert rig.events[i + 1] == ("shutter", "closed")
    assert not np.isnan(imgs["seconds"].values).any()
    assert (tmp_path / "GFP").is_dir()


def test_camera_none_keeps_the_current_one(rig, tmp_path):  # noqa: F811
    tile(rig, tmp_path / "a", camera=None)
    assert rig.runs[0]["camera"] == "camera"  # the Zyla, as selected


def test_bad_arguments_fail_before_anything_moves(rig, tmp_path):  # noqa: F811
    with pytest.raises(ValueError, match="'egfp' is not one of"):
        tile(rig, tmp_path / "a", channel="egfp")
    with pytest.raises(ValueError, match="'kinetics'"):
        tile(rig, tmp_path / "a", camera="kinetics")
    with pytest.raises(ValueError, match="exposure"):
        tile(rig, tmp_path / "a", exposure=0)
    (tmp_path / "exists").mkdir()
    with pytest.raises(FileExistsError):
        tile(rig, tmp_path / "exists")
    assert rig.events == [] and rig.runs == []
