import numpy as np
import pytest
from test_binding import PS18K_V3, rig, valve_events  # noqa: F401  (rig is a fixture)

from prismo.protocols import assays, binding

NECKS = {PS18K_V3["neckL"], PS18K_V3["neckR"]}
BUTTONS = {PS18K_V3["butL"], PS18K_V3["butR"]}
SANDWICHES = {PS18K_V3["sandL"], PS18K_V3["sandR"]}


@pytest.fixture(autouse=True)
def fakes(rig, monkeypatch):  # noqa: F811
    monkeypatch.setattr(assays, "tiled_acq", binding.tiled_acq)  # the rig's fake
    monkeypatch.setattr(assays, "tqdm", None)
    for v in SANDWICHES:
        rig.driver.state[v] = "open"  # e.g. dead-ended in the ligand


def run(rig, file, **kwargs):  # noqa: F811
    kw = dict(incubations_s=[30, 60], top_left=[0, 0], bot_right=[10, 20])
    return assays.snap_labeling_kinetics(rig.ctrl, rig.chip, str(file), **{**kw, **kwargs})


def buttons_at_snaps(rig):  # noqa: F811
    """Button states at every snap."""
    names = {v: k for k, v in PS18K_V3.items()}
    state, out = {}, []
    for e in rig.events:
        if e[0] == "valve":
            state[names[e[1]]] = e[2]
        elif e[0] == "snap":
            out.append((state.get("butL"), state.get("butR")))
    return out


def test_buttons_open_only_between_passes(rig, tmp_path):  # noqa: F811
    session = run(rig, tmp_path / "k")
    imgs = session.arrays["cy5_500ms"]

    closed, opened = ("butL", "closed"), ("butL", "open")
    v = valve_events(rig)
    assert v[:4] == [("sandL", "closed"), ("sandR", "closed"), closed, ("butR", "closed")]
    assert [e for e in v if e[0] == "butL"] == [closed, opened, closed, opened, closed, closed]
    assert rig.driver.written <= BUTTONS | SANDWICHES  # nothing else, never a neck
    assert set(buttons_at_snaps(rig)) == {("closed", "closed")}
    assert {k: s for k, s in rig.chip.valves.items() if s == "open"} == {
        "neckL": "open",
        "neckR": "open",
    }

    assert rig.runs[0]["camera"] == "kinetix" and rig.dev("lightpath").state == "kinetix"
    assert rig.dev("filter").state == "cy5" and rig.ctrl.exposure == 500
    assert rig.dev("shutter").state == "closed"
    assert sorted(np.unique(imgs.values)) == list(range(1, 13))  # every tile once per pass
    np.testing.assert_allclose(imgs["incubation_s"], [0, 30, 60])
    np.testing.assert_allclose(imgs["buttons_open_s"], [0, 30, 90])
    # 4 one-second tiles per pass, then the incubation.
    np.testing.assert_allclose(imgs["seconds"].values.min(axis=(1, 2)), [0, 34, 98])
    assert imgs.attrs["incubations_s"] == [30, 60] and "started_at" in imgs.attrs


def test_quit_while_reacting_closes_the_buttons(rig, tmp_path):  # noqa: F811
    rig.stop_after = 10  # 4 tiles, then 6 s into the first incubation
    session = run(rig, tmp_path / "k")
    imgs = session.arrays["cy5_500ms"]

    assert valve_events(rig)[-2:] == [("butL", "closed"), ("butR", "closed")]
    assert rig.chip["butL"] == "closed" and rig.chip["butR"] == "closed"
    assert rig.dev("shutter").state == "closed"
    assert np.isfinite(imgs["seconds"].values).sum() == 4
    assert np.isnan(imgs["incubation_s"].values[1:]).all()
    assert not rig.driver.written & NECKS


def test_an_error_while_reacting_closes_the_buttons(rig, tmp_path, monkeypatch):  # noqa: F811
    def broken_wait(t_end):
        yield
        raise RuntimeError("stage error")

    monkeypatch.setattr(assays, "_wait_until", broken_wait)
    with pytest.raises(RuntimeError, match="stage error"):
        run(rig, tmp_path / "k")
    assert rig.chip["butL"] == "closed" and rig.chip["butR"] == "closed"
    assert rig.dev("shutter").state == "closed"


def test_camera_choice(rig, tmp_path):  # noqa: F811
    run(rig, tmp_path / "k", camera="zyla", incubations_s=[5])
    assert rig.runs[-1]["camera"] == "camera" and rig.dev("lightpath").state == "zyla"


def test_bad_arguments_move_nothing(rig, tmp_path):  # noqa: F811
    with pytest.raises(ValueError, match="'egfp' is not one of"):
        run(rig, tmp_path / "k", channel="egfp")
    with pytest.raises(ValueError, match="incubations_s"):
        run(rig, tmp_path / "k", incubations_s=[])
    with pytest.raises(ValueError, match="incubations_s"):
        run(rig, tmp_path / "k", incubations_s=[30, 0])
    with pytest.raises(ValueError, match="neck"):
        run(rig, tmp_path / "k", buttons=("butL", "neckR"))
    with pytest.raises(ValueError, match="no 'butX' valve"):
        run(rig, tmp_path / "k", buttons=("butX",))
    with pytest.raises(ValueError, match="'kinetics'"):
        run(rig, tmp_path / "k", camera="kinetics")
    (tmp_path / "exists").mkdir()
    with pytest.raises(FileExistsError):
        run(rig, tmp_path / "exists")
    assert rig.events == [] and rig.runs == []


def run_kinetics(rig, file, **kwargs):  # noqa: F811
    kw = dict(channels=("cy5", "gfp"), top_left=[0, 0], bot_right=[10, 20])
    if "wait_s" not in kwargs:
        kw["n_passes"] = 2
    return assays.tiled_kinetics(rig.ctrl, str(file), **{**kw, **kwargs})


def test_kinetics_snaps_every_channel_before_moving(rig, tmp_path):  # noqa: F811
    session = run_kinetics(rig, tmp_path / "k", camera=None, light_source=None)
    imgs = session.arrays["kinetics"]

    assert list(session.arrays) == ["kinetics"]  # one array holds everything
    assert imgs.dims == ("time", "channel", "row", "col", "y", "x")
    assert list(imgs["channel"].values) == ["cy5", "gfp"]
    # At each tile: move, snap cy5, snap gfp, then the next move.
    kinds = ("xy", "snap", "filter")
    tile = [e[1] if e[0] == "filter" else e[0] for e in rig.events if e[0] in kinds]
    assert tile[:6] == ["xy", "cy5", "snap", "gfp", "snap", "xy"]
    assert tile.count("xy") == 8 and tile.count("snap") == 16  # 4 tiles x 2 passes, 2 channels
    assert sorted(np.unique(imgs.values)) == list(range(1, 17))  # every image once
    assert rig.driver.written == set()  # imaging only: no valve commands at all
    assert rig.dev("shutter").state == "closed"


def test_kinetics_waits_the_listed_times_between_passes(rig, tmp_path):  # noqa: F811
    t0 = rig.clock.now - 30  # buttons opened 30 s before the call
    session = run_kinetics(rig, tmp_path / "k", t0=t0, wait_s=[100, 50], exposure=[200, 500])
    imgs = session.arrays["kinetics"]

    assert imgs.sizes["time"] == 3  # one more pass than pauses
    # Each pass is 4 tiles x 2 one-second snaps, then the pause before the next.
    np.testing.assert_allclose(imgs["pass_start_s"], [30, 30 + 8 + 100, 30 + 16 + 150])
    np.testing.assert_allclose(imgs["wait_s"], [0, 100, 50])
    secs = imgs["seconds"]
    assert secs.sel(channel="gfp").values.min() > secs.sel(channel="cy5").values.min()
    assert list(imgs["exposure_ms"].values) == [200, 500]
    assert np.isfinite(imgs["seconds"].values).all()


def test_kinetics_same_wait_after_every_pass(rig, tmp_path):  # noqa: F811
    session = run_kinetics(rig, tmp_path / "k", n_passes=3, wait_s=20)
    imgs = session.arrays["kinetics"]
    np.testing.assert_allclose(np.diff(imgs["pass_start_s"]), [28, 28])


def test_kinetics_quit_while_waiting_stops(rig, tmp_path):  # noqa: F811
    rig.stop_after = 6  # 4 tiles, then 2 s into the first pause
    session = run_kinetics(rig, tmp_path / "k", wait_s=[600])
    imgs = session.arrays["kinetics"]
    assert np.isfinite(imgs["seconds"].values).sum() == 8  # only the first pass
    assert rig.dev("shutter").state == "closed"


def test_kinetics_quit_closes_the_shutter(rig, tmp_path):  # noqa: F811
    rig.stop_after = 3
    session = run_kinetics(rig, tmp_path / "k")
    imgs = session.arrays["kinetics"]
    assert np.isfinite(imgs["seconds"].values).sum() == 6  # 3 tiles x 2 channels
    assert rig.dev("shutter").state == "closed"
    assert rig.driver.written == set()


def test_kinetics_bad_arguments_move_nothing(rig, tmp_path):  # noqa: F811
    with pytest.raises(ValueError, match="'egfp' is not one of"):
        run_kinetics(rig, tmp_path / "k", channels=("cy5", "egfp"))
    with pytest.raises(ValueError, match="each once"):
        run_kinetics(rig, tmp_path / "k", channels=("cy5", "cy5"))
    with pytest.raises(ValueError, match="one per channel"):
        run_kinetics(rig, tmp_path / "k", exposure=[500])
    with pytest.raises(ValueError, match="n_passes"):
        run_kinetics(rig, tmp_path / "k", n_passes=0)
    with pytest.raises(ValueError, match="n_passes must be 3"):
        run_kinetics(rig, tmp_path / "k", wait_s=[10, 10], n_passes=5)
    with pytest.raises(ValueError, match="wait_s at least 0"):
        run_kinetics(rig, tmp_path / "k", wait_s=[10, -1])
    assert rig.events == [] and rig.runs == []
