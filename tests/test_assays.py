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
