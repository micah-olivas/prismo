import itertools
import json
import os

import pytest
from test_binding import PS18K_V3, rig, valve_events  # noqa: F401  (rig is a fixture)

from prismo.protocols import _common, washes

NECKS = {PS18K_V3["neckL"], PS18K_V3["neckR"]}


@pytest.fixture(autouse=True)
def fast(monkeypatch, tmp_path):
    """No real waiting; images and the protocol log go to a temp dir."""

    def no_sleep(seconds):
        yield

    monkeypatch.setattr(_common, "sleep", no_sleep)
    monkeypatch.setattr(_common, "tqdm", None)
    monkeypatch.chdir(tmp_path)


def series(rig, **kwargs):  # noqa: F811
    kw = dict(sds="in2", pbs="in1", waste="waste7", top_left=[0, 0], bot_right=[10, 20])
    washes.sds_wash_series(rig.ctrl, rig.chip, **{**kw, **kwargs})


def test_each_wash_is_imaged_and_the_series_ends_dead_ended_in_pbs(rig):  # noqa: F811
    series(rig, n_washes=2, exposures=[500, 1000])

    assert [r["file"] for r in rig.runs] == [
        "GFP/egfp_post_SDS_wash1_kinetix_500ms",
        "GFP/egfp_post_SDS_wash1_kinetix_1000ms",
        "GFP/egfp_post_SDS_wash2_kinetix_500ms",
        "GFP/egfp_post_SDS_wash2_kinetix_1000ms",
    ]
    assert "egfp_1000ms" in rig.runs[1]["session"].arrays
    assert not rig.driver.written & NECKS
    v = valve_events(rig)
    assert ("butL", "open") not in v and ("butR", "open") not in v
    assert v.count(("waste7", "open")) == 2  # SDS and PBS purged once, in the first wash

    # wash, images, wash, images: no valve moves while tiling.
    kinds = [k for k, _ in itertools.groupby(e[0] for e in rig.events if e[0] in ("valve", "snap"))]
    assert kinds == ["valve", "snap", "valve", "snap"]

    is_open = {k for k, s in rig.chip.valves.items() if s == "open"}  # necks: as they were
    assert is_open == {"neckL", "neckR", "sandL", "sandR", "inlet", "in1"}
    with open("patterning_log.json") as f:
        log = json.load(f)
    assert [(e["info"]["wash"], e["completed"]) for e in log] == [(1, True), (2, True)]


def test_necks_are_left_as_they_are(rig):  # noqa: F811
    rig.driver.state[PS18K_V3["neckR"]] = "closed"
    series(rig, n_washes=1)
    assert not rig.driver.written & NECKS
    assert rig.chip["neckL"] == "open" and rig.chip["neckR"] == "closed"


def test_bad_arguments_move_nothing(rig):  # noqa: F811
    with pytest.raises(ValueError, match="'egfp' is not one of"):
        series(rig, channel="egfp")
    with pytest.raises(ValueError, match="waste='waste' does not exist"):
        series(rig, waste="waste")
    with pytest.raises(ValueError, match="neck valve"):
        series(rig, outlet="neckL")
    os.makedirs("GFP/egfp_post_SDS_wash3_kinetix_500ms")
    with pytest.raises(FileExistsError, match="wash3"):
        series(rig)
    assert rig.events == [] and rig.runs == []


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_series_stops_if_a_wash_does_not_finish(rig, monkeypatch):  # noqa: F811
    driver_cls = type(rig.driver)
    set_valve = driver_cls.__setitem__

    def failing(self, idx, state):
        if idx == PS18K_V3["out"] and state == "open":
            raise ConnectionError("valve controller not responding")
        set_valve(self, idx, state)

    monkeypatch.setattr(driver_cls, "__setitem__", failing)
    with pytest.raises(RuntimeError, match="Wash 1 did not finish"):
        series(rig)
    assert rig.runs == [] and not rig.driver.written & NECKS
