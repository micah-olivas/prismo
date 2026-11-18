import itertools
import json
import os
import threading
import time
from types import SimpleNamespace

import pytest
from test_binding import PS18K_V3, rig, valve_events  # noqa: F401  (rig is a fixture)

import prismo.control
from prismo.protocols import _common, washes
from prismo.session import Session

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
    kw = dict(
        sds="in2", pbs="in1", waste="waste7", top_left=[0, 0], bot_right=[10, 20], blocking=True
    )
    return washes.sds_wash_series(rig.ctrl, rig.chip, **{**kw, **kwargs})


def test_each_wash_is_imaged_and_the_series_ends_dead_ended_in_pbs(rig, monkeypatch):  # noqa: F811
    monkeypatch.setattr(prismo.control, "control", SimpleNamespace(setup="setup7"))
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
    assert {e["setup"] for e in log} == {"setup7"}


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
def test_series_stops_if_a_wash_does_not_finish(rig, monkeypatch, caplog):  # noqa: F811
    driver_cls = type(rig.driver)
    set_valve = driver_cls.__setitem__

    def failing(self, idx, state):
        if idx == PS18K_V3["out"] and state == "open":
            raise ConnectionError("valve controller not responding")
        set_valve(self, idx, state)

    monkeypatch.setattr(driver_cls, "__setitem__", failing)
    series(rig)
    assert "Wash 1 did not finish" in caplog.text
    assert rig.runs == [] and not rig.driver.written & NECKS


def test_quit_stops_the_running_wash_and_leaves_the_valves(rig, monkeypatch):  # noqa: F811
    in_wash = threading.Event()

    def endless(seconds):  # a wash step that only ends when quit
        in_wash.set()
        while True:
            threading.Event().wait(0.01)
            yield

    monkeypatch.setattr(_common, "sleep", endless)
    g = series(rig, blocking=False)  # returns while the first wash runs
    assert in_wash.wait(5)
    g.quit()
    g.join()  # the series waits for its wash to stop

    assert rig.runs == []  # never imaged
    with open("patterning_log.json") as f:
        assert [e["completed"] for e in json.load(f)] == [False]
    assert rig.chip["in2"] == "open" and rig.chip["waste7"] == "open"  # as they were: purging
    assert not rig.driver.written & NECKS


def flush(rig, **kwargs):  # noqa: F811
    kw = dict(buffer="in1", waste="waste7", flush_min=2, n_waits=2, wait_min=10, blocking=True)
    return washes.iterated_flush(rig.chip, **{**kw, **kwargs})


def flush_log():
    with open("patterning_log.json") as f:
        (entry,) = json.load(f)
    return entry


def drop_outlet_closes(rig, monkeypatch, n):  # noqa: F811
    """Lose the first `n` commands that close the open outlet, as if they never arrived."""
    driver_cls = type(rig.driver)
    set_valve = driver_cls.__setitem__
    drops = iter([True] * n)

    def lossy(self, idx, state):
        closing_open_outlet = idx == PS18K_V3["out"] and state == "closed" and self[idx] == "open"
        if closing_open_outlet and next(drops, False):
            return
        set_valve(self, idx, state)

    monkeypatch.setattr(driver_cls, "__setitem__", lossy)


def test_flush_alternates_flushes_and_dead_ended_waits(rig):  # noqa: F811
    flush(rig)
    outlet = [s for v, s in valve_events(rig) if v == "out"]
    assert outlet == ["closed", "open", "closed", "open", "closed", "open", "closed"]
    entry = flush_log()
    assert [(s["step"], s["seconds"]) for s in entry["steps"]] == [
        ("purge", 5),
        ("flush 1/3", 120),
        ("wait 1/2", 600),
        ("flush 2/3", 120),
        ("wait 2/2", 600),
        ("flush 3/3", 120),
    ]
    assert entry["completed"] and entry["n_waits"] == 2

    # Ends dead-ended in the buffer, necks as they were, buttons never opened.
    is_open = {k for k, s in rig.chip.valves.items() if s == "open"}
    assert is_open == {"neckL", "neckR", "sandL", "sandR", "inlet", "in1"}
    assert not rig.driver.written & NECKS
    v = valve_events(rig)
    assert ("butL", "open") not in v and ("butR", "open") not in v


def test_a_lost_outlet_close_is_caught_and_retried(rig, monkeypatch, caplog):  # noqa: F811
    drop_outlet_closes(rig, monkeypatch, 2)
    flush(rig, n_waits=0)
    assert "closing out failed (2/5): it still reads open" in caplog.text
    assert rig.chip["out"] == "closed" and flush_log()["completed"]


def test_an_outlet_that_will_not_close_stops_the_flush(rig, monkeypatch, caplog):  # noqa: F811
    drop_outlet_closes(rig, monkeypatch, 99)
    flush(rig, n_waits=1)
    assert "closing out failed (5/5)" in caplog.text
    # Stopped after the first flush: no wait, no second flush.
    entry = flush_log()
    assert not entry["completed"]
    assert [s["step"] for s in entry["steps"]] == ["purge", "flush 1/2"]


def test_flush_bad_arguments_move_nothing(rig):  # noqa: F811
    with pytest.raises(ValueError, match="neck valve"):
        flush(rig, outlet="neckR")
    with pytest.raises(ValueError, match="wait_min > 0"):
        flush(rig, wait_min=0)
    with pytest.raises(ValueError, match="butL"):
        flush(rig, buffer="butL")
    assert rig.events == []


def running_assay():
    """A session that runs until the returned event is set, like an assay in the background."""
    done = threading.Event()

    def steps():
        while not done.is_set():
            time.sleep(0.01)
            yield

    assay = Session()
    assay.worker(steps)
    assay.start()
    return assay, done


def test_a_queued_flush_waits_for_the_running_assay(rig):  # noqa: F811
    assay, done = running_assay()
    g = flush(rig, n_waits=0, after=assay, blocking=False)
    time.sleep(0.5)
    assert rig.events == []  # nothing moves while the assay runs

    done.set()
    g.join()
    assert flush_log()["completed"] and rig.chip["out"] == "closed"
    assert not rig.driver.written & NECKS


def test_quitting_a_queued_flush_touches_no_valve(rig):  # noqa: F811
    assay, done = running_assay()
    g = flush(rig, after=assay, blocking=False)
    time.sleep(0.3)
    g.quit()
    g.join()
    done.set()
    assay.join()
    time.sleep(0.3)
    assert rig.events == [] and not os.path.exists("patterning_log.json")


def test_buttons_must_read_closed_before_anything_flows(rig, monkeypatch, caplog):  # noqa: F811
    driver_cls = type(rig.driver)
    set_valve = driver_cls.__setitem__

    def stuck_button(self, idx, state):
        if idx != PS18K_V3["butL"]:  # butL stays open, whatever it is sent
            set_valve(self, idx, state)

    rig.driver.state[PS18K_V3["butL"]] = "open"
    monkeypatch.setattr(driver_cls, "__setitem__", stuck_button)
    flush(rig)
    assert "closing butL failed (5/5)" in caplog.text
    assert not flush_log()["completed"]
    v = valve_events(rig)
    assert ("in1", "open") not in v and ("out", "open") not in v


def test_open_necks_opens_them_before_the_flush_and_never_closes_them(rig):  # noqa: F811
    for neck in NECKS:
        rig.driver.state[neck] = "closed"
    flush(rig, open_necks=True)
    v = valve_events(rig)
    assert [e for e in v if e[0].startswith("neck")] == [("neckL", "open"), ("neckR", "open")]
    assert v.index(("neckR", "open")) < v.index(("out", "open"))
    assert rig.chip["neckL"] == "open" and rig.chip["neckR"] == "open"
    assert flush_log()["open_necks"] and flush_log()["completed"]


def test_by_default_the_necks_get_no_command(rig):  # noqa: F811
    rig.driver.state[PS18K_V3["neckR"]] = "closed"
    flush(rig)
    assert not rig.driver.written & NECKS
    assert rig.chip["neckL"] == "open" and rig.chip["neckR"] == "closed"


def test_open_necks_still_never_closes_a_neck(rig):  # noqa: F811
    g = _common._GuardedChip(rig.chip, forbid_open=[], open_necks=True)
    with pytest.raises(RuntimeError, match="necks are left alone"):
        g["neckL"] = "closed"
    assert not rig.driver.written & NECKS
