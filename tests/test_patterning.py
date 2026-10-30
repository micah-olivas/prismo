import json

import pytest

from prismo.devices.microfluidic import Chip, ValveDriver
from prismo.protocols import _common, patterning, washes

PS18K_V2_D1 = {
    "in9": 0,
    "in8": 1,
    "in7": 2,
    "in6": 3,
    "in5": 4,
    "in4": 5,
    "in3": 6,
    "in2": 7,
    "waste": 8,
    "inlet": 9,
    "butL": 10,
    "sandL": 11,
    "sandR": 12,
    "butR": 13,
    "neck": 14,
    "out": 15,
}
PS18K_V2_D2 = {k: v + 16 for k, v in PS18K_V2_D1.items()}
MINICHIP = {
    "in0": 16,
    "in1": 17,
    "in2": 18,
    "in3": 19,
    "in4": 20,
    "waste": 21,
    "inlet": 22,
    "but": 23,
    "sand": 24,
    "2_1": 25,
    "2_0": 26,
    "1_1": 27,
    "1_0": 28,
    "0_1": 29,
    "0_0": 30,
    "out": 31,
}


class FakeDriver(ValveDriver):
    """Records valve states in memory instead of talking to the valve controller."""

    def __init__(self, initial: dict[int, str] | None = None):
        self.name = "fake"
        self.state = dict(initial or {})
        self.history = []  # snapshot of every open valve index after each command
        self.written = set()  # every valve index that was sent a command

    def __getitem__(self, idx):
        return self.state.get(idx, "closed")

    def __setitem__(self, idx, state):
        self.state[idx] = state
        self.written.add(idx)
        self.history.append({i for i, s in self.state.items() if s == "open"})


def make_chip(mapping, initial=None):
    driver = FakeDriver(initial)
    return Chip("test_chip", driver, dict(mapping)), driver


def opened(history, mapping):
    """Each snapshot in `history` as the set of open valve names."""
    names = {v: k for k, v in mapping.items()}
    return [{names[i] for i in snap} for snap in history]


@pytest.fixture(autouse=True)
def fast(monkeypatch, tmp_path):
    """No real waiting, and the patterning log goes to a temp dir."""

    def no_sleep(seconds):
        yield

    monkeypatch.setattr(_common, "sleep", no_sleep)
    monkeypatch.setattr(_common, "tqdm", None)
    monkeypatch.chdir(tmp_path)


HIGH_LOW = dict(bbsa="in6", na="in5", anti_egfp_high="in4", anti_egfp_low="in3", pbs="in2")


def run_two_anti_egfp(**kwargs):
    chip, driver = make_chip(PS18K_V2_D1)
    patterning.pattern_anti_egfp_high_low(chip, **{**HIGH_LOW, **kwargs}, blocking=True)
    assert PS18K_V2_D1["neck"] not in driver.written  # the neck is never sent any command
    return driver, opened(driver.history, PS18K_V2_D1)


def test_two_anti_egfp_splits_only_the_anti_egfp_step():
    _, snaps = run_two_anti_egfp(high_side="L")

    assert all("neck" not in s for s in snaps)
    # bBSA and NA flow into both sides.
    for reagent in ("in6", "in5"):
        flowing = [s for s in snaps if {reagent, "inlet"} <= s]
        assert flowing and all({"sandL", "sandR"} <= s for s in flowing)
    # Low anti-eGFP only reaches the right side, high anti-eGFP only the left.
    low = [s for s in snaps if {"in3", "inlet"} <= s]
    high = [s for s in snaps if {"in4", "inlet"} <= s]
    assert low and all("sandR" in s and not s & {"sandL", "butL"} for s in low)
    assert high and all("sandL" in s and not s & {"sandR", "butR"} for s in high)
    assert any("butR" in s for s in low) and any("butL" in s for s in high)
    # Low first by default.
    assert snaps.index(low[0]) < snaps.index(high[0])
    # Ends dead-ended in PBS on both sides.
    assert snaps[-1] == {"sandL", "sandR", "inlet", "in2"}


def test_two_anti_egfp_high_side_right_and_log(tmp_path):
    _, snaps = run_two_anti_egfp(high_side="R", info={"anti_egfp_high_ug_ml": 10})
    high = [s for s in snaps if {"in4", "inlet"} <= s]
    assert high and all("sandR" in s and not s & {"sandL", "butL"} for s in high)

    (entry,) = json.loads((tmp_path / patterning.LOG_FILE).read_text())
    assert entry["protocol"] == "pattern_anti_egfp_high_low"
    assert entry["completed"] is True
    assert entry["high_side"] == "R" and entry["info"] == {"anti_egfp_high_ug_ml": 10}
    assert len(entry["steps"]) == 10 + 2 * 5


def dedupe(states):
    return [s for i, s in enumerate(states) if i == 0 or s != states[i - 1]]


def test_two_devices_run_in_lockstep(tmp_path):
    _, solo = run_two_anti_egfp(high_side="L")

    driver = FakeDriver()
    d1 = Chip("d1", driver, dict(PS18K_V2_D1))
    d2 = Chip("d2", driver, dict(PS18K_V2_D2))
    patterning.pattern_anti_egfp_high_low([d1, d2], **HIGH_LOW, high_side="L", blocking=True)

    # Neither neck is ever sent a command.
    assert not driver.written & {PS18K_V2_D1["neck"], PS18K_V2_D2["neck"]}

    # Each device sees exactly the single-device valve sequence.
    for mapping in (PS18K_V2_D1, PS18K_V2_D2):
        snaps = opened([s & set(mapping.values()) for s in driver.history], mapping)
        assert dedupe(snaps) == dedupe(solo)
    entries = json.loads((tmp_path / patterning.LOG_FILE).read_text())
    assert entries[-1]["chip"] == "d1+d2" and entries[-1]["completed"] is True


def test_two_devices_must_be_distinct_and_not_overlap():
    driver = FakeDriver()
    d1 = Chip("d1", driver, dict(PS18K_V2_D1))
    args = HIGH_LOW
    with pytest.raises(ValueError, match="different chips"):
        patterning.pattern_anti_egfp_high_low([d1, d1], **args)
    overlap = Chip("d2", driver, {**PS18K_V2_D2, "neck": PS18K_V2_D1["in3"]})
    with pytest.raises(ValueError, match="shares hardware valves"):
        patterning.pattern_anti_egfp_high_low([d1, overlap], **args)
    renamed = Chip("d2", driver, {**PS18K_V2_D2, "flow": PS18K_V2_D2["inlet"]})
    with pytest.raises(ValueError, match="same valve names"):
        patterning.pattern_anti_egfp_high_low([d1, renamed], **args)
    assert driver.history == []

    # The guard covers every chip in the group.
    d2 = Chip("d2", driver, dict(PS18K_V2_D2))
    g = patterning._GuardedChip(patterning._as_chip([d1, d2]), forbid_open=[])
    for state in ("open", "closed"):
        with pytest.raises(RuntimeError, match="left alone"):
            g["neck"] = state
    assert driver.history == []
    g.close_all()
    assert driver.written and not driver.written & {14, 30}
    # One chip in a list is just that chip.
    assert patterning._as_chip([d1]) is d1


def test_bad_arguments_fail_before_any_valve_moves():
    chip, driver = make_chip(PS18K_V2_D1)
    with pytest.raises(ValueError, match="neck"):
        patterning.pattern_anti_egfp_high_low(
            chip, bbsa="in6", na="in5", anti_egfp_high="in4", anti_egfp_low="in3", pbs="neck"
        )
    with pytest.raises(ValueError, match="does not exist"):
        patterning.pattern_anti_egfp(chip, bbsa="in6", na="in5", anti_egfp="in1", pbs="in2")
    with pytest.raises(ValueError, match="its own valve"):
        patterning.pattern_anti_egfp(chip, bbsa="in6", na="in5", anti_egfp="in5", pbs="in2")
    assert driver.history == []


def test_guard_never_commands_necks_and_refuses_open_all():
    chip, driver = make_chip(PS18K_V2_D1, initial={14: "open"})
    g = patterning._GuardedChip(chip, forbid_open=[])
    for state in ("open", "closed"):
        with pytest.raises(RuntimeError, match="left alone"):
            g.neck = state
        with pytest.raises(RuntimeError, match="left alone"):
            g["neck"] = state
    with pytest.raises(RuntimeError):
        g.open_all()
    g.close_all()
    g.close_all_except(["sandL"])
    assert 14 not in driver.written and driver.state[14] == "open"


def test_patterning_refuses_to_start_if_a_neck_reads_open():
    for initial in ({14: "open"}, {30: "open"}):
        driver = FakeDriver(initial=initial)
        d1 = Chip("d1", driver, dict(PS18K_V2_D1))
        d2 = Chip("d2", driver, dict(PS18K_V2_D2))
        with pytest.raises(RuntimeError, match="not closed"):
            patterning.pattern_anti_egfp_high_low([d1, d2], **HIGH_LOW)
        with pytest.raises(RuntimeError, match="not closed"):
            patterning.pattern_anti_egfp([d1, d2], bbsa="in6", na="in5", anti_egfp="in4", pbs="in2")
        assert driver.written == set()


def test_pattern_anti_egfp_uses_both_sides():
    chip, driver = make_chip(PS18K_V2_D1)
    patterning.pattern_anti_egfp(
        chip, bbsa="in6", na="in5", anti_egfp="in4", pbs="in2", blocking=True
    )
    snaps = opened(driver.history, PS18K_V2_D1)
    anti_egfp = [s for s in snaps if {"in4", "inlet"} <= s]
    assert anti_egfp and all({"sandL", "sandR"} <= s for s in anti_egfp)
    assert any({"butL", "butR"} <= s for s in anti_egfp)
    assert 14 not in driver.written
    assert snaps[-1] == {"sandL", "sandR", "inlet", "in2"}


def test_sds_wash_leaves_open_neck_alone():
    chip, driver = make_chip(PS18K_V2_D1, initial={14: "open"})
    washes.sds_wash(chip, sds="in9", pbs="in2", blocking=True)
    assert 14 not in driver.written
    snaps = opened(driver.history, PS18K_V2_D1)
    assert all("neck" in s for s in snaps)
    assert all(not s & {"butL", "butR"} for s in snaps)
    assert snaps[-1] == {"neck", "sandL", "sandR", "inlet", "in2"}


def test_minichip_load_opens_only_its_lane():
    chip, driver = make_chip(MINICHIP)
    patterning.pattern_minichip_to_anti_egfp(chip, bbsa="in4", na="in3", pbs="in1", blocking=True)
    io = set(v for vs in patterning.MINICHIP_IO_LANES.values() for v in vs)
    dead_end = {"sand", "inlet", "in1", *io}
    assert opened(driver.history, MINICHIP)[-1] == dead_end

    driver.history.clear()
    patterning.load_minichip_anti_egfp(chip, io=3, anti_egfp="in2", pbs="in1", blocking=True)
    snaps = opened(driver.history, MINICHIP)
    flowing = [s for s in snaps if {"in2", "inlet"} <= s]
    assert flowing and all(s & io == {"0_0", "1_1", "2_1"} for s in flowing)
    assert snaps[-1] == dead_end
