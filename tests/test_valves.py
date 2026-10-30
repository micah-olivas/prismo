import pytest
from test_patterning import MINICHIP, PS18K_V2_D1, PS18K_V2_D2, FakeDriver, dedupe, opened

from prismo.devices.microfluidic import Chip
from prismo.protocols import valves

PS_INLETS = {"in9", "in8", "in7", "in6", "in5", "in4", "in3", "in2", "waste"}
MINICHIP_INLETS = {"in0", "in1", "in2", "in3", "in4", "waste"}
# PS1.8k v3: split necks, inlets in0..in6 and waste7.
PS18K_V3 = {
    **{f"in{i}": i for i in range(7)},
    "waste7": 7,
    "inlet": 8,
    "neckL": 9,
    "butL": 10,
    "sandL": 11,
    "sandR": 12,
    "butR": 13,
    "neckR": 14,
    "out": 15,
}


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(valves.time, "sleep", lambda s: None)
    monkeypatch.setattr(valves, "tqdm", None)


def two_ps(initial=None):
    driver = FakeDriver(initial)
    return driver, Chip("d1", driver, dict(PS18K_V2_D1)), Chip("d2", driver, dict(PS18K_V2_D2))


def test_d1_and_d2_actuate_their_inlets_and_end_closed():
    driver, d1, d2 = two_ps()
    valves.actuate_valves([d1, d2], cycles=3)

    assert not driver.written & {PS18K_V2_D1["neck"], PS18K_V2_D2["neck"]}
    for mapping in (PS18K_V2_D1, PS18K_V2_D2):
        snaps = opened([s & set(mapping.values()) for s in driver.history], mapping)
        assert dedupe(snaps).count(PS_INLETS) == 3  # all inlets open together once per cycle
        assert all(s <= PS_INLETS for s in snaps) and snaps[-1] == set()


def test_each_chip_uses_its_own_mapping():
    driver = FakeDriver()
    d1 = Chip("d1", driver, dict(PS18K_V2_D1))
    mini = Chip("d2_minichip", driver, dict(MINICHIP))
    valves.actuate_valves([d1, mini], cycles=2)
    ps = opened([s & set(PS18K_V2_D1.values()) for s in driver.history], PS18K_V2_D1)
    mc = opened([s & set(MINICHIP.values()) for s in driver.history], MINICHIP)
    assert dedupe(ps).count(PS_INLETS) == 2 and dedupe(mc).count(MINICHIP_INLETS) == 2
    assert driver.state and all(s == "closed" for s in driver.state.values())


def test_default_valves_for_ps18k_v2_v3_and_minichip():
    driver = FakeDriver()
    v3 = Chip("v3", driver, dict(PS18K_V3))
    assert valves.inlet_valves(v3) == [*(f"in{i}" for i in range(7)), "waste7"]
    assert set(valves.inlet_valves(Chip("v2", driver, dict(PS18K_V2_D1)))) == PS_INLETS
    assert set(valves.inlet_valves(Chip("mc", driver, dict(MINICHIP)))) == MINICHIP_INLETS
    valves.actuate_valves(v3, cycles=2)
    assert driver.written == set(range(8))  # neither neck, nor inlet/out


def test_explicit_valves_and_bad_arguments_move_nothing():
    driver, d1, d2 = two_ps()
    valves.actuate_valves(d1, valves=["butL", "butR"], cycles=1)
    assert driver.written == {PS18K_V2_D1["butL"], PS18K_V2_D1["butR"]}

    driver, d1, d2 = two_ps()
    with pytest.raises(ValueError, match="left alone"):
        valves.actuate_valves([d1, d2], valves=["in2", "neck"])
    with pytest.raises(ValueError, match="do not exist"):
        valves.actuate_valves(d1, valves=["in1"])
    with pytest.raises(ValueError, match="different chips"):
        valves.actuate_valves([d1, d1])
    overlap = Chip("d2", driver, {**PS18K_V2_D2, "in9": PS18K_V2_D1["neck"]})
    with pytest.raises(ValueError, match="shares hardware"):
        valves.actuate_valves([d1, overlap])
    assert driver.written == set()


def test_open_neck_means_nothing_moves():
    driver, d1, d2 = two_ps(initial={PS18K_V2_D2["neck"]: "open"})
    with pytest.raises(RuntimeError, match="not closed"):
        valves.actuate_valves([d1, d2])
    assert driver.written == set()


def test_interrupt_leaves_valves_closed(monkeypatch):
    calls = []

    def interrupted(s):
        calls.append(s)
        if len(calls) == 3:  # mid-cycle, with the inlets open
            raise KeyboardInterrupt

    monkeypatch.setattr(valves.time, "sleep", interrupted)
    driver, d1, d2 = two_ps()
    with pytest.raises(KeyboardInterrupt):
        valves.actuate_valves([d1, d2])
    assert driver.history[-1] == set()
    assert not driver.written & {PS18K_V2_D1["neck"], PS18K_V2_D2["neck"]}
