from unittest.mock import Mock

import pytest
from pymmcore import CMMCore

from prismo.devices import ti


@pytest.fixture
def core():
    c = Mock(spec=CMMCore)
    c.getLoadedDevices.return_value = ["ti_scope"]
    c.getNumberOfStates.return_value = 3
    c.getStateLabel.return_value = "cy5"
    return c


@pytest.fixture
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(ti.time, "sleep", calls.append)
    return calls


def test_wait_settles_only_after_a_change(core, sleeps):
    f = ti.Filter("filter", core, filter=1, states=["open", "egfp", "cy5"], settle_s=0.7)
    f.wait()
    assert sleeps == []  # nothing commanded yet
    f.state = "egfp"
    f.wait()
    assert sleeps == [0.7]
    core.waitForDevice.assert_called_with("filter")
    f.wait()
    assert sleeps == [0.7]  # already settled


def test_reading_a_failing_state_never_moves_the_turret(core, sleeps):
    f = ti.Filter("filter", core, filter=1, states=["open", "egfp", "cy5"])
    core.getStateLabel.side_effect = RuntimeError("busy")
    core.setStateLabel.reset_mock()
    assert f.state == "unknown"
    core.setStateLabel.assert_not_called()
