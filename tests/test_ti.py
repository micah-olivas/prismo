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
    monkeypatch.setattr("time.sleep", calls.append)
    return calls


def test_reading_a_failing_state_never_moves_the_turret(core, sleeps):
    f = ti.Filter("filter", core, filter=1, states=["open", "egfp", "cy5"])
    core.getStateLabel.side_effect = RuntimeError("busy")
    core.setStateLabel.reset_mock()
    assert f.state == "unknown"
    core.setStateLabel.assert_not_called()
