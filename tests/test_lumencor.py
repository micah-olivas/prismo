from unittest.mock import Mock

import pytest
from pymmcore import CMMCore

from prismo.devices import lumencor


def light(level=50):
    """A SolaLight on a mock core, without loading the device."""
    lt = lumencor.SolaLight.__new__(lumencor.SolaLight)
    core = Mock(spec=CMMCore)
    core.getProperty.return_value = str(level)
    lt.name, lt._core = "light", core
    lt._gated, lt._gate_open, lt._level = False, True, 0
    return lt


def test_opening_waits_for_the_light_then_settles(monkeypatch):
    sleeps = []
    monkeypatch.setattr(lumencor.time, "sleep", sleeps.append)
    lt = light()
    shutter = lumencor.LightShutter("shutter", lt, settle_s=0.05)
    assert sleeps == []  # starting closed doesn't pause

    lt._core.reset_mock()
    shutter.state = "open"
    calls = [c[0] for c in lt._core.method_calls]
    assert calls == ["setProperty", "waitForDevice"]  # light up, then wait for it
    assert lt._core.setProperty.call_args.args == ("light", "White_Level", 50)
    assert sleeps == [0.05]

    shutter.state = "open"  # already open: no extra pause
    shutter.state = "closed"  # closing never pauses
    assert sleeps == [0.05]
    assert lt._core.setProperty.call_args.args == ("light", "White_Level", 0)


def test_settle_can_be_turned_off_and_must_not_be_negative(monkeypatch):
    sleeps = []
    monkeypatch.setattr(lumencor.time, "sleep", sleeps.append)
    shutter = lumencor.LightShutter("shutter", light(), settle_s=0)
    shutter.state = "open"
    assert sleeps == [0]
    with pytest.raises(ValueError, match="settle_s"):
        lumencor.LightShutter("shutter", light(), settle_s=-1)
