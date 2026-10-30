from unittest.mock import Mock

import numpy as np
import pytest
from pymmcore import CMMCore

from prismo.control import Control
from prismo.devices import protocols
from prismo.devices.manual import Objective


@pytest.fixture
def mock_core():
    core = Mock(spec=CMMCore)
    core.getLoadedDevices.return_value = []
    core.getNumberOfStates.return_value = 4
    return core


def test_init_with_devices(mock_core):
    """Control initializes with device list."""
    device = Mock()
    device.name = "test"
    c = Control(mock_core, devices=[device])

    assert device in c.devices


def test_finds_camera(mock_core):
    """Control finds camera device."""
    camera = Mock(spec=protocols.Camera)
    camera.name = "camera"
    c = Control(mock_core, devices=[camera])
    c.snap()
    assert camera.snap.called


def test_finds_stage(mock_core):
    """Control finds stage device."""
    stage = Mock(spec=protocols.Stage)
    stage.name = "stage"
    c = Control(mock_core, devices=[stage])

    assert c._stage is stage


def test_no_camera_is_none(mock_core):
    """Control without camera has None."""
    stage = Mock(spec=protocols.Stage)
    stage.name = "stage"
    c = Control(mock_core, devices=[stage])

    assert c._camera is None


def test_snap(mock_core):
    """Snap returns image from camera."""
    camera = Mock(spec=protocols.Camera)
    camera.name = "cam"
    expected = np.arange(100 * 100, dtype=np.uint16).reshape(100, 100)
    camera.snap.return_value = expected
    c = Control(mock_core, devices=[camera])

    img = c.snap()

    np.testing.assert_array_equal(img, expected)


def test_exposure(mock_core):
    """Exposure can be read and set."""
    camera = Mock(spec=protocols.Camera)
    camera.name = "cam"
    camera.exposure = 10.0
    c = Control(mock_core, devices=[camera])

    assert c.exposure == 10.0

    c.exposure = 100.0

    assert camera.exposure == 100.0


def test_px_len(mock_core):
    """Pixel length accounts for camera and zoom devices."""
    camera = Mock(spec=protocols.Camera)
    camera.name = "cam"
    camera.px_len = 1.0
    c = Control(mock_core, devices=[camera])

    assert c.px_len == 1.0

    obj = Objective("4x", zoom=4.0)
    c = Control(mock_core, devices=[camera, obj])

    assert c.px_len == 0.25

    obj2 = Objective("1.5x", zoom=1.5)
    c = Control(mock_core, devices=[camera, obj, obj2])

    assert abs(c.px_len - 1 / 6) < 0.001


def test_stage_position(mock_core):
    """Stage x, y, xy can be read and set."""
    stage = Mock(spec=protocols.Stage)
    stage.name = "stage"
    stage.x = 100.0
    stage.y = 200.0
    stage.xy = (100.0, 200.0)
    c = Control(mock_core, devices=[stage])

    assert c.x == 100.0
    assert c.y == 200.0
    assert c.xy == (100.0, 200.0)

    c.x = 150.0
    c.y = 250.0
    c.xy = (300.0, 400.0)

    assert stage.x == 150.0
    assert stage.y == 250.0
    assert stage.xy == (300.0, 400.0)


def test_state_device_access(mock_core):
    """State devices can be read and set by name."""
    device = Mock(spec=protocols.State)
    device.name = "emission"
    device.state = "gfp"
    c = Control(mock_core, devices=[device])

    assert c.emission == "gfp"

    c.emission = "dapi"

    assert device.state == "dapi"


def test_getattr_non_state_device_returns_device(mock_core):
    """Getting non-state device by name returns the device."""
    camera = Mock(spec=protocols.Camera)
    camera.name = "mycam"
    c = Control(mock_core, devices=[camera])

    result = c.mycam

    assert result is camera


def test_getattr_unknown_raises(mock_core):
    """Getting unknown attribute raises AttributeError."""
    c = Control(mock_core, devices=[])
    with pytest.raises(AttributeError):
        _ = c.nonexistent_device


def test_wait_calls_all_wait_devices(mock_core):
    """Wait calls wait on all Wait-protocol devices."""
    wait1 = Mock(spec=protocols.Wait)
    wait1.name = "wait1"
    wait2 = Mock(spec=protocols.Wait)
    wait2.name = "wait2"
    non_wait = Mock(spec=protocols.State)
    non_wait.name = "non_wait"
    c = Control(mock_core, devices=[wait1, non_wait, wait2])

    c.wait()

    wait1.wait.assert_called_once()
    wait2.wait.assert_called_once()
    assert not hasattr(non_wait, "wait") or not non_wait.wait.called


def test_close_resets_core(mock_core):
    """close() resets the core."""
    c = Control(mock_core, devices=[])

    c.close()

    mock_core.reset.assert_called()


def test_device_setters(mock_core):
    """Camera and stage can be switched by name."""
    cam1 = Mock(spec=protocols.Camera)
    cam1.name = "cam1"
    cam1.snap.return_value = np.arange(100, dtype=np.uint16).reshape(10, 10)
    cam2 = Mock(spec=protocols.Camera)
    cam2.name = "cam2"
    cam2.snap.return_value = np.arange(100, 200, dtype=np.uint16).reshape(10, 10)
    stage1 = Mock(spec=protocols.Stage)
    stage1.name = "stage1"
    stage1.x = 100.0
    stage2 = Mock(spec=protocols.Stage)
    stage2.name = "stage2"
    stage2.x = 200.0
    c = Control(mock_core, devices=[cam1, cam2, stage1, stage2])

    np.testing.assert_array_equal(c.snap(), cam1.snap.return_value)
    assert c.x == 100.0

    c.camera = "cam2"
    c.stage = "stage2"

    np.testing.assert_array_equal(c.snap(), cam2.snap.return_value)
    assert c.x == 200.0


def test_setup_name(mock_core, monkeypatch):
    """The setup name given to load is kept; otherwise it is the computer name."""
    import platform

    import prismo.control

    assert Control(mock_core, devices=[], setup="setup7").setup == "setup7"
    assert Control(mock_core, devices=[]).setup == platform.node()

    monkeypatch.setattr(prismo.control, "control", None)
    assert prismo.control.setup_name() == platform.node()
    monkeypatch.setattr(prismo.control, "control", Control(mock_core, devices=[], setup="setup7"))
    assert prismo.control.setup_name() == "setup7"
