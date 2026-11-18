from unittest.mock import Mock

import numpy as np
import pytest
from pymmcore import CMMCore
from test_binding import Events, StateDevice

from prismo.control import Control
from prismo.devices import protocols
from prismo.devices.kinetix import Camera
from prismo.protocols import imaging


@pytest.fixture
def mock_core():
    core = Mock(spec=CMMCore)
    core.getImage.return_value = np.arange(6, dtype=np.uint16).reshape(2, 3)
    core.popNextImage.return_value = core.getImage.return_value
    core.getRemainingImageCount.return_value = 2
    core.getProperty.return_value = "2x2"
    core.getExposure.return_value = 10.0
    return core


class SequenceCore:
    """The CMMCore calls a snap makes. Logs them; a sequence of n serves the first n `frames`."""

    def __init__(self, frames, exposure=500.0):
        self.frames, self.exposure, self.props = frames, exposure, {}
        self.events, self._buffer = Events(), []

    def loadDevice(self, *args):
        pass

    def initializeDevice(self, *args):
        pass

    def setCameraDevice(self, name):
        pass

    def getExposure(self, name):
        return self.exposure

    def getProperty(self, name, prop):
        return self.props.get(prop, "No")

    def setProperty(self, name, prop, value):
        self.props[prop] = value

    def clearCircularBuffer(self):
        self._buffer = []

    def startSequenceAcquisition(self, name, n, interval_ms, stop_on_overflow):
        self.events("start", n)
        self._buffer = list(self.frames[:n])

    def stopSequenceAcquisition(self, name):
        self.events("stop")

    def getRemainingImageCount(self):
        return len(self._buffer)

    def popNextImage(self):
        return self._buffer.pop(0)

    def snapImage(self):
        self.events("snap")

    def getImage(self):
        return self.frames[0]


def clean():
    return np.full((256, 64), 1000, dtype=np.uint16)


def banded():
    img = clean()
    img[(np.arange(256) // 32) % 2 == 1] = 100  # every other 32-row block at the offset
    return img


def test_snap_keeps_the_second_frame_of_a_smart_sequence(caplog):
    core = SequenceCore([banded(), clean()])
    img = Camera("kinetix", core).snap()
    np.testing.assert_array_equal(img, clean())
    assert core.props == {
        "SMARTStreamingValues[ms]": "1.000;500.000",
        "SMARTStreamingEnabled": "Yes",
    }
    assert core.events == [("start", 2), ("stop",)]
    assert "banding" not in caplog.text


def test_snap_shuttered_opens_the_shutter_during_the_throwaway_frame():
    core = SequenceCore([banded(), clean()])
    img = Camera("kinetix", core).snap_shuttered(lambda is_open: core.events("shutter", is_open))
    np.testing.assert_array_equal(img, clean())
    assert core.props["SMARTStreamingValues[ms]"] == "75.000;500.000"
    assert core.events == [("start", 2), ("shutter", True), ("shutter", False), ("stop",)]


def test_clean_snap_off_goes_back_to_single_snaps():
    core = SequenceCore([banded(), clean()])
    cam = Camera("kinetix", core)
    cam.snap()
    cam.clean_snap = False
    img = cam.snap_shuttered(lambda is_open: core.events("shutter", is_open))
    np.testing.assert_array_equal(img, banded())
    assert core.props["SMARTStreamingEnabled"] == "No"
    assert core.events[2:] == [("shutter", True), ("snap",), ("shutter", False)]


def test_a_banded_frame_that_gets_through_is_reported(caplog):
    Camera("kinetix", SequenceCore([clean(), banded()])).snap()
    assert "128 of 256 rows came back dark" in caplog.text


def test_tiling_lets_the_kinetix_time_the_shutter():
    core = SequenceCore([banded(), clean()])
    cam = Camera("kinetix", core, flip="both")
    shutter = StateDevice("shutter", core.events, ["open", "closed"], "closed")
    ctrl = object.__new__(Control)
    object.__setattr__(ctrl, "devices", [cam, shutter])
    object.__setattr__(ctrl, "_camera", cam)
    img = imaging.snap(ctrl)
    np.testing.assert_array_equal(img, clean())
    assert core.events == [("start", 2), ("shutter", "open"), ("shutter", "closed"), ("stop",)]


def test_loads_pvcam_device(mock_core):
    """Camera loads through the PVCAM adapter with the given label."""
    Camera("kinetix", mock_core, camera_label="Camera-2")
    mock_core.loadDevice.assert_called_once_with("kinetix", "PVCAM", "Camera-2")
    mock_core.initializeDevice.assert_called_once_with("kinetix")


def test_is_camera(mock_core):
    """Camera satisfies the Camera and Wait protocols."""
    cam = Camera("kinetix", mock_core)
    assert isinstance(cam, protocols.Camera)
    assert isinstance(cam, protocols.Wait)


@pytest.mark.parametrize(
    "flip, expected",
    [
        ("none", lambda img: img),
        ("ud", np.flipud),
        ("lr", np.fliplr),
        ("both", lambda img: np.flipud(np.fliplr(img))),
    ],
)
def test_snap_flip(mock_core, flip, expected):
    """Snapped images are flipped as configured."""
    cam = Camera("kinetix", mock_core, flip=flip)
    img = cam.snap()
    mock_core.setCameraDevice.assert_called_with("kinetix")
    np.testing.assert_array_equal(img, expected(mock_core.getImage.return_value))


def test_binning(mock_core):
    """Binning reads and writes the NxN property."""
    cam = Camera("kinetix", mock_core)
    assert cam.binning == 2
    cam.binning = 4
    mock_core.setProperty.assert_called_with("kinetix", "Binning", "4x4")


def test_exposure(mock_core):
    """Exposure is passed through to the core."""
    cam = Camera("kinetix", mock_core)
    assert cam.exposure == 10.0
    cam.exposure = 25
    mock_core.setExposure.assert_called_with("kinetix", 25.0)


def test_px_len_uses_pixel_size(mock_core):
    """Pixel length scales the configured pixel size by binning."""
    assert Camera("kinetix", mock_core).px_len == pytest.approx(13.0)
    assert Camera("kinetix", mock_core, pixel_size=4.25).px_len == pytest.approx(8.5)
