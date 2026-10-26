from unittest.mock import Mock

import numpy as np
import pytest
from pymmcore import CMMCore

from prismo.devices import protocols
from prismo.devices.kinetix import Camera


@pytest.fixture
def mock_core():
    core = Mock(spec=CMMCore)
    core.getImage.return_value = np.arange(6, dtype=np.uint16).reshape(2, 3)
    core.getProperty.return_value = "2x2"
    core.getExposure.return_value = 10.0
    return core


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
