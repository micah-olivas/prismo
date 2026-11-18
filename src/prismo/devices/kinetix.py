import logging
import time
from collections.abc import Callable
from typing import Literal

import numpy as np
from pymmcore import CMMCore

logger = logging.getLogger(__name__)

# The first frame the camera reads after it starts acquiring can come back banded: rows at the
# offset level, in blocks locked to sensor rows (a camera fault, 2026-10/11), while later frames
# of a sequence read every row. So a clean snap is a 2-frame SMART-streaming sequence, a short
# throwaway frame and then the exposure, keeping the second frame.
_THROWAWAY_MS = 1.0
# snap_shuttered starts the sequence before opening the shutter, so the throwaway frame covers
# the shutter opening: the Ti2 shutter is fully open 60-85 ms after its command (75 ms gave an
# even frame, 50 ms didn't; 2026-11-05). That keeps the shutter open no longer than for a snap.
_SHUTTER_THROWAWAY_MS = 75.0
_OFFSET = 100  # camera offset (ADU), for the banding check


class Camera:
    def __init__(
        self,
        name: str,
        core: CMMCore,
        flip: Literal["none", "ud", "lr", "both"] = "none",
        pixel_size: float = 6.5,
        camera_label: str = "Camera-1",
        clean_snap: bool = True,
    ):
        self.name = name
        self._core = core
        self._flip = flip
        self._pixel_size = pixel_size
        # False: plain single snaps (banded while the camera fault lasts), e.g. to compare.
        self.clean_snap = clean_snap
        self._smart_values = None  # SMART-streaming exposures last set
        core.loadDevice(name, "PVCAM", camera_label)
        core.initializeDevice(name)

    def snap(self) -> np.ndarray:
        """One frame: with `clean_snap`, the second frame of a 2-frame sequence."""
        self._core.setCameraDevice(self.name)
        if not self.clean_snap:
            self._single_snaps()
            self._core.snapImage()
            return self._oriented(self._core.getImage())
        return self._oriented(self._sequence(_THROWAWAY_MS))

    def snap_shuttered(self, shutter: Callable[[bool], None]) -> np.ndarray:
        """One frame with `shutter` (called with True to open, False to close) open only around
        it. With `clean_snap`, the sequence starts first and the shutter opens during its
        throwaway frame; it closes once the frame is read."""
        self._core.setCameraDevice(self.name)
        if not self.clean_snap:
            self._single_snaps()
            shutter(True)
            try:
                self._core.snapImage()
            finally:
                shutter(False)
            return self._oriented(self._core.getImage())
        return self._oriented(self._sequence(_SHUTTER_THROWAWAY_MS, shutter))

    def _sequence(
        self, throwaway_ms: float, shutter: Callable[[bool], None] | None = None
    ) -> np.ndarray:
        core = self._core
        exposure = core.getExposure(self.name)
        values = f"{throwaway_ms:.3f};{exposure:.3f}"
        if values != self._smart_values:
            core.setProperty(self.name, "SMARTStreamingValues[ms]", values)
            self._smart_values = values
        if core.getProperty(self.name, "SMARTStreamingEnabled") != "Yes":
            core.setProperty(self.name, "SMARTStreamingEnabled", "Yes")
        core.clearCircularBuffer()
        core.startSequenceAcquisition(self.name, 2, 0, True)
        try:
            if shutter is not None:
                shutter(True)
            deadline = time.time() + 10 + 2 * (throwaway_ms + exposure) / 1000
            while core.getRemainingImageCount() < 2:
                if time.time() > deadline:
                    raise TimeoutError(f"{self.name} returned no frames.")
                time.sleep(0.0005)
        finally:
            if shutter is not None:
                shutter(False)
            core.stopSequenceAcquisition(self.name)
        core.popNextImage()  # the throwaway frame
        img = core.popNextImage()
        self._check_banding(img)
        return img

    def _single_snaps(self):
        if self._smart_values is not None:
            self._core.setProperty(self.name, "SMARTStreamingEnabled", "No")
            self._smart_values = None

    def _check_banding(self, img: np.ndarray):
        """Warn if whole rows came back unexposed even so."""
        rows = img[:, ::16].mean(axis=1)
        lit = np.median(rows) - _OFFSET
        if lit < 20:  # too dark to tell, e.g. the shutter was closed
            return
        dark = int((rows - _OFFSET < 0.2 * lit).sum())
        if dark > 32:
            logger.warning(f"{self.name}: {dark} of {len(rows)} rows came back dark (banding).")

    def _oriented(self, img: np.ndarray) -> np.ndarray:
        if self._flip == "ud":
            img = np.flipud(img)
        elif self._flip == "lr":
            img = np.fliplr(img)
        elif self._flip == "both":
            img = np.flipud(np.fliplr(img))
        return img

    def wait(self):
        self._core.waitForDevice(self.name)

    @property
    def binning(self) -> int:
        return int(self._core.getProperty(self.name, "Binning")[-1])

    @binning.setter
    def binning(self, new_binning: int):
        self._core.setProperty(self.name, "Binning", f"{new_binning}x{new_binning}")

    @property
    def exposure(self) -> float:
        return self._core.getExposure(self.name)

    @exposure.setter
    def exposure(self, new_exposure: float):
        self._core.setExposure(self.name, float(new_exposure))

    @property
    def px_len(self) -> float:
        return self.binning * self._pixel_size
