from typing import Literal

import numpy as np
from pymmcore import CMMCore


class Camera:
    def __init__(
        self,
        name: str,
        core: CMMCore,
        flip: Literal["none", "ud", "lr", "both"] = "none",
        pixel_size: float = 6.5,
        camera_label: str = "Camera-1",
    ):
        self.name = name
        self._core = core
        self._flip = flip
        self._pixel_size = pixel_size
        core.loadDevice(name, "PVCAM", camera_label)
        core.initializeDevice(name)

    def snap(self) -> np.ndarray:
        self._core.setCameraDevice(self.name)
        self._core.snapImage()
        img = self._core.getImage()
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
