import time
from typing import Literal

from pymmcore import CMMCore

from . import utils


class SolaLight:
    def __init__(
        self,
        name: str,
        core: CMMCore,
        port: str | None = None,
        version: Literal["sola", "spectra"] = "sola",
    ):
        self.name = name
        self._core = core

        port = utils.load_port(core, vid=0x0403, pid=0x6001, port=port)
        core.loadDevice(name, "LumencorSpectra", "Spectra")
        core.setProperty(name, "SetLE_Type", version.capitalize())
        core.setProperty(name, "Port", port)
        core.initializeDevice(name)

        # Set by a LightShutter. While gated, the level is only sent to the light when the gate is
        # open, and the light is held at 0 otherwise.
        self._gated = False
        self._gate_open = True
        self._level = 0

    def wait(self):
        self._core.waitForDevice(self.name)

    @property
    def state(self) -> int:
        if self._gated:
            return self._level
        return int(self._core.getProperty(self.name, "White_Level"))

    @state.setter
    def state(self, new_state: int):
        if self._gated:
            self._level = new_state
            if not self._gate_open:
                return
        self._core.setProperty(self.name, "White_Level", new_state)

    def gate(self, open: bool):
        if not self._gated:
            level = int(self._core.getProperty(self.name, "White_Level"))
            # Default to full power so opening the shutter doesn't leave the light off.
            self._level = level if level > 0 else 100
            self._gated = True
        self._gate_open = open
        self._core.setProperty(self.name, "White_Level", self._level if open else 0)


class LightShutter:
    """Shutter control for microscopes with no mechanical shutter, such as the Ti.

    There is no physical shutter. Setting the state to "closed" sets the light's power to 0, and
    "open" sets it back to the level it was last given (``c.light``), which is kept while closed.
    Opening pauses `settle_s` so the light is fully on before a snap starts: at short exposures
    the Zyla otherwise starts exposing before the light comes up, leaving a dark band across
    the middle of the image (where its exposure starts).
    """

    states: list[str | int | float] = ["open", "closed"]

    def __init__(self, name: str, light: SolaLight, settle_s: float = 0.05):
        if settle_s < 0:
            raise ValueError(f"settle_s must be at least 0, got {settle_s}.")
        self.name = name
        self._light = light
        self._settle_s = settle_s
        # Start closed, like a real shutter.
        self.state = "closed"

    @property
    def state(self) -> str:
        return "open" if self._light._gate_open else "closed"

    @state.setter
    def state(self, new_state: str):
        if new_state not in self.states:
            raise ValueError(f"Shutter state must be one of {self.states}, got {new_state!r}.")
        was_open = self._light._gated and self._light._gate_open
        self._light.gate(new_state == "open")
        if new_state == "open" and not was_open:
            self._light.wait()
            time.sleep(self._settle_s)


class RetraLight:
    def __init__(self, name: str, core: CMMCore, ip: str):
        self.name = name
        self._core = core

        core.loadDevice(name, "Lumencor", "LightEngine")
        core.setProperty(name, "Model", "GEN3")
        core.setProperty(name, "Connection", ip)
        core.initializeDevice(name)

    def wait(self):
        self._core.waitForDevice(self.name)

    @property
    def state(self) -> float:
        return float(self._core.getProperty(self.name, "UV340_Intensity")) / 10.0

    @state.setter
    def state(self, new_state: float):
        self._core.setProperty(self.name, "UV340", new_state > 0.0)
        self._core.setShutterOpen(self.name, new_state > 0.0)
        self._core.setProperty(self.name, "UV340_Intensity", new_state * 10.0)
