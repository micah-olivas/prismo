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
    """A software shutter for scopes without one: closing it turns the light down to 0.

    The light keeps the level it was last set to and goes back to it when the shutter opens.
    """

    states: list[str | int | float] = ["open", "closed"]

    def __init__(self, name: str, light: SolaLight):
        self.name = name
        self._light = light
        # Start closed, like a real shutter.
        self.state = "closed"

    @property
    def state(self) -> str:
        return "open" if self._light._gate_open else "closed"

    @state.setter
    def state(self, new_state: str):
        if new_state not in self.states:
            raise ValueError(f"Shutter state must be one of {self.states}, got {new_state!r}.")
        self._light.gate(new_state == "open")


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
