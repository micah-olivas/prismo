import time
from collections.abc import Callable

import numpy as np

from .ports import Port

VID = 0x10C4
PID = 0xEA60


def _wait(cmd: Callable[[str], str], timeout: float = 60.0):
    # Controllers answer the status command "/" with "N" (not busy) or "B" (busy), or with ":A"
    # when idle on some firmware. Fail instead of spinning forever if neither shows up.
    deadline = time.monotonic() + timeout
    response = ""
    while time.monotonic() < deadline:
        response = cmd("/")
        if response.startswith(":A") or response == "N":
            return
    raise TimeoutError(f"ASI controller not ready after {timeout}s; last reply: {response!r}")


class Stage:
    def __init__(
        self,
        name: str,
        port: str | None = None,
        units_per_um: float = 1.0,
        counts_per_um: float = 1.0,
        flip_x: bool = False,
    ):
        self.name = name
        # Stage units per micron: 10 on controllers that count in 0.1 um. Positions stay in stage
        # units; this is only used to convert distances measured in microns, like tile sizes.
        self.units_per_um = units_per_um
        # Or convert positions themselves: controller counts per micron (10 for 0.1 um counts), so
        # x, y and moves are in microns. Use one or the other, or tile steps get scaled twice.
        if counts_per_um <= 0:
            raise ValueError(f"counts_per_um must be positive, got {counts_per_um}.")
        if counts_per_um != 1 and units_per_um != 1:
            raise ValueError("Set units_per_um or counts_per_um for an ASI stage, not both.")
        self._counts_per_um = counts_per_um
        # flip_x mirrors the x axis (e.g. after a controller reset reversed it).
        self._x_sign = -1 if flip_x else 1
        self._port = Port(VID, PID, port, baudrate=9600, timeout=2.0)

    def _to_counts(self, x: float, y: float) -> tuple[float, float]:
        return self._x_sign * float(x) * self._counts_per_um, float(y) * self._counts_per_um

    def _from_counts(self, x: float, y: float) -> tuple[float, float]:
        return self._x_sign * x / self._counts_per_um, y / self._counts_per_um

    def _cmd(self, cmd: str) -> str:
        response = self._port.write_readline((cmd + "\r").encode()).decode().strip()
        if response.startswith(":N"):
            raise RuntimeError(f"ASI error: {response}")
        return response

    def wait(self):
        _wait(self._cmd)

    def close(self):
        self._port.close()

    @property
    def x(self) -> float:
        resp = self._cmd("WHERE X")
        return self._from_counts(float(resp.split(" ")[1]), 0)[0]

    @x.setter
    def x(self, new_x: float):
        self._cmd(f"MOVE X={self._to_counts(new_x, 0)[0]:.4f}")

    @property
    def y(self) -> float:
        resp = self._cmd("WHERE Y")
        return self._from_counts(0, float(resp.split(" ")[1]))[1]

    @y.setter
    def y(self, new_y: float):
        self._cmd(f"MOVE Y={self._to_counts(0, new_y)[1]:.4f}")

    @property
    def xy(self) -> np.ndarray:
        resp = self._cmd("WHERE X Y").split(" ")
        return np.array(self._from_counts(float(resp[1]), float(resp[2])))

    @xy.setter
    def xy(self, new_xy: tuple[float, float]):
        x, y = self._to_counts(*new_xy)
        self._cmd(f"MOVE X={x:.4f} Y={y:.4f}")

    @property
    def x_speed(self) -> float:
        resp = self._cmd("VE X?")
        return float(resp.split("X=")[1].split()[0])

    @x_speed.setter
    def x_speed(self, speed: float):
        self._cmd(f"VE X={float(speed)}")

    @property
    def y_speed(self) -> float:
        resp = self._cmd("VE Y?")
        return float(resp.split("Y=")[1].split()[0])

    @y_speed.setter
    def y_speed(self, speed: float):
        self._cmd(f"VE Y={float(speed)}")

    def set_xy_speed(self, vx: float, vy: float):
        self._cmd(f"VE X={float(vx)} Y={float(vy)}")

    def __iadd__(self, delta: tuple[float, float]) -> "Stage":
        dx, dy = self._to_counts(*delta)
        self._cmd(f"MOVREL X={dx:.4f} Y={dy:.4f}")
        return self

    def __isub__(self, delta: tuple[float, float]) -> "Stage":
        dx, dy = self._to_counts(*delta)
        self._cmd(f"MOVREL X={-dx:.4f} Y={-dy:.4f}")
        return self


class Focus:
    def __init__(self, name: str, port: str | None = None):
        self.name = name
        self._port = Port(VID, PID, port, baudrate=9600, timeout=2.0)

    def _cmd(self, cmd: str) -> str:
        response = self._port.write_readline((cmd + "\r").encode()).decode().strip()
        if response.startswith(":N"):
            raise RuntimeError(f"ASI error: {response}")
        return response

    def wait(self):
        _wait(self._cmd)

    def close(self):
        self._port.close()

    @property
    def z(self) -> float:
        resp = self._cmd("WHERE Z")
        return float(resp.split(" ")[1])

    @z.setter
    def z(self, new_z: float):
        self._cmd(f"MOVE Z={float(new_z):.4f}")

    def __iadd__(self, delta: float) -> "Focus":
        self._cmd(f"MOVREL Z={float(delta):.4f}")
        return self

    def __isub__(self, delta: float) -> "Focus":
        self._cmd(f"MOVREL Z={-float(delta):.4f}")
        return self
