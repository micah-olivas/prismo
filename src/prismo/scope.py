"""The microscope's settings, saved with the images of every acquisition (see run.py).

Filter cube specs can't be read from the scope. Set them once after ``pm.load`` to save them too::

    c.filter_cubes = filter_cube_params  # {"cy5": {"ex": "618/50", "em": "698/70", ...}, ...}
"""

import datetime
from collections.abc import Callable
from typing import Any

import numpy as np
import pymmcore

from . import devices as dev
from .control import Control

# The fluidics are never read here, so saving the settings can't send the valve controller anything.
_FLUIDICS = ("prismo.devices.microfluidic", "prismo.devices.fluidic", "prismo.devices.fluigent")
# The Zyla's device name in older configs (as in protocols.imaging).
_CAMERA_NAMES = {"camera": "zyla"}


def scope_attrs(ctrl: Control) -> dict[str, Any]:
    """The microscope's settings now, as zarr attrs.

    ``objective``, ``filter`` and its ``filter_cube`` specs (from ``c.filter_cubes``; None if
    unset or not listed), ``light`` (source and power), ``lightpath``, ``camera`` (name, binning,
    exposure, orientation and every Micro-Manager property), ``px_len_um`` at the sample, the
    focus ``z``, ``device_states`` (the state of every filter, light, shutter, light path...) and
    ``scope_read_at``. Only the optics are read: never the stage or the fluidics, and nothing is
    moved. A value that can't be read is saved as "<unreadable: ...>".
    """
    states = {d.name: _read(lambda d=d: _state(d)) for d in _optics(ctrl)}
    attrs: dict[str, Any] = dict(
        scope_read_at=datetime.datetime.now().isoformat(timespec="seconds")
    )
    for name in ("objective", "filter", "lightpath"):
        if name in states:
            attrs[name] = states[name]
    if "filter" in states:
        cubes = getattr(ctrl, "filter_cubes", None) or {}
        attrs["filter_cube"] = _read(lambda: cubes.get(states["filter"]))
    if "light" in states:
        attrs["light"] = dict(source=states.get("light_source"), power=states["light"])
    attrs["px_len_um"] = _read(lambda: _px_len(ctrl))
    attrs["camera"] = _read(lambda: _camera(ctrl.camera))
    if getattr(ctrl, "focus", None) is not None:
        attrs["z"] = _read(lambda: float(ctrl.z))
    attrs["device_states"] = states
    return _plain(attrs)


def _optics(ctrl: Control) -> list[Any]:
    """The devices with a state, except the fluidics."""
    return [
        d
        for d in ctrl.devices
        if not type(d).__module__.startswith(_FLUIDICS)
        and not isinstance(d, dev.Valved)
        and isinstance(d, dev.State)
    ]


def _state(d: Any) -> Any:
    """`d.state`, except that a Micro-Manager state device (filter turret, nosepiece, light path)
    is read straight from the core: the Ti filter and objective `state` move to the first position
    when a read fails, and reading must never move anything."""
    core = getattr(d, "_core", None)
    if core is None or core.getDeviceType(d.name) != pymmcore.StateDevice:
        return d.state
    i = core.getState(d.name)
    states = getattr(d, "states", None)
    return states[i] if isinstance(states, list) and 0 <= i < len(states) else i


def _px_len(ctrl: Control) -> float:
    """`ctrl.px_len` (µm per pixel at the sample), with the objective read as in `_state`."""
    zoom = 1.0
    for d in ctrl.devices:
        if isinstance(d, dev.Zoom):
            zooms = getattr(d, "zooms", None)
            zoom *= zooms[_state(d)] if isinstance(zooms, dict) else d.zoom
    return float(ctrl.camera.px_len / zoom)


def _camera(cam: Any) -> dict[str, Any]:
    out = dict(
        name=_CAMERA_NAMES.get(cam.name, cam.name),
        device=cam.name,
        binning=_read(lambda: cam.binning),
        exposure_ms=_read(lambda: cam.exposure),
        sensor_px_len_um=_read(lambda: cam.px_len),  # before the objective
    )
    # Orientation and snap mode, where the camera has them.
    for key, attr in (("flip", "_flip"), ("rotation", "_rotation"), ("clean_snap", "clean_snap")):
        if hasattr(cam, attr):
            out[key] = getattr(cam, attr)
    core = getattr(cam, "_core", None)
    if core is not None:
        out["properties"] = _read(lambda: _properties(core, cam.name))
    return out


def _properties(core: Any, name: str) -> dict[str, Any]:
    """Every Micro-Manager property of device `name` (readout port, rate, gain, temperature...)."""
    return {
        p: _read(lambda p=p: core.getProperty(name, p)) for p in core.getDevicePropertyNames(name)
    }


def _read(get: Callable[[], Any]) -> Any:
    try:
        return get()
    except Exception as e:  # a device that can't be read never stops the acquisition
        return f"<unreadable: {e}>"


def _plain(value: Any) -> Any:
    """`value` with JSON types only, as zarr attrs need."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, str | bool | int | float):
        return value
    return str(value)
