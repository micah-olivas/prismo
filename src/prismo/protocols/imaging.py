"""Imaging: ``tile_image`` tiles the device once, in any channel. Imaging only: it never touches a
valve. Also the pieces the other imaging protocols share (optics checks, shutter-gated snaps,
snake tiling).

Example (post-expression eGFP image on the Kinetix)::

    from prismo.protocols.imaging import tile_image

    g = tile_image(
        c,
        "GFP/egfp_post_expression_kinetix_500ms",
        channel="gfp",
        top_left=d1_top_left,
        bot_right=d1_bottom_right,
    )
"""

import datetime
import json
import os
import time
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from ..control import Control
from ..devices import Camera
from ..run import tiled_acq
from ..session import Session

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional.
    tqdm = None


def tile_image(
    ctrl: Control,
    file: str,
    *,
    channel: str,
    top_left: Sequence[float],
    bot_right: Sequence[float],
    exposure: int = 500,
    power: int = 100,
    light_source: str | None = "sola",
    camera: str | None = "kinetix",
    overlap: float = 0.1,
    info: dict[str, Any] | None = None,
) -> Session:
    """Tile the device once into array ``{channel}_{exposure}ms`` of `file`. Never touches a valve.

    Parameters:
    -----------
    ctrl :
        The controller (``c``).
    file :
        Zarr file for the images, e.g. ``"GFP/egfp_post_expression_kinetix_500ms"``. Must not
        exist yet.
    channel :
        Filter cube, e.g. "gfp" or "cy5".
    top_left, bot_right :
        Corners of the device for `camera`, e.g. ``d1_top_left`` and ``d1_bottom_right``.
    exposure, power, light_source :
        Exposure (ms), light power and light source (None if there is no ``light_source``
        device). The shutter only opens while a tile is snapped.
    camera :
        Camera to image with, selected (and the lightpath sent to it) before tiling, like
        ``use_camera``: "kinetix" or "zyla". None keeps the current camera.
    overlap :
        Tile overlap, as in ``pm.tiled_acq``.
    info :
        Free-form notes saved with the images.

    Returns the acquisition session; ``g.join()`` waits for it and ``g.quit()`` stops it.
    """
    if exposure <= 0:
        raise ValueError("exposure must be positive.")
    if os.path.exists(file):
        raise FileExistsError(f"{file} already exists; pick a new name.")
    check_optics(ctrl, channel, light_source)
    check_camera(ctrl, camera)

    acq_func = tile_acq_func(
        ctrl,
        name=channel,
        exposure=exposure,
        channel=channel,
        power=power,
        light_source=light_source,
        attrs=dict(info=json.dumps(info or {})),
    )
    if os.path.dirname(file):
        os.makedirs(os.path.dirname(file), exist_ok=True)
    use_camera(ctrl, camera)  # before tiled_acq, which sizes the tiles from a snap
    return tiled_acq(
        ctrl, file, acq_func, overlap=overlap, top_left=tuple(top_left), bot_right=tuple(bot_right)
    )


def device(ctrl: Control, name: str) -> Any:
    for d in ctrl.devices:
        if d.name == name:
            return d
    # Control silently stores unknown attributes, so a missing device would never be set.
    raise ValueError(f"No device named {name!r} in the config.")


def check_optics(ctrl: Control, channel: str, light_source: str | None):
    """Fail before anything moves if the filter cube, light source, shutter or light is wrong."""
    checks = {"filter": channel, "light_source": light_source}
    for name, state in checks.items():
        if state is None:
            continue
        states = list(getattr(device(ctrl, name), "states", []))
        if state not in states:
            raise ValueError(f"{name}={state!r} is not one of {states}.")
    for name in ("shutter", "light"):
        device(ctrl, name)


def set_optics(ctrl: Control, channel: str, exposure: int, power: int, light_source: str | None):
    ctrl.filter = channel
    if light_source is not None:
        ctrl.light_source = light_source
    ctrl.light = power
    ctrl.exposure = exposure


def snap(ctrl: Control) -> np.ndarray:
    """Snap with the shutter open only for the exposure."""
    ctrl.shutter = "open"
    try:
        return ctrl.snap()
    finally:
        ctrl.shutter = "closed"


# Camera (and lightpath) name -> camera device name, where they differ: the Zyla is 'camera'.
_CAMERA_DEVICES = {"zyla": "camera"}


def check_camera(ctrl: Control, camera: str | None):
    """Fail before anything moves if `camera` (e.g. "kinetix", "zyla") can't be selected."""
    if camera is None:
        return
    if not isinstance(device(ctrl, _CAMERA_DEVICES.get(camera, camera)), Camera):
        raise ValueError(f"{camera!r} is not a camera.")
    lightpath = next((d for d in ctrl.devices if d.name == "lightpath"), None)
    if lightpath is not None and camera not in list(lightpath.states):
        raise ValueError(f"lightpath={camera!r} is not one of {list(lightpath.states)}.")


def use_camera(ctrl: Control, camera: str | None):
    """Select `camera` and send the light to it, like the notebooks' use_camera. None: keep."""
    if camera is None:
        return
    ctrl.camera = _CAMERA_DEVICES.get(camera, camera)
    if any(d.name == "lightpath" for d in ctrl.devices):
        ctrl.lightpath = camera
    ctrl.wait()


def camera_name(ctrl: Control) -> str:
    """The selected camera, as named in file names ("zyla", "kinetix")."""
    names = {v: k for k, v in _CAMERA_DEVICES.items()}
    return names.get(ctrl.camera.name, ctrl.camera.name)


def snake(xs: np.ndarray, ys: np.ndarray) -> Iterator[tuple[float, float]]:
    """Tile positions row by row, alternating direction; the same order every time."""
    for i, y in enumerate(ys):
        for x in xs if i % 2 == 0 else xs[::-1]:
            yield x, y


def add_tile_times(imgs: Any, dims: tuple[str, ...]):
    """Per-tile `seconds` (NaN until imaged) and wall-clock `timestamp` coordinates."""
    shape = tuple(imgs.sizes[d] for d in dims)
    imgs.coords["seconds"] = (dims, np.full(shape, np.nan))
    imgs.coords["timestamp"] = (dims, np.full(shape, np.datetime64("NaT"), dtype="datetime64[ns]"))
    imgs.coords["timestamp"].encoding["units"] = "microseconds since 1970-01-01"
    imgs.coords["timestamp"].encoding["dtype"] = "int64"


def mark_tile(imgs: Any, index: tuple, seconds: float):
    imgs["seconds"].loc[index] = seconds
    imgs["timestamp"].loc[index] = np.datetime64(datetime.datetime.now())


def tile_acq_func(
    ctrl: Control,
    *,
    name: str,
    exposure: int,
    channel: str,
    power: int,
    light_source: str | None,
    attrs: dict[str, Any] | None = None,
):
    """A ``tiled_acq`` acq_func that tiles the device once into array ``{name}_{exposure}ms``."""

    def acq_func(session, xs, ys):
        imgs = session.array(f"{name}_{exposure}ms", row=ys, col=xs)
        add_tile_times(imgs, ("row", "col"))
        imgs.attrs.update(channel=channel, exposure_ms=exposure, power=power, **(attrs or {}))
        start = time.time()
        bar = (
            tqdm(total=len(ys) * len(xs), desc=f"{exposure} ms tiles", unit="tile")
            if tqdm
            else None
        )
        try:
            set_optics(ctrl, channel, exposure, power, light_source)
            for x, y in snake(xs, ys):
                ctrl.xy = (x, y)
                ctrl.wait()
                mark_tile(imgs, (y, x), time.time() - start)
                imgs.loc[y, x] = snap(ctrl)
                if bar is not None:
                    bar.update()
                yield
        finally:
            ctrl.shutter = "closed"
            if bar is not None:
                bar.close()

    return acq_func
