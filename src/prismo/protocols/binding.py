"""Binding: tile the device again and again to follow protein binding at the buttons.

Imaging only: this never touches a valve (it isn't given the chip). Set the valves yourself
first (e.g. open the buttons, sandwiches closed), then run it. Run it after the post-expression
image and before the SDS wash. Every pass goes into one zarr array (``time`` x ``row`` x ``col``),
with the seconds since imaging started and the wall-clock time of every tile.

Example (Kinetix, 30 back-to-back eGFP tilings)::

    from prismo.protocols.binding import track_binding

    g = track_binding(
        c,
        "GFP/egfp_binding_kinetix_500ms",
        top_left=d1_top_left,
        bot_right=d1_bottom_right,
        passes=30,
    )

The corners must be for the camera used (the Kinetix by default; ``camera="zyla"`` with the Zyla
corners for the Zyla).

``g.quit()`` stops it early; the passes taken so far are kept.
"""

import datetime
import json
import logging
import os
import time
from collections.abc import Iterator, Sequence
from typing import Any

from ..control import Control
from ..run import tiled_acq
from ..session import Session
from .imaging import (
    add_tile_times,
    check_camera,
    check_optics,
    mark_tile,
    set_optics,
    snake,
    snap,
    use_camera,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional; without it each pass is only logged.
    tqdm = None

logger = logging.getLogger(__name__)


def _wait_until(t_end: float) -> Iterator[None]:
    """Wait until `t_end`, yielding about once a second so the session can still quit."""
    while (left := t_end - time.time()) > 0:
        time.sleep(min(left, 1.0))
        yield


def track_binding(
    ctrl: Control,
    file: str,
    *,
    top_left: Sequence[float],
    bot_right: Sequence[float],
    passes: int = 30,
    interval_s: float = 0,
    exposure: int = 500,
    channel: str = "gfp",
    power: int = 100,
    light_source: str | None = "sola",
    name: str = "egfp",
    camera: str | None = "kinetix",
    overlap: float = 0.1,
    info: dict[str, Any] | None = None,
) -> Session:
    """Tile the device `passes` times. Never touches a valve.

    Parameters:
    -----------
    ctrl :
        The controller (``c``).
    file :
        Zarr file for the images, e.g. ``"GFP/egfp_binding_kinetix_500ms"``. Must not exist yet.
    top_left, bot_right :
        Corners of the device for `camera`, e.g. ``d1_top_left`` and ``d1_bottom_right``.
    passes :
        Number of tilings.
    interval_s :
        Seconds from the start of one pass to the start of the next. 0 tiles back-to-back; if a
        pass takes longer than this, the next one starts right away.
    exposure, channel, power, light_source :
        Exposure (ms), filter cube, light power and light source (None if there is no
        ``light_source`` device). The shutter only opens while a tile is snapped.
    name :
        The images are saved as array ``{name}_{exposure}ms`` in `file`.
    camera :
        Camera to image with, selected (and the lightpath sent to it) before tiling, like
        ``use_camera``: "kinetix" or "zyla". None keeps the current camera.
    overlap :
        Tile overlap, as in ``pm.tiled_acq``.
    info :
        Free-form notes saved with the images, e.g. ``{"buttons_opened": "14:05"}``.

    Returns the acquisition session; ``g.join()`` waits for it and ``g.quit()`` stops it.
    """
    if passes < 1 or interval_s < 0 or exposure <= 0:
        raise ValueError("passes must be at least 1, interval_s at least 0, exposure positive.")
    if os.path.exists(file):
        raise FileExistsError(f"{file} already exists; pick a new name.")
    check_optics(ctrl, channel, light_source)
    check_camera(ctrl, camera)

    acq_func = _binding_acq_func(
        ctrl,
        passes=passes,
        interval_s=interval_s,
        exposure=exposure,
        channel=channel,
        power=power,
        light_source=light_source,
        name=name,
        info=info,
    )
    if os.path.dirname(file):
        os.makedirs(os.path.dirname(file), exist_ok=True)
    use_camera(ctrl, camera)  # before tiled_acq, which sizes the tiles from a snap
    return tiled_acq(
        ctrl, file, acq_func, overlap=overlap, top_left=tuple(top_left), bot_right=tuple(bot_right)
    )


def _binding_acq_func(
    ctrl: Control,
    *,
    passes: int,
    interval_s: float,
    exposure: int,
    channel: str,
    power: int,
    light_source: str | None,
    name: str,
    info: dict[str, Any] | None,
):
    """The tiled_acq acq_func behind `track_binding`."""

    def acq_func(session, xs, ys):
        imgs = session.array(f"{name}_{exposure}ms", time=passes, row=ys, col=xs)
        add_tile_times(imgs, ("time", "row", "col"))  # seconds since imaging started
        imgs.attrs.update(
            info=json.dumps(info or {}),
            channel=channel,
            exposure_ms=exposure,
            power=power,
            interval_s=interval_s,
        )
        logger.info(
            f"Binding imaging: {passes} passes of {len(ys)}x{len(xs)} tiles, {channel} "
            f"{exposure} ms"
        )

        bar = None
        try:
            start = time.time()
            imgs.attrs["started_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            if tqdm is not None:
                bar = tqdm(total=passes * len(ys) * len(xs), desc="Binding", unit="tile")
            for t in range(passes):
                yield from _wait_until(start + t * interval_s)
                logger.info(f"Pass {t + 1}/{passes}: {(time.time() - start) / 60:.1f} min")
                set_optics(ctrl, channel, exposure, power, light_source)
                # Same snake order every pass, so each tile is imaged at a steady interval.
                for x, y in snake(xs, ys):
                    ctrl.xy = (x, y)
                    ctrl.wait()
                    mark_tile(imgs, (t, y, x), time.time() - start)
                    imgs.loc[t, y, x] = snap(ctrl)
                    if bar is not None:
                        bar.set_postfix(t=f"{t + 1}/{passes}")
                        bar.update()
                    yield
            logger.info(f"Binding imaging done after {(time.time() - start) / 60:.1f} min.")
        finally:
            ctrl.shutter = "closed"
            if bar is not None:
                bar.close()

    return acq_func
