"""Assays: on-chip kinetics read out by tiling the device.

``snap_labeling_kinetics``: SNAP-tag labeling. Tile with the buttons closed (the baseline), then
open the buttons for each incubation, close them and tile again.

Example (1 uM Cy5-BG, Kinetix)::

    from prismo.protocols.assays import snap_labeling_kinetics

    g = snap_labeling_kinetics(
        c,
        chip_d1,
        "Cy5/cy5_SNAP_1uM_BG_kinetix_500ms",
        incubations_s=[15, 30, 60, 120, 120, 120, 120],
        top_left=d1_top_left,
        bot_right=d1_bottom_right,
        info={"ligand": "Cy5-BG", "conc_uM": 1},
    )

The corners must be for the camera used (the Kinetix by default; ``camera="zyla"`` with the Zyla
corners for the Zyla).
"""

import datetime
import json
import logging
import os
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..control import Control
from ..devices.microfluidic import Chip
from ..run import tiled_acq
from ..session import Session
from ._common import _check, _GuardedChip, _necks, _set_all
from .binding import _wait_until
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


def snap_labeling_kinetics(
    ctrl: Control,
    chip: Chip,
    file: str,
    *,
    incubations_s: Sequence[float],
    top_left: Sequence[float],
    bot_right: Sequence[float],
    exposure: int = 500,
    channel: str = "cy5",
    power: int = 100,
    light_source: str | None = "sola",
    name: str = "cy5",
    camera: str | None = "kinetix",
    buttons: Sequence[str] = ("butL", "butR"),
    sandwiches: Sequence[str] = ("sandL", "sandR"),
    overlap: float = 0.1,
    info: dict[str, Any] | None = None,
) -> Session:
    """SNAP labeling kinetics: react with the buttons open, close them to image, repeat.

    Flow the ligand on before calling this. Only the buttons and sandwiches are commanded: the
    sandwiches are closed at the start and never opened, and the necks are never commanded. The
    buttons are closed however it ends (done, ``g.quit()`` or an error).

    Every pass goes into one array ``{name}_{exposure}ms`` of `file` (``time`` x ``row`` x
    ``col``); pass 0 is the baseline before any reaction. Per pass, ``buttons_open_s`` is the total
    buttons-open time so far (the reaction time; it pauses while imaging) and ``incubation_s`` the
    open time just before it (NaN for passes not reached). Per tile, ``seconds`` since the start and
    the wall-clock ``timestamp``.

    Parameters:
    -----------
    ctrl :
        The controller (``c``).
    chip :
        The device, e.g. ``chip_d1``.
    file :
        Zarr file for the images, e.g. ``"Cy5/cy5_SNAP_1uM_BG_kinetix_500ms"``. Must not exist yet.
    incubations_s :
        Buttons-open time (s) before each image after the baseline, e.g. ``[15, 30, 60, 120]``.
    top_left, bot_right :
        Corners of the device for `camera`, e.g. ``d1_top_left`` and ``d1_bottom_right``.
    exposure, channel, power, light_source :
        Exposure (ms), filter cube, light power and light source (None if there is no
        ``light_source`` device). The shutter only opens while a tile is snapped.
    name :
        The images are saved as array ``{name}_{exposure}ms`` in `file`.
    camera :
        Camera to image with, selected (and the lightpath sent to it) before tiling, like
        ``use_camera``: "kinetix" or "zyla". None keeps the current camera.
    buttons, sandwiches :
        The device's button and sandwich valves.
    overlap :
        Tile overlap, as in ``pm.tiled_acq``.
    info :
        Free-form notes saved with the images, e.g. ``{"ligand": "Cy5-BG", "conc_uM": 1}``.

    Returns the acquisition session; ``g.join()`` waits for it and ``g.quit()`` stops it (and
    closes the buttons). Pausing during an incubation keeps the buttons open until it resumes;
    ``incubation_s`` records the real open time.
    """
    if not incubations_s or min(incubations_s) <= 0 or exposure <= 0:
        raise ValueError("incubations_s needs at least one time, all positive; exposure too.")
    if os.path.exists(file):
        raise FileExistsError(f"{file} already exists; pick a new name.")
    _check(chip, {}, [*buttons, *sandwiches])
    if set(_necks(chip)) & {*buttons, *sandwiches}:
        raise ValueError("A neck can't be a button or sandwich valve; necks are left alone.")
    check_optics(ctrl, channel, light_source)
    check_camera(ctrl, camera)

    acq_func = _labeling_acq_func(
        ctrl,
        _GuardedChip(chip, forbid_open=sandwiches),
        incubations_s=list(incubations_s),
        exposure=exposure,
        channel=channel,
        power=power,
        light_source=light_source,
        name=name,
        buttons=buttons,
        sandwiches=sandwiches,
        info=info,
    )
    if os.path.dirname(file):
        os.makedirs(os.path.dirname(file), exist_ok=True)
    use_camera(ctrl, camera)  # before tiled_acq, which sizes the tiles from a snap
    return tiled_acq(
        ctrl, file, acq_func, overlap=overlap, top_left=tuple(top_left), bot_right=tuple(bot_right)
    )


def _labeling_acq_func(
    ctrl: Control,
    chip: _GuardedChip,
    *,
    incubations_s: list[float],
    exposure: int,
    channel: str,
    power: int,
    light_source: str | None,
    name: str,
    buttons: Sequence[str],
    sandwiches: Sequence[str],
    info: dict[str, Any] | None,
):
    """The tiled_acq acq_func behind `snap_labeling_kinetics`."""
    passes = len(incubations_s) + 1  # +1 for the baseline before any reaction

    def acq_func(session, xs, ys):
        imgs = session.array(f"{name}_{exposure}ms", time=passes, row=ys, col=xs)
        add_tile_times(imgs, ("time", "row", "col"))
        imgs.coords["buttons_open_s"] = ("time", np.full(passes, np.nan))
        imgs.coords["incubation_s"] = ("time", np.full(passes, np.nan))
        imgs.attrs.update(
            info=json.dumps(info or {}),
            chip=chip.name,
            channel=channel,
            exposure_ms=exposure,
            power=power,
            incubations_s=incubations_s,
        )
        logger.info(
            f"SNAP labeling: {passes} passes of {len(ys)}x{len(xs)} tiles, incubations "
            f"{incubations_s} s, {channel} {exposure} ms"
        )

        bar = None
        try:
            # Isolate the chambers and start with the buttons closed. Necks are left alone.
            _set_all(chip, sandwiches, "closed")
            _set_all(chip, buttons, "closed")
            start = time.time()
            imgs.attrs["started_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            react_s = 0.0
            if tqdm is not None:
                bar = tqdm(total=passes * len(ys) * len(xs), desc="SNAP labeling", unit="tile")
            for t in range(passes):
                open_s = 0.0
                if t > 0:
                    inc = incubations_s[t - 1]
                    logger.info(f"Pass {t}/{passes - 1}: buttons open for {inc} s")
                    if bar is not None:
                        bar.set_postfix(t=f"{t + 1}/{passes}", step=f"reacting {inc} s")
                    _set_all(chip, buttons, "open")
                    t_open = time.time()
                    yield from _wait_until(t_open + inc)
                    _set_all(chip, buttons, "closed")
                    open_s = time.time() - t_open
                    react_s += open_s
                imgs["incubation_s"].loc[t] = open_s
                imgs["buttons_open_s"].loc[t] = react_s
                logger.info(
                    f"Pass {t}: buttons open {react_s:.1f} s in all, "
                    f"{(time.time() - start) / 60:.1f} min since the start"
                )

                set_optics(ctrl, channel, exposure, power, light_source)
                if bar is not None:
                    bar.set_postfix(t=f"{t + 1}/{passes}", step="imaging")
                for x, y in snake(xs, ys):
                    ctrl.xy = (x, y)
                    ctrl.wait()
                    mark_tile(imgs, (t, y, x), time.time() - start)
                    imgs.loc[t, y, x] = snap(ctrl)
                    if bar is not None:
                        bar.update()
                    yield
            logger.info("SNAP labeling done; buttons closed.")
        finally:
            try:
                _set_all(chip, buttons, "closed")  # stop the reaction however this ends
            finally:
                ctrl.shutter = "closed"
                if bar is not None:
                    bar.close()

    return acq_func
