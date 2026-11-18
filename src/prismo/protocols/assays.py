"""Assays: on-chip kinetics read out by tiling the device.

``labeling_kinetics``: labeling of the protein on the buttons, with any label (SNAP-tag + BG
dye, HaloTag ligand, click chemistry...). Tile with the buttons closed (the baseline), then open
the buttons for each incubation, close them and tile again. Every pass goes into one zarr array.

Example (1 uM Cy5-BG on SNAP, Kinetix)::

    from prismo.protocols.assays import labeling_kinetics

    g = labeling_kinetics(
        c,
        chip_d1,
        "Cy5/cy5_SNAP_1uM_BG_kinetix_500ms",
        incubations_s=[1, 5, 5, 10, 10, 10, 120, 120, 120, 300, 300, 300, 300],
        channel="cy5",
        top_left=d1_top_left,
        bot_right=d1_bottom_right,
        info={"label": "Cy5-BG", "tag": "SNAP", "conc_uM": 1},
    )

``tiled_kinetics``: imaging-only kinetics in several channels. Set the valves yourself first,
then tile the device back to back, snapping every channel at each tile before moving on.

Example (Cy5 then eGFP at every tile, 7 passes with growing pauses, on the Ti)::

    from prismo.protocols.assays import tiled_kinetics

    chip_d1.sandL = chip_d1.sandR = "closed"
    time.sleep(2)
    chip_d1.butL = chip_d1.butR = "open"
    t0 = time.time()
    g = tiled_kinetics(
        c,
        "Assay/two_color_kinetics_zyla",
        channels=("cy5", "egfp"),
        wait_s=[0, 30, 60, 120, 300, 300],  # pause before each pass after the first
        top_left=d1_top_left,
        bot_right=d1_bottom_right,
        t0=t0,
        info={"substrate": "BzP", "conc_uM": 78.125},
    )

The corners must be for the camera used (the Kinetix by default; ``camera="zyla"`` with the Zyla
corners for the Zyla).

``simulate_labeling`` plans the waits: the labeled fraction expected at each image for one or more
half times, e.g. ``simulate_labeling([30, 120, 600], incubations_s)``.
"""

import datetime
import json
import logging
import os
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..control import Control
from ..devices.microfluidic import Chip
from ..run import tiled_acq
from ..session import Session
from ._common import _check, _close_verified, _GuardedChip, _necks, _set_all
from .binding import _wait_until
from .imaging import (
    add_tile_times,
    check_camera,
    check_optics,
    mark_tile,
    resolve_defaults,
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


def labeling_kinetics(
    ctrl: Control,
    chip: Chip,
    file: str,
    *,
    incubations_s: Sequence[float] | np.ndarray,
    channel: str,
    top_left: Sequence[float],
    bot_right: Sequence[float],
    exposure: int = 500,
    power: int = 100,
    light_source: str | None = "auto",
    name: str | None = None,
    camera: str | None = "auto",
    buttons: Sequence[str] = ("butL", "butR"),
    sandwiches: Sequence[str] = ("sandL", "sandR"),
    overlap: float = 0.1,
    info: dict[str, Any] | None = None,
) -> Session:
    """Labeling kinetics: react with the buttons open, close them to image, repeat.

    Flow the label on before calling this. Only the buttons and sandwiches are commanded: the
    sandwiches are closed at the start and never opened, and the necks are never commanded. Every
    close before imaging is read back (and retried), so no tile is taken with a button open. The
    buttons are closed however it ends (done, ``g.quit()`` or an error).

    Every pass goes into one array ``{name}_{exposure}ms`` of `file` (``time`` x ``row`` x
    ``col``); pass 0 is the baseline before any reaction. Per pass, ``buttons_open_s`` is the total
    buttons-open time so far (the reaction time; it pauses while imaging) and ``incubation_s`` the
    open time just before it (NaN for passes not reached). Per tile, ``seconds`` since the start and
    the wall-clock ``timestamp``. The scope settings are saved with it.

    Parameters:
    -----------
    ctrl :
        The controller (``c``).
    chip :
        The device, e.g. ``chip_d1``.
    file :
        Zarr file for the images, e.g. ``"Cy5/cy5_SNAP_1uM_BG_kinetix_500ms"``. Must not exist yet.
    incubations_s :
        Buttons-open times (s), in order: one per pass after the baseline, e.g.
        ``[15, 30, 60, 120]`` or a numpy array such as ``np.repeat([10, 60, 300], 3)``.
    channel :
        Filter cube for the label, e.g. "cy5" or "gfp".
    top_left, bot_right :
        Corners of the device for `camera`, e.g. ``d1_top_left`` and ``d1_bottom_right``.
    exposure, power, light_source :
        Exposure (ms), light power and light source (None if there is no ``light_source``
        device; "auto" uses "sola" if there is one). The shutter only opens while a tile is
        snapped.
    name :
        The images are saved as array ``{name}_{exposure}ms`` in `file`; `channel` by default.
    camera :
        Camera to image with, selected (and the lightpath sent to it) before tiling, like
        ``use_camera``: "kinetix" or "zyla". None keeps the current camera; "auto"
        picks the Kinetix if there is one, else keeps the current camera.
    buttons, sandwiches :
        The device's button and sandwich valves.
    overlap :
        Tile overlap, as in ``pm.tiled_acq``.
    info :
        Free-form notes saved with the images, e.g. ``{"label": "Cy5-BG", "conc_uM": 1}``.

    Returns the acquisition session; ``g.join()`` waits for it and ``g.quit()`` stops it (and
    closes the buttons). Pausing during an incubation keeps the buttons open until it resumes;
    ``incubation_s`` records the real open time.
    """
    incubations = _seconds(incubations_s, "incubations_s")
    if exposure <= 0:
        raise ValueError("exposure must be positive.")
    if os.path.exists(file):
        raise FileExistsError(f"{file} already exists; pick a new name.")
    _check(chip, {}, [*buttons, *sandwiches])
    if set(_necks(chip)) & {*buttons, *sandwiches}:
        raise ValueError("A neck can't be a button or sandwich valve; necks are left alone.")
    camera, light_source = resolve_defaults(ctrl, camera, light_source)
    check_optics(ctrl, channel, light_source)
    check_camera(ctrl, camera)

    acq_func = _labeling_acq_func(
        ctrl,
        _GuardedChip(chip, forbid_open=sandwiches),
        incubations_s=incubations,
        exposure=exposure,
        channel=channel,
        power=power,
        light_source=light_source,
        name=channel if name is None else name,
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
    """The tiled_acq acq_func behind `labeling_kinetics`."""
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
            f"Labeling kinetics: {passes} passes of {len(ys)}x{len(xs)} tiles, incubations "
            f"{incubations_s} s, {channel} {exposure} ms"
        )

        bar = None
        try:
            # Isolate the chambers and start with the buttons closed. Necks are left alone.
            for valve in (*sandwiches, *buttons):
                yield from _close_verified(chip, valve)
            start = time.time()
            imgs.attrs["started_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            react_s = 0.0
            if tqdm is not None:
                bar = tqdm(total=passes * len(ys) * len(xs), desc="Labeling", unit="tile")
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
                    for button in buttons:
                        yield from _close_verified(chip, button)
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
            logger.info("Labeling kinetics done; buttons closed.")
        finally:
            try:
                _set_all(chip, buttons, "closed")  # stop the reaction however this ends
            finally:
                ctrl.shutter = "closed"
                if bar is not None:
                    bar.close()

    return acq_func


def snap_labeling_kinetics(*args: Any, channel: str = "cy5", **kwargs: Any) -> Session:
    """`labeling_kinetics` under its earlier name, imaging in cy5 unless told otherwise."""
    return labeling_kinetics(*args, channel=channel, **kwargs)


# Categorical colors, in order (the dataviz reference palette, light mode).
_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
# Log time axis ticks (s): 1 s, 5 s, 10 s, 30 s, 1 min, 5 min, 10 min, 30 min, 1 h, 5 h, 10 h.
_LOG_TICKS_S = (1, 5, 10, 30, 60, 300, 600, 1800, 3600, 18000, 36000)


def simulate_labeling(
    half_times_s: float | Sequence[float] | np.ndarray,
    incubations_s: Sequence[float] | np.ndarray,
    *,
    plot: bool = True,
    log_time: bool = False,
    ax: Any = None,
    return_table: bool = False,
) -> pd.DataFrame | None:
    """The labeled fraction expected at each image of `labeling_kinetics`, to plan the waits.

    First-order labeling: fraction = 1 - 2^(-t / t½), with t the total buttons-open time before
    the image (the reaction pauses while imaging).

    Parameters:
    -----------
    half_times_s :
        Expected labeling half time (s), or several (up to 8) to see a range.
    incubations_s :
        Buttons-open times (s) in order, as passed to ``labeling_kinetics``.
    plot :
        Draw the curves, with a dot at each image.
    log_time :
        Log time axis, to spread out early images when the waits go from seconds to many
        minutes. The baseline (t = 0) isn't shown on it.
    ax :
        Matplotlib axes to draw on (a new figure if None).
    return_table :
        Also return the numbers: one row per pass (0 is the baseline) with ``incubation_s``,
        ``buttons_open_s`` and the labeled fraction for each half time, in columns named like
        ``"t½=60s"``.
    """
    half_times = _seconds(half_times_s, "half_times_s")
    incubations = _seconds(incubations_s, "incubations_s")
    if plot and len(half_times) > len(_COLORS):
        raise ValueError(f"Plot at most {len(_COLORS)} half times at once.")
    open_s = np.concatenate([[0.0], np.cumsum(incubations)])
    table = pd.DataFrame({"incubation_s": [0.0, *incubations], "buttons_open_s": open_s})
    table.index.name = "pass"
    for h in half_times:
        table[_half_time_label(h)] = _labeled(open_s, h)
    if plot:
        _plot_labeling(table, half_times, log_time, ax)
    return table if return_table else None


def _seconds(times: float | Sequence[float] | np.ndarray, name: str) -> list[float]:
    out = [float(s) for s in np.ravel(times)]
    if not out or min(out) <= 0:
        raise ValueError(f"{name} needs at least one time, all positive.")
    return out


def _labeled(t: np.ndarray, half_time: float) -> np.ndarray:
    return 1 - 2 ** (-t / half_time)


def _half_time_label(half_time: float) -> str:
    return f"t½={half_time:g}s"


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:g} s"
    if seconds < 3600:
        return f"{seconds / 60:g} min"
    return f"{seconds / 3600:g} h"


def _plot_labeling(table: pd.DataFrame, half_times: list[float], log: bool, ax: Any):
    import matplotlib.pyplot as plt  # only for plotting; not a prismo dependency

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.2), layout="constrained")
    open_s = table["buttons_open_s"].to_numpy()
    first, last = open_s[1], open_s[-1]
    if log:
        scale, xlabel = 1, "Buttons-open time"
        t = np.geomspace(first / 1.5, last * 1.5, 400)
        shown = slice(1, None)  # the baseline (t = 0) has no place on a log axis
    else:
        scale, unit = (60, "min") if last > 300 else (1, "s")
        xlabel = f"Buttons-open time ({unit})"
        t = np.linspace(0, last, 400)
        shown = slice(None)
    font = dict(fontfamily="Arial", color="black")
    for h, color in zip(half_times, _COLORS, strict=False):
        column = _half_time_label(h)
        ax.plot(t / scale, _labeled(t, h), color=color, lw=2, label=f"{h:g} s", zorder=2)
        # A dot at each image, ringed so dots stay apart where curves meet.
        dots = (open_s[shown] / scale, table[column].to_numpy()[shown])
        ax.plot(*dots, "o", color=color, ms=7, mec="white", mew=1.5, zorder=3, clip_on=False)
    if log:
        ax.set_xscale("log")
        ax.set_xlim(t[0], t[-1])
        # Round durations; every other one when the axis spans too many.
        ticks = [s for s in _LOG_TICKS_S if t[0] <= s <= t[-1]]
        if len(ticks) > 6:
            ticks = [s for s in ticks if s in _LOG_TICKS_S[::2]]
        ax.set_xticks(ticks, [_duration(s) for s in ticks])
        ax.xaxis.set_minor_locator(plt.NullLocator())
    else:
        ax.set_xlim(-0.02 * last / scale, last / scale * 1.03)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel(xlabel, **font)
    ax.set_ylabel("Fraction labeled", **font)
    ax.grid(False)
    ax.tick_params(width=1.5, colors="black", labelfontfamily="Arial")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    # Outside the axes, so the legend never covers a curve.
    legend = ax.legend(
        title="t½",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        prop=dict(family="Arial"),
        title_fontproperties=dict(family="Arial"),
        handlelength=1.5,
    )
    for text in [legend.get_title(), *legend.get_texts()]:
        text.set_color("black")


def tiled_kinetics(
    ctrl: Control,
    file: str,
    *,
    channels: Sequence[str],
    top_left: Sequence[float],
    bot_right: Sequence[float],
    n_passes: int | None = None,
    wait_s: float | Sequence[float] = 0,
    exposure: int | Sequence[int] = 500,
    power: int | Sequence[int] = 100,
    settle_s: float = 0.5,
    t0: float | None = None,
    light_source: str | None = "auto",
    camera: str | None = "auto",
    overlap: float = 0.1,
    name: str = "kinetics",
    info: dict[str, Any] | None = None,
) -> Session:
    """Tile the device again and again, snapping every channel (in order) at each tile before
    moving on. Imaging only: it never touches a valve, so set them before calling this.

    Everything goes into one array `name` of `file` (``time`` x ``channel`` x ``row`` x ``col``),
    one compressed chunk per image. ``exposure_ms`` and ``power`` are per-channel coordinates. Per
    image, ``seconds`` since `t0` (or since the first pass started) and the wall-clock
    ``timestamp``; NaN for images not reached. Every pass starts at the top-left tile, so each
    tile is sampled at the same point in every pass.

    Parameters:
    -----------
    ctrl :
        The controller (``c``).
    file :
        Zarr file for the images. Must not exist yet.
    channels :
        Filter cubes to snap at every tile, in order, e.g. ``("cy5", "egfp")``.
    top_left, bot_right :
        Corners of the device for `camera`, e.g. ``d1_top_left`` and ``d1_bottom_right``.
    n_passes :
        Number of times to tile the device. Can be left out when `wait_s` is a list.
    wait_s :
        Pause (s) after each pass before the next starts: one number for every pause, or a list
        with one per pause (so ``len(wait_s) + 1`` passes), e.g. ``[0, 30, 60, 120]``.
    exposure, power :
        Exposure (ms) and light power, one for all channels or one per channel.
    settle_s :
        Pause (s) after switching channels before snapping. The turret can report it is done
        while still turning, and with no stage move in between it would blur the image.
    t0 :
        When the reaction started (``time.time()`` when you opened the buttons). Saved with the
        images, and ``seconds`` counts from it. None counts from the first pass.
    light_source :
        Light source (None if there is no ``light_source`` device; "auto" uses "sola" if there is
        one). The shutter only opens while an image is snapped.
    camera :
        Camera to image with, like ``use_camera``: "kinetix" or "zyla". None keeps the current
        camera; "auto" picks the Kinetix if there is one, else keeps the current camera.
    overlap :
        Tile overlap, as in ``pm.tiled_acq``.
    name :
        Name of the array in `file`.
    info :
        Free-form notes saved with the images, e.g. ``{"substrate": "BzP", "conc_uM": 78.125}``.

    Returns the acquisition session; ``g.join()`` waits for it and ``g.quit()`` stops it.
    """
    channels = list(channels)
    if not channels or len(set(channels)) != len(channels):
        raise ValueError(f"channels needs at least one filter cube, each once, got {channels}.")
    exposures = _per_channel(exposure, channels, "exposure")
    powers = _per_channel(power, channels, "power")
    waits = _waits(wait_s, n_passes)
    n_passes = len(waits) + 1
    if min(exposures) <= 0 or min(waits, default=0) < 0 or settle_s < 0:
        raise ValueError("Exposures must be positive and wait_s and settle_s at least 0.")
    if os.path.exists(file):
        raise FileExistsError(f"{file} already exists; pick a new name.")
    camera, light_source = resolve_defaults(ctrl, camera, light_source)
    for channel in channels:
        check_optics(ctrl, channel, light_source)
    check_camera(ctrl, camera)

    acq_func = _kinetics_acq_func(
        ctrl,
        name=name,
        channels=channels,
        exposures=exposures,
        powers=powers,
        n_passes=n_passes,
        waits=waits,
        settle_s=settle_s,
        t0=t0,
        light_source=light_source,
        info=info,
    )
    if os.path.dirname(file):
        os.makedirs(os.path.dirname(file), exist_ok=True)
    use_camera(ctrl, camera)  # before tiled_acq, which sizes the tiles from a snap
    return tiled_acq(
        ctrl, file, acq_func, overlap=overlap, top_left=tuple(top_left), bot_right=tuple(bot_right)
    )


def _waits(wait_s: float | Sequence[float], n_passes: int | None) -> list[float]:
    """One pause per gap between passes."""
    if isinstance(wait_s, (int, float)):
        if n_passes is None or n_passes < 1:
            raise ValueError("Give n_passes (at least 1), or a list of wait_s.")
        return [float(wait_s)] * (n_passes - 1)
    if n_passes is not None and n_passes != len(wait_s) + 1:
        raise ValueError(
            f"wait_s has {len(wait_s)} pauses, so n_passes must be {len(wait_s) + 1} or left out."
        )
    return [float(w) for w in wait_s]


def _per_channel(value: int | Sequence[int], channels: list[str], what: str) -> list[int]:
    if isinstance(value, (int, float)):
        return [value] * len(channels)
    if len(value) != len(channels):
        raise ValueError(f"Give one {what} for all channels or one per channel ({channels}).")
    return list(value)


def _kinetics_acq_func(
    ctrl: Control,
    *,
    name: str,
    channels: list[str],
    exposures: list[int],
    powers: list[int],
    n_passes: int,
    waits: list[float],
    settle_s: float,
    t0: float | None,
    light_source: str | None,
    info: dict[str, Any] | None,
):
    """The tiled_acq acq_func behind `tiled_kinetics`."""

    def acq_func(session, xs, ys):
        imgs = session.array(name, time=n_passes, channel=channels, row=ys, col=xs)
        add_tile_times(imgs, ("time", "channel", "row", "col"))
        imgs.coords["exposure_ms"] = ("channel", exposures)
        imgs.coords["power"] = ("channel", powers)
        imgs.coords["pass_start_s"] = ("time", np.full(n_passes, np.nan))
        imgs.coords["wait_s"] = ("time", [0.0, *waits])  # pause before each pass
        imgs.attrs.update(
            info=json.dumps(info or {}),
            n_passes=n_passes,
            settle_s=settle_s,
            px_len_um=float(ctrl.px_len),
        )
        logger.info(
            f"Kinetics: {n_passes} passes of {len(ys)}x{len(xs)} tiles in {channels}, "
            f"exposures {exposures} ms"
        )

        bar = None
        try:
            start = time.time() if t0 is None else t0
            imgs.attrs["t0"] = datetime.datetime.fromtimestamp(start).isoformat(timespec="seconds")
            if tqdm is not None:
                bar = tqdm(total=n_passes * len(ys) * len(xs), desc="Kinetics", unit="tile")
            current = None  # the channel the turret was last sent to
            for t in range(n_passes):
                if t > 0 and waits[t - 1] > 0:
                    logger.info(f"Waiting {waits[t - 1]} s before pass {t + 1}")
                    if bar is not None:
                        bar.set_postfix(t=f"{t + 1}/{n_passes}", step=f"waiting {waits[t - 1]} s")
                    yield from _wait_until(time.time() + waits[t - 1])
                pass_start = time.time()
                imgs["pass_start_s"].loc[t] = pass_start - start
                logger.info(f"Pass {t + 1}/{n_passes}: {(pass_start - start) / 60:.1f} min")
                if bar is not None:
                    bar.set_postfix(t=f"{t + 1}/{n_passes}")
                for x, y in snake(xs, ys):
                    ctrl.xy = (x, y)
                    for channel, exposure, power in zip(channels, exposures, powers):
                        set_optics(ctrl, channel, exposure, power, light_source)
                        ctrl.wait()  # the stage move and the filter change
                        if channel != current:
                            time.sleep(settle_s)
                            current = channel
                        mark_tile(imgs, (t, channel, y, x), time.time() - start)
                        imgs.loc[t, channel, y, x] = snap(ctrl)
                    if bar is not None:
                        bar.update()
                    yield
            logger.info("Kinetics done.")
        finally:
            ctrl.shutter = "closed"
            if bar is not None:
                bar.close()

    return acq_func
