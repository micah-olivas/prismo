"""SDS washes and buffer flushes for single-common-inlet devices (PS1.8k, minichip).

Example (d2: flush buffer for 2 min, then dead-end it for 10 min; three waits, then a final
flush; queued to start once the running assay ``g`` finishes)::

    from prismo.protocols.washes import iterated_flush

    g_flush = iterated_flush(
        chip_d2,
        buffer="in1",
        waste="waste7",
        flush_min=2,
        n_waits=3,
        wait_min=10,
        after=g,
    )

Example (ps_2block_v3, imaged on the Kinetix: SDS wash, PBS wash, tile; four times)::

    from prismo.protocols.washes import sds_wash_series

    g = sds_wash_series(
        c,
        chip_d1,
        sds="in2",
        pbs="in1",
        waste="waste7",
        top_left=d1_top_left,
        bot_right=d1_bottom_right,
    )

The series runs in the background: ``g.quit()`` stops it. Necks are never commanded and buttons
stay closed. Every wash is appended to ``patterning_log.json``.
"""

import logging
import os
from collections.abc import Iterator, Sequence
from typing import Any

from ..control import Control
from ..devices.microfluidic import Chip
from ..run import run_async, tiled_acq
from ..session import Session
from ..utils import sleep
from ._common import (
    LOG_FILE,
    _check,
    _close_verified,
    _GuardedChip,
    _launch,
    _Progress,
    _purge,
    _set_all,
    protocol_running,
)
from .imaging import camera_name, check_camera, check_optics, tile_acq_func, use_camera

logger = logging.getLogger(__name__)

# SDS wash step times (seconds).
_SDS_PURGE_S = 5
_SDS_S = 15 * 60
_SDS_PBS_S = 10 * 60
# Air purge before a buffer flush (seconds).
_FLUSH_PURGE_S = 5


def iterated_flush(
    chip: Chip,
    *,
    buffer: str,
    flush_min: float,
    n_waits: int,
    wait_min: float,
    waste: str = "waste",
    outlet: str = "out",
    inlet: str = "inlet",
    buttons: Sequence[str] = ("butL", "butR"),
    sandwiches: Sequence[str] = ("sandL", "sandR"),
    purge: bool = True,
    open_necks: bool = False,
    after: Session | None = None,
    info: dict[str, Any] | None = None,
    log_file: str | None = LOG_FILE,
    blocking: bool = False,
) -> Session:
    """Flush `buffer` through the device, then hold it dead-ended; `n_waits` times, with a final
    flush after the last wait: flush, (wait, flush) x `n_waits`.

    Every valve but the necks is closed first; the buttons are read back closed before anything
    flows, then the sandwiches are opened. Each flush runs `flush_min` with the outlet open. Each
    wait keeps `buffer` and the inlet open with the outlet closed (dead-ended) for `wait_min`.
    Every outlet close is read back and retried. Buttons stay closed. Necks are never commanded,
    unless `open_necks` opens them before the first flush. Even then they are never closed.
    Ends dead-ended in `buffer`, outlet closed.

    Pass ``after=g`` to queue the flush behind a running assay's session `g`: nothing moves until
    `g` has finished (or stopped), then the flush starts.

    Returns the flush's session: ``quit()`` cancels a queued flush, or stops a running one and
    leaves the valves as they are.

    Parameters:
    -----------
    chip :
        A PS1.8k v2 or ps_2block_v3 Chip, e.g. ``chip_d2``.
    buffer :
        Inlet holding the buffer.
    flush_min :
        Minutes of each flush.
    n_waits :
        Number of dead-ended waits (0 for a single flush).
    wait_min :
        Minutes of each wait.
    waste :
        Inlet used to purge air from the buffer line.
    outlet :
        Outlet valve.
    inlet, buttons, sandwiches :
        The device's common inlet, button and sandwich valves.
    purge :
        Purge air from the buffer line to waste (5 s) before the first flush.
    open_necks :
        Open the necks before the first flush and leave them open: they are never sent "closed".
        Off by default, and then the necks are never sent any command.
    after :
        A running protocol or acquisition session to wait for before starting.
    info :
        Free-form notes saved in the patterning log.
    log_file :
        JSON log appended to after the run (None to disable).
    blocking :
        Wait for the flush to finish before returning.
    """
    if flush_min <= 0 or n_waits < 0 or (n_waits and wait_min <= 0):
        raise ValueError("flush_min must be > 0, n_waits >= 0 and, with waits, wait_min > 0.")
    roles = dict(buffer=buffer, waste=waste, outlet=outlet)
    _check(chip, roles, [inlet, *buttons, *sandwiches])
    flush_s, wait_s = round(flush_min * 60), round(wait_min * 60)

    def body(g: _GuardedChip, prog: _Progress) -> Iterator[None]:
        g.close_all()  # skips the necks
        # Nothing flows until the buttons read closed.
        for button in buttons:
            yield from _close_verified(g, button)
        _set_all(g, sandwiches, "open")
        if open_necks:
            _set_all(g, sorted(g._necks), "open")
        if purge:
            yield from _purge(g, buffer, waste, inlet, _FLUSH_PURGE_S, prog, "purge")
        else:
            g[buffer] = "open"
        g[inlet] = "open"
        for i in range(n_waits + 1):
            if i:
                yield from prog.wait(wait_s, f"wait {i}/{n_waits}")
            g[outlet] = "open"
            yield from prog.wait(flush_s, f"flush {i + 1}/{n_waits + 1}")
            yield from _close_verified(g, outlet)

    record = dict(
        **roles, flush_min=flush_min, n_waits=n_waits, wait_min=wait_min, open_necks=open_necks
    )
    total = (_FLUSH_PURGE_S if purge else 0) + (n_waits + 1) * flush_s + n_waits * wait_s
    done = f"Flush done; dead-ended in {buffer}, outlet closed"
    done += ", necks open." if open_necks else "."
    forbid = list(buttons)

    def launch(blocking: bool) -> Session:
        return _launch(
            chip,
            "iterated_flush",
            record,
            info,
            log_file,
            total,
            forbid,
            body,
            done,
            blocking,
            open_necks=open_necks,
        )

    if after is None:
        return launch(blocking)

    logger.info(f"Flush of {chip.name} queued; it starts once the running session finishes.")

    @run_async
    def queued() -> Iterator[None]:
        flush = None
        try:
            yield from _wait(after)
            logger.info(f"The running session finished; starting the flush of {chip.name}.")
            flush = launch(False)
            yield from _wait(flush)
        except GeneratorExit:  # quit()
            if flush is None:
                logger.warning("Queued flush cancelled before it started; no valve was touched.")
            else:
                logger.warning("Flush stopped; valves left as they are.")
                flush.quit()
                flush.join()
            raise

    return queued(blocking=blocking)


def sds_wash(
    chip: Chip,
    *,
    sds: str,
    pbs: str,
    waste: str = "waste",
    outlet: str = "out",
    inlet: str = "inlet",
    buttons: Sequence[str] = ("butL", "butR"),
    sandwiches: Sequence[str] = ("sandL", "sandR"),
    purge: bool = True,
    info: dict[str, Any] | None = None,
    log_file: str | None = LOG_FILE,
    blocking: bool = False,
) -> Session:
    """Wash a single-common-inlet device once with SDS (15 min), then PBS (10 min).

    Buttons stay closed and necks are never commanded. Ends dead-ended in PBS, outlet closed.

    Parameters:
    -----------
    chip :
        A PS1.8k v2 or ps_2block_v3 Chip.
    sds, pbs :
        Inlets holding SDS and wash PBS.
    waste :
        Inlet used to purge air from the other inlets.
    outlet :
        Outlet valve.
    inlet, buttons, sandwiches :
        The device's common inlet, button and sandwich valves.
    purge :
        Purge air from the SDS and PBS lines to waste (5 s each) before flowing them.
    info :
        Free-form notes saved in the patterning log.
    log_file :
        JSON log appended to after the run (None to disable).
    blocking :
        Wait for the wash to finish before returning.
    """
    roles = dict(sds=sds, pbs=pbs, waste=waste, outlet=outlet)
    _check(chip, roles, [inlet, *buttons, *sandwiches])

    def body(g: _GuardedChip, prog: _Progress) -> Iterator[None]:
        def flow(reagent: str, seconds: int, desc: str) -> Iterator[None]:
            if purge:
                yield from _purge(g, reagent, waste, inlet, _SDS_PURGE_S, prog, f"{desc}: purge")
            else:
                g[reagent] = "open"
            g[inlet] = "open"
            g[outlet] = "open"
            yield from prog.wait(seconds, desc)
            g[reagent] = "closed"
            g[outlet] = "closed"

        g.close_all()  # skips the necks
        _set_all(g, sandwiches, "open")
        yield from flow(sds, _SDS_S, "SDS wash")
        yield from flow(pbs, _SDS_PBS_S, "PBS wash")
        g[inlet] = "open"
        g[pbs] = "open"

    total = (2 * _SDS_PURGE_S if purge else 0) + _SDS_S + _SDS_PBS_S
    done = "SDS wash done; dead-ended in PBS."
    forbid = list(buttons)
    return _launch(chip, "sds_wash", roles, info, log_file, total, forbid, body, done, blocking)


def sds_wash_series(
    ctrl: Control,
    chip: Chip,
    *,
    sds: str,
    pbs: str,
    top_left: Sequence[float],
    bot_right: Sequence[float],
    n_washes: int = 4,
    exposures: Sequence[int] = (500,),
    channel: str = "gfp",
    power: int = 100,
    light_source: str | None = "sola",
    out_dir: str = "GFP",
    name: str = "egfp",
    camera: str | None = "kinetix",
    waste: str = "waste",
    outlet: str = "out",
    inlet: str = "inlet",
    buttons: Sequence[str] = ("butL", "butR"),
    sandwiches: Sequence[str] = ("sandL", "sandR"),
    overlap: float = 0.1,
    info: dict[str, Any] | None = None,
    log_file: str | None = LOG_FILE,
    blocking: bool = False,
) -> Session:
    """Repeat (SDS wash, PBS wash, tile the device) `n_washes` times, in the background.

    Images with `camera` (the Kinetix by default). Images go to
    ``{out_dir}/{name}_post_SDS_wash{n}_{camera}_{exposure}ms``. The SDS and PBS lines are purged
    once, in the first wash. Everything is checked before the first valve moves, and the series
    stops if a wash doesn't finish.

    Returns the series' session: ``g.quit()`` stops the running wash or tiling and leaves the
    valves as they are; ``g.join()`` waits for the series. Don't start other imaging while it
    runs.

    Parameters:
    -----------
    sds, pbs, waste, outlet, inlet, buttons, sandwiches :
        As in ``sds_wash``.
    top_left, bot_right :
        Corners of the device for `camera`, e.g. ``d1_top_left`` and ``d1_bottom_right``.
    n_washes :
        Number of wash + image rounds.
    exposures, channel, power, light_source :
        Exposures (ms) to tile after each wash, filter cube, light power and light source.
    out_dir, name :
        Where the images go, and the name they start with.
    camera :
        Camera to image with, selected (and the lightpath sent to it) before the first wash, like
        ``use_camera``: "kinetix" or "zyla". None keeps the current camera.
    info :
        Free-form notes saved in the patterning log with every wash.
    blocking :
        Wait for the series to finish before returning.
    """
    if n_washes < 1 or not exposures:
        raise ValueError("n_washes must be at least 1, with at least one exposure.")
    _check(chip, dict(sds=sds, pbs=pbs, waste=waste, outlet=outlet), [inlet, *buttons, *sandwiches])
    check_optics(ctrl, channel, light_source)
    check_camera(ctrl, camera)
    cam = camera if camera is not None else camera_name(ctrl)
    paths = {
        (w, e): f"{out_dir}/{name}_post_SDS_wash{w}_{cam}_{e}ms"
        for w in range(1, n_washes + 1)
        for e in exposures
    }
    existing = [p for p in paths.values() if os.path.exists(p)]
    if existing:
        raise FileExistsError(f"{existing} already exist; change `name` or `out_dir`.")
    os.makedirs(out_dir, exist_ok=True)
    use_camera(ctrl, camera)

    logger.info(f"SDS series: {n_washes} washes, imaging with {cam} at {list(exposures)} ms")

    @run_async
    def series() -> Iterator[None]:
        step = None  # the wash or tiling running now
        try:
            for w in range(1, n_washes + 1):
                logger.info(f"=== Wash {w}/{n_washes} ===")
                step = sds_wash(
                    chip,
                    sds=sds,
                    pbs=pbs,
                    waste=waste,
                    outlet=outlet,
                    inlet=inlet,
                    buttons=buttons,
                    sandwiches=sandwiches,
                    purge=(w == 1),
                    info={**(info or {}), "wash": w, "n_washes": n_washes},
                    log_file=log_file,
                )
                yield from _wait(step)
                if not step.log_entry["completed"]:
                    logger.error(f"Wash {w} did not finish; stopping the series before imaging.")
                    return
                # Dead-ended in PBS now; image it.
                for e in exposures:
                    acq_func = tile_acq_func(
                        ctrl,
                        name=name,
                        exposure=e,
                        channel=channel,
                        power=power,
                        light_source=light_source,
                        attrs=dict(chip=chip.name, wash=w),
                    )
                    step = tiled_acq(
                        ctrl,
                        paths[w, e],
                        acq_func,
                        overlap=overlap,
                        top_left=tuple(top_left),
                        bot_right=tuple(bot_right),
                    )
                    yield from _wait(step)
            logger.info("SDS series done.")
        except GeneratorExit:  # g.quit()
            logger.warning(
                "SDS series stopped: stopping the current step; valves left as they are."
            )
            if step is not None:
                step.quit()  # a tiling closes the shutter as it stops
                step.join()
            raise

    return series(blocking=blocking)


def _wait(session: Session) -> Iterator[None]:
    """Wait for a step's session to finish, yielding so the series can be quit meanwhile."""
    while protocol_running(session):
        yield from sleep(0.2)
