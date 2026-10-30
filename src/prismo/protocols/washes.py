"""SDS washes after expression and binding, for single-common-inlet devices (PS1.8k, minichip).

Example (ps_2block_v3, imaged on the Kinetix: SDS wash, PBS wash, tile; four times)::

    from prismo.protocols.washes import sds_wash_series

    sds_wash_series(
        c,
        chip_d1,
        sds="in2",
        pbs="in1",
        waste="waste7",
        top_left=d1_top_left,
        bot_right=d1_bottom_right,
    )

Necks are never commanded and buttons stay closed. Every wash is appended to
``patterning_log.json``.
"""

import logging
import os
from collections.abc import Iterator, Sequence
from typing import Any

from ..control import Control
from ..devices.microfluidic import Chip
from ..run import tiled_acq
from ..session import Session
from ._common import LOG_FILE, _check, _GuardedChip, _launch, _Progress, _purge, _set_all
from .imaging import camera_name, check_camera, check_optics, tile_acq_func, use_camera

logger = logging.getLogger(__name__)

# SDS wash step times (seconds).
_SDS_PURGE_S = 5
_SDS_S = 5 * 60
_SDS_PBS_S = 10 * 60


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
    """Wash a single-common-inlet device once with SDS (5 min), then PBS (10 min).

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
):
    """Repeat (SDS wash, PBS wash, tile the device) `n_washes` times. Blocks until done.

    Images with `camera` (the Kinetix by default). Images go to
    ``{out_dir}/{name}_post_SDS_wash{n}_{camera}_{exposure}ms``. The SDS and PBS lines are purged
    once, in the first wash. Everything is checked before the first valve moves, and the series
    stops if a wash doesn't finish. Interrupting the cell stops the running step and leaves the
    valves as they are.

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
    g = None
    try:
        for w in range(1, n_washes + 1):
            logger.info(f"=== Wash {w}/{n_washes} ===")
            g = sds_wash(
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
            g.join()
            if not g.log_entry["completed"]:
                raise RuntimeError(f"Wash {w} did not finish; stopping before imaging.")
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
                g = tiled_acq(
                    ctrl,
                    paths[w, e],
                    acq_func,
                    overlap=overlap,
                    top_left=tuple(top_left),
                    bot_right=tuple(bot_right),
                )
                g.join()
        logger.info("SDS series done.")
    except KeyboardInterrupt:
        logger.warning("Interrupted: stopping the current step; valves are left as they are.")
        if g is not None:
            g.quit()
        ctrl.shutter = "closed"
        raise
