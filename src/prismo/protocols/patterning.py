"""Patterning protocols for single-common-inlet chips and the minichip.

Covers devices where every reagent enters through one common ``inlet`` valve:

* PS1.8k:
    - 2-block:
        - v2: one ``neck`` valve
        - v3: split ``neckL`` ``neckR``
    - 4-block:
* Minichip (``but``/``sand`` plus a 3-bit IO multiplexer, ``0_0`` ... ``2_1``).

Valve names are explicit arguments and must match the chip mapping in your notebook config.
Arguments are checked when you call a protocol, so a bad valve name fails in the cell before any
valve moves. Any valve whose name starts with ``neck`` is left alone: these protocols never send
it any command, not even "closed" (``open_all`` is refused too). Patterning only reads the necks,
and refuses to start unless every neck already reads closed.

Every run is appended to ``patterning_log.json`` (in the working directory by default) with the
inlets, step times, the start time of each step, and whether it completed.

Example (PS1.8k v2, high anti-eGFP on the left side, low anti-eGFP on the right)::

    from prismo.protocols.patterning import pattern_anti_egfp_high_low

    p = pattern_anti_egfp_high_low(
        c.d1_ps_2block_v2,
        bbsa="in6",
        na="in5",
        anti_egfp_high="in4",
        anti_egfp_low="in3",
        pbs="in2",
        high_side="L",
    )

Pass a list of chips, e.g. ``[c.d1_ps_2block_v2, c.d2_ps_2block_v2]``, to pattern several
devices at once in lockstep. They must have the same valve names and the same inlets, and every
valve command goes to all of them.
"""

import logging
from collections.abc import Iterator, Sequence
from typing import Any

from ..devices.microfluidic import Chip
from ..session import Session
from ._common import (
    LOG_FILE,
    _as_chip,
    _check,
    _ChipGroup,
    _GuardedChip,
    _launch,
    _members,
    _necks,
    _Progress,
    _purge,
    _set_all,
    protocol_running,  # noqa: F401  (imported from here by the notebooks)
)

logger = logging.getLogger(__name__)


# Step times in seconds. STANDARD_TIMES matches the full-length prismo pattern_anti_gfp.
STANDARD_TIMES = dict(
    bbsa_purge=5,
    bbsa_closed=5 * 60,
    bbsa_open=35 * 60,
    pbs_purge=30,
    pbs_wash1=10 * 60,
    na_purge=30,
    na=30 * 60,
    pbs_wash2=10 * 60,
    quench=35 * 60,
    pbs_wash3=10 * 60,
    anti_egfp_purge=30,
    anti_egfp_closed_pre=30,
    anti_egfp_open=int(13.3 * 60),
    anti_egfp_closed_post=30,
    pbs_wash_anti_egfp=10 * 60,
)
# Shortened minichip patterning (from 2026-07), with every buttons-open step at 5 min.
MINICHIP_SHORT_TIMES = dict(
    bbsa_purge=30,
    bbsa_closed=2 * 60,
    bbsa_open=5 * 60,
    pbs_purge=30,
    pbs_wash1=2 * 60,
    na_purge=30,
    na=5 * 60,
    pbs_wash2=3 * 60,
    quench=10 * 60,
    pbs_wash3=3 * 60,
    anti_egfp_purge=30,
    anti_egfp_closed_pre=30,
    anti_egfp_open=5 * 60,
    anti_egfp_closed_post=30,
    pbs_wash_anti_egfp=4 * 60,
)
_TO_ANTI_EGFP_STEPS = (
    "bbsa_purge",
    "bbsa_closed",
    "bbsa_open",
    "pbs_purge",
    "pbs_wash1",
    "na_purge",
    "na",
    "pbs_wash2",
    "quench",
    "pbs_wash3",
)
_ANTI_EGFP_STEPS = (
    "anti_egfp_purge",
    "anti_egfp_closed_pre",
    "anti_egfp_open",
    "anti_egfp_closed_post",
    "pbs_wash_anti_egfp",
)

# (sandwich, button) valves of each side of a PS1.8k v2 / ps_2block_v3 device.
PS_SIDES = {"L": ("sandL", "butL"), "R": ("sandR", "butR")}

# Minichip IO multiplexer: the valves to open for each of the 8 lanes.
MINICHIP_IO_LANES = {
    0: ("0_0", "1_0", "2_0"),
    1: ("0_0", "1_0", "2_1"),
    2: ("0_0", "1_1", "2_0"),
    3: ("0_0", "1_1", "2_1"),
    4: ("0_1", "1_0", "2_0"),
    5: ("0_1", "1_0", "2_1"),
    6: ("0_1", "1_1", "2_0"),
    7: ("0_1", "1_1", "2_1"),
}


def _check_times(times: dict[str, int], steps: Sequence[str]):
    missing = [k for k in steps if k not in times]
    if missing:
        raise ValueError(f"times is missing {missing}; start from STANDARD_TIMES.")


def _require_necks_closed(chip: Chip | _ChipGroup | _GuardedChip):
    """Read-only: the necks are never commanded, so patterning refuses to start unless they read
    closed already."""
    inner = chip._chip if isinstance(chip, _GuardedChip) else chip
    for ch in _members(inner):
        not_closed = [n for n in _necks(ch) if ch[n] != "closed"]
        if not_closed:
            raise RuntimeError(
                f"{not_closed} on {ch.name} not closed. Patterning never touches the necks, so "
                "nothing was started; set them yourself if that is intended."
            )


def _bbsa_na_quench(
    chip: _GuardedChip,
    t: dict[str, int],
    prog: _Progress,
    *,
    bbsa: str,
    na: str,
    pbs: str,
    waste: str,
    inlet: str,
    buttons: Sequence[str],
) -> Iterator[None]:
    """bBSA -> PBS -> neutravidin -> PBS -> bBSA quench (buttons closed) -> PBS. Ends PBS closed."""
    yield from _purge(chip, bbsa, waste, inlet, t["bbsa_purge"], prog, "Step 1: purge bBSA")
    chip[inlet] = "open"
    yield from prog.wait(t["bbsa_closed"], "Step 1: bBSA, buttons closed")
    _set_all(chip, buttons, "open")
    yield from prog.wait(t["bbsa_open"], "Step 1: bBSA, buttons open")
    chip[bbsa] = "closed"

    yield from _purge(chip, pbs, waste, inlet, t["pbs_purge"], prog, "Step 1: purge PBS")
    chip[inlet] = "open"
    yield from prog.wait(t["pbs_wash1"], "Step 1: PBS wash")
    chip[pbs] = "closed"

    yield from _purge(chip, na, waste, inlet, t["na_purge"], prog, "Step 2: purge NA")
    chip[inlet] = "open"
    yield from prog.wait(t["na"], "Step 2: neutravidin, buttons open")
    chip[na] = "closed"
    chip[pbs] = "open"
    yield from prog.wait(t["pbs_wash2"], "Step 2: PBS wash")
    chip[pbs] = "closed"

    _set_all(chip, buttons, "closed")
    chip[bbsa] = "open"
    yield from prog.wait(t["quench"], "Step 3: bBSA quench, buttons closed")
    chip[bbsa] = "closed"
    chip[pbs] = "open"
    yield from prog.wait(t["pbs_wash3"], "Step 3: PBS wash")
    chip[pbs] = "closed"


def _flow_anti_egfp(
    chip: _GuardedChip,
    t: dict[str, int],
    prog: _Progress,
    *,
    anti_egfp: str,
    pbs: str,
    waste: str,
    inlet: str,
    buttons: Sequence[str],
    keep_closed: Sequence[str],
    label: str,
) -> Iterator[None]:
    """Flow `anti_egfp` (buttons closed / `buttons` open / closed), then PBS-wash. Ends PBS closed.

    Raises if any `keep_closed` valve is not closed while the anti-eGFP is flowing.
    """

    def assert_sealed():
        bad = [v for v in keep_closed if chip[v] != "closed"]
        if bad:
            raise RuntimeError(f"{label}: {bad} must be closed while {anti_egfp} flows.")

    yield from _purge(chip, anti_egfp, waste, inlet, t["anti_egfp_purge"], prog, f"{label}: purge")
    assert_sealed()
    chip[inlet] = "open"
    yield from prog.wait(t["anti_egfp_closed_pre"], f"{label}: buttons closed")
    _set_all(chip, buttons, "open")
    assert_sealed()
    yield from prog.wait(t["anti_egfp_open"], f"{label}: buttons open")
    _set_all(chip, buttons, "closed")
    yield from prog.wait(t["anti_egfp_closed_post"], f"{label}: buttons closed")
    chip[anti_egfp] = "closed"
    chip[pbs] = "open"
    yield from prog.wait(t["pbs_wash_anti_egfp"], f"{label}: PBS wash")
    chip[pbs] = "closed"


def _deadend(chip: _GuardedChip, pbs: str, inlet: str, open_valves: Sequence[str]):
    """Close everything, then dead-end fill from `pbs` through `open_valves` (outlet closed)."""
    chip.close_all()
    _set_all(chip, open_valves, "open")
    chip[inlet] = "open"
    chip[pbs] = "open"


def pattern_anti_egfp(
    chip: Chip | Sequence[Chip],
    *,
    bbsa: str,
    na: str,
    anti_egfp: str,
    pbs: str,
    waste: str = "waste",
    outlet: str = "out",
    inlet: str = "inlet",
    buttons: Sequence[str] = ("butL", "butR"),
    sandwiches: Sequence[str] = ("sandL", "sandR"),
    times: dict[str, int] = STANDARD_TIMES,
    info: dict[str, Any] | None = None,
    log_file: str | None = LOG_FILE,
    blocking: bool = False,
) -> Session:
    """Pattern a bBSA-NA-anti-eGFP pedestal under every button of a single-common-inlet device.

    Same steps and timings as prismo's pattern_anti_gfp, but through one common `inlet` valve.
    The necks are left alone (never commanded) and must already read closed. Ends dead-end
    filled with PBS.

    Parameters:
    -----------
    chip :
        A PS1.8k v2 or ps_2block_v3 Chip, or a list of such chips (e.g. [d1, d2]) to pattern in
        lockstep: same valve names and inlets on each, every valve command sent to all of them.
    bbsa, na, anti_egfp, pbs :
        Inlets holding biotinylated BSA, neutravidin, biotinylated anti-eGFP and PBS.
    waste :
        Inlet used to purge air from the other inlets.
    outlet :
        Outlet valve.
    inlet, buttons, sandwiches :
        The device's common inlet, button and sandwich valves.
    times :
        Step times in seconds (see STANDARD_TIMES).
    info :
        Free-form notes saved in the patterning log, e.g. {"anti_egfp_ug_ml": 10}.
    log_file :
        JSON log appended to after the run (None to disable).
    blocking :
        Wait for the protocol to finish before returning.
    """
    chip = _as_chip(chip)
    roles = dict(bbsa=bbsa, na=na, anti_egfp=anti_egfp, pbs=pbs, waste=waste, outlet=outlet)
    _check(chip, roles, [inlet, *buttons, *sandwiches])
    _check_times(times, _TO_ANTI_EGFP_STEPS + _ANTI_EGFP_STEPS)
    necks = _necks(chip)
    _require_necks_closed(chip)

    def body(g: _GuardedChip, prog: _Progress) -> Iterator[None]:
        _require_necks_closed(g)  # read-only; the necks are never commanded
        g.close_all()
        _set_all(g, sandwiches, "open")
        g[outlet] = "open"
        yield from _bbsa_na_quench(
            g, times, prog, bbsa=bbsa, na=na, pbs=pbs, waste=waste, inlet=inlet, buttons=buttons
        )
        yield from _flow_anti_egfp(
            g,
            times,
            prog,
            anti_egfp=anti_egfp,
            pbs=pbs,
            waste=waste,
            inlet=inlet,
            buttons=buttons,
            keep_closed=necks,
            label="Step 4: anti-eGFP",
        )
        _deadend(g, pbs, inlet, sandwiches)

    total = sum(times[k] for k in _TO_ANTI_EGFP_STEPS + _ANTI_EGFP_STEPS)
    done = f"Patterning done; dead-end filled with {pbs}."
    record = {**roles, "times": times}
    return _launch(
        chip, "pattern_anti_egfp", record, info, log_file, total, necks, body, done, blocking
    )


def pattern_anti_egfp_high_low(
    chip: Chip | Sequence[Chip],
    *,
    bbsa: str,
    na: str,
    anti_egfp_high: str,
    anti_egfp_low: str,
    pbs: str,
    high_side: str = "L",
    waste: str = "waste",
    outlet: str = "out",
    inlet: str = "inlet",
    sides: dict[str, tuple[str, str]] = PS_SIDES,
    low_first: bool = True,
    times: dict[str, int] = STANDARD_TIMES,
    info: dict[str, Any] | None = None,
    log_file: str | None = LOG_FILE,
    blocking: bool = False,
) -> Session:
    """Pattern two anti-eGFP densities: `anti_egfp_high` on one side, `anti_egfp_low` on the other.

    bBSA, neutravidin and the bBSA quench run on both sides exactly as in pattern_anti_egfp. Only
    the anti-eGFP step is split: each anti-eGFP flows with the OTHER side's sandwiches closed
    (sandR closed to pattern the left side, sandL closed to pattern the right), and the other
    side's buttons stay closed too. A PBS wash separates the two anti-eGFP flows. The necks are
    left alone (never commanded) and must already read closed. Ends dead-end filled with PBS
    (both sides).

    Parameters:
    -----------
    chip :
        A PS1.8k v2 or ps_2block_v3 Chip, or a list of such chips (e.g. [d1, d2]) to pattern in
        lockstep: same valve names, inlets and high_side on each, every valve command sent to all.
    bbsa, na, pbs :
        Inlets holding biotinylated BSA, neutravidin and PBS.
    anti_egfp_high, anti_egfp_low :
        Inlets holding the high- and low-concentration anti-eGFP.
    high_side :
        Side that gets `anti_egfp_high` ("L" or "R"); the other side gets `anti_egfp_low`.
    waste :
        Inlet used to purge air from the other inlets.
    outlet :
        Outlet valve.
    inlet :
        The device's common inlet valve.
    sides :
        (sandwich, button) valves of each side (see PS_SIDES).
    low_first :
        Flow `anti_egfp_low` first, so any carryover is low anti-eGFP reaching the high side.
    times :
        Step times in seconds (see STANDARD_TIMES); both anti-eGFP flows use the anti_egfp_* times.
    info :
        Free-form notes saved in the patterning log, e.g. {"anti_egfp_high_ug_ml": 10}.
    log_file :
        JSON log appended to after the run (None to disable).
    blocking :
        Wait for the protocol to finish before returning.
    """
    if set(sides) != {"L", "R"} or high_side not in sides:
        raise ValueError(
            f"sides needs keys 'L' and 'R' and high_side must be one of them, "
            f"got {list(sides)} and {high_side!r}."
        )
    chip = _as_chip(chip)
    low_side = "R" if high_side == "L" else "L"
    sandwiches = [sides["L"][0], sides["R"][0]]
    buttons = [sides["L"][1], sides["R"][1]]
    roles = dict(
        bbsa=bbsa,
        na=na,
        anti_egfp_high=anti_egfp_high,
        anti_egfp_low=anti_egfp_low,
        pbs=pbs,
        waste=waste,
        outlet=outlet,
    )
    _check(chip, roles, [inlet, *buttons, *sandwiches])
    _check_times(times, _TO_ANTI_EGFP_STEPS + _ANTI_EGFP_STEPS)
    necks = _necks(chip)
    _require_necks_closed(chip)

    flows = [("high", anti_egfp_high, high_side), ("low", anti_egfp_low, low_side)]
    if low_first:
        flows.reverse()

    def body(g: _GuardedChip, prog: _Progress) -> Iterator[None]:
        _require_necks_closed(g)  # read-only; the necks are never commanded
        g.close_all()
        _set_all(g, sandwiches, "open")
        g[outlet] = "open"
        yield from _bbsa_na_quench(
            g, times, prog, bbsa=bbsa, na=na, pbs=pbs, waste=waste, inlet=inlet, buttons=buttons
        )
        for level, anti_egfp, side in flows:
            sand, but = sides[side]
            other_sand, other_but = sides["R" if side == "L" else "L"]
            # Seal the other side; only this side's chambers see the anti-eGFP.
            g[other_sand] = "closed"
            g[other_but] = "closed"
            g[sand] = "open"
            yield from _flow_anti_egfp(
                g,
                times,
                prog,
                anti_egfp=anti_egfp,
                pbs=pbs,
                waste=waste,
                inlet=inlet,
                buttons=[but],
                keep_closed=[other_sand, other_but, *necks],
                label=f"Step 4: {level} anti-eGFP ({anti_egfp}) -> side {side}",
            )
        _deadend(g, pbs, inlet, sandwiches)

    total = sum(times[k] for k in _TO_ANTI_EGFP_STEPS) + 2 * sum(times[k] for k in _ANTI_EGFP_STEPS)
    done = f"Patterning done; dead-end filled with {pbs}."
    record = {
        **roles,
        "high_side": high_side,
        "low_side": low_side,
        "sides": sides,
        "low_first": low_first,
        "times": times,
    }
    return _launch(
        chip,
        "pattern_anti_egfp_high_low",
        record,
        info,
        log_file,
        total,
        necks,
        body,
        done,
        blocking,
    )


def pattern_minichip_to_anti_egfp(
    chip: Chip,
    *,
    bbsa: str,
    na: str,
    pbs: str,
    waste: str = "waste",
    outlet: str = "out",
    inlet: str = "inlet",
    button: str = "but",
    sandwich: str = "sand",
    lanes: dict[int, tuple[str, ...]] = MINICHIP_IO_LANES,
    times: dict[str, int] = MINICHIP_SHORT_TIMES,
    info: dict[str, Any] | None = None,
    log_file: str | None = LOG_FILE,
    blocking: bool = False,
) -> Session:
    """Minichip: bBSA, neutravidin and the bBSA quench through every IO lane, no anti-eGFP yet.

    Ends dead-ended in PBS with the buttons closed and all lanes open, so the anti-eGFP can be
    loaded next with load_minichip_anti_egfp (one lane per condition, or all lanes at once).

    Parameters:
    -----------
    chip :
        A minichip Chip.
    bbsa, na, pbs :
        Inlets holding biotinylated BSA, neutravidin and PBS.
    waste :
        Inlet used to purge air from the other inlets.
    outlet :
        Outlet valve.
    inlet, button, sandwich :
        The device's common inlet, button and sandwich valves.
    lanes :
        IO multiplexer valves to open for each lane (see MINICHIP_IO_LANES).
    times :
        Step times in seconds (see MINICHIP_SHORT_TIMES).
    info :
        Free-form notes saved in the patterning log.
    log_file :
        JSON log appended to after the run (None to disable).
    blocking :
        Wait for the protocol to finish before returning.
    """
    io_valves = sorted({v for vs in lanes.values() for v in vs})
    roles = dict(bbsa=bbsa, na=na, pbs=pbs, waste=waste, outlet=outlet)
    _check(chip, roles, [inlet, button, sandwich, *io_valves])
    _check_times(times, _TO_ANTI_EGFP_STEPS)
    necks = _necks(chip)
    _require_necks_closed(chip)

    def body(g: _GuardedChip, prog: _Progress) -> Iterator[None]:
        _require_necks_closed(g)  # read-only; the necks are never commanded
        g.close_all()
        _set_all(g, io_valves, "open")
        g[sandwich] = "open"
        g[outlet] = "open"
        yield from _bbsa_na_quench(
            g, times, prog, bbsa=bbsa, na=na, pbs=pbs, waste=waste, inlet=inlet, buttons=[button]
        )
        _deadend(g, pbs, inlet, [sandwich, *io_valves])

    total = sum(times[k] for k in _TO_ANTI_EGFP_STEPS)
    done = f"Ready for anti-eGFP: dead-ended in PBS ({pbs}), buttons closed."
    record = {**roles, "times": times}
    return _launch(
        chip,
        "pattern_minichip_to_anti_egfp",
        record,
        info,
        log_file,
        total,
        necks,
        body,
        done,
        blocking,
    )


def load_minichip_anti_egfp(
    chip: Chip,
    *,
    io: int | Sequence[int],
    anti_egfp: str,
    pbs: str,
    waste: str = "waste",
    outlet: str = "out",
    inlet: str = "inlet",
    button: str = "but",
    sandwich: str = "sand",
    lanes: dict[int, tuple[str, ...]] = MINICHIP_IO_LANES,
    times: dict[str, int] = MINICHIP_SHORT_TIMES,
    info: dict[str, Any] | None = None,
    log_file: str | None = LOG_FILE,
    blocking: bool = False,
) -> Session:
    """Minichip: flow the anti-eGFP in `anti_egfp` through IO lane(s) `io` only, then PBS-wash them.

    Run after pattern_minichip_to_anti_egfp. Swap the tube in the `anti_egfp` inlet by hand between
    conditions. Ends dead-ended in PBS again (buttons closed, all lanes open).

    Parameters:
    -----------
    chip :
        A minichip Chip.
    io :
        Lane (0-7), or a list of lanes, to load, e.g. 3 or range(8) for every lane.
    anti_egfp, pbs :
        Inlets holding this condition's anti-eGFP and PBS.
    waste :
        Inlet used to purge air from the other inlets.
    outlet :
        Outlet valve.
    inlet, button, sandwich :
        The device's common inlet, button and sandwich valves.
    lanes :
        IO multiplexer valves to open for each lane (see MINICHIP_IO_LANES).
    times :
        Step times in seconds (see MINICHIP_SHORT_TIMES).
    info :
        This condition's recipe, saved in the patterning log, e.g. {"bbsa_mg_ml": 0.5}.
    log_file :
        JSON log appended to after the run (None to disable).
    blocking :
        Wait for the protocol to finish before returning.
    """
    io_list = [io] if isinstance(io, int) else list(io)
    if not io_list or any(i not in lanes for i in io_list):
        raise ValueError(f"io must be lane(s) from {list(lanes)}, got {io!r}.")
    io_valves = sorted({v for vs in lanes.values() for v in vs})
    lane_valves = sorted({v for i in io_list for v in lanes[i]})
    others = [v for v in io_valves if v not in lane_valves]
    roles = dict(anti_egfp=anti_egfp, pbs=pbs, waste=waste, outlet=outlet)
    _check(chip, roles, [inlet, button, sandwich, *io_valves])
    _check_times(times, _ANTI_EGFP_STEPS)
    necks = _necks(chip)

    def body(g: _GuardedChip, prog: _Progress) -> Iterator[None]:
        # Stop the PBS dead-end and route the flow through the chosen lane(s) only.
        g[button] = "closed"
        g[pbs] = "closed"
        _set_all(g, others, "closed")
        _set_all(g, lane_valves, "open")
        g[sandwich] = "open"
        g[outlet] = "open"
        yield from _flow_anti_egfp(
            g,
            times,
            prog,
            anti_egfp=anti_egfp,
            pbs=pbs,
            waste=waste,
            inlet=inlet,
            buttons=[button],
            keep_closed=[*others, *necks],
            label=f"Lane(s) {io_list}: anti-eGFP ({anti_egfp})",
        )
        _deadend(g, pbs, inlet, [sandwich, *io_valves])

    total = sum(times[k] for k in _ANTI_EGFP_STEPS)
    done = f"Done with lane(s) {io_list}; dead-ended in PBS."
    record = {**roles, "io": io_list, "io_valves": lane_valves, "times": times}
    return _launch(
        chip, "load_minichip_anti_egfp", record, info, log_file, total, necks, body, done, blocking
    )
