"""Device setup: open and close valves repeatedly to loosen them before first use.

Only meant for setting up a new device, before patterning. By default each chip's numbered
inlets and waste are actuated (PS1.8k v2: in2..in9 + waste, v3: in0..in6 + waste7, minichip:
in0..in4 + waste).

Example (both PS1.8k v2 devices, each chip's own inlet and waste valves)::

    from prismo.protocols.valves import actuate_valves

    actuate_valves([c.d1_ps_2block_v2, c.d2_ps_2block_v2])

Works with any chip mapping, and the chips in a list don't need the same valve names. Necks are
left alone: they are never commanded, and nothing moves unless every neck already reads closed.
"""

import re
import time
from collections.abc import Sequence

from ..devices.microfluidic import Chip
from ._common import _GuardedChip, _hardware_valves, _necks

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional; without it each cycle is printed.
    tqdm = None

# Default valves to actuate: numbered inlets and the waste, e.g. in0..in9, waste, waste7.
_INLET_VALVE = re.compile(r"(in\d+|waste\d*)")


def inlet_valves(chip: Chip) -> list[str]:
    """The numbered inlet and waste valves of `chip` (in0..in9, waste...), in mapping order."""
    return [v for v in chip.valve_states if _INLET_VALVE.fullmatch(v)]


def actuate_valves(
    chips: Chip | Sequence[Chip],
    valves: Sequence[str] | None = None,
    cycles: int = 20,
    interval_s: float = 0.5,
):
    """Open and close valves together `cycles` times, `interval_s` seconds in each state.

    Parameters:
    -----------
    chips :
        One chip or a list, e.g. ``[c.d1_ps_2block_v2, c.d2_ps_2block_v2]``. Every chip cycles
        together, each with its own mapping.
    valves :
        Valve names to actuate on every chip. Default: each chip's numbered inlets and waste
        (``inlet_valves``). Pass them for other names, e.g. ``["waste1", "bBSA2", "na3"]``.
    cycles :
        Number of open/close cycles.
    interval_s :
        Seconds open, then seconds closed, in each cycle.

    Notes:
    ------
    Blocks until done. Every actuated valve ends closed, also when interrupted.
    """
    chips = [chips] if isinstance(chips, Chip) else list(chips)
    names = [ch.name for ch in chips]
    if not chips or len(set(names)) != len(names):
        raise ValueError(f"Pass one or more different chips, got {names}.")
    if cycles < 1 or interval_s <= 0:
        raise ValueError("cycles must be at least 1 and interval_s positive.")

    seen: set[tuple[int, int]] = set()
    plan: list[tuple[_GuardedChip, list[str]]] = []
    for ch in chips:
        hw = _hardware_valves(ch)
        if hw & seen:
            # Otherwise one chip's valve could be another chip's neck.
            raise ValueError(f"{ch.name} shares hardware valves with another chip; check mappings.")
        seen |= hw
        vs = list(valves) if valves is not None else inlet_valves(ch)
        if not vs:
            raise ValueError(f"{ch.name} has no in<N>/waste valves; pass `valves`.")
        missing = [v for v in vs if v not in ch.valve_states]
        if missing:
            raise ValueError(f"{missing} do not exist in {ch.name}.")
        necks = _necks(ch)
        if set(vs) & set(necks):
            raise ValueError(f"{sorted(set(vs) & set(necks))} are necks; necks are left alone.")
        # Read-only: an open neck means the user opened it (e.g. expression), so don't touch.
        not_closed = [n for n in necks if ch[n] != "closed"]
        if not_closed:
            raise RuntimeError(
                f"{not_closed} on {ch.name} not closed, so nothing was actuated. Necks are "
                "never commanded; close them yourself first if that is intended."
            )
        plan.append((_GuardedChip(ch, forbid_open=[]), vs))

    for g, vs in plan:
        print(f"{g.name}: {', '.join(vs)}")
    bar = tqdm(total=cycles, desc="actuating valves") if tqdm is not None else None
    try:
        for i in range(cycles):
            for state in ("open", "closed"):
                for g, vs in plan:
                    for v in vs:
                        g[v] = state
                time.sleep(interval_s)
            if bar is not None:
                bar.update()
            else:
                print(f"cycle {i + 1}/{cycles}")
    finally:
        for g, vs in plan:
            for v in vs:
                g[v] = "closed"
        if bar is not None:
            bar.close()
