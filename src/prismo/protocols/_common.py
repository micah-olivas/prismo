"""Shared machinery for the protocols: the valve guard, lockstep chip groups, the progress bar,
the protocol log and running a protocol in the background.

Any valve whose name starts with ``neck`` is left alone: the guard refuses to send it any command,
not even "closed".
"""

import datetime
import json
import logging
import os
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from ..devices.microfluidic import Chip
from ..run import run_async
from ..session import Session
from ..utils import sleep

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional; without it progress is only logged.
    tqdm = None

logger = logging.getLogger(__name__)


# Every protocol run (patterning and washes) is appended here, in the working directory.
LOG_FILE = "patterning_log.json"


def protocol_running(session: Any) -> bool:
    """True while a protocol session started from this module still has steps to run."""
    return any(w.is_alive() for w in getattr(session, "_workers", []))


def _hardware_valves(chip: Chip) -> set[tuple[int, int]]:
    """(driver id, valve index) of every valve in `chip`'s mapping."""
    out = set()
    for v in chip._mapping.values():
        for idx in getattr(v, "_valves", None) or getattr(v, "_all", []):
            out.add((id(v._driver), idx))
    return out


class _ChipGroup:
    """Several chips with the same valve names, driven in lockstep (e.g. d1 and d2 together).

    Setting a valve sets it on every chip in turn. Reading one gives "closed" or "open" only when
    every chip agrees and "mixed" otherwise, so a `!= "closed"` safety check fails if any chip is
    not closed.
    """

    def __init__(self, chips: Sequence[Chip]):
        names = [c.name for c in chips]
        if len(chips) < 2 or len(set(names)) != len(names):
            raise ValueError(f"Pass two or more different chips, got {names}.")
        valve_names = set(chips[0].valve_states)
        seen: set[tuple[int, int]] = set()
        for c in chips:
            if set(c.valve_states) != valve_names:
                raise ValueError(f"{names[0]} and {c.name} must have the same valve names.")
            hw = _hardware_valves(c)
            if hw & seen:
                # Lockstep would otherwise move one chip's valve (maybe a neck) for another's.
                raise ValueError(
                    f"{c.name} shares hardware valves with another chip; check mappings."
                )
            seen |= hw
        self.chips = list(chips)
        self.name = "+".join(names)

    @property
    def valve_states(self) -> dict[str, list[str | int]]:
        return self.chips[0].valve_states

    def __getitem__(self, key: str) -> str:
        states = set()
        for c in self.chips:
            v = c[key]
            states.add("closed" if v == "closed" else "open" if v == "open" else "mixed")
        return states.pop() if len(states) == 1 else "mixed"

    def __setitem__(self, key: str, state: str):
        for c in self.chips:
            c[key] = state


def _as_chip(chip: Chip | Sequence[Chip]) -> Chip | _ChipGroup:
    if isinstance(chip, Chip):
        return chip
    return chip[0] if len(chip) == 1 else _ChipGroup(chip)


def _members(chip: Chip | _ChipGroup) -> list[Chip]:
    return chip.chips if isinstance(chip, _ChipGroup) else [chip]


class _GuardedChip:
    """Wraps a Chip: logs every valve command, never commands a neck valve (not even "closed"),
    and refuses to open any `forbid_open` valve."""

    def __init__(self, chip: Chip | _ChipGroup, forbid_open: Sequence[str]):
        object.__setattr__(self, "_chip", chip)
        object.__setattr__(self, "_necks", set(_necks(chip)))
        object.__setattr__(self, "_forbid", set(forbid_open))

    def _set(self, key: str, state: str):
        if key.split("[")[0] in self._necks:
            raise RuntimeError(f"Refusing to set {key} to {state!r}: necks are left alone.")
        if state != "closed" and key.split("[")[0] in self._forbid:
            raise RuntimeError(f"Refusing to set {key} to {state!r}: it must stay closed.")
        logger.info(f"    {key} -> {state}")
        self._chip[key] = state

    def __getattr__(self, key: str) -> Any:
        return getattr(self._chip, key)

    def __getitem__(self, key: str) -> Any:
        return self._chip[key]

    def __setattr__(self, key: str, state: str):
        self._set(key, state)

    def __setitem__(self, key: str, state: str):
        self._set(key, state)

    def close_all(self):
        """Close every valve except the necks, which are left alone."""
        self.close_all_except([])

    def close_all_except(self, keep: Sequence[str]):
        """Close every valve except `keep` and the necks, which are left alone."""
        logger.info(f"    all valves except {sorted({*keep, *self._necks})} -> closed")
        for v in self._chip.valve_states:
            base = v.split("[")[0]
            if base not in keep and base not in self._necks:
                self._chip[v] = "closed"

    def open_all(self):
        raise RuntimeError("open_all is not allowed in patterning protocols.")


class _Progress:
    """One progress bar for a whole protocol; its label shows the current step.

    Also records the start time of every timed step in `steps`.
    """

    def __init__(self, total_s: int, desc: str, steps: list[dict[str, Any]]):
        self.steps = steps
        self.bar = None
        if tqdm is not None:
            self.bar = tqdm(
                total=total_s,
                desc=desc,
                unit="s",
                bar_format="{desc} {percentage:3.0f}%|{bar}| {elapsed}<{remaining} {postfix}",
            )

    def wait(self, seconds: int, desc: str) -> Iterator[None]:
        seconds = int(seconds)
        logger.info(f"{desc} ({seconds // 60}:{seconds % 60:02d})")
        self.steps.append(dict(step=desc, start=_now(), seconds=seconds))
        if self.bar is None:
            yield from sleep(seconds)
            return
        self.bar.set_description(desc)
        start, done = time.time(), 0
        for _ in sleep(seconds):
            step = int(min(seconds, time.time() - start)) - done
            if step > 0:
                self.bar.update(step)
                done += step
                left = seconds - done
                self.bar.set_postfix_str(f"step {left // 60}:{left % 60:02d} left")
            yield
        self.bar.update(seconds - done)

    def close(self, desc: str):
        if self.bar is not None:
            self.bar.set_description(desc)
            self.bar.set_postfix_str("")
            self.bar.close()


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _necks(chip: Chip | _ChipGroup) -> list[str]:
    return sorted({v.split("[")[0] for v in chip.valve_states if v.startswith("neck")})


def _check(chip: Chip | _ChipGroup, roles: dict[str, str], structural: Sequence[str]):
    """Validate the reagent/outlet `roles` and the `structural` valves of the device(s)."""
    if len(set(roles.values())) != len(roles):
        raise ValueError(f"Each role needs its own valve, got {roles}.")
    for ch in _members(chip):
        valves = ch.valve_states
        necks = _necks(ch)
        for role, valve in roles.items():
            if valve not in valves:
                raise ValueError(f"{role}={valve!r} does not exist in {ch.name}.")
            if valve in necks:
                raise ValueError(f"{role}={valve!r} is a neck valve; necks are left alone.")
            if valve in structural:
                raise ValueError(
                    f"{role}={valve!r} is a device valve, not a reagent inlet or outlet."
                )
        for valve in structural:
            if valve not in valves:
                raise ValueError(f"{ch.name} has no {valve!r} valve; check the valve arguments.")


def _append_log(path: str | None, entry: dict[str, Any]):
    if path is None:
        return
    try:
        log = []
        if os.path.exists(path):
            with open(path) as f:
                log = json.load(f)
        log.append(entry)
        with open(path, "w") as f:
            json.dump(log, f, indent=1, default=str)
    except Exception as e:  # Never let logging mask the protocol's own outcome.
        logger.error(f"Could not write {path}: {e}\n{entry}")


def _launch(
    chip: Chip | _ChipGroup,
    protocol: str,
    record: dict[str, Any],
    info: dict[str, Any] | None,
    log_file: str | None,
    total_s: int,
    forbid_open: Sequence[str],
    body: Callable[..., Iterator[None]],
    done_msg: str,
    blocking: bool,
) -> Session:
    """Run `body(guarded_chip, progress)` in a background session with logging and the guard."""
    logger.info(f"{protocol} on {chip.name}: { {k: v for k, v in record.items() if k != 'times'} }")
    entry = dict(
        protocol=protocol, chip=chip.name, **record, info=info or {}, steps=[], completed=False
    )
    g = _GuardedChip(chip, forbid_open=forbid_open)

    def steps() -> Iterator[None]:
        entry["start"] = _now()
        prog = _Progress(total_s, protocol, entry["steps"])
        try:
            yield from body(g, prog)
            entry["completed"] = True
            logger.info(done_msg)
        finally:
            entry["end"] = _now()
            prog.close(done_msg if entry["completed"] else f"{protocol} stopped")
            _append_log(log_file, entry)

    session = run_async(steps)(blocking=blocking)
    session.log_entry = entry  # e.g. log_entry["completed"] once it has finished
    return session


def _set_all(chip: _GuardedChip, valves: Sequence[str], state: str):
    for v in valves:
        chip[v] = state


def _purge(
    chip: _GuardedChip, flow: str, waste: str, inlet: str, seconds: int, prog: _Progress, desc: str
) -> Iterator[None]:
    """Purge air from `flow` to `waste` with the common inlet closed. Leaves `flow` open."""
    chip[inlet] = "closed"
    chip[flow] = "open"
    chip[waste] = "open"
    yield from prog.wait(seconds, desc)
    chip[waste] = "closed"
