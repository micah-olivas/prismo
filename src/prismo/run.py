import threading
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import dill
import numpy as np
import zarr as zr

from .control import Control
from .session import Session
from .views import AcquisitionView, LiveView
from .widgets import init_widgets


def run_async[**P](func: Callable[P, Iterator[Any]]) -> Callable[P, Any]:
    # `blocking` is keyword-only so that positional args pass through to `func`.
    def wrapper(*args: P.args, blocking: bool = False, **kwargs: P.kwargs) -> Any:
        def run_func(_session):
            yield from func(*args, **kwargs)

        session = run(run_func)
        if blocking:
            session.join()
        return session

    return wrapper


def run(run_func: Callable[[Session], Iterator[Any]]) -> Session:
    session = Session()

    @session.worker
    def worker():
        yield from run_func(session)

    session.start()
    return session


def live(ctrl: Control) -> Session:
    widgets, widget_routes = init_widgets(ctrl)
    session = Session(lambda v, r: LiveView(v, r, widgets=widgets))

    img = ctrl.snap()

    @session.worker
    def snap():
        while True:
            img[:] = ctrl.snap()
            yield

    session.route("img", lambda: img)
    for name, func in widget_routes.items():
        session.route(name, func)

    session.start()

    return session


def acq(ctrl: Control, file: str, acq_func: Callable[[Session], Iterator[Any]]) -> Session:
    widgets, widget_routes = init_widgets(ctrl)
    session = Session(
        lambda v, r: AcquisitionView(v, r, file=file, widgets=widgets),
        file,
        ctrl.snap(),
        dict(acq_func=dill.source.getsource(acq_func)),
    )

    @session.worker
    def acq():
        store = zr.storage.LocalStore(file)
        for _ in acq_func(session):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Consolidated metadata")
                for name, xp in session.arrays.items():
                    xp.to_dataset(promote_attrs=True, name="tile").to_zarr(
                        store, group=name, compute=False, mode="a"
                    )
            yield

    session.route("arrays", lambda: set(session.arrays.keys()))
    for name, func in widget_routes.items():
        session.route(name, func)

    session.start()
    return session


def multi_acq(
    ctrl: Control,
    file: str,
    acq_func: Callable[[Session, list[tuple[float, float]]], Iterator[Any]],
    overlap: float = 0.0,
) -> Session:
    tile = ctrl.snap()
    pos: list[list[tuple[float, float]] | None] = [None]
    acq_event = threading.Event()

    widgets, widget_routes = init_widgets(ctrl)
    session = Session(
        lambda v, r: AcquisitionView(v, r, file=file, widgets=widgets, multi=True),
        file,
        ctrl.snap(),
        dict(overlap=overlap, acq_func=dill.source.getsource(acq_func)),
    )

    @session.worker
    def acq():
        while not acq_event.is_set():
            tile[:] = ctrl.snap()
            yield

        store = zr.storage.LocalStore(file)
        assert pos[0] is not None
        for _ in acq_func(session, pos[0]):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Consolidated metadata")
                for name, xp in session.arrays.items():
                    xp.to_dataset(promote_attrs=True, name="tile").to_zarr(
                        store, group=name, compute=False, mode="a"
                    )
            yield

    @session.route("start_acq")
    def start_acq(xys):
        pos[0] = xys
        acq_event.set()

    session.route("img", lambda: tile)
    session.route("xy", lambda: ctrl.xy)
    session.route("arrays", lambda: set(session.arrays.keys()))
    for name, func in widget_routes.items():
        session.route(name, func)

    session.start()
    return session


def tiled_acq(
    ctrl: Control,
    file: str,
    acq_func: Callable[[Session, np.ndarray, np.ndarray], Iterator[Any]],
    overlap: float,
    top_left: tuple[float, float] | None = None,
    bot_right: tuple[float, float] | None = None,
) -> Session:
    tile = ctrl.snap()
    pos: list[np.ndarray | None] = [None, None]
    acq_event = threading.Event()

    widgets, widget_routes = init_widgets(ctrl)
    # TODO: Write additional attrs e.g. px_len.
    session = Session(
        lambda v, r: AcquisitionView(
            v, r, file=file, widgets=widgets, tiled=top_left is None or bot_right is None
        ),
        file,
        ctrl.snap(),
        dict(overlap=overlap, acq_func=dill.source.getsource(acq_func)),
    )

    @session.worker
    def acq():
        if top_left is None or bot_right is None:
            while not acq_event.is_set():
                tile[:] = ctrl.snap()
                yield
            xs, ys = pos
            assert xs is not None and ys is not None
        else:
            xs, ys = tile_coords(ctrl, top_left, bot_right, overlap)

        store = zr.storage.LocalStore(file)
        for _ in acq_func(session, xs, ys):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Consolidated metadata")
                for name, xp in session.arrays.items():
                    xp.to_dataset(promote_attrs=True, name="tile").to_zarr(
                        store, group=name, compute=False, mode="a"
                    )
            yield

    @session.route("start_acq")
    def start_acq(top_left, bot_right):
        xs, ys = tile_coords(ctrl, top_left, bot_right, overlap)
        pos[0] = xs
        pos[1] = ys
        acq_event.set()

    session.route("img", lambda: tile)
    session.route("xy", lambda: ctrl.xy)
    session.route("arrays", lambda: set(session.arrays.keys()))
    for name, func in widget_routes.items():
        session.route(name, func)

    session.start()
    return session


def tile_coords(
    ctrl: Control,
    top_left: tuple[float, float],
    bot_right: tuple[float, float],
    overlap: float,
) -> tuple[np.ndarray, np.ndarray]:
    tile = ctrl.snap()
    width = tile.shape[1]
    height = tile.shape[0]
    overlap_x = int(round(overlap * width))
    overlap_y = int(round(overlap * height))
    delta_x = (width - overlap_x) * ctrl.px_len
    delta_y = (height - overlap_y) * ctrl.px_len
    xs = np.arange(top_left[0], bot_right[0] + delta_x - 1, delta_x)
    ys = np.arange(top_left[1], bot_right[1] + delta_y - 1, delta_y)
    return xs, ys
