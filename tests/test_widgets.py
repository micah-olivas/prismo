import datetime
import os
import re

import numpy as np
import pytest

from prismo import devices, widgets
from prismo.control import Control
from prismo.devices.microfluidic import Chip, ValveDriver
from prismo.session import Relay

CUBES = ["gfp", "cy5", "NADHTi", "SOLA_DANAC", "empty5", "empty6"]


class FakeCore:
    """Just the CMMCore calls the Ti2 filter and shutter make; records every set."""

    def __init__(self, state: int | None = 0, shutter_open: bool = False):
        self.state = state  # None means the read fails, e.g. while the turret moves
        self.shutter_open = shutter_open
        self.sets = []

    def getState(self, name):
        if self.state is None:
            raise RuntimeError("turret busy")
        return self.state

    def getStateLabel(self, name):
        return CUBES[self.getState(name)]

    def setState(self, name, state):
        self.sets.append(state)
        self.state = state

    def setStateLabel(self, name, label):
        self.setState(name, CUBES.index(label))

    def getShutterOpen(self, name):
        return self.shutter_open

    def setShutterOpen(self, name, state):
        self.sets.append(state)
        self.shutter_open = state


def ti2_filter(core):
    f = devices.ti2.Filter.__new__(devices.ti2.Filter)
    f.name, f._core, f.states = "filter", core, list(CUBES)
    return f


def ti2_shutter(core):
    s = devices.ti2.Shutter.__new__(devices.ti2.Shutter)
    s.name, s._core = "shutter", core
    return s


class FakeDriver(ValveDriver):
    def __init__(self):
        self.name = "valves"
        self._num_valves = 2
        self.state = {}

    def __getitem__(self, idx):
        return self.state.get(idx, "closed")

    def __setitem__(self, idx, state):
        self.state[idx] = state


class FakeCamera:
    def __init__(self, name, exposure):
        self.name, self._exposure, self.sets = name, exposure, []

    @property
    def exposure(self):
        return self._exposure

    @exposure.setter
    def exposure(self, ms):
        self.sets.append(ms)
        self._exposure = ms


class FakeStage:
    """Moves instantly; records every target and every wait."""

    def __init__(self, xy=(0.0, 0.0)):
        self.name, self._xy, self.moves, self.waits = "stage", tuple(xy), [], 0

    def wait(self):
        self.waits += 1

    @property
    def xy(self):
        return self._xy

    @xy.setter
    def xy(self, new_xy):
        self.moves.append(tuple(new_xy))
        self._xy = tuple(new_xy)


class BrokenStage(FakeStage):
    @property
    def xy(self):
        raise RuntimeError("ASI error: :N-1")


def make_ctrl(devs, camera=None, stage=None):
    ctrl = object.__new__(Control)
    object.__setattr__(ctrl, "devices", devs)
    if camera is not None:
        object.__setattr__(ctrl, "_camera", camera)
    if stage is not None:
        object.__setattr__(ctrl, "_stage", stage)
    return ctrl


class FakeRelay(Relay):
    """Calls the routes in-process instead of over the pipe to the viewer."""

    def __init__(self, routes, path=""):
        self._routes, self._path = routes, path

    def subpath(self, path):
        return FakeRelay(self._routes, self._path + path + "/")

    def get(self, route, *args, **kwargs):
        return self._routes[self._path + route](*args, **kwargs)

    def post(self, route, *args, **kwargs):
        self._routes[self._path + route](*args, **kwargs)


def test_server_reads_and_sets_filter_and_shutter():
    core = FakeCore(state=2)
    server = widgets.FilterShutterControllerServer([ti2_filter(core), ti2_shutter(core)])
    assert server.get_options() == {"filter": CUBES, "shutter": ["open", "closed"]}
    assert server.get_states() == {"filter": "NADHTi", "shutter": "closed"}
    server.set_state("filter", "cy5")
    server.set_state("shutter", "open")
    assert server.get_states() == {"filter": "cy5", "shutter": "open"}


def test_server_polling_never_moves_the_turret(capsys):
    # ti2.Filter.state sets the first cube when its read fails; the poll must not.
    core = FakeCore(state=None)
    server = widgets.FilterShutterControllerServer([ti2_filter(core)])
    assert server.get_states() == {"filter": None}
    assert core.sets == []

    # A failed set is reported, not raised, since a raising route ends the session.
    server.set_state("filter", "not a cube")
    assert "Couldn't set filter" in capsys.readouterr().out


def test_filters_and_shutters_get_their_own_panel_before_the_valves():
    core = FakeCore()
    chip = Chip("chip", FakeDriver(), {"in1": 0, "out": 1})
    ws, routes = widgets.init_widgets(make_ctrl([chip, ti2_filter(core), ti2_shutter(core)]))
    assert list(ws) == [
        widgets.FILTER_SHUTTER_WIDGET,
        widgets.OUTLET_TIMER_WIDGET,
        "chip controller",
    ]
    assert "widget/filters_shutters/set_state" in routes
    assert list(widgets.init_widgets(make_ctrl([chip]))[0]) == [
        widgets.OUTLET_TIMER_WIDGET,
        "chip controller",
    ]
    # A camera alone (for its exposure) is enough for the panel.
    ws, _ = widgets.init_widgets(make_ctrl([chip], camera=FakeCamera("kinetix", 100.0)))
    assert list(ws) == [
        widgets.FILTER_SHUTTER_WIDGET,
        widgets.OUTLET_TIMER_WIDGET,
        "chip controller",
    ]


@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_buttons_set_the_devices(qapp):
    core = FakeCore(state=0)
    ws, routes = widgets.init_widgets(make_ctrl([ti2_filter(core), ti2_shutter(core)]))
    panel = ws[widgets.FILTER_SHUTTER_WIDGET](FakeRelay(routes))

    panel._choices["filter"]["NADHTi"].click()
    assert core.state == CUBES.index("NADHTi")
    toggle = panel._toggles["shutter"]
    assert toggle.text() == "closed"
    toggle.click()
    assert core.shutter_open and toggle.text() == "open"
    toggle.click()
    assert not core.shutter_open

    # Changes made elsewhere (e.g. the notebook) show up on the next poll.
    core.shutter_open = True
    panel.update_states()
    assert toggle.text() == "open"


def test_exposure_box_shows_and_sets_the_current_cameras_exposure(qapp):
    kinetix = FakeCamera("kinetix", 100.0)
    ctrl = make_ctrl([ti2_filter(FakeCore()), ti2_shutter(FakeCore())], camera=kinetix)
    ws, routes = widgets.init_widgets(ctrl)
    panel = ws[widgets.FILTER_SHUTTER_WIDGET](FakeRelay(routes))
    box = panel._exposure
    assert box.value() == 100 and panel._exposure_label.text() == "exposure (kinetix)"

    box.setValue(250)
    assert kinetix.sets == [250]

    # A camera switched in the notebook shows up on the next poll, without being set.
    zyla = FakeCamera("zyla", 30.0)
    object.__setattr__(ctrl, "_camera", zyla)
    panel.update_states()
    assert box.value() == 30 and panel._exposure_label.text() == "exposure (zyla)"
    assert zyla.sets == [] and kinetix.sets == [250]


def test_no_camera_no_exposure_box(qapp):
    ws, routes = widgets.init_widgets(make_ctrl([ti2_filter(FakeCore())]))
    panel = ws[widgets.FILTER_SHUTTER_WIDGET](FakeRelay(routes))
    assert panel._exposure is None
    # Every get must reply: the viewer waits for one.
    assert widgets.FilterShutterControllerServer([]).get_exposure() == {"camera": None, "ms": None}


def test_valve_panels_show_the_chip_name(qapp):
    chip = Chip("d1_ps_2block_v3", FakeDriver(), {"in1": 0, "out": 1})
    ws, routes = widgets.init_widgets(make_ctrl([chip]))
    panel = ws["d1_ps_2block_v3 controller"](FakeRelay(routes))
    assert panel._title.text() == "d1_ps_2block_v3"


def test_stage_gets_its_own_panel_between_filters_and_valves():
    chip = Chip("chip", FakeDriver(), {"in1": 0, "out": 1})
    ctrl = make_ctrl([chip, ti2_filter(FakeCore())], stage=FakeStage())
    ws, routes = widgets.init_widgets(ctrl)
    assert list(ws) == [
        widgets.FILTER_SHUTTER_WIDGET,
        widgets.STAGE_WIDGET,
        widgets.OUTLET_TIMER_WIDGET,
        "chip controller",
    ]
    assert "widget/stage_dpad/move" in routes
    assert widgets.STAGE_WIDGET not in widgets.init_widgets(make_ctrl([chip]))[0]


def test_dpad_steps_the_stage_the_way_the_image_is_laid_out(qapp):
    stage = FakeStage((1000.0, 2000.0))
    ws, routes = widgets.init_widgets(make_ctrl([], stage=stage))
    panel = ws[widgets.STAGE_WIDGET](FakeRelay(routes))
    assert panel._position.text() == "x 1000.0   y 2000.0"

    panel._btns["►"].click()
    panel._step.setValue(25)
    panel._btns["▼"].click()
    panel._btns["◄"].click()
    panel._btns["▲"].click()
    assert stage.moves == [(1100, 2000), (1100, 2025), (1075, 2025), (1075, 2000)]
    # Each step waits for the last one, so quick clicks add up.
    assert stage.waits == 4

    # Moves made elsewhere (e.g. the notebook) show up on the next poll.
    stage.xy = (-34118.34, 12000.0)
    panel.update_position()
    assert panel._position.text() == "x -34118.3   y 12000.0"


def test_stage_errors_are_reported_not_raised(capsys):
    server = widgets.StageDPadServer(make_ctrl([], stage=BrokenStage()))
    # Every get must reply: the viewer waits for one.
    assert server.get_xy() == {"x": None, "y": None}
    # A raising route would end the session, so a failed move is only printed.
    server.move(100, 0)
    assert "Couldn't move the stage" in capsys.readouterr().out


class FlakyDriver(FakeDriver):
    """Drops the first `fails` commands, like a lost connection."""

    def __init__(self, fails):
        super().__init__()
        self.fails = fails

    def __setitem__(self, idx, state):
        if self.fails:
            self.fails -= 1
            raise ConnectionError("Modbus timeout")
        super().__setitem__(idx, state)


def timer_chip():
    drv = FakeDriver()
    drv._num_valves = 8
    return Chip("d1", drv, {"inlet": 0, "neckL": 1, "out": [2, 3], "waste7": 4}), drv


def timer_panel(chip):
    ws, routes = widgets.init_widgets(make_ctrl([chip]))
    return ws[widgets.OUTLET_TIMER_WIDGET](FakeRelay(routes))


def test_outlet_timer_never_offers_or_closes_a_neck(capsys):
    chip, drv = timer_chip()
    server = widgets.OutletTimerServer([chip])
    assert server.get_valves() == [("d1", "inlet"), ("d1", "out"), ("d1", "waste7")]
    assert server.close("d1", "neckL") == {"ok": False, "error": "necks are left alone"}
    # Not even "closed" was sent.
    assert 1 not in drv.state


def test_outlet_timer_closes_the_valve_when_time_is_up(qapp):
    chip, drv = timer_chip()
    chip.out = "open"
    panel = timer_panel(chip)
    # Nothing is picked to start with, so there's no default to miss.
    assert panel._valve.currentText() == "pick a valve" and panel._state.text() == "-"
    panel._btn.click()
    assert panel._deadline is None and panel._status.text() == "pick the valve to close first"

    panel._valve.setCurrentText("d1 · out")
    assert panel._state.text() == "open"
    panel._minutes.setValue(1)
    panel._btn.click()
    assert panel._btn.text() == "cancel" and not panel._valve.isEnabled()
    panel._deadline -= 30
    panel.tick()
    assert panel._countdown.text() == "0:30" and chip.valves["out"] == "open"

    panel._deadline -= 30
    panel.tick()
    assert panel._btn.text() == "start" and panel._status.text().startswith("closed d1 · out at")
    panel.update_state()
    assert panel._state.text() == "closed"
    # Only the outlet was touched.
    assert drv.state == {2: "closed", 3: "closed"}


def test_outlet_timer_will_not_start_on_a_closed_valve(qapp):
    # The mix-up of 11/13: d2 flowing, d1's outlet (already closed) picked.
    d1, d1_drv = timer_chip()
    d2_drv = FakeDriver()
    d2 = Chip("d2", d2_drv, {"out": 0})
    d2.out = "open"
    ws, routes = widgets.init_widgets(make_ctrl([d1, d2]))
    panel = ws[widgets.OUTLET_TIMER_WIDGET](FakeRelay(routes))

    panel._valve.setCurrentText("d1 · out")
    assert panel._state.text() == "closed"
    panel._btn.click()
    assert panel._deadline is None and panel._btn.text() == "start"
    assert panel._status.text() == "d1 · out is already closed; not started"

    panel._valve.setCurrentText("d2 · out")
    assert panel._state.text() == "open"
    panel._btn.click()
    assert panel._deadline is not None
    assert d1_drv.state == {} and d2_drv.state == {0: "open"}  # starting moves nothing


def test_outlet_timer_cancel_leaves_the_valve_alone(qapp):
    chip, drv = timer_chip()
    chip.out = "open"
    panel = timer_panel(chip)
    panel._valve.setCurrentText("d1 · out")
    panel._btn.click()
    panel._btn.click()
    assert panel._deadline is None and panel._btn.text() == "start"
    assert panel._status.text() == "cancelled; d1 · out left as it was"
    assert drv.state == {2: "open", 3: "open"}


def test_outlet_timer_retries_a_failed_close_until_it_works(qapp, capsys):
    drv = FlakyDriver(fails=1)
    drv.state[0] = "open"
    panel = timer_panel(Chip("d1", drv, {"out": 0}))
    panel._valve.setCurrentText("d1 · out")
    panel._btn.click()
    panel._deadline -= 10 * 60
    panel.tick()
    assert panel._status.text() == "couldn't close d1 · out; retrying"
    assert panel._status.toolTip() == "Modbus timeout" and panel._btn.text() == "cancel"

    panel._deadline -= widgets.OutletTimer._RETRY_S
    panel.tick()
    assert drv.state == {0: "closed"} and panel._status.text().startswith("closed d1 · out")
    out = capsys.readouterr().out
    assert "couldn't close d1 out: Modbus timeout" in out and "closed d1 out" in out


def test_outlet_timer_tells_the_time_on_a_12_hour_clock(qapp):
    t = datetime.datetime
    assert widgets._civilian_time(t(2026, 11, 15, 15, 5)) == "3:05 PM"
    assert widgets._civilian_time(t(2026, 11, 15, 0, 30)) == "12:30 AM"
    assert widgets._civilian_time(t(2026, 11, 15, 12, 0)) == "12:00 PM"
    assert widgets._civilian_time(t(2026, 11, 15, 9, 7, 3), seconds=True) == "9:07:03 AM"

    chip, _ = timer_chip()
    chip.out = "open"
    panel = timer_panel(chip)
    panel._valve.setCurrentText("d1 · out")
    panel._btn.click()
    assert re.fullmatch(r"closes d1 · out at 1?\d:\d\d [AP]M( tomorrow)?", panel._status.text())
    panel._btn.click()


def test_outlet_timer_fits_fonts_set_after_it_is_made(qapp):
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QDockWidget, QMainWindow

    panel = timer_panel(timer_chip()[0])
    win = QMainWindow()
    dock = QDockWidget(widgets.OUTLET_TIMER_WIDGET)
    dock.setWidget(panel)
    win.addDockWidget(Qt.RightDockWidgetArea, dock)
    # As napari does, the stylesheet comes after the panel is made; here a bigger font, as on a
    # scaled screen. A height fixed when it was made cut off the countdown.
    win.setStyleSheet("QWidget { font-size: 20pt; }")
    win.resize(600, 900)
    win.show()
    qapp.processEvents()
    assert panel.height() == panel.sizeHint().height()  # neither squeezed nor stretched
    win.close()


D2 = dict(rows=4, cols=6, top_left=(1000.0, 2000.0), bot_right=(1500.0, 2300.0))


def test_chambers_lie_evenly_between_the_corner_chambers():
    array = widgets.ChipArray(**D2)
    assert array.xy(0, 0) == (1000, 2000) and array.xy(3, 5) == (1500, 2300)
    assert array.xy(1, 2) == (1200, 2100)
    assert array.chamber(1200, 2100) == pytest.approx((1, 2))
    assert array.aspect == pytest.approx(1.5)

    # A chip rotated on the stage: the third corner tilts the rows and columns.
    tilted = widgets.ChipArray(
        rows=4, cols=6, top_left=(0, 0), top_right=(500, 10), bot_right=(490, 310)
    )
    assert tilted.xy(0, 5) == pytest.approx((500, 10))
    assert tilted.xy(3, 0) == pytest.approx((-10, 300))
    assert tilted.chamber(*tilted.xy(2, 4)) == pytest.approx((2, 4))


def test_a_bad_chip_map_is_left_off_without_stopping_the_session(capsys):
    ctrl = make_ctrl([], stage=FakeStage())
    ctrl.chip_maps = {
        "d2": D2,
        "d1": dict(D2, rows=0),
        "d3": dict(D2, top_left=(1, 2, 3)),
        "d4": dict(D2, row=4),
    }
    ws, routes = widgets.init_widgets(ctrl)
    assert list(ws) == [widgets.STAGE_WIDGET, widgets.CHIP_MAP_WIDGET]
    assert list(routes["widget/chip_map/arrays"]()) == ["d2"]
    out = capsys.readouterr().out
    assert "Chip map 'd1' left out: rows must be a whole number" in out
    assert "Chip map 'd3' left out: top_left must be a stage (x, y)" in out
    assert "Chip map 'd4' left out" in out

    ctrl.chip_maps = [D2]
    assert widgets.CHIP_MAP_WIDGET not in widgets.init_widgets(ctrl)[0]
    assert "c.chip_maps should be a dict" in capsys.readouterr().out
    # Nothing to drive without a stage.
    no_stage = make_ctrl([])
    no_stage.chip_maps = {"d2": D2}
    assert widgets.init_widgets(no_stage)[0] == {}


def map_panel(stage, maps=None):
    ctrl = make_ctrl([], stage=stage)
    ctrl.chip_maps = maps or {"d2": D2}
    ws, routes = widgets.init_widgets(ctrl)
    panel = ws[widgets.CHIP_MAP_WIDGET](FakeRelay(routes))
    panel._grid.resize(240, 160)
    return panel


def mouse(grid, kind, row=0, col=0):
    """A mouse event over the chamber at (row, col)."""
    from qtpy.QtCore import QEvent, Qt
    from qtpy.QtGui import QMouseEvent
    from qtpy.QtWidgets import QApplication

    button = Qt.LeftButton if kind == "press" else Qt.NoButton
    kind = {"move": QEvent.MouseMove, "press": QEvent.MouseButtonPress}[kind]
    event = QMouseEvent(kind, grid.center(row, col), button, button, Qt.NoModifier)
    QApplication.sendEvent(grid, event)


def test_clicking_a_chamber_centers_the_stage_on_it(qapp):
    from qtpy.QtCore import QEvent
    from qtpy.QtWidgets import QApplication

    stage = FakeStage((1000.0, 2000.0))
    panel = map_panel(stage)
    assert panel._status.text() == "stage at row 1, col 1"

    mouse(panel._grid, "move", 1, 2)
    assert panel._status.text() == "row 2, col 3"
    mouse(panel._grid, "press", 1, 2)
    assert stage.moves == [(1200, 2100)] and stage.waits == 1
    assert panel._status.text() == "moving to row 2, col 3"
    mouse(panel._grid, "press", 3, 5)
    assert stage.moves[-1] == (1500, 2300)

    # Once the cursor leaves, the next poll shows where the stage is.
    QApplication.sendEvent(panel._grid, QEvent(QEvent.Leave))
    panel.update_position()
    assert panel._status.text() == "stage at row 4, col 6"
    stage.xy = (5000.0, 2000.0)
    panel.update_position()
    assert panel._status.text() == "stage off the array"


def test_each_chip_has_its_own_map(qapp):
    stage = FakeStage((0.0, 0.0))
    d1 = dict(rows=2, cols=2, top_left=(0, 0), bot_right=(100, 100))
    panel = map_panel(stage, {"d1": d1, "d2": D2})
    assert panel._status.text() == "stage at row 1, col 1"
    panel._chip.setCurrentText("d2")
    assert panel._status.text() == "stage off the array"
    mouse(panel._grid, "press", 0, 0)
    assert stage.moves == [(1000, 2000)]


def test_the_chip_map_only_drives_to_chambers(capsys):
    stage = FakeStage((1000.0, 2000.0))
    server = widgets.ChipMapServer(make_ctrl([], stage=stage), {"d2": widgets.ChipArray(**D2)})
    server.go("d2", 4, 0)
    server.go("d2", 0, -1)
    server.go("d1", 0, 0)
    assert stage.moves == []
    out = capsys.readouterr().out
    assert out.count("Couldn't move the stage") == 3 and "d2 has no chamber there" in out
    assert server.where("d2") == pytest.approx({"row": 0, "col": 0})

    # Every get must reply, and a raising route would end the session.
    broken = widgets.ChipMapServer(
        make_ctrl([], stage=BrokenStage()), {"d2": widgets.ChipArray(**D2)}
    )
    assert broken.where("d2") == {"row": None, "col": None}
    broken.go("d2", 0, 0)
    assert "Couldn't move the stage to d2 row 1, col 1" in capsys.readouterr().out


def test_the_chip_map_polls_the_stage_only_while_shown(qapp):
    panel = map_panel(FakeStage())
    assert not panel._timer.isActive()
    panel.show()
    assert panel._timer.isActive()
    panel.hide()
    assert not panel._timer.isActive()


class FakeWindow:
    def __init__(self):
        self.docks = {}

    def add_dock_widget(self, widget, name, tabify=False, area="left"):
        self.docks[name] = widget

    def remove_dock_widget(self, widget):
        self.docks = {n: w for n, w in self.docks.items() if w is not widget}


class FakeViewer:
    def __init__(self):
        self.window, self.layers = FakeWindow(), []

    def add_image(self, img, name, **kwargs):
        self.layers.append(name)


def test_the_chip_map_is_gone_before_imaging_starts(qapp):
    from prismo import views

    stage = FakeStage((1000.0, 2000.0))
    ctrl = make_ctrl([], stage=stage)
    ctrl.chip_maps = {"d2": D2}
    ws, routes = widgets.init_widgets(ctrl)
    started = []
    routes.update(
        img=lambda: np.zeros((4, 4), np.uint16),
        xy=lambda: stage.xy,
        arrays=lambda: set(),
        start_acq=lambda *corners: started.append(corners),
    )
    relay = FakeRelay(routes)

    # Protocols (corners given) never show it.
    viewer = FakeViewer()
    view = views.AcquisitionView(viewer, relay, "k.zarr", ws)
    assert widgets.STAGE_WIDGET in viewer.window.docks
    assert widgets.CHIP_MAP_WIDGET not in viewer.window.docks
    view._refresh_timer.stop()

    # Corners picked by hand: it's there while they're picked, and gone once imaging starts.
    viewer = FakeViewer()
    view = views.AcquisitionView(viewer, relay, "k.zarr", ws, tiled=True)
    assert widgets.CHIP_MAP_WIDGET in viewer.window.docks
    view.start_acq((0, 0), (10, 10))
    view._refresh_timer.stop()
    assert widgets.CHIP_MAP_WIDGET not in viewer.window.docks and started == [((0, 0), (10, 10))]
