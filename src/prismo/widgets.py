import datetime
import functools
import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from qtpy.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from qtpy.QtGui import QBrush, QColor, QDoubleValidator, QPainter, QPen
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import devices
from .control import Control
from .session import Relay

# Dock names of the panels that views give their own panel instead of a tab.
FILTER_SHUTTER_WIDGET = "filters & shutters"
STAGE_WIDGET = "stage"
CHIP_MAP_WIDGET = "chip map"
OUTLET_TIMER_WIDGET = "outlet timer"
OWN_PANEL_WIDGETS = (FILTER_SHUTTER_WIDGET, STAGE_WIDGET, CHIP_MAP_WIDGET, OUTLET_TIMER_WIDGET)
# Panels that views dock on the right, so the left column still fits on the screen.
RIGHT_PANEL_WIDGETS = (STAGE_WIDGET, CHIP_MAP_WIDGET, OUTLET_TIMER_WIDGET)
_FILTERS = (devices.ti.Filter, devices.ti2.Filter, devices.sutter.Filter, devices.demo.Filter)
_SHUTTERS = (devices.ti2.Shutter, devices.sutter.Shutter)


def init_widgets(
    ctrl: "Control",
) -> tuple[dict[str, Callable[["Relay"], QWidget]], dict[str, Callable[..., Any]]]:
    widgets: dict[str, Callable[[Relay], QWidget]] = {}
    routes: dict[str, Callable[..., Any]] = {}

    optics = [d for d in ctrl.devices if isinstance(d, _FILTERS + _SHUTTERS)]
    if optics or getattr(ctrl, "camera", None) is not None:
        path = "widget/filters_shutters"
        widgets[FILTER_SHUTTER_WIDGET] = lambda r, path=path: FilterShutterController(
            r.subpath(path)
        )
        server = FilterShutterControllerServer(optics, ctrl)
        routes = {**routes, **server.routes(path)}

    if getattr(ctrl, "stage", None) is not None:
        path = "widget/stage_dpad"
        widgets[STAGE_WIDGET] = lambda r, path=path: StageDPad(r.subpath(path))
        server = StageDPadServer(ctrl)
        routes = {**routes, **server.routes(path)}

        arrays = _chip_arrays(ctrl)
        if arrays:
            path = "widget/chip_map"
            widgets[CHIP_MAP_WIDGET] = lambda r, path=path: ChipMap(r.subpath(path))
            server = ChipMapServer(ctrl, arrays)
            routes = {**routes, **server.routes(path)}

    chips = [d for d in ctrl.devices if isinstance(d, devices.microfluidic.Chip)]
    if chips:
        path = "widget/outlet_timer"
        widgets[OUTLET_TIMER_WIDGET] = lambda r, path=path: OutletTimer(r.subpath(path))
        server = OutletTimerServer(chips)
        routes = {**routes, **server.routes(path)}

    for device in ctrl.devices:
        if isinstance(device, devices.Valved):
            path = f"widget/{device.name}"
            # We need to set dummy default arguments so their values get captured by the lambda.
            widgets[f"{device.name} controller"] = lambda r, path=path, title=device.name: (
                ValveController(r.subpath(path), title=title)
            )
            server = ValveControllerServer(device)
            routes = {**routes, **server.routes(path)}

    # Only devices with a discrete list of states can be shown as a dropdown.
    state_devices = [
        d for d in ctrl.devices if isinstance(d, devices.State) and hasattr(d, "states")
    ]
    if state_devices:
        path = "widget/states"
        widgets["state controller"] = lambda r, path=path: StateController(r.subpath(path))
        server = StateControllerServer(state_devices)
        routes = {**routes, **server.routes(path)}

    if ctrl.stage is not None:
        path = "widget/stage"
        widgets["stage controller"] = lambda r, path=path: StageController(r.subpath(path))
        server = StageControllerServer(ctrl.stage)
        routes = {**routes, **server.routes(path)}

    return widgets, routes


class BoundarySelector(QWidget):
    def __init__(
        self,
        relay: "Relay",
        next_step: Callable[[tuple[float, float], tuple[float, float]], Any],
    ):
        super().__init__()
        self.setMaximumHeight(150)
        layout = QGridLayout(self)

        self.left_x = QLineEdit()
        self.left_x.setValidator(QDoubleValidator())
        self.left_y = QLineEdit()
        self.left_y.setValidator(QDoubleValidator())
        self.left_btn = QPushButton("Set")
        self.left_btn.setMinimumWidth(50)

        self.right_x = QLineEdit()
        self.right_x.setValidator(QDoubleValidator())
        self.right_y = QLineEdit()
        self.right_y.setValidator(QDoubleValidator())
        self.right_btn = QPushButton("Set")
        self.right_btn.setMinimumWidth(50)

        continue_btn = QPushButton("Continue")

        layout.addWidget(QLabel("x"), 0, 1, alignment=Qt.AlignHCenter)
        layout.addWidget(QLabel("y"), 0, 2, alignment=Qt.AlignHCenter)
        layout.addWidget(QLabel("Top Left"), 1, 0)
        layout.addWidget(QLabel("Bottom Right"), 2, 0)
        layout.addWidget(self.left_x, 1, 1)
        layout.addWidget(self.left_y, 1, 2)
        layout.addWidget(self.left_btn, 1, 3)

        layout.addWidget(self.right_x, 2, 1)
        layout.addWidget(self.right_y, 2, 2)
        layout.addWidget(self.right_btn, 2, 3)

        layout.addWidget(continue_btn, 3, 0)

        layout.setColumnMinimumWidth(3, 60)
        layout.setHorizontalSpacing(10)

        self.left_btn.clicked.connect(self.set_left)
        self.right_btn.clicked.connect(self.set_right)
        continue_btn.clicked.connect(self.next_step)

        self._relay = relay
        self._next_step = next_step

    def set_left(self):
        xy = self._relay.get("xy")
        self.left_x.setText(f"{xy[0]:.2f}")
        self.left_y.setText(f"{xy[1]:.2f}")

    def set_right(self):
        xy = self._relay.get("xy")
        self.right_x.setText(f"{xy[0]:.2f}")
        self.right_y.setText(f"{xy[1]:.2f}")

    def next_step(self):
        if (
            self.left_x.text()
            and self.left_y.text()
            and self.right_x.text()
            and self.right_y.text()
        ):
            self.close()
            self._next_step(
                (float(self.left_x.text()), float(self.left_y.text())),
                (float(self.right_x.text()), float(self.right_y.text())),
            )
        else:
            for w in [self.left_x, self.left_y, self.right_x, self.right_y]:
                if not w.text():
                    w.setStyleSheet("border: 1px solid red;")
                else:
                    w.setStyleSheet("border: 0px;")


class PositionSelector(QWidget):
    def __init__(self, relay: "Relay", next_step: Callable[[list[tuple[float, float]]], Any]):
        super().__init__()
        layout = QVBoxLayout(self)
        self.setMaximumHeight(150)

        self.rows = QVBoxLayout()
        layout.addLayout(self.rows)
        btns = QHBoxLayout()
        add_btn = QPushButton("Add")
        continue_btn = QPushButton("Continue")
        btns.addWidget(add_btn)
        btns.addWidget(continue_btn)
        layout.addLayout(btns)

        self.add_row()
        add_btn.clicked.connect(self.add_row)
        continue_btn.clicked.connect(self.next_step)

        self._relay = relay
        self._next_step = next_step

    def add_row(self):
        row = QHBoxLayout()
        self.rows.addLayout(row)
        x = QLineEdit()
        x.setValidator(QDoubleValidator())
        y = QLineEdit()
        y.setValidator(QDoubleValidator())
        set_btn = QPushButton("Set")
        set_btn.setMinimumWidth(50)
        delete_btn = QPushButton("Rem")
        delete_btn.setMinimumWidth(50)

        row.addWidget(x)
        row.addWidget(y)
        row.addWidget(set_btn)
        row.addWidget(delete_btn)

        delete_btn.clicked.connect(lambda: self.delete(row))
        set_btn.clicked.connect(lambda: self.set(row))

    def set(self, row: QHBoxLayout):
        x = row.itemAt(0).widget()
        y = row.itemAt(1).widget()
        xy = self._relay.get("xy")
        x.setText(f"{xy[0]:.2f}")
        y.setText(f"{xy[1]:.2f}")

    def delete(self, row: QHBoxLayout):
        self.rows.removeItem(row)

    def next_step(self):
        valid = True
        xys = []

        for i in range(self.rows.count()):
            row = self.rows.itemAt(i).layout()
            x = row.itemAt(0).widget()
            y = row.itemAt(1).widget()
            for w in [x, y]:
                if not w.text():
                    valid = False
                    w.setStyleSheet("border: 1px solid red;")
                else:
                    w.setStyleSheet("border: 0px;")

            if x.text() and y.text():
                xys.append((float(x.text()), float(y.text())))

        if valid:
            self.close()
            self._next_step(xys)


class StateController(QWidget):
    def __init__(self, relay: "Relay"):
        super().__init__()
        self._relay = relay
        self._states: dict[str, str | int | float] = self._relay.get("states")
        self._options: dict[str, list[str | int | float]] = self._relay.get("options")
        self._btns: dict[str, dict[str, QPushButton]] = {}
        self._shown: dict[str, str] = {}
        grid = QGridLayout(self)
        grid.setHorizontalSpacing(0)
        grid.setVerticalSpacing(0)
        self._timer = QTimer()
        self._timer.timeout.connect(self.update_states)
        self._timer.start(10)

        # One row per device: its name, then a button for each state.
        for row, name in enumerate(self._states):
            grid.addWidget(QLabel(name), row, 0)
            self._btns[name] = {}
            for col, opt in enumerate(self._options[name], start=1):
                btn = QPushButton(str(opt))
                btn.setMinimumWidth(10)
                btn.clicked.connect(functools.partial(self.set_state, name, opt))
                grid.addWidget(btn, row, col)
                self._btns[name][str(opt)] = btn
        self.update_states()

    def update_states(self):
        self._states = self._relay.get("states")
        for name, state in self._states.items():
            # Only restyle when the state changes, since this runs every 10ms.
            if self._shown.get(name) == str(state):
                continue
            self._shown[name] = str(state)
            for opt, btn in self._btns[name].items():
                # Highlight the current state with the same style as a closed valve.
                btn.setStyleSheet(
                    ValveController.button_stylesheet("closed" if opt == str(state) else "open")
                )

    def set_state(self, name: str, state: str | int | float):
        self._relay.post("set_state", name, state)


class StateControllerServer:
    def __init__(self, state_devices: list[devices.State]):
        self._devices = {d.name: d for d in state_devices}

    def routes(self, path: str) -> dict[str, Callable[..., Any]]:
        return {
            path + "/states": self.get_states,
            path + "/options": self.get_options,
            path + "/set_state": self.set_state,
        }

    def get_states(self) -> dict[str, str | int | float]:
        return {name: d.state for name, d in self._devices.items()}

    def get_options(self) -> dict[str, list[str | int | float]]:
        return {name: d.states for name, d in self._devices.items()}

    def set_state(self, name: str, state: str | int | float):
        self._devices[name].state = state


class StageController(QWidget):
    _PAD_R = 50
    _THUMB_R = 16
    _MAX_SPEED = 5.7459

    def __init__(self, relay: "Relay"):
        super().__init__()
        self._relay = relay
        self._thumb = QPointF(0.0, 0.0)
        self._last_posted = QPointF(0.0, 0.0)
        self._dragging = False
        pad_size = (self._PAD_R + self._THUMB_R + 4) * 2
        self.setFixedSize(pad_size, pad_size)
        self._timer = QTimer()
        self._timer.timeout.connect(self._poll)
        self._timer.start(100)

    def _poll(self):
        if self._thumb != self._last_posted:
            self._last_posted = QPointF(self._thumb.x(), self._thumb.y())
            self._relay.post(
                "set_xy_speed",
                self._thumb.x() * self._MAX_SPEED,
                -self._thumb.y() * self._MAX_SPEED,
            )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self.width() / 2, self.height() / 2
        painter.setPen(QPen(QColor(80, 80, 80), 2))
        painter.setBrush(QBrush(QColor(40, 40, 40)))
        painter.drawEllipse(QPointF(cx, cy), self._PAD_R, self._PAD_R)
        tx = cx + self._thumb.x() * self._PAD_R
        ty = cy + self._thumb.y() * self._PAD_R
        color = QColor(220, 220, 220) if self._dragging else QColor(130, 130, 130)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(tx, ty), self._THUMB_R, self._THUMB_R)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._move_thumb(event.pos())

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._move_thumb(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._thumb = QPointF(0.0, 0.0)
            self._last_posted = QPointF(0.0, 0.0)
            self._relay.post("set_xy_speed", 0.0, 0.0)
            self.update()

    def _move_thumb(self, pos):
        cx, cy = self.width() / 2, self.height() / 2
        dx = (pos.x() - cx) / self._PAD_R
        dy = (pos.y() - cy) / self._PAD_R
        mag = (dx**2 + dy**2) ** 0.5
        if mag > 1.0:
            dx /= mag
            dy /= mag
        self._thumb = QPointF(dx, dy)
        self.update()


class StageControllerServer:
    def __init__(self, stage: devices.Stage):
        self._stage = stage

    def routes(self, path: str) -> dict[str, Callable[..., Any]]:
        return {
            path + "/set_xy_speed": self.set_xy_speed,
        }

    def set_xy_speed(self, vx: float, vy: float):
        try:
            self._stage.set_xy_speed(vx, vy)
        except Exception:
            pass


class ValveController(QWidget):
    def __init__(self, relay: "Relay", title: str | None = None):
        super().__init__()
        self._relay = relay
        self._valves: dict[str, str | int] = self._relay.get("valves")
        self._valve_states: dict[str, list[str | int]] = self._relay.get("valve_states")
        self._valve_widgets: dict[str, QPushButton | QComboBox] = {}
        self._btns: list[QPushButton] = []
        self.setMaximumHeight(300)
        outer = QVBoxLayout(self)
        self._timer = QTimer()
        self._timer.timeout.connect(self.update_valves)
        self._timer.start(10)

        self._btn_grid = QGridLayout()
        self._btn_grid.setHorizontalSpacing(0)
        self._btn_grid.setVerticalSpacing(0)

        for k, v in self._valves.items():
            states = self._valve_states[k]
            if set(states) == {"open", "closed"}:
                btn = QPushButton(str(k))
                btn.setCheckable(True)
                btn.setChecked(v == "open")
                btn.setStyleSheet(self.button_stylesheet(v))
                btn.setMinimumWidth(10)
                btn.clicked.connect(functools.partial(self.toggle_valve, k))
                self._btns.append(btn)
                self._valve_widgets[k] = btn
            else:
                row = QHBoxLayout()
                row.addWidget(QLabel(str(k)))
                combo = QComboBox()
                for state in states:
                    combo.addItem(str(state))
                combo.setCurrentText(str(v))
                combo.currentTextChanged.connect(functools.partial(self.set_valve, k))
                row.addWidget(combo)
                outer.insertLayout(0, row)
                self._valve_widgets[k] = combo

        if title is not None:
            # Large and bold so devices (e.g. d1 and d2) are easy to tell apart.
            self._title = QLabel(title)
            self._title.setStyleSheet("font-size: 16px; font-weight: bold; color: #f0f0f0;")
            outer.insertWidget(0, self._title)

        outer.addLayout(self._btn_grid)
        self._relayout_buttons()

    def update_valves(self):
        self._valves = self._relay.get("valves")
        for k, v in self._valves.items():
            widget = self._valve_widgets[k]
            if isinstance(widget, QPushButton):
                widget.setStyleSheet(self.button_stylesheet(v))
            else:
                widget.blockSignals(True)
                widget.setCurrentText(str(v))
                widget.blockSignals(False)

    def _relayout_buttons(self):
        if not self._btns:
            return
        btn_width = max(btn.sizeHint().width() for btn in self._btns)
        btn_width = max(btn_width, 40)
        cols = max(1, self.width() // btn_width)
        while self._btn_grid.count():
            self._btn_grid.takeAt(0)
        for i, btn in enumerate(self._btns):
            self._btn_grid.addWidget(btn, i // cols, i % cols)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout_buttons()

    def toggle_valve(self, key: str, checked: bool):
        state: str | int = "open" if checked else "closed"
        self._valves[key] = state
        self._relay.post("set_valve", key, state)
        self._valve_widgets[key].setStyleSheet(self.button_stylesheet(state))

    def set_valve(self, key: str, state_str: str):
        states = self._valve_states[key]
        state: str | int = next((s for s in states if str(s) == state_str), state_str)
        self._valves[key] = state
        self._relay.post("set_valve", key, state)

    @staticmethod
    def button_stylesheet(state: str | int) -> str:
        if state == "closed":
            bg = "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1e3045, stop:1 #2a4060)"
            bg_hover = "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #243850, stop:1 #305070)"
            border_tl, border_br = "#162335", "#3d6080"
        else:
            bg = "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a4f5c, stop:1 #30343c)"
            bg_hover = "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #545966, stop:1 #3a3e47)"
            border_tl, border_br = "#5a606e", "#1a1d22"
        return (
            f"QPushButton {{"
            f" color: #f0f0f0;"
            f" background-color: {bg};"
            f" border-style: solid; border-width: 1px; border-radius: 3px;"
            f" border-top-color: {border_tl}; border-left-color: {border_tl};"
            f" border-right-color: {border_br}; border-bottom-color: {border_br};"
            f" padding: 3px 1px; margin: 1px;"
            f"}}"
            f"QPushButton:hover {{ background-color: {bg_hover}; }}"
        )


class ValveControllerServer:
    def __init__(self, valves: devices.Valved):
        self._valves = valves

    def routes(self, path: str) -> dict[str, Callable[..., Any]]:
        return {
            path + "/valves": self.get_valves,
            path + "/valve_states": self.get_valve_states,
            path + "/set_valve": self.set_valve,
        }

    def get_valves(self) -> dict[str, str | int]:
        return self._valves.valves

    def get_valve_states(self) -> dict[str, list[str | int]]:
        return self._valves.valve_states

    def set_valve(self, key: str, state: str | int):
        self._valves[key] = state


class FilterShutterController(QWidget):
    _COLS = 2

    def __init__(self, relay: "Relay"):
        super().__init__()
        self._relay = relay
        self._options: dict[str, list[str | int]] = self._relay.get("options")
        self._states: dict[str, str | int | None] = self._relay.get("states")
        # Shutters get one open/closed toggle; filters get one button per cube.
        self._toggles: dict[str, QPushButton] = {}
        self._choices: dict[str, dict[str | int, QPushButton]] = {}
        layout = QGridLayout(self)
        layout.setHorizontalSpacing(0)
        layout.setVerticalSpacing(0)

        row = 0
        # Exposure of the current camera, if there is one.
        self._exposure_label = QLabel()
        self._exposure: QDoubleSpinBox | None = None
        exposure = self._relay.get("exposure")
        if exposure["camera"] is not None:
            self._exposure = QDoubleSpinBox()
            self._exposure.setRange(0.1, 60000)
            self._exposure.setDecimals(1)
            self._exposure.setSingleStep(10)
            self._exposure.setSuffix(" ms")
            # Set on Enter, an arrow click or leaving the box, not on every keystroke.
            self._exposure.setKeyboardTracking(False)
            self._exposure.valueChanged.connect(self.set_exposure)
            layout.addWidget(self._exposure_label, row, 0)
            layout.addWidget(self._exposure, row, 1, 1, self._COLS)
            self._show_exposure(exposure)
            row += 1

        for name, options in self._options.items():
            layout.addWidget(QLabel(name), row, 0)
            if set(options) == {"open", "closed"}:
                btn = QPushButton()
                btn.setMinimumWidth(10)
                btn.clicked.connect(functools.partial(self.toggle, name))
                layout.addWidget(btn, row, 1, 1, self._COLS)
                self._toggles[name] = btn
                row += 1
            else:
                self._choices[name] = {}
                for i, opt in enumerate(options):
                    btn = QPushButton(str(opt))
                    btn.setMinimumWidth(10)
                    btn.clicked.connect(functools.partial(self.set_state, name, opt))
                    layout.addWidget(btn, row + i // self._COLS, 1 + i % self._COLS)
                    self._choices[name][opt] = btn
                row += -(-len(options) // self._COLS)
        for col in range(1, self._COLS + 1):
            layout.setColumnStretch(col, 1)

        self._restyle()
        self.setMaximumHeight(self.sizeHint().height())
        # Poll slowly: these are serial/Ti2 reads that share the core with acquisitions.
        self._timer = QTimer()
        self._timer.timeout.connect(self.update_states)
        self._timer.start(500)

    def update_states(self):
        states = self._relay.get("states")
        if states != self._states:
            self._states = states
            self._restyle()
        if self._exposure is not None:
            self._show_exposure(self._relay.get("exposure"))

    def _show_exposure(self, exposure: dict[str, Any]):
        self._exposure_label.setText(f"exposure ({exposure['camera']})")
        # Don't overwrite a value that is being typed.
        if exposure["ms"] is not None and not self._exposure.hasFocus():
            self._exposure.blockSignals(True)
            self._exposure.setValue(exposure["ms"])
            self._exposure.blockSignals(False)

    def set_exposure(self, ms: float):
        self._relay.post("set_exposure", ms)

    def _restyle(self):
        for name, btn in self._toggles.items():
            state = self._states[name]
            btn.setText("?" if state is None else str(state))
            btn.setStyleSheet(self.button_stylesheet(state == "open"))
        for name, btns in self._choices.items():
            for opt, btn in btns.items():
                btn.setStyleSheet(self.button_stylesheet(self._states[name] == opt))

    def toggle(self, name: str, _checked: bool = False):
        self.set_state(name, "closed" if self._states[name] == "open" else "open")

    def set_state(self, name: str, state: str | int, _checked: bool = False):
        self._states[name] = state
        self._relay.post("set_state", name, state)
        self._restyle()

    @staticmethod
    def button_stylesheet(on: bool) -> str:
        # Amber for an open shutter / the active cube, so it doesn't read as a valve state.
        if on:
            bg = "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #8a6d24, stop:1 #6b5218)"
            bg_hover = "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #957830, stop:1 #765c20)"
            border_tl, border_br = "#5a4510", "#a8883a"
        else:
            bg = "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a4f5c, stop:1 #30343c)"
            bg_hover = "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #545966, stop:1 #3a3e47)"
            border_tl, border_br = "#5a606e", "#1a1d22"
        return (
            f"QPushButton {{"
            f" color: #f0f0f0;"
            f" background-color: {bg};"
            f" border-style: solid; border-width: 1px; border-radius: 3px;"
            f" border-top-color: {border_tl}; border-left-color: {border_tl};"
            f" border-right-color: {border_br}; border-bottom-color: {border_br};"
            f" padding: 3px 1px; margin: 1px;"
            f"}}"
            f"QPushButton:hover {{ background-color: {bg_hover}; }}"
        )


class FilterShutterControllerServer:
    def __init__(self, optics: list[Any], ctrl: Any = None):
        self._devices = {d.name: d for d in optics}
        self._ctrl = ctrl

    def routes(self, path: str) -> dict[str, Callable[..., Any]]:
        return {
            path + "/states": self.get_states,
            path + "/options": self.get_options,
            path + "/set_state": self.set_state,
            path + "/exposure": self.get_exposure,
            path + "/set_exposure": self.set_exposure,
        }

    def get_exposure(self) -> dict[str, Any]:
        # Never None: the viewer waits for a reply to every get.
        camera = getattr(self._ctrl, "camera", None)
        if camera is None:
            return {"camera": None, "ms": None}
        try:
            return {"camera": camera.name, "ms": float(camera.exposure)}
        except Exception:
            return {"camera": camera.name, "ms": None}

    def set_exposure(self, ms: float):
        # A route that raises ends the whole session (acquisition included), so just report it.
        try:
            self._ctrl.exposure = ms
        except Exception as e:
            print(f"Couldn't set the exposure to {ms} ms: {e}")

    def get_states(self) -> dict[str, str | int | None]:
        return {name: self._read(d) for name, d in self._devices.items()}

    def get_options(self) -> dict[str, list[str | int]]:
        return {name: list(d.states) for name, d in self._devices.items()}

    def set_state(self, name: str, state: str | int):
        # A route that raises ends the whole session (acquisition included), so just report it.
        try:
            self._devices[name].state = state
        except Exception as e:
            print(f"Couldn't set {name} to {state!r}: {e}")

    @staticmethod
    def _read(device: Any) -> str | int | None:
        try:
            if isinstance(device, _FILTERS) and hasattr(device, "_core"):
                # Read through the core: ti2.Filter.state moves the turret to its first cube
                # when the read fails, and polling must never move anything.
                return device.states[device._core.getState(device.name)]
            return device.state
        except Exception:
            return None


class StageDPad(QWidget):
    # Arrows follow the image, the way tiled acquisitions lay out tiles: ► is +x, ▼ is +y.
    # Each maps to (dx, dy, grid row, grid col).
    _ARROWS = {
        "▲": (0, -1, 1, 1),
        "◄": (-1, 0, 2, 0),
        "►": (1, 0, 2, 2),
        "▼": (0, 1, 3, 1),
    }

    def __init__(self, relay: "Relay"):
        super().__init__()
        self._relay = relay
        layout = QGridLayout(self)
        layout.setHorizontalSpacing(0)
        layout.setVerticalSpacing(0)

        self._step = QDoubleSpinBox()
        self._step.setRange(1, 10000)
        self._step.setDecimals(0)
        self._step.setSingleStep(50)
        self._step.setValue(100)
        self._step.setSuffix(" µm")
        layout.addWidget(QLabel("step"), 0, 0)
        layout.addWidget(self._step, 0, 1, 1, 2)

        self._btns: dict[str, QPushButton] = {}
        for arrow, (dx, dy, row, col) in self._ARROWS.items():
            btn = QPushButton(arrow)
            btn.setMinimumWidth(10)
            btn.setStyleSheet(FilterShutterController.button_stylesheet(False))
            btn.clicked.connect(functools.partial(self.move, dx, dy))
            layout.addWidget(btn, row, col)
            self._btns[arrow] = btn

        self._position = QLabel()
        self._position.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self._position, 4, 0, 1, 3)
        for col in range(3):
            layout.setColumnStretch(col, 1)

        self._show_position(self._relay.get("xy"))
        self.setMaximumHeight(self.sizeHint().height())
        # Poll slowly: each read is a serial command on the port that acquisitions move through.
        self._timer = QTimer()
        self._timer.timeout.connect(self.update_position)
        self._timer.start(500)

    def update_position(self):
        self._show_position(self._relay.get("xy"))

    def _show_position(self, xy: dict[str, float | None]):
        if xy["x"] is None:
            self._position.setText("x ?   y ?")
        else:
            self._position.setText(f"x {xy['x']:.1f}   y {xy['y']:.1f}")

    def move(self, dx: int, dy: int, _checked: bool = False):
        step = self._step.value()
        self._relay.post("move", dx * step, dy * step)


class StageDPadServer:
    def __init__(self, ctrl: Any):
        self._ctrl = ctrl

    def routes(self, path: str) -> dict[str, Callable[..., Any]]:
        return {
            path + "/xy": self.get_xy,
            path + "/move": self.move,
        }

    def get_xy(self) -> dict[str, float | None]:
        # Never None: the viewer waits for a reply to every get.
        try:
            x, y = self._ctrl.stage.xy
            return {"x": float(x), "y": float(y)}
        except Exception:
            return {"x": None, "y": None}

    def move(self, dx: float, dy: float):
        # A route that raises ends the whole session (acquisition included), so just report it.
        try:
            stage = self._ctrl.stage
            # Let the last step finish first, so quick clicks add up instead of starting mid-move.
            if isinstance(stage, devices.Wait):
                stage.wait()
            x, y = stage.xy
            stage.xy = (x + dx, y + dy)
        except Exception as e:
            print(f"Couldn't move the stage by ({dx}, {dy}) µm: {e}")


class ChipArray:
    """Where each chamber of a chip's array is on the stage, from the stage xy of the centers of
    its corner chambers. Rows and columns count from 0 at `top_left`.

    With only `top_left` and `bot_right`, the rows are taken to run along the stage's x. Give
    `top_right` too for a chip that sits rotated on the stage: 1° across a 16 mm array is about
    one chamber."""

    def __init__(
        self,
        rows: int,
        cols: int,
        top_left: Any,
        bot_right: Any,
        top_right: Any = None,
    ):
        for n, label in ((rows, "rows"), (cols, "cols")):
            if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 1:
                raise ValueError(f"{label} must be a whole number of at least 1, not {n!r}")
        self.rows, self.cols = int(rows), int(cols)
        tl, br = _stage_xy(top_left, "top_left"), _stage_xy(bot_right, "bot_right")
        tr = np.array([br[0], tl[1]]) if top_right is None else _stage_xy(top_right, "top_right")
        self._origin = tl
        self._col_step = (tr - tl) / max(self.cols - 1, 1)
        self._row_step = (br - tr) / max(self.rows - 1, 1)

    def xy(self, row: int, col: int) -> tuple[float, float]:
        x, y = self._origin + col * self._col_step + row * self._row_step
        return float(x), float(y)

    def chamber(self, x: float, y: float) -> tuple[float, float] | None:
        """The (row, col) at stage (x, y), in fractions of a chamber; None if the corners don't
        span an area (one row or column)."""
        steps = np.column_stack([self._row_step, self._col_step])
        if abs(np.linalg.det(steps)) < 1e-9:
            return None
        row, col = np.linalg.solve(steps, np.array([x, y]) - self._origin)
        return float(row), float(col)

    @property
    def aspect(self) -> float:
        """Width over height of the array on the stage."""
        col_pitch, row_pitch = np.hypot(*self._col_step), np.hypot(*self._row_step)
        col_pitch, row_pitch = col_pitch or row_pitch or 1.0, row_pitch or col_pitch or 1.0
        return float(self.cols * col_pitch / (self.rows * row_pitch))


class ChipMap(QWidget):
    """A chip's chamber array: click a chamber to center the stage on it.

    Set each chip's array once after ``pm.load``, with the stage xy of the centers of its corner
    chambers (and ``top_right`` if the chip sits rotated; see ChipArray)::

        c.chip_maps = {
            "d2": dict(rows=32, cols=56, top_left=(45948.2, -406.5), bot_right=(62491.6, 16779.8))
        }

    Chambers are named from row 1, col 1 at the top left. Hovering names the chamber under the
    cursor; the amber ring is where the stage is. Views show it in live views and while positions
    are picked for an acquisition, never while one is imaging."""

    def __init__(self, relay: "Relay"):
        super().__init__()
        self._relay = relay
        self._arrays: dict[str, dict[str, float]] = self._relay.get("arrays")
        layout = QVBoxLayout(self)
        layout.setSpacing(2)

        self._chip = QComboBox()
        self._chip.addItems(list(self._arrays))
        self._chip.currentTextChanged.connect(self.pick)
        layout.addWidget(self._chip)

        self._grid = ChamberGrid(self._show_status, self.go)
        layout.addWidget(self._grid)

        self._status = QLabel()
        self._status.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self._status)
        # As tall as the map is at this width, never stretched with blank space above and below.
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        self._name = ""
        self.pick(self._chip.currentText())
        # Poll slowly, and only while shown: each read is a serial command on the stage's port.
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update_position)

    def showEvent(self, event: Any):
        super().showEvent(event)
        self.update_position()
        self._timer.start(1000)

    def hideEvent(self, event: Any):
        super().hideEvent(event)
        self._timer.stop()

    def pick(self, name: str):
        self._name = name
        array = self._arrays[name]
        self._grid.set_array(int(array["rows"]), int(array["cols"]), array["aspect"])
        self.update_position()

    def update_position(self):
        where = self._relay.get("where", self._name)
        self._grid.set_stage(where["row"], where["col"])
        self._show_status()

    def go(self, row: int, col: int):
        self._relay.post("go", self._name, row, col)
        self._status.setText(f"moving to {_chamber_name(row, col)}")

    def _show_status(self):
        if self._grid.hovered is not None:
            self._status.setText(_chamber_name(*self._grid.hovered))
        elif self._grid.stage_chamber is not None:
            self._status.setText(f"stage at {_chamber_name(*self._grid.stage_chamber)}")
        elif self._grid.stage is None:
            self._status.setText("stage ?")
        else:
            self._status.setText("stage off the array")


class ChamberGrid(QWidget):
    """The chambers as dots, laid out like the image: row 1 at the top, col 1 at the left."""

    _DOT, _HOVER, _STAGE = QColor("#80869a"), QColor("#f0f0f0"), QColor("#e0b050")

    def __init__(
        self,
        on_hover: Callable[[], Any],
        on_click: Callable[[int, int], Any],
    ):
        super().__init__()
        self._on_hover, self._on_click = on_hover, on_click
        self._rows, self._cols, self._aspect = 1, 1, 1.0
        self.hovered: tuple[int, int] | None = None
        # Where the stage is, as a fractional (row, col); None if it can't be read.
        self.stage: tuple[float, float] | None = None
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    def set_array(self, rows: int, cols: int, aspect: float):
        self._rows, self._cols, self._aspect = rows, cols, aspect
        self.hovered = None
        self.updateGeometry()
        self.update()

    def set_stage(self, row: float | None, col: float | None):
        self.stage = None if row is None or col is None else (row, col)
        self.update()

    @property
    def stage_chamber(self) -> tuple[int, int] | None:
        """The chamber the stage is on (within half a chamber of its center)."""
        if self.stage is None:
            return None
        row, col = (round(v) for v in self.stage)
        return (row, col) if 0 <= row < self._rows and 0 <= col < self._cols else None

    def center(self, row: float, col: float) -> QPointF:
        area = self._area()
        return QPointF(
            area.left() + (col + 0.5) * area.width() / self._cols,
            area.top() + (row + 0.5) * area.height() / self._rows,
        )

    def chamber_at(self, pos: QPointF) -> tuple[int, int] | None:
        area = self._area()
        if not area.contains(pos):
            return None
        col = int((pos.x() - area.left()) / area.width() * self._cols)
        row = int((pos.y() - area.top()) / area.height() * self._rows)
        return min(row, self._rows - 1), min(col, self._cols - 1)

    def _area(self) -> QRectF:
        """Where the array is drawn: as big as fits, in the chip's proportions."""
        bounds = QRectF(self.rect()).adjusted(6, 6, -6, -6)
        width = max(min(bounds.width(), bounds.height() * self._aspect), 1.0)
        height = width / self._aspect
        return QRectF(
            bounds.center().x() - width / 2, bounds.center().y() - height / 2, width, height
        )

    def sizeHint(self) -> QSize:
        # The array's height at the width it's shown at, so widening the panel enlarges the map.
        width = self.width() if self.isVisible() else 240
        return QSize(240, min(max(round(width / self._aspect) + 12, 60), 600))

    def minimumSizeHint(self) -> QSize:
        return QSize(100, min(max(round(100 / self._aspect), 40), 200))

    def resizeEvent(self, event: Any):
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            self.updateGeometry()

    def paintEvent(self, event: Any):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        area = self._area()
        pitch = min(area.width() / self._cols, area.height() / self._rows)
        dot = max(pitch * 0.3, 0.75)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._DOT)
        for row in range(self._rows):
            for col in range(self._cols):
                painter.drawEllipse(self.center(row, col), dot, dot)
        if self.hovered is not None:
            painter.setBrush(self._HOVER)
            painter.drawEllipse(self.center(*self.hovered), dot * 1.8, dot * 1.8)
        if self.stage_chamber is not None:
            painter.setPen(QPen(self._STAGE, 2))
            painter.setBrush(Qt.NoBrush)
            ring = max(pitch * 0.9, 4.0)
            painter.drawEllipse(self.center(*self.stage), ring, ring)

    def mouseMoveEvent(self, event: Any):
        self._hover(self.chamber_at(QPointF(event.pos())))

    def leaveEvent(self, event: Any):
        self._hover(None)

    def mousePressEvent(self, event: Any):
        chamber = self.chamber_at(QPointF(event.pos()))
        if event.button() == Qt.LeftButton and chamber is not None:
            self._on_click(*chamber)

    def _hover(self, chamber: tuple[int, int] | None):
        if chamber != self.hovered:
            self.hovered = chamber
            self.update()
            self._on_hover()


class ChipMapServer:
    def __init__(self, ctrl: Any, arrays: dict[str, ChipArray]):
        self._ctrl = ctrl
        self._arrays = arrays

    def routes(self, path: str) -> dict[str, Callable[..., Any]]:
        return {
            path + "/arrays": self.get_arrays,
            path + "/where": self.where,
            path + "/go": self.go,
        }

    def get_arrays(self) -> dict[str, dict[str, float]]:
        return {
            name: dict(rows=a.rows, cols=a.cols, aspect=a.aspect)
            for name, a in self._arrays.items()
        }

    def where(self, name: str) -> dict[str, float | None]:
        # Never None: the viewer waits for a reply to every get.
        try:
            x, y = self._ctrl.stage.xy
            chamber = self._arrays[name].chamber(float(x), float(y))
        except Exception:
            chamber = None
        if chamber is None:
            return {"row": None, "col": None}
        return {"row": chamber[0], "col": chamber[1]}

    def go(self, name: str, row: int, col: int):
        # A route that raises ends the whole session (acquisition included), so just report it.
        try:
            array = self._arrays[name]
            # Only ever to a chamber: inside the corners given.
            if not (0 <= row < array.rows and 0 <= col < array.cols):
                raise ValueError(f"{name} has no chamber there")
            stage = self._ctrl.stage
            if isinstance(stage, devices.Wait):
                stage.wait()
            stage.xy = array.xy(row, col)
        except Exception as e:
            print(f"Couldn't move the stage to {name} {_chamber_name(row, col)}: {e}")


def _chip_arrays(ctrl: Any) -> dict[str, ChipArray]:
    """``c.chip_maps`` as ChipArrays. A bad entry is reported and left off the map, so it can't
    stop a session from starting."""
    maps = getattr(ctrl, "chip_maps", None) or {}
    if not isinstance(maps, dict):
        print(f"c.chip_maps should be a dict of chip name to array, not {type(maps).__name__}")
        return {}
    arrays = {}
    for name, spec in maps.items():
        try:
            arrays[str(name)] = ChipArray(**spec)
        except Exception as e:
            print(f"Chip map {name!r} left out: {e}")
    return arrays


def _stage_xy(value: Any, name: str) -> np.ndarray:
    xy = np.asarray(value, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError(f"{name} must be a stage (x, y), not {value!r}")
    return xy


def _chamber_name(row: int, col: int) -> str:
    return f"row {row + 1}, col {col + 1}"


class OutletTimer(QWidget):
    """Closes the picked valve after a set number of minutes, e.g. to flow buffer and walk away.

    Nothing is picked to start with, and it only starts on a valve that reads open, so picking
    the other chip's (closed) outlet is caught. The countdown runs in the viewer, so closing the
    viewer cancels it."""

    # A failed close (e.g. a dropped connection) is tried again this often until it works.
    _RETRY_S = 5

    def __init__(self, relay: "Relay"):
        super().__init__()
        self._relay = relay
        self._choices: list[tuple[str, str]] = [tuple(c) for c in self._relay.get("valves")]
        self._target: tuple[str, str] | None = None
        # time.monotonic() of the next close attempt; None when not counting down.
        self._deadline: float | None = None
        self._failed = False
        layout = QGridLayout(self)
        layout.setVerticalSpacing(0)

        self._valve = QComboBox()
        self._valve.addItem("pick a valve")
        for chip, valve in self._choices:
            self._valve.addItem(f"{chip} · {valve}")
        self._valve.currentIndexChanged.connect(self.update_state)
        layout.addWidget(QLabel("close"), 0, 0)
        layout.addWidget(self._valve, 0, 1)

        # The picked valve's state, as read from the valve controller.
        self._state = QLabel()
        layout.addWidget(QLabel("now"), 1, 0)
        layout.addWidget(self._state, 1, 1)

        self._minutes = QDoubleSpinBox()
        self._minutes.setRange(0.1, 24 * 60)
        self._minutes.setDecimals(1)
        self._minutes.setSingleStep(5)
        self._minutes.setValue(10)
        self._minutes.setSuffix(" min")
        layout.addWidget(QLabel("after"), 2, 0)
        layout.addWidget(self._minutes, 2, 1)

        self._btn = QPushButton()
        self._btn.setMinimumWidth(10)
        self._btn.setEnabled(bool(self._choices))
        self._btn.clicked.connect(self.toggle)
        layout.addWidget(self._btn, 3, 0, 1, 2)

        self._countdown = QLabel("--:--")
        self._countdown.setAlignment(Qt.AlignHCenter)
        self._countdown.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(self._countdown, 4, 0, 1, 2)
        self._status = QLabel()
        self._status.setAlignment(Qt.AlignHCenter)
        layout.addWidget(self._status, 5, 0, 1, 2)
        layout.setColumnStretch(1, 1)

        # The countdown is local; only state reads and the close itself go through the relay.
        self._timer = QTimer()
        self._timer.timeout.connect(self.tick)
        self._finish("not running")
        self.update_state()
        # Never taller than its rows, measured as it's shown. A height fixed here, before
        # napari's stylesheet and the screen's scaling set the fonts, cut off the countdown.
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self._poll = QTimer()
        self._poll.timeout.connect(self.update_state)
        self._poll.start(500)

    def update_state(self, _index: int = 0) -> str:
        picked = self._picked()
        state = "-" if picked is None else self._relay.get("state", *picked)
        self._state.setText(state)
        self._state.setStyleSheet("color: #e0b050; font-weight: bold;" if state == "open" else "")
        return state

    def toggle(self, _checked: bool = False):
        if self._deadline is None:
            self.start()
        else:
            self._finish(f"cancelled; {self._name()} left as it was")

    def start(self):
        picked = self._picked()
        if picked is None:
            self._show_status("pick the valve to close first", "#e06c6c")
            return
        state = self.update_state()
        if state != "open":
            # E.g. the other chip's outlet: closing it later would leave this one flowing.
            problem = "is already closed" if state == "closed" else "can't be read"
            self._show_status(f"{picked[0]} · {picked[1]} {problem}; not started", "#e06c6c")
            return
        self._target = picked
        self._deadline = time.monotonic() + self._minutes.value() * 60
        self._failed = False
        now = datetime.datetime.now()
        at = now + datetime.timedelta(minutes=self._minutes.value())
        day = "" if at.date() == now.date() else " tomorrow"
        self._set_running(True)
        self._show_status(f"closes {self._name()} at {_civilian_time(at)}{day}")
        self.tick()
        self._timer.start(200)

    def tick(self):
        remaining = self._deadline - time.monotonic()
        if remaining > 0:
            if not self._failed:
                self._countdown.setText(self._clock(remaining))
            return
        self._countdown.setText(self._clock(0))
        result = self._relay.get("close", *self._target)
        if result["ok"]:
            now = _civilian_time(datetime.datetime.now())
            self._finish(f"closed {self._name()} at {now}", "#7fc97f")
        else:
            # Closing is the safe direction, so keep trying until it works or is cancelled.
            self._failed = True
            self._deadline = time.monotonic() + self._RETRY_S
            self._show_status(f"couldn't close {self._name()}; retrying", "#e06c6c")
            self._status.setToolTip(result["error"])

    def _finish(self, status: str, color: str | None = None):
        self._timer.stop()
        self._deadline = None
        self._set_running(False)
        self._show_status(status, color)

    def _set_running(self, running: bool):
        self._btn.setText("cancel" if running else "start")
        # Amber while counting down, like an open shutter.
        self._btn.setStyleSheet(FilterShutterController.button_stylesheet(running))
        self._valve.setEnabled(not running)
        self._minutes.setEnabled(not running)

    def _show_status(self, text: str, color: str | None = None):
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {color};" if color else "")
        self._status.setToolTip("")

    def _picked(self) -> tuple[str, str] | None:
        # Item 0 is "pick a valve".
        index = self._valve.currentIndex()
        return self._choices[index - 1] if index > 0 else None

    def _name(self) -> str:
        chip, valve = self._target
        return f"{chip} · {valve}"

    @staticmethod
    def _clock(seconds: float) -> str:
        # Round up, so it reads 0:01 until the moment it closes.
        m, s = divmod(math.ceil(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class OutletTimerServer:
    def __init__(self, chips: list[Any]):
        self._chips = {c.name: c for c in chips}

    def routes(self, path: str) -> dict[str, Callable[..., Any]]:
        return {
            path + "/valves": self.get_valves,
            path + "/state": self.get_state,
            path + "/close": self.close,
        }

    def get_state(self, chip: str, valve: str) -> str:
        # Never None: the viewer waits for a reply to every get.
        try:
            return "closed" if self._chips[chip][valve] == "closed" else "open"
        except Exception:
            return "?"

    def get_valves(self) -> list[tuple[str, str]]:
        # Whole open/closed valves only, and never a neck, so the timer can't send one a command.
        return [
            (name, valve)
            for name, chip in self._chips.items()
            for valve, states in chip.valve_states.items()
            if set(states) == {"open", "closed"} and "[" not in valve and not _is_neck(valve)
        ]

    def close(self, chip: str, valve: str) -> dict[str, Any]:
        # Never None (the viewer waits for a reply) and never raises (that ends the session).
        if _is_neck(valve):
            error = "necks are left alone"
        else:
            try:
                device = self._chips[chip]
                device[valve] = "closed"
                state = device.valves[valve]
                error = None if state == "closed" else f"it still reads {state!r}"
            except Exception as e:
                error = str(e) or type(e).__name__
        stamp = _civilian_time(datetime.datetime.now(), seconds=True)
        if error is None:
            print(f"{stamp} outlet timer: closed {chip} {valve}")
        else:
            print(f"{stamp} outlet timer: couldn't close {chip} {valve}: {error}")
        return {"ok": error is None, "error": error}


def _is_neck(valve: str) -> bool:
    return valve.split("[")[0].startswith("neck")


def _civilian_time(t: datetime.datetime, seconds: bool = False) -> str:
    """The time of day on a 12-hour clock, e.g. 3:05 PM."""
    hm = f"{t.hour % 12 or 12}:{t:%M:%S}" if seconds else f"{t.hour % 12 or 12}:{t:%M}"
    return f"{hm} {'AM' if t.hour < 12 else 'PM'}"
