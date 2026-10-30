import functools
from collections.abc import Callable
from typing import Any

from qtpy.QtCore import QPointF, Qt, QTimer
from qtpy.QtGui import QBrush, QColor, QDoubleValidator, QPainter, QPen
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import devices
from .control import Control
from .session import Relay


def init_widgets(
    ctrl: "Control",
) -> tuple[dict[str, Callable[["Relay"], QWidget]], dict[str, Callable[..., Any]]]:
    widgets: dict[str, Callable[[Relay], QWidget]] = {}
    routes: dict[str, Callable[..., Any]] = {}

    for device in ctrl.devices:
        if isinstance(device, devices.Valved):
            path = f"widget/{device.name}"
            # We need to set a dummy default argument so path's value gets captured by the lambda.
            widgets[f"{device.name} controller"] = lambda r, path=path: ValveController(
                r.subpath(path)
            )
            server = ValveControllerServer(device)
            routes = {**routes, **server.routes(path)}

    state_devices = [d for d in ctrl.devices if isinstance(d, devices.State)]
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
        self._combos: dict[str, QComboBox] = {}
        layout = QVBoxLayout(self)
        self._timer = QTimer()
        self._timer.timeout.connect(self.update_states)
        self._timer.start(10)

        for name, state in self._states.items():
            row = QHBoxLayout()
            row.addWidget(QLabel(name))
            combo = QComboBox()
            for opt in self._options[name]:
                combo.addItem(str(opt))
            combo.setCurrentText(str(state))
            combo.currentTextChanged.connect(functools.partial(self.set_state, name))
            row.addWidget(combo)
            layout.addLayout(row)
            self._combos[name] = combo

    def update_states(self):
        self._states = self._relay.get("states")
        for name, state in self._states.items():
            combo = self._combos[name]
            combo.blockSignals(True)
            combo.setCurrentText(str(state))
            combo.blockSignals(False)

    def set_state(self, name: str, state_str: str):
        options = self._options[name]
        state: str | int | float = next((s for s in options if str(s) == state_str), state_str)
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
    def __init__(self, relay: "Relay"):
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

    def button_stylesheet(self, state: str | int) -> str:
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
