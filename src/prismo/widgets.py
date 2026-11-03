import functools
import time
from collections.abc import Callable
from typing import Any

from qtpy.QtCore import QPointF, QRectF, Qt, QTimer
from qtpy.QtGui import QColor, QDoubleValidator, QPainter, QPen
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
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

    # One panel for the optics: a row of buttons per device with discrete states (filter,
    # shutter, ...), a slider per light, since lights have a level (c.light = 0-100), and the
    # camera's exposure.
    state_devices = [
        d for d in ctrl.devices if isinstance(d, devices.State) and hasattr(d, "states")
    ]
    lights = [d for d in ctrl.devices if isinstance(d, devices.lumencor.SolaLight)]
    has_camera = ctrl.camera is not None
    if state_devices or lights or has_camera:
        path = "widget/states"
        light_names = [d.name for d in lights]
        widgets["optics"] = lambda r, path=path: StateController(
            r.subpath(path), light_names, exposure=has_camera
        )
        server = StateControllerServer(state_devices)
        routes = {**routes, **server.routes(path)}
        for light in lights:
            routes = {**routes, **LightControllerServer(light).routes(f"{path}/{light.name}")}
        if has_camera:
            routes = {**routes, **ExposureControllerServer(ctrl).routes(f"{path}/exposure")}

    if ctrl.stage is not None:
        path = "widget/stage"
        widgets["stage controller"] = lambda r, path=path: StageController(r.subpath(path))
        server = StageControllerServer(ctrl.stage, ctrl)
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
    """The optics panel: a row of buttons per state device, a power slider per light, and the
    camera's exposure."""

    def __init__(self, relay: "Relay", lights: list[str] | None = None, exposure: bool = False):
        super().__init__()
        self._relay = relay
        self._states: dict[str, str | int | float] = self._relay.get("states")
        self._options: dict[str, list[str | int | float]] = self._relay.get("options")
        self._btns: dict[str, dict[str, QPushButton]] = {}
        self._shown: dict[str, str] = {}
        grid = QGridLayout(self)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        grid.setColumnStretch(1, 1)
        self._timer = QTimer()
        self._timer.timeout.connect(self.update_states)
        self._timer.start(10)

        # The light's power and the exposure go right under the shutter, since they're used
        # together.
        rows: list[tuple[str, str]] = [(name, "state") for name in self._states]
        at = next((i + 1 for i, (n, _) in enumerate(rows) if n == "shutter"), len(rows))
        extra = [(name, "light") for name in lights or []]
        if exposure:
            extra.append(("exposure", "exposure"))
        rows[at:at] = extra

        # One row per device: its name, then its buttons (sharing the row equally), slider or box.
        for row, (name, kind) in enumerate(rows):
            label = QLabel(name)
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(label, row, 0)
            if kind == "light":
                grid.addWidget(LightController(self._relay.subpath(name)), row, 1)
                continue
            if kind == "exposure":
                grid.addWidget(ExposureController(self._relay.subpath("exposure")), row, 1)
                continue
            line = QHBoxLayout()
            line.setSpacing(4)
            self._btns[name] = {}
            for opt in self._options[name]:
                btn = QPushButton(str(opt))
                btn.setMinimumSize(44, 26)
                btn.clicked.connect(functools.partial(self.set_state, name, opt))
                line.addWidget(btn, 1)
                self._btns[name][str(opt)] = btn
            grid.addLayout(line, row, 1)
        grid.setRowStretch(len(rows), 1)  # keep the rows together at the top
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
                style = ValveController.button_stylesheet("closed" if opt == str(state) else "open")
                btn.setStyleSheet(style + "QPushButton { padding: 4px 6px; margin: 0px; }")

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


class LightController(QWidget):
    """A slider and box for the light level (c.light), e.g. to turn the power down while imaging."""

    def __init__(self, relay: "Relay"):
        super().__init__()
        self._relay = relay
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 100)
        self._box = QSpinBox()
        self._box.setRange(0, 100)
        self._box.setSuffix(" %")
        self._box.setMinimumWidth(64)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._box)

        level = self._relay.get("level")
        self._slider.setValue(level)
        self._box.setValue(level)
        # Only send the level once the slider is let go, so dragging doesn't flood the light.
        self._slider.sliderReleased.connect(lambda: self.set_level(self._slider.value()))
        self._slider.valueChanged.connect(self._slider_moved)
        self._box.editingFinished.connect(lambda: self.set_level(self._box.value()))

        # Show changes made from the notebook too (e.g. a protocol setting the power).
        self._timer = QTimer()
        self._timer.timeout.connect(self.update_level)
        self._timer.start(250)

    def _slider_moved(self, value: int):
        self._box.blockSignals(True)
        self._box.setValue(value)
        self._box.blockSignals(False)
        # Clicks and arrow keys move the slider without a release.
        if not self._slider.isSliderDown():
            self.set_level(value)

    def update_level(self):
        if self._slider.isSliderDown() or self._box.hasFocus():
            return
        level = self._relay.get("level")
        for w in (self._slider, self._box):
            if w.value() != level:
                w.blockSignals(True)
                w.setValue(level)
                w.blockSignals(False)

    def set_level(self, level: int):
        self._relay.post("set_level", level)


class ExposureController(QWidget):
    """A box for the camera's exposure (c.exposure), in ms, with buttons to halve or double it."""

    def __init__(self, relay: "Relay"):
        super().__init__()
        self._relay = relay
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._box = QDoubleSpinBox()
        self._box.setRange(0.01, 100000.0)
        self._box.setDecimals(2)
        self._box.setSuffix(" ms")
        self._box.setMinimumWidth(96)
        self._box.setValue(self._relay.get("ms"))
        self._box.editingFinished.connect(lambda: self.set_ms(self._box.value()))
        layout.addWidget(self._box, 1)
        for text, factor in (("÷2", 0.5), ("×2", 2.0)):
            btn = QPushButton(text)
            btn.setMinimumSize(36, 26)
            btn.setStyleSheet(
                ValveController.button_stylesheet("open")
                + "QPushButton { padding: 4px 6px; margin: 0px; }"
            )
            btn.clicked.connect(functools.partial(self.scale, factor))
            layout.addWidget(btn)

        # Show changes made from the notebook too (e.g. a protocol setting the exposure).
        self._timer = QTimer()
        self._timer.timeout.connect(self.update_ms)
        self._timer.start(250)

    def update_ms(self):
        if self._box.hasFocus():
            return
        ms = self._relay.get("ms")
        if abs(self._box.value() - ms) > 1e-6:
            self._box.blockSignals(True)
            self._box.setValue(ms)
            self._box.blockSignals(False)

    def scale(self, factor: float):
        ms = min(max(self._box.value() * factor, self._box.minimum()), self._box.maximum())
        self._box.setValue(ms)
        self.set_ms(ms)

    def set_ms(self, ms: float):
        self._relay.post("set_ms", ms)


class ExposureControllerServer:
    def __init__(self, ctrl: "Control"):
        self._ctrl = ctrl

    def routes(self, path: str) -> dict[str, Callable[..., Any]]:
        return {path + "/ms": self.get_ms, path + "/set_ms": self.set_ms}

    def get_ms(self) -> float:
        return float(self._ctrl.exposure)

    def set_ms(self, ms: float):
        self._ctrl.exposure = float(ms)


class LightControllerServer:
    def __init__(self, light: devices.lumencor.SolaLight):
        self._light = light

    def routes(self, path: str) -> dict[str, Callable[..., Any]]:
        return {path + "/level": self.get_level, path + "/set_level": self.set_level}

    def get_level(self) -> int:
        return int(self._light.state)

    def set_level(self, level: int):
        self._light.state = int(level)


class GridMap(QWidget):
    """A small map of a tiled acquisition's grid, with a box for the part napari is showing.

    Works from the stitched image's size and its rows and columns alone, so it fits any tiled
    acquisition. It only repaints when the view moves, and clicking or dragging on it moves the
    view there.
    """

    _ACCENT = QColor("#4fc3f7")

    def __init__(self, viewer: Any, rows: int, cols: int, height: int, width: int):
        super().__init__()
        self._viewer = viewer
        self._rows, self._cols = rows, cols
        self._h, self._w = height, width  # stitched image size, in pixels
        self._view: tuple[float, float, float, float] | None = None  # y0, x0, y1, x1
        self._center: tuple[float, float] = (height / 2, width / 2)
        self.setMinimumSize(120, 120)

        # Camera events fire on every frame of a pan or zoom; repaint at most every 30 ms.
        self._pending = QTimer()
        self._pending.setSingleShot(True)
        self._pending.setInterval(30)
        self._pending.timeout.connect(self._update_view)
        viewer.camera.events.center.connect(self._schedule)
        viewer.camera.events.zoom.connect(self._schedule)
        canvas = getattr(viewer, "canvas", None)
        if canvas is not None:
            canvas.events.size.connect(self._schedule)
        self._update_view()

    def _schedule(self, event: Any = None):
        if not self._pending.isActive():
            self._pending.start()

    def _canvas_size(self) -> tuple[float, float]:
        canvas = getattr(self._viewer, "canvas", None)
        size = canvas.size if canvas is not None else getattr(self._viewer, "_canvas_size", None)
        return tuple(size) if size else (600.0, 800.0)

    def _update_view(self):
        cy, cx = self._viewer.camera.center[-2:]
        zoom = self._viewer.camera.zoom or 1.0
        ch, cw = self._canvas_size()
        vh, vw = ch / zoom, cw / zoom
        self._center = (cy, cx)
        self._view = (cy - vh / 2, cx - vw / 2, cy + vh / 2, cx + vw / 2)
        self.update()

    def tile_at_center(self) -> tuple[int, int] | None:
        """The (row, col) shown in the middle of the view, from 0, or None if off the grid."""
        cy, cx = self._center
        if not (0 <= cy < self._h and 0 <= cx < self._w):
            return None
        return int(cy / self._h * self._rows), int(cx / self._w * self._cols)

    def _map_rect(self) -> tuple[float, float, float, float]:
        """Where the grid is drawn: left, top, width, height, keeping its aspect ratio."""
        pad, label_h = 6.0, 18.0
        aw, ah = self.width() - 2 * pad, self.height() - 2 * pad - label_h
        s = min(aw / self._w, ah / self._h)
        w, h = self._w * s, self._h * s
        return pad + (aw - w) / 2, pad + label_h + (ah - h) / 2, w, h

    def paintEvent(self, event: Any):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        left, top, w, h = self._map_rect()
        cell_w, cell_h = w / self._cols, h / self._rows

        tile = self.tile_at_center()
        label = (
            f"row {tile[0] + 1}/{self._rows}   col {tile[1] + 1}/{self._cols}"
            if tile is not None
            else f"off the grid ({self._rows} x {self._cols} tiles)"
        )
        p.setPen(QColor("#d0d0d0"))
        p.drawText(int(left), int(top) - 5, label)

        # The grid: an outline, plus the tile lines when they're far enough apart to see.
        p.setBrush(QColor(255, 255, 255, 18))
        p.setPen(QPen(QColor("#8a8f99"), 1))
        p.drawRect(QRectF(left, top, w, h))
        p.setPen(QPen(QColor(138, 143, 153, 90), 1))
        if cell_w >= 3:
            for c in range(1, self._cols):
                x = left + c * cell_w
                p.drawLine(QPointF(x, top), QPointF(x, top + h))
        if cell_h >= 3:
            for r in range(1, self._rows):
                y = top + r * cell_h
                p.drawLine(QPointF(left, y), QPointF(left + w, y))

        if tile is not None:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(79, 195, 247, 110))
            p.drawRect(QRectF(left + tile[1] * cell_w, top + tile[0] * cell_h, cell_w, cell_h))

        # The part of the image in view, clipped to the grid.
        if self._view is not None:
            y0, x0, y1, x1 = self._view
            sx, sy = w / self._w, h / self._h
            rx0, ry0 = max(left, left + x0 * sx), max(top, top + y0 * sy)
            rx1, ry1 = min(left + w, left + x1 * sx), min(top + h, top + y1 * sy)
            if rx1 > rx0 and ry1 > ry0:
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.setPen(QPen(self._ACCENT, 2))
                p.drawRect(QRectF(rx0, ry0, rx1 - rx0, ry1 - ry0))
        p.end()

    def _jump(self, pos: Any):
        left, top, w, h = self._map_rect()
        y = (pos.y() - top) / h * self._h
        x = (pos.x() - left) / w * self._w
        y, x = min(max(y, 0), self._h), min(max(x, 0), self._w)
        center = tuple(self._viewer.camera.center)
        self._viewer.camera.center = (*center[:-2], y, x)

    def mousePressEvent(self, event: Any):
        self._jump(event.position() if hasattr(event, "position") else event.pos())

    def mouseMoveEvent(self, event: Any):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._jump(event.position() if hasattr(event, "position") else event.pos())


class StageController(QWidget):
    """A d-pad that moves the stage a fixed step per press (held down, it repeats).

    Right is +x and up is +y on the stage, as with the old joystick. The step is picked below
    the pad; "field" is one camera field of view.
    """

    _STEPS = ["1 µm", "10 µm", "100 µm", "1 mm", "field"]
    _ARROWS = {"up": ("▲", 0, 1, 0, 1), "left": ("◀", 1, 0, -1, 0),
               "right": ("▶", 1, 2, 1, 0), "down": ("▼", 2, 1, 0, -1)}

    def __init__(self, relay: "Relay"):
        super().__init__()
        self._relay = relay
        self._step = "100 µm"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        pad = QGridLayout()
        pad.setSpacing(4)
        for name, (arrow, row, col, dx, dy) in self._ARROWS.items():
            btn = QPushButton(arrow)
            btn.setFixedSize(44, 44)
            btn.setAutoRepeat(True)
            btn.setAutoRepeatDelay(400)
            btn.setAutoRepeatInterval(250)
            btn.setStyleSheet(ValveController.button_stylesheet("open"))
            btn.clicked.connect(functools.partial(self.move, dx, dy))
            pad.addWidget(btn, row, col)
        # Keep the pad a tight cross in the middle, however wide the panel is.
        centered = QHBoxLayout()
        centered.addStretch(1)
        centered.addLayout(pad)
        centered.addStretch(1)
        layout.addLayout(centered)

        steps = QHBoxLayout()
        steps.setSpacing(4)
        self._step_btns: dict[str, QPushButton] = {}
        for step in self._STEPS:
            btn = QPushButton(step)
            btn.setMinimumSize(44, 26)
            btn.clicked.connect(functools.partial(self.set_step, step))
            steps.addWidget(btn, 1)
            self._step_btns[step] = btn
        layout.addLayout(steps)
        layout.addStretch(1)
        self.set_step(self._step)

    def set_step(self, step: str):
        self._step = step
        for s, btn in self._step_btns.items():
            style = ValveController.button_stylesheet("closed" if s == step else "open")
            btn.setStyleSheet(style + "QPushButton { padding: 4px 6px; margin: 0px; }")

    def move(self, dx: int, dy: int):
        self._relay.post("move", dx, dy, self._step)


class StageControllerServer:
    # Successive presses add to where the last one was headed, rather than to a position read
    # mid-move, unless the stage has been left alone this long.
    _TARGET_S = 2.0

    def __init__(self, stage: devices.Stage, ctrl: "Control | None" = None):
        self._stage = stage
        self._ctrl = ctrl
        self._target: tuple[float, float] | None = None
        self._last_move = 0.0

    def routes(self, path: str) -> dict[str, Callable[..., Any]]:
        return {path + "/move": self.move}

    def step_um(self, step: str) -> tuple[float, float]:
        """The step (x, y) in µm. "field" is the camera's field of view, from its pixel size."""
        if step == "field":
            core = self._ctrl._core
            px = float(self._ctrl.px_len)
            return core.getImageWidth() * px, core.getImageHeight() * px
        value, unit = step.split()
        um = float(value) * (1000.0 if unit == "mm" else 1.0)
        return um, um

    def move(self, dx: int, dy: int, step: str):
        sx, sy = self.step_um(step)
        units = getattr(self._stage, "units_per_um", 1.0)
        now = time.monotonic()
        if self._target is None or now - self._last_move > self._TARGET_S:
            x, y = self._stage.xy
        else:
            x, y = self._target
        self._target = (x + dx * sx * units, y + dy * sy * units)
        self._last_move = now
        self._stage.xy = self._target


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
