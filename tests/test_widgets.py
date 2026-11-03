import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # no windows

import pytest
from napari.components import ViewerModel
from qtpy.QtWidgets import QApplication

from prismo import widgets


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class FakeStage:
    def __init__(self, units_per_um=10.0):
        self.name, self.units_per_um, self.xy, self.moves = "stage", units_per_um, (0.0, 0.0), []

    def __setattr__(self, name, value):
        if name == "xy" and hasattr(self, "moves"):
            self.moves.append(value)
        super().__setattr__(name, value)


def test_dpad_steps_in_stage_units_and_adds_up_quick_presses():
    stage = FakeStage(units_per_um=10.0)  # ASI: 0.1 µm per count
    server = widgets.StageControllerServer(stage)
    server.move(1, 0, "10 µm")
    server.move(1, 0, "10 µm")  # before the first move finished: adds to its target
    server.move(0, -1, "1 mm")
    assert stage.moves == [(100.0, 0.0), (200.0, 0.0), (200.0, -10000.0)]


def test_dpad_field_step_is_the_camera_field_of_view():
    class Core:
        def getImageWidth(self):
            return 2048

        def getImageHeight(self):
            return 1024

    class Ctrl:
        _core, px_len = Core(), 1.625

    server = widgets.StageControllerServer(FakeStage(units_per_um=1.0), Ctrl())
    assert server.step_um("field") == (2048 * 1.625, 1024 * 1.625)


def test_dpad_widget_posts_moves_with_the_chosen_step(app):
    posts = []

    class Relay:
        def post(self, *args):
            posts.append(args)

    pad = widgets.StageController(Relay())
    pad.set_step("1 mm")
    pad.move(0, 1)
    assert posts == [("move", 0, 1, "1 mm")]


def test_grid_map_follows_the_view_and_jumps_on_click(app):
    viewer = ViewerModel()
    rows, cols, tile = 36, 12, 512
    gm = widgets.GridMap(viewer, rows, cols, rows * tile, cols * tile)
    gm.resize(200, 260)
    viewer.camera.center = (0, 10 * tile + 100, 3 * tile + 50)
    gm._update_view()  # what the 30 ms timer does
    assert gm.tile_at_center() == (10, 3)
    viewer.camera.center = (0, -5.0, 0.0)
    gm._update_view()
    assert gm.tile_at_center() is None  # off the grid

    class Pos:
        def __init__(self, x, y):
            self._x, self._y = x, y

        def x(self):
            return self._x

        def y(self):
            return self._y

    left, top, w, h = gm._map_rect()
    gm._jump(Pos(left + w / 2, top + h / 4))
    y, x = viewer.camera.center[-2:]
    assert y == pytest.approx(rows * tile / 4) and x == pytest.approx(cols * tile / 2)
