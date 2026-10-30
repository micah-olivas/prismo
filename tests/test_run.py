import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
import xarray as xr

from prismo.run import acq, run, tile_coords, tiled_acq


@pytest.fixture
def mock_ctrl():
    c = Mock()
    c.snap.return_value = np.zeros((100, 100), dtype=np.uint16)
    c.px_len = 1.0
    c.setup = "setup7"
    return c


def test_run_executes_function():
    results = []

    def foo(session):
        results.append(1)
        yield
        results.append(2)
        yield

    session = run(foo)
    session.join()
    assert results == [1, 2]


def test_run_provides_session_to_function():
    captured_session = []

    def my_func(session):
        captured_session.append(session)
        yield

    session = run(my_func)
    session.join()
    assert captured_session[0] is session


def test_run_can_be_quit():
    results = []

    def my_func(session):
        for i in range(1000):
            results.append(i)
            time.sleep(0.001)
            yield

    session = run(my_func)
    time.sleep(0.01)
    session.quit()
    session.join()
    assert len(results) < 1000


def test_tile_coords_basic(mock_ctrl):
    xs, ys = tile_coords(mock_ctrl, top_left=(0, 0), bot_right=(200, 200), overlap=0)
    assert len(xs) == 3
    assert len(ys) == 3
    np.testing.assert_array_almost_equal(xs, [0, 100, 200])
    np.testing.assert_array_almost_equal(ys, [0, 100, 200])


def test_tile_coords_with_overlap(mock_ctrl):
    xs, ys = tile_coords(mock_ctrl, top_left=(0, 0), bot_right=(200, 200), overlap=0.1)
    assert xs[1] - xs[0] == 90
    assert ys[1] - ys[0] == 90


def test_tile_coords_with_px_len(mock_ctrl):
    mock_ctrl.px_len = 0.5
    xs, ys = tile_coords(mock_ctrl, top_left=(0, 0), bot_right=(100, 100), overlap=0)
    assert xs[1] - xs[0] == 50
    assert ys[1] - ys[0] == 50


def test_tile_coords_rectangular_tiles():
    c = Mock()
    c.snap.return_value = np.zeros((50, 100), dtype=np.uint16)
    c.px_len = 1.0
    xs, ys = tile_coords(c, top_left=(0, 0), bot_right=(200, 100), overlap=0)
    assert xs[1] - xs[0] == 100
    assert ys[1] - ys[0] == 50


def test_tile_coords_small_area(mock_ctrl):
    xs, ys = tile_coords(mock_ctrl, top_left=(0, 0), bot_right=(50, 50), overlap=0)
    assert len(xs) == 2
    assert len(ys) == 2


def test_acq_writes_data_to_zarr(tmp_path, mock_ctrl):
    zarr_path = str(tmp_path / "test.zarr")
    frame0 = np.arange(100 * 100, dtype=np.uint16).reshape(100, 100)
    frame1 = np.arange(100 * 100, 2 * 100 * 100, dtype=np.uint16).reshape(100, 100)

    def my_acq(session):
        arr = session.array("data", time=2)
        arr[{"time": 0}] = frame0
        yield
        arr[{"time": 1}] = frame1
        yield

    with patch("prismo.run.init_widgets", return_value=({}, {})):
        session = acq(mock_ctrl, zarr_path, my_acq)
        session.join()

    ds = xr.open_zarr(zarr_path, group="data")
    np.testing.assert_array_equal(ds["tile"][0], frame0)
    np.testing.assert_array_equal(ds["tile"][1], frame1)


def test_acq_writes_multiple_arrays(tmp_path, mock_ctrl):
    zarr_path = str(tmp_path / "test.zarr")
    gfp_data = np.arange(100 * 100, dtype=np.uint16).reshape(100, 100)
    rfp_data = np.arange(100 * 100, 2 * 100 * 100, dtype=np.uint16).reshape(100, 100)

    def my_acq(session):
        gfp = session.array("gfp", time=1)
        rfp = session.array("rfp", time=1)
        gfp[{"time": 0}] = gfp_data
        rfp[{"time": 0}] = rfp_data
        yield

    with patch("prismo.run.init_widgets", return_value=({}, {})):
        session = acq(mock_ctrl, zarr_path, my_acq)
        session.join()

    gfp_ds = xr.open_zarr(zarr_path, group="gfp")
    rfp_ds = xr.open_zarr(zarr_path, group="rfp")
    np.testing.assert_array_equal(gfp_ds["tile"][0], gfp_data)
    np.testing.assert_array_equal(rfp_ds["tile"][0], rfp_data)


def test_tiled_acq_passes_correct_coordinates(tmp_path, mock_ctrl):
    zarr_path = str(tmp_path / "test.zarr")
    captured_coords = []

    def my_acq(session, xs, ys):
        captured_coords.append((xs.copy(), ys.copy()))
        yield

    with patch("prismo.run.init_widgets", return_value=({}, {})):
        session = tiled_acq(
            mock_ctrl, zarr_path, my_acq, overlap=0, top_left=(0, 0), bot_right=(200, 200)
        )
        session.join()

    xs, ys = captured_coords[0]
    np.testing.assert_array_almost_equal(xs, [0, 100, 200])
    np.testing.assert_array_almost_equal(ys, [0, 100, 200])


def test_tiled_acq_writes_tiled_data(tmp_path, mock_ctrl):
    zarr_path = str(tmp_path / "test.zarr")

    def my_acq(session, xs, ys):
        arr = session.array("tiles", row=len(ys), col=len(xs))
        for i, _y in enumerate(ys):
            for j, _x in enumerate(xs):
                arr[{"row": i, "col": j}] = np.full((100, 100), i * 10 + j, dtype=np.uint16)
            yield

    with patch("prismo.run.init_widgets", return_value=({}, {})):
        session = tiled_acq(
            mock_ctrl, zarr_path, my_acq, overlap=0, top_left=(0, 0), bot_right=(200, 200)
        )
        session.join()

    ds = xr.open_zarr(zarr_path, group="tiles")
    result = ds["tile"]

    assert result.shape == (3, 3, 100, 100)
    assert result[0, 0, 0, 0] == 0
    assert result[0, 1, 0, 0] == 1
    assert result[0, 2, 0, 0] == 2
    assert result[1, 0, 0, 0] == 10
    assert result[2, 2, 0, 0] == 22


def test_acq_saves_coordinates_and_attrs(tmp_path, mock_ctrl):
    zarr_path = str(tmp_path / "test.zarr")
    channels = ["gfp", "rfp", "dapi"]
    frame = np.arange(100 * 100, dtype=np.uint16).reshape(100, 100)

    def my_acq(session):
        arr = session.array("data", channel=channels)
        arr.attrs["exposure"] = 50.0
        arr.attrs["objective"] = "20x"
        for i, ch in enumerate(channels):
            arr.loc[ch] = frame * (i + 1)
        yield

    with patch("prismo.run.init_widgets", return_value=({}, {})):
        session = acq(mock_ctrl, zarr_path, my_acq)
        session.join()

    ds = xr.open_zarr(zarr_path, group="data")
    result = ds["tile"]

    assert list(result.coords["channel"].values) == channels
    assert ds.attrs["exposure"] == 50.0
    assert ds.attrs["objective"] == "20x"
    assert ds.attrs["setup"] == "setup7"
    np.testing.assert_array_equal(result[0], frame)
    np.testing.assert_array_equal(result[1], frame * 2)
    np.testing.assert_array_equal(result[2], frame * 3)
