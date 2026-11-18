from unittest.mock import patch

import numpy as np
import pymmcore
import xarray as xr
from test_binding import CUBES, PS18K_V3, Events, StateDevice
from test_patterning import FakeDriver

from prismo import scope
from prismo.control import Control
from prismo.devices import ti2
from prismo.devices.microfluidic import Chip
from prismo.run import acq

CUBE_SPECS = {"cy5": {"cube_idx": 23, "ex": "618/50", "em": "698/70", "di": "652"}}


class Core:
    """A Micro-Manager core with the nosepiece as its one state device, and the camera's
    properties."""

    def __init__(self, fail=False):
        self.fail, self.moves = fail, []

    def getDeviceType(self, name):
        return pymmcore.StateDevice if name == "objective" else pymmcore.CameraDevice

    def getState(self, name):
        if self.fail:
            raise RuntimeError("no answer")
        return 1

    def setState(self, name, i):
        self.moves.append(i)

    def getStateLabel(self, name):
        raise RuntimeError("no answer")  # Objective.state would now move the nosepiece

    def getDevicePropertyNames(self, name):
        return ("PortSpeedGain", "Temperature")

    def getProperty(self, name, prop):
        if prop == "Temperature":
            raise RuntimeError("busy")
        return "Port0-Speed0-100MHz-16bit-HDR"


class Kinetix:
    name, _flip, clean_snap = "kinetix", "both", True

    def __init__(self, core):
        self._core, self.exposure, self.binning = core, 500.0, 1

    @property
    def px_len(self):
        return 6.5 * self.binning

    def snap(self):
        return np.ones((4, 4), np.uint16)


class Focus:
    name, z = "focus", 1234.5


class ReadLoggingDriver(FakeDriver):
    def __init__(self):
        super().__init__()
        self.reads = []

    def __getitem__(self, idx):
        self.reads.append(idx)
        return super().__getitem__(idx)


def microscope(core):
    events = Events()
    nosepiece = object.__new__(ti2.Objective)
    nosepiece.name, nosepiece._core = "objective", core
    nosepiece.states = ["Apo4x", "10x", "empty1", "40x", "empty2", "Fluor4x"]
    nosepiece.zooms = dict(zip(nosepiece.states, [4, 10, 0, 40, 0, 4], strict=True))
    driver = ReadLoggingDriver()
    camera = Kinetix(core)
    devs = [
        StateDevice("filter", events, CUBES, "cy5"),
        StateDevice("shutter", events, ["open", "closed"], "closed"),
        StateDevice("light", events, None, 100),
        StateDevice("light_source", events, ["sola", "retra"], "sola"),
        StateDevice("lightpath", events, ["eye", "kinetix", "aux", "zyla"], "kinetix"),
        nosepiece,
        camera,
        Focus(),
        driver,
        Chip("d1", driver, dict(PS18K_V3)),
    ]
    ctrl = object.__new__(Control)
    object.__setattr__(ctrl, "devices", devs)
    for name, value in dict(_camera=camera, _focus=devs[7], _stage=None, setup="setup7").items():
        object.__setattr__(ctrl, name, value)
    return ctrl, driver


def test_every_acquisition_saves_the_scope_settings(tmp_path):
    core = Core()
    ctrl, driver = microscope(core)
    ctrl.filter_cubes = CUBE_SPECS
    file = str(tmp_path / "k.zarr")

    def my_acq(session):
        imgs = session.array("cy5_500ms", time=2)
        imgs.attrs["light"] = "set by the acq_func"
        imgs.loc[0] = ctrl.snap()
        yield
        ctrl.filter = "gfp"  # read once, at the first tile: this doesn't change them
        imgs.loc[1] = ctrl.snap()
        yield

    with patch("prismo.run.init_widgets", return_value=({}, {})):
        session = acq(ctrl, file, my_acq)
        session.join()
        session.quit()  # closes its viewer

    attrs = xr.open_zarr(file, group="cy5_500ms").attrs
    assert attrs["filter"] == "cy5" and attrs["filter_cube"] == CUBE_SPECS["cy5"]
    assert attrs["objective"] == "10x" and attrs["px_len_um"] == 0.65
    assert attrs["lightpath"] == "kinetix" and attrs["z"] == 1234.5
    assert attrs["light"] == "set by the acq_func"  # an acq_func's own attrs are kept
    assert attrs["device_states"]["light_source"] == "sola"
    assert attrs["device_states"]["shutter"] == "closed"
    camera = attrs["camera"]
    assert camera["name"] == "kinetix" and camera["exposure_ms"] == 500.0
    assert camera["binning"] == 1 and camera["flip"] == "both" and camera["clean_snap"]
    assert camera["properties"]["PortSpeedGain"] == "Port0-Speed0-100MHz-16bit-HDR"
    assert camera["properties"]["Temperature"] == "<unreadable: busy>"
    assert attrs["setup"] == "setup7" and "scope_read_at" in attrs

    # Never the valves, and nothing moved.
    assert driver.reads == [] and driver.written == set()
    assert {"d1", "fake"}.isdisjoint(attrs["device_states"]) and core.moves == []


def test_a_failed_read_moves_nothing():
    core = Core(fail=True)
    ctrl, driver = microscope(core)
    attrs = scope.scope_attrs(ctrl)
    assert attrs["objective"] == "<unreadable: no answer>"
    assert attrs["px_len_um"] == "<unreadable: no answer>"
    assert attrs["filter_cube"] is None  # no c.filter_cubes
    assert attrs["light"] == {"source": "sola", "power": 100}
    assert core.moves == [] and driver.reads == []


def test_the_acquisition_goes_on_if_the_settings_cant_be_read(tmp_path, monkeypatch, caplog):
    ctrl, _ = microscope(Core())
    file = str(tmp_path / "k.zarr")

    def broken(ctrl):
        raise RuntimeError("core busy")

    def my_acq(session):
        imgs = session.array("img", time=2)
        for t in range(2):
            imgs.loc[t] = ctrl.snap() * (t + 1)
            yield

    monkeypatch.setattr("prismo.run.scope_attrs", broken)
    with patch("prismo.run.init_widgets", return_value=({}, {})):
        session = acq(ctrl, file, my_acq)
        session.join()
        session.quit()  # closes its viewer

    ds = xr.open_zarr(file, group="img")
    np.testing.assert_array_equal(ds["tile"].values[:, 0, 0], [1, 2])
    assert ds.attrs["scope_read_at"] == "<unreadable: core busy>"
    assert caplog.text.count("Could not read the scope settings") == 1
