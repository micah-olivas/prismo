import numpy as np
import pytest
import xarray as xr
from test_patterning import FakeDriver

from prismo.control import Control
from prismo.devices.microfluidic import Chip
from prismo.protocols import binding, imaging, washes
from prismo.session import Session

# PS1.8k v3: split necks.
PS18K_V3 = {
    **{f"in{i}": i for i in range(7)},
    "waste7": 7,
    "inlet": 8,
    "neckL": 9,
    "butL": 10,
    "sandL": 11,
    "sandR": 12,
    "butR": 13,
    "neckR": 14,
    "out": 15,
}
CUBES = ["gfp", "cy5", "NADHTi", "SOLA_DANAC", "empty5", "empty6"]


class Clock:
    """Virtual time: sleeping and snapping advance it, nothing really waits."""

    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def sleep(self, s):
        self.now += s


class Events(list):
    def __call__(self, *event):
        self.append(event)


class StateDevice:
    def __init__(self, name, events, states=None, state=None):
        self.name, self._events, self.states, self._state = name, events, states, state

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, s):
        self._events(self.name, s)
        self._state = s


class Camera:
    def __init__(self, name, events, clock):
        self.name, self._events, self._clock, self.exposure, self.n = name, events, clock, 0, 0

    def snap(self):
        self._clock.sleep(1.0)  # each tile takes a second
        self.n += 1
        self._events("snap", self.name, self.n)
        return np.full((2, 2), self.n, dtype=np.uint16)


class Stage:
    def __init__(self, events):
        self.name, self._events, self.x, self.y = "stage", events, 0.0, 0.0

    @property
    def xy(self):
        return self.x, self.y

    @xy.setter
    def xy(self, xy):
        self._events("xy", tuple(xy))
        self.x, self.y = xy

    def wait(self):
        pass


class LoggedDriver(FakeDriver):
    def __init__(self, events, initial=None):
        super().__init__(initial)
        self._events = events

    def __setitem__(self, idx, state):
        super().__setitem__(idx, state)
        self._events("valve", idx, state)


class FakeSession(Session):
    """In-memory arrays instead of zarr, and no viewer."""

    def __init__(self):
        super().__init__()
        self.fake_arrays = {}

    @property
    def arrays(self):
        return self.fake_arrays

    def array(self, name, **dims):
        shape = tuple(x if isinstance(x, int) else len(x) for x in dims.values()) + (2, 2)
        xp = xr.DataArray(np.zeros(shape, np.uint16), dims=(*dims, "y", "x"), name=name)
        for d, coords in dims.items():
            if not isinstance(coords, int):
                xp.coords[d] = np.asarray(coords)
        self.fake_arrays[name] = xp
        return xp


XS, YS = np.array([0.0, 10.0]), np.array([0.0, 20.0])


@pytest.fixture
def rig(monkeypatch):
    clock, events = Clock(), Events()
    monkeypatch.setattr(binding.time, "time", clock.time)
    monkeypatch.setattr(binding.time, "sleep", clock.sleep)
    monkeypatch.setattr(binding, "tqdm", None)
    monkeypatch.setattr(imaging, "tqdm", None)

    zyla, kinetix = Camera("camera", events, clock), Camera("kinetix", events, clock)
    stage = Stage(events)
    devs = [
        StateDevice("filter", events, CUBES, "cy5"),
        StateDevice("shutter", events, ["open", "closed"], "closed"),
        StateDevice("light", events),
        StateDevice("light_source", events, ["sola", "retra"], "retra"),
        StateDevice("lightpath", events, ["eye", "kinetix", "aux", "zyla"], "zyla"),
        zyla,
        kinetix,
        stage,
    ]
    ctrl = object.__new__(Control)
    object.__setattr__(ctrl, "devices", devs)
    object.__setattr__(ctrl, "_camera", zyla)
    object.__setattr__(ctrl, "_stage", stage)
    driver = LoggedDriver(events, initial={PS18K_V3["neckL"]: "open", PS18K_V3["neckR"]: "open"})
    chip = Chip("d1", driver, dict(PS18K_V3))

    runs = []

    def fake_tiled_acq(ctrl, file, acq_func, overlap, top_left, bot_right):
        session = FakeSession()
        runs.append(dict(file=file, top_left=top_left, camera=ctrl.camera.name, session=session))
        gen = acq_func(session, XS, YS)
        for k, _ in enumerate(gen, 1):
            if rig.stop_after is not None and k == rig.stop_after:
                gen.close()  # what g.quit() does to the acquisition
                break
        return session

    monkeypatch.setattr(binding, "tiled_acq", fake_tiled_acq)
    monkeypatch.setattr(washes, "tiled_acq", fake_tiled_acq)
    monkeypatch.setattr(imaging, "tiled_acq", fake_tiled_acq)

    class Rig:
        def dev(self, name):
            return next(d for d in devs if d.name == name)

    rig = Rig()
    rig.__dict__.update(
        ctrl=ctrl, chip=chip, driver=driver, events=events, clock=clock, runs=runs, stop_after=None
    )
    return rig


def run(rig, file="binding", **kwargs):
    kwargs.setdefault("top_left", [0, 0])
    kwargs.setdefault("bot_right", [10, 20])
    return binding.track_binding(rig.ctrl, str(file), **kwargs)


def valve_events(rig):
    names = {v: k for k, v in PS18K_V3.items()}
    return [(names[e[1]], e[2]) for e in rig.events if e[0] == "valve"]


def test_only_images_every_pass_on_the_kinetix(rig, tmp_path):
    session = run(rig, tmp_path / "b", passes=3)
    imgs = session.arrays["egfp_500ms"]

    assert rig.driver.written == set()  # never touches a valve
    assert rig.runs[0]["camera"] == "kinetix" and rig.dev("lightpath").state == "kinetix"
    snaps = [e for e in rig.events if e[0] == "snap"]
    assert len(snaps) == 12 and {e[1] for e in snaps} == {"kinetix"}
    assert sorted(np.unique(imgs.values)) == list(range(1, 13))  # every tile once per pass

    # The shutter is open only for each snap.
    for i, e in enumerate(rig.events):
        if e[0] == "snap":
            assert rig.events[i - 1] == ("shutter", "open")
            assert rig.events[i + 1] == ("shutter", "closed")
    assert rig.dev("shutter").state == "closed"
    assert ("filter", "gfp") in rig.events and ("light_source", "sola") in rig.events
    assert rig.ctrl.exposure == 500

    seconds = imgs["seconds"].values
    assert not np.isnan(seconds).any() and seconds.min() == 0
    assert np.all(np.diff(np.sort(seconds.ravel())) > 0)
    assert "started_at" in imgs.attrs and imgs.attrs["channel"] == "gfp"
    assert rig.runs[0]["top_left"] == (0, 0)  # lists are passed on as tuples


def test_camera_choice(rig, tmp_path):
    run(rig, tmp_path / "b", passes=1, camera="zyla")
    assert rig.runs[-1]["camera"] == "camera" and rig.dev("lightpath").state == "zyla"
    rig.ctrl.camera = "kinetix"
    run(rig, tmp_path / "c", passes=1, camera=None)  # keep the current camera
    assert rig.runs[-1]["camera"] == "kinetix"


def test_same_snake_order_every_pass(rig, tmp_path):
    run(rig, tmp_path / "b", passes=2)
    xys = [e[1] for e in rig.events if e[0] == "xy"]
    assert xys[:4] == [(0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0)]
    assert xys[4:] == xys[:4]


def test_interval_is_from_pass_start_to_pass_start(rig, tmp_path):
    session = run(rig, tmp_path / "b", passes=3, interval_s=60)
    starts = session.arrays["egfp_500ms"]["seconds"].values.min(axis=(1, 2))
    np.testing.assert_allclose(starts, [0, 60, 120])

    # A pass longer than the interval: the next one starts right away.
    session = run(rig, tmp_path / "c", passes=2, interval_s=1)
    starts = session.arrays["egfp_500ms"]["seconds"].values.min(axis=(1, 2))
    np.testing.assert_allclose(starts, [0, 4])


def test_quit_keeps_the_passes_so_far_and_closes_the_shutter(rig, tmp_path):
    rig.stop_after = 5
    session = run(rig, tmp_path / "b", passes=3)
    imgs = session.arrays["egfp_500ms"]
    assert np.isnan(imgs["seconds"].values).sum() == 12 - 5
    assert rig.dev("shutter").state == "closed"
    assert rig.driver.written == set()


def test_bad_arguments_fail_before_anything_moves(rig, tmp_path):
    with pytest.raises(ValueError, match="'egfp' is not one of"):
        run(rig, tmp_path / "b", channel="egfp")
    with pytest.raises(ValueError, match="light_source"):
        run(rig, tmp_path / "b", light_source="lamp")
    with pytest.raises(ValueError, match="'kinetics'"):
        run(rig, tmp_path / "b", camera="kinetics")
    with pytest.raises(ValueError, match="not a camera"):
        run(rig, tmp_path / "b", camera="stage")
    with pytest.raises(ValueError, match="passes"):
        run(rig, tmp_path / "b", passes=0)
    (tmp_path / "exists").mkdir()
    with pytest.raises(FileExistsError):
        run(rig, tmp_path / "exists")
    rig.ctrl.devices.remove(rig.dev("shutter"))
    with pytest.raises(ValueError, match="'shutter'"):
        run(rig, tmp_path / "b")
    assert rig.events == [] and rig.runs == []
