import numpy as np
import pytest

from prismo.devices import asi


class FakePort:
    """Stands in for the COM port: answers WHERE with fixed counts and records every command."""

    def __init__(self, *args, **kwargs):
        self.sent = []
        self.counts = {"X": 1000.0, "Y": 500.0}

    def write_readline(self, data: bytes) -> bytes:
        cmd = data.decode().strip()
        self.sent.append(cmd)
        if cmd.startswith("WHERE"):
            axes = cmd.split()[1:]
            return (":A " + " ".join(f"{self.counts[a]:g}" for a in axes) + "\r\n").encode()
        return b":A\r\n"

    def close(self):
        pass


@pytest.fixture(autouse=True)
def fake_port(monkeypatch):
    monkeypatch.setattr(asi, "Port", FakePort)


def test_by_default_positions_are_controller_counts():
    stage = asi.Stage("stage")
    np.testing.assert_allclose(stage.xy, [1000, 500])
    assert (stage.x, stage.y) == (1000, 500)
    stage.xy = (10, 20)
    stage += (1, 2)
    stage -= (1, 2)
    assert stage._port.sent[-3:] == [
        "MOVE X=10.0000 Y=20.0000",
        "MOVREL X=1.0000 Y=2.0000",
        "MOVREL X=-1.0000 Y=-2.0000",
    ]
    assert stage.units_per_um == 1.0


def test_counts_per_um_and_flip_x_give_microns_with_x_mirrored():
    # The Ti2 setup: 0.1 um counts and a reversed x axis.
    stage = asi.Stage("stage", counts_per_um=10, flip_x=True)
    np.testing.assert_allclose(stage.xy, [-100, 50])
    assert (stage.x, stage.y) == (-100, 50)
    stage.xy = (-100, 50)
    stage.x = 3
    stage.y = 4
    stage += (5, 2)
    stage -= (5, 2)
    assert stage._port.sent[-5:] == [
        "MOVE X=1000.0000 Y=500.0000",
        "MOVE X=-30.0000",
        "MOVE Y=40.0000",
        "MOVREL X=-50.0000 Y=20.0000",
        "MOVREL X=50.0000 Y=-20.0000",
    ]
    assert stage.units_per_um == 1.0  # positions are already in microns: tiles need no scaling


def test_units_per_um_leaves_positions_in_counts():
    stage = asi.Stage("stage", units_per_um=10)
    np.testing.assert_allclose(stage.xy, [1000, 500])
    assert stage.units_per_um == 10


def test_bad_scaling_settings_are_refused():
    with pytest.raises(ValueError, match="not both"):
        asi.Stage("stage", units_per_um=10, counts_per_um=10)
    with pytest.raises(ValueError, match="positive"):
        asi.Stage("stage", counts_per_um=0)
