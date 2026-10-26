import os
import weakref
from typing import Any

from pymmcore import CMMCore

import prismo.devices as dev

control = None


def load(
    config: dict[str, dict[str, Any] | str], path: str | None = None, singleton: bool = True
) -> "Control":
    # Make sure we close out the previous control before instantiating a new one.
    global control
    if singleton and control is not None:
        control.close()

    core = CMMCore()
    if path is None:
        if os.name == "nt":
            path = "C:/Program Files/Micro-Manager-2.0"
        else:
            path = "/usr/local/lib/micro-manager"

    os.environ["PATH"] += os.pathsep + path
    core.setDeviceAdapterSearchPaths([path])

    devices = []
    valves = None
    for name, params in config.items():
        # TODO: Pull out valves so config isn't order dependent.
        params: dict[str, Any] = {"device": params} if isinstance(params, str) else params
        device = params.pop("device")
        match device:
            case "asi_stage":
                devices.append(dev.asi.Stage(name, **params))
            case "asi_zstage":
                devices.append(dev.asi.Focus(name, **params))
            case "bsi_camera":
                devices.append(dev.bsi.Camera(name, core, **params))
            case "demo_camera":
                devices.append(dev.demo.Camera(name))
            case "demo_filter":
                devices.append(dev.demo.Filter(name, **params))
            case "demo_stage":
                devices.append(dev.demo.Stage(name))
            case "demo_valves":
                devices.append(dev.demo.Valves(name, **params))
            case "fluidic_sipper":
                devices.append(dev.fluidic.Sipper(name, **params))
            case "kinetix_camera":
                devices.append(dev.kinetix.Camera(name, core, **params))
            case "lambda_filter1":
                devices.append(dev.sutter.Filter(name, core, filter="A", **params))
            case "lambda_filter2":
                devices.append(dev.sutter.Filter(name, core, filter="B", **params))
            case "lambda_filter3":
                devices.append(dev.sutter.Filter(name, core, filter="C", **params))
            case "lambda_shutter1":
                devices.append(dev.sutter.Shutter(name, core, shutter="A", **params))
            case "lambda_shutter2":
                devices.append(dev.sutter.Shutter(name, core, shutter="B", **params))
            case "manual_objective":
                devices.append(dev.manual.Objective(name, **params))
            case "microfluidic_chip":
                devices.append(dev.microfluidic.Chip(name, valves, **params))
            case "microfluidic_valves":
                devices.append(dev.microfluidic.ValveDriver(name, **params))
                valves = devices[-1]
            case "retra_light":
                devices.append(dev.lumencor.RetraLight(name, core, **params))
            case "sola_light":
                devices.append(dev.lumencor.SolaLight(name, core, version="sola", **params))
            case "spectra_light":
                devices.append(dev.lumencor.SolaLight(name, core, version="spectra", **params))
            case "ti_filter1":
                devices.append(dev.ti.Filter(name, core, filter=1, **params))
            case "ti_filter2":
                devices.append(dev.ti.Filter(name, core, filter=2, **params))
            case "ti_lightpath":
                devices.append(dev.ti.LightPath(name, core, **params))
            case "ti_focus":
                devices.append(dev.ti.Focus(name, core))
            case "ti_objective":
                devices.append(dev.ti.Objective(name, core, **params))
            case "ti2_filter1":
                devices.append(dev.ti2.Filter(name, core, filter=1, **params))
            case "ti2_filter2":
                devices.append(dev.ti2.Filter(name, core, filter=2, **params))
            case "ti2_overheadlight":
                devices.append(dev.ti2.OverheadLight(name, core, **params))
            case "ti2_shutter1":
                devices.append(dev.ti2.Shutter(name, core, shutter=1))
            case "ti2_shutter2":
                devices.append(dev.ti2.Shutter(name, core, shutter=2))
            case "ti2_lightpath":
                devices.append(dev.ti2.LightPath(name, core, **params))
            case "ti2_lightselector":
                devices.append(dev.ti2.LightSelector(name, core, **params))
            case "ti2_focus":
                devices.append(dev.ti2.Focus(name, core))
            case "ti2_objective":
                devices.append(dev.ti2.Objective(name, core, **params))
            case "zyla_camera":
                devices.append(dev.zyla.Camera(name, core, **params))
            case _:
                raise ValueError(f"Device {device} is not recognized.")

    if singleton:
        control = Control(core, devices=devices)
        return control
    else:
        return Control(core, devices=devices)


class Control:
    def __init__(self, core: CMMCore, devices: list[Any]):
        # We can't directly set self.devices = devices since our overriden method
        # depends on self.devices being set.
        super().__setattr__("devices", devices)

        self._core = core
        self._core.setTimeoutMs(100000)

        self._camera = None
        for device in self.devices:
            if isinstance(device, dev.Camera):
                self._camera = device
                break

        self._stage = None
        for device in self.devices:
            if isinstance(device, dev.Stage):
                self._stage = device
                break

        self._focus = None
        for device in self.devices:
            if isinstance(device, dev.Focus):
                self._focus = device
                break

        weakref.finalize(self, self.close)

    def wait(self):
        for device in self.devices:
            if isinstance(device, dev.Wait):
                device.wait()

    @property
    def camera(self):
        return self._camera

    @camera.setter
    def camera(self, name: str):
        for device in self.devices:
            if device.name == name and isinstance(device, dev.Camera):
                self._camera = device
                return
        raise ValueError(f"Camera '{name}' not found.")

    def snap(self):
        return self._camera.snap()

    @property
    def px_len(self):
        zoom_total = 1
        for device in self.devices:
            if isinstance(device, dev.Zoom):
                zoom_total *= device.zoom
        return self._camera.px_len / zoom_total

    @property
    def exposure(self):
        return self._camera.exposure

    @exposure.setter
    def exposure(self, new_exposure):
        self._camera.exposure = new_exposure

    @property
    def focus(self):
        return self._focus

    @focus.setter
    def focus(self, name: str):
        for device in self.devices:
            if device.name == name and isinstance(device, dev.Focus):
                self._focus = device
                return
        raise ValueError(f"Focus '{name}' not found.")

    @property
    def z(self):
        return self._focus.z

    @z.setter
    def z(self, new_z):
        self._focus.z = new_z

    @property
    def stage(self):
        return self._stage

    @stage.setter
    def stage(self, name: str):
        for device in self.devices:
            if device.name == name and isinstance(device, dev.Stage):
                self._stage = device
                return
        raise ValueError(f"Stage '{name}' not found.")

    @property
    def x(self):
        return self._stage.x

    @x.setter
    def x(self, new_x):
        self._stage.x = new_x

    @property
    def y(self):
        return self._stage.y

    @y.setter
    def y(self, new_y):
        self._stage.y = new_y

    @property
    def xy(self):
        return self._stage.xy

    @xy.setter
    def xy(self, new_xy):
        self._stage.xy = new_xy

    def __getattr__(self, name):
        for device in self.devices:
            if name == device.name:
                if isinstance(device, dev.State):
                    return device.state
                else:
                    return device
        return self.__getattribute__(name)

    def __setattr__(self, name, value):
        for device in self.devices:
            if name == device.name and isinstance(device, dev.State):
                device.state = value
                return
        super().__setattr__(name, value)

    def close(self):
        self._core.reset()
        for device in self.devices:
            if isinstance(device, dev.Close):
                device.close()
