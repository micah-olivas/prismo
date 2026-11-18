from collections.abc import Callable
from typing import Any

import xarray as xr
from napari import Viewer
from qtpy.QtCore import QTimer

from .session import Relay
from .widgets import (
    CHIP_MAP_WIDGET,
    OWN_PANEL_WIDGETS,
    RIGHT_PANEL_WIDGETS,
    BoundarySelector,
    PositionSelector,
)


class LiveView:
    def __init__(self, viewer: Viewer, relay: Relay, widgets: dict[str, Callable[[Relay], Any]]):
        self._viewer = viewer
        self._relay = relay
        img = self._relay.get("img")
        self._viewer.add_image(img, name="live")
        self._timer = QTimer()
        self._timer.timeout.connect(self.update_img)
        self._timer.start(1000 // 30)
        tabify = False
        for name, widget in widgets.items():
            # The stage and outlet timer go on the right so the left column still fits on screen.
            area = "right" if name in RIGHT_PANEL_WIDGETS else "left"
            self._viewer.window.add_dock_widget(
                widget(self._relay), name=name, tabify=tabify, area=area
            )
            # Filters/shutters and the stage get their own panels; the valve controllers share a
            # tabbed one.
            tabify = name not in OWN_PANEL_WIDGETS

    def update_img(self):
        img = self._relay.get("img")
        self._viewer.layers[0].data = img


class AcquisitionView:
    def __init__(
        self,
        viewer: Viewer,
        relay: Relay,
        file: str,
        widgets: dict[str, Callable[[Relay], Any]],
        tiled: bool = False,
        multi: bool = False,
    ):
        self._viewer = viewer
        self._relay = relay
        self._file = file
        self._live_timer = QTimer()
        self._refresh_timer = QTimer()
        self._arrays: set[str] = set()
        self._imgs: dict[str, xr.DataArray] = {}
        self._contrast_set: set[str] = set()

        if tiled or multi:
            img = self._relay.get("img")
            self._viewer.add_image(img, name="live")
            self._live_timer.timeout.connect(self.update_img)
            self._live_timer.start(1000 // 30)
            if tiled:
                self._viewer.window.add_dock_widget(
                    BoundarySelector(self._relay, self.start_acq),
                    name="Acquisition Boundaries",
                    tabify=False,
                )
            elif multi:
                self._viewer.window.add_dock_widget(
                    PositionSelector(self._relay, self.start_acq),
                    name="Acquisition Positions",
                    tabify=False,
                )
        else:
            self._refresh_timer.timeout.connect(self.refresh)
            self._refresh_timer.start(1000)

        # A click on the chip map moves the stage, so it's only here while positions are picked,
        # and start_acq takes it away before imaging starts.
        self._chip_map = None
        tabify = False
        for name, widget in widgets.items():
            if name == CHIP_MAP_WIDGET and not (tiled or multi):
                continue
            # The stage and outlet timer go on the right so the left column still fits on screen.
            area = "right" if name in RIGHT_PANEL_WIDGETS else "left"
            panel = widget(self._relay)
            self._viewer.window.add_dock_widget(panel, name=name, tabify=tabify, area=area)
            if name == CHIP_MAP_WIDGET:
                self._chip_map = panel
            # Filters/shutters and the stage get their own panels; the valve controllers share a
            # tabbed one.
            tabify = name not in OWN_PANEL_WIDGETS

    def start_acq(self, *args: Any):
        if self._chip_map is not None:
            self._viewer.window.remove_dock_widget(self._chip_map)
            self._chip_map = None
        self._live_timer.disconnect()
        self._live_timer.stop()
        self._viewer.layers.remove("live")

        self._relay.post("start_acq", *args)
        self._refresh_timer.timeout.connect(self.refresh)
        self._refresh_timer.start(1000)

    def refresh(self):
        arrays = self._relay.get("arrays")
        new_arrays = arrays - self._arrays
        self._arrays = arrays.union(self._arrays)
        for arr in new_arrays:
            xp = xr.open_zarr(self._file, group=arr)
            xp = xp["tile"].assign_attrs(xp.attrs)
            img = tiles_to_image(xp)

            layer_names = arr
            if "channel" in img.dims:
                layer_names = [f"{arr}: {c}" for c in xp.coords["channel"].to_numpy()]

            viewer_dims = self._viewer.dims.axis_labels[:-2] + ("y", "x")
            img = img.expand_dims([d for d in viewer_dims if d not in img.dims])
            img = img.transpose("channel", ..., *viewer_dims, missing_dims="ignore")

            self._viewer.add_image(
                img,
                channel_axis=0 if "channel" in img.dims else None,
                name=layer_names,
                multiscale=False,
                cache=False,
            )
            new_dims = tuple(d for d in img.dims if d not in viewer_dims and d != "channel")
            self._viewer.dims.axis_labels = new_dims + viewer_dims
            # Make sure new dimension sliders get initialized to be 0.
            self._viewer.dims.current_step = (0,) * len(new_dims) + self._viewer.dims.current_step[
                -len(new_dims) :
            ]
            # Save this array so we can set the contrast limits once a nonzero element gets added.
            self._imgs[arr] = img

        """
        # Set contrast limits when a layer gets updated for the first time.
        for arr, img in self._imgs.items():
            if "channel" in img.dims:
                for c in img.coords["channel"].to_numpy():
                    layer_name = f"{arr}: {c}"
                    subimg = img.sel(channel=c)
                    if layer_name not in self._contrast_set and subimg.any():
                        self._viewer.layers[layer_name].contrast_limits = (
                                0, subimg.max().to_numpy())
                        self._contrast_set.add(layer_name)
            else:
                if arr not in self._contrast_set and img.any():
                    self._viewer.layers[arr].contrast_limits = (0, img.max().to_numpy())
                    self._contrast_set.add(arr)
        """

        # Update each of the image layers.
        for layer in self._viewer.layers:
            layer.refresh()

    def update_img(self):
        img = self._relay.get("img")
        self._viewer.layers[0].data = img


def tiles_to_image(xp: xr.DataArray) -> xr.DataArray:
    if "overlap" not in xp.attrs:
        return xp.transpose(..., "y", "x")

    if xp.attrs["overlap"] != 0:
        overlap_y = int(round(xp.attrs["overlap"] * xp.shape[-2]))
        overlap_x = int(round(xp.attrs["overlap"] * xp.shape[-1]))
        img = xp[..., :-overlap_y, :-overlap_x]
    else:
        img = xp

    if "row" in img.dims:
        img = img.transpose("row", "y", ...)
        img = xr.concat(img, dim="y")
    if "col" in img.dims:
        img = img.transpose("col", "x", ...)
        img = xr.concat(img, dim="x")

    img = img.transpose(..., "y", "x")
    return img
