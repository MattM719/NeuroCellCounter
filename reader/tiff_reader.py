#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Reads nd2 images and extracts layers of interest"""

from pathlib import Path

from typing import Optional, Any
from xarray import DataArray, Coordinates

import imagej
import numpy as np
import scyjava
from scyjava import config, jimport, JavaMap, JavaList


config.enable_headless_mode()
config.add_repositories(
    {"scijava.public": "https://maven.scijava.org/content/groups/public"}
)
config.endpoints.append("net.imagej:imagej:2.1.0")

DefaultDataset = jimport("net.imagej.DefaultDataset")
DefaultLinearAxis = jimport("net.imagej.axis.DefaultLinearAxis")
DefaultAxisType = jimport("net.imagej.axis.DefaultAxisType")
PlaneSeparatorMetadata = jimport("io.scif.filters.PlaneSeparatorMetadata")
ChannelFillerMetadata = jimport("io.scif.filters.ChannelFillerMetadata")
ChannelFiller = jimport("io.scif.filters.ChannelFiller")
ColorTable8 = jimport("net.imglib2.display.ColorTable8")
DefaultMetaTable = jimport("io.scif.DefaultMetaTable")


def _parse_meta(meta: dict[str, Any], ij) -> dict[str, Any]:
    """Parses dicts of meta data"""
    meta_new = {}
    for key, child in meta.items():
        if isinstance(child, (int, bool)) or child is None:
            c_new = child
        elif isinstance(child, JavaMap):
            for k in child.jobj:
                assert isinstance(k, DefaultAxisType)
                assert isinstance(child[k], int)
            c_new: dict[str, int] = {str(k): int(child[k]) for k in child.jobj}
        elif isinstance(child, JavaList):
            for tmp in child.jobj:
                assert isinstance(tmp, DefaultLinearAxis)
            c_new = "[DefaultLinearAxis]"
        elif isinstance(child, ChannelFillerMetadata):
            table: JavaMap = ij.py.from_java(child.getTable())
            c_new: dict[str, str] = {str(k): str(table[k]).strip() for k in table.jobj}
        elif key == "impl_cls":
            c_new = None
        else:
            tmp = ij.py.from_java(child)
            c_new = None
            if isinstance(tmp, (int, float, str, bool)):
                c_new = tmp
        meta_new[key] = c_new

    return meta_new


def _get_important_meta(global_meta: dict[str, Any], n_ch: int) -> dict[str, Any]:
    """Pulls the important meta from the global metadata dict"""
    meta = {
        "ImageLength": int(global_meta.get("ImageLength", -1)),  # 512
        "ImageWidth": int(global_meta.get("ImageWidth", -1)),  # 512
        "dCalibration": float(global_meta.get("dCalibration", "nan")),
        "dAspect": float(global_meta.get("dAspect", "nan")),
    }

    # validate initial set of values
    if meta["ImageLength"] == -1 or meta["ImageWidth"] == -1:
        raise ValueError("Image Length/Width not found.")
    if meta["dAspect"] != meta["dAspect"]:
        raise NotImplementedError("Aspect ratio ('dAspect') not found.")
    if meta["ImageLength"] != meta["ImageWidth"]:
        raise NotImplementedError("Unprepared for non-square image.")
    if abs(meta["dAspect"] - 1) > 0.0001:
        raise NotImplementedError("Unprepared for a non-unity aspect ratio.")

    # get stain info
    dye_name_keys = [f"CH{i+1}ChannelDyeName" for i in range(n_ch)]
    color_code_keys = [f"CH{i+1}ChannelColor" for i in range(n_ch)]
    dye_names = {x: global_meta.get(x, None) for x in dye_name_keys}
    color_codes = {x: global_meta.get(x, None) for x in color_code_keys}

    # update meta
    meta.update(dye_names)
    meta.update(color_codes)

    return meta


def read_tiff_ij(
    path: str | Path,
) -> tuple[
    np.ndarray, dict[str, Any], tuple[str, dict[str, Any], dict[str, Any], Coordinates]
]:
    """Reads a TIFF file containing ImageJ meta data"""
    ij = imagej.init()
    scyjava.config.add_options("-Xmx6g")
    dataset = ij.io().open(str(path))  # DefaultDataset

    # read to an xarray DataArray
    xr_dataset: DataArray = ij.py.from_java(dataset)

    # get the name of the file
    fname: str = xr_dataset.name

    # verify dimensions are in the expected order
    dims: tuple[str, ...] = xr_dataset.dims  # ('row', 'col', 'ch')
    dims_list = list(dims)
    row_loc = dims_list.index("row")
    col_loc = dims_list.index("col")
    ch_loc = dims_list.index("ch")
    assert len(dims) == 3

    # read image stack, with channels indexed as indicated by `dims`
    img_stack: np.ndarray = xr_dataset.data  # see dims, (row, col, ch)
    images = img_stack.transpose((ch_loc, row_loc, col_loc)).astype(dtype=np.int64)
    n_ch, n_row, n_col = images.shape

    # read the coordinates of each pixel in the image stack
    coords: Coordinates = xr_dataset.coords

    # read the image's meta data
    meta: dict[str, Optional[Any]] = xr_dataset.attrs
    if "tables" in meta:
        _tables = meta.pop("tables")
        assert _tables is None
    if "rois" in meta:
        _rois = meta.pop("rois")
        assert _rois is None

    global_meta_raw = meta.pop("scifio.metadata.global")
    image_meta_raw = meta.pop("scifio.metadata.image")
    assert len(meta) == 0

    global_meta = _parse_meta(global_meta_raw, ij=ij)
    image_meta = _parse_meta(image_meta_raw, ij=ij)
    important_meta = _get_important_meta(global_meta["metadata"], n_ch=n_ch)

    return (images, important_meta, (fname, global_meta, image_meta, coords))


def get_dapi_tiff_ij(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """gets the DAPI stain from a TIFF, ImageJ-based image"""
    images, important_meta, _ = read_tiff_ij(path)

    def filter_dyes(s: str) -> bool:
        """Selects dye key names"""
        return bool("ChannelDyeName" in s)

    def filter_colors(s: str) -> bool:
        """Selects dye colors"""
        return bool("ChannelColor" in s)

    dye_keys = sorted(filter(filter_dyes, important_meta.keys()))
    dye_names = [str(important_meta[k]).upper().strip() for k in dye_keys]

    if "DAPI" not in dye_names:
        raise NameError(f"Could not find a DAPI-stained image: {dye_names}")

    # find DAPI index and corresponding color info
    dapi_idx = dye_names.index("DAPI")
    color_keys = sorted(filter(filter_colors, important_meta.keys()))
    color_codes = [str(important_meta[k]) for k in color_keys]

    meta = {
        dye_keys[dapi_idx]: dye_names[dapi_idx],
        color_keys[dapi_idx]: color_codes[dapi_idx],
        "ImageLength": important_meta["ImageLength"],
        "ImageWidth": important_meta["ImageWidth"],
        "dAspect": important_meta["dAspect"],
        "dCalibration": important_meta["dCalibration"],
    }

    return (np.array(images[dapi_idx, ...], dtype=np.int64), meta)


def main():
    """main"""
    input_path = Path(...)  # FIXME: provide path to image directory for debugging
    input_path /= "IMAGE_NAME.tif"  # FIXME: image file name
    # read_tiff_ij(path=input_path)
    image, meta = get_dapi_tiff_ij(input_path)

    print(f"{image=}")
    print(f"{meta=}")


if __name__ == "__main__":
    main()
