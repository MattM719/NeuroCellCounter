#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simple, generic tools"""

from typing import Optional
import warnings

import numpy as np

from .models import Cell, Region


def retrieve_cell_region_uid(objs: list, uid: int) -> Optional[Cell | Region]:
    """Returns cell or region with requested uid from list"""
    for obj in objs:
        obj: Cell | Region = obj
        if int(obj.uid) == uid:
            return obj
    return None


def validate_meta(
    file_name: str, meta: dict, img: np.ndarray, expect_square: bool = True
) -> None:
    """Confirms meta data are being interpretted correctly"""
    # check keys
    keys = list(meta.keys())
    expected = [
        "name",
        "axes_calibrated",
        "axes_calibration",
        "axes_interpretation",
        "voxel_count",
    ]
    for key in expected:
        assert (
            key in keys
        ), f"Could not find '{key}' in meta data for file\n{file_name}\n{meta}"

    # check image dimensions
    dims = np.shape(img)
    if len(dims) != 2:
        raise IndexError(f"not prepared to validate a {len(dims)}-dimensional image")

    assert meta["axes_calibrated"] == (
        True,
        True,
        False,
    ), f"*** {file_name}\nUnexpected axes were calibrated {meta['axes_calibrated']}"

    voxel_count = (int(meta["voxel_count"][0]), int(meta["voxel_count"][1]))

    if dims != voxel_count:
        raise IndexError(
            f"*** {file_name}\nImage shape {dims} did not "
            f"align with voxels from meta {meta['voxel_count']}"
        )
    if voxel_count[0] != voxel_count[1] and expect_square:
        warnings.warn(
            f"Image has non-square dimensions: {meta['voxel_count']}",
            category=UserWarning,
        )

    interpretation = (
        str(meta["axes_interpretation"][0]),
        str(meta["axes_interpretation"][1]),
    )
    if interpretation != ("distance", "distance"):
        raise ValueError(
            f"*** {file_name}\nUnexpected axes interpretation "
            f"{meta['axes_interpretation']}. Should be ('distance','distance')."
        )
    try:
        calibration = (
            float(meta["axes_calibration"][0]),
            float(meta["axes_calibration"][0]),
        )
    except Exception:
        raise ValueError(
            f"*** {file_name}\nInvalid pixel calibration {meta['axes_calibration']}"
        )
    assert (calibration[0] > 0.0) and (
        calibration[1] > 0.0
    ), f"*** {file_name}\nUnacceptable axes calibration {calibration}"
    if not bool(np.isclose(calibration[0], calibration[1])) and expect_square:
        warnings.warn(
            f"Voxels have non-square dimensions: {meta['axes_calibration']}",
            category=UserWarning,
        )

    # successfully completed checks
    return None
