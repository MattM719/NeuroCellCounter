#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Stores types"""

from typing import Literal, Tuple, TypeAlias, TypedDict

from nd2.structures import Color

_axes_interp: TypeAlias = Literal["distance"]


class MetaLike_ND2(TypedDict):
    name: str
    color: Color
    index: int
    emission_lambda: float
    excitation_lambda: float
    objective_magnification: float
    zoom_magnification: float
    pinhole_diameter_um: float
    axes_calibrated: Tuple[bool, ...]
    axes_calibration: Tuple[float, ...]
    axes_interpretation: Tuple[_axes_interp, ...]
    voxel_count: Tuple[int, ...]
