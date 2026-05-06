#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simple filters for image processing or filtering lists"""

import os
from pathlib import Path
from typing import Any, Optional

from .models import Cell


def filter_none(value: Optional[Any]) -> bool:
    """Returns True if value is not None"""
    return value is not None


def filter_tile_scan(path: str | Path) -> bool:
    """Returns false if tilescan is in the path name"""
    if isinstance(path, str):
        name = str(os.path.basename(path)).lower()
    elif isinstance(path, Path):
        name = str(path.name).lower()
    return "tilescan" not in name and "_ts" not in name and "4x" not in name


def filter_edge(cell: Cell) -> bool:
    """Returns True if cell does not border an edge"""
    return not cell.is_touching_edge


def filter_small_sigma(cell: Cell) -> bool:
    """Returns True if sigma is a large enough number of pixels"""
    return cell.sigma_um > 2 * 0.8631674575031096


def filter_small_radius(cell: Cell) -> bool:
    """Returns True if ideal radius > threshold

    Expect true nuclei to be ≥5 µm in diameter
    """
    return cell.ideal_radius > 2.25


def filter_low_intensity(cell: Cell) -> bool:
    """Returns True if average intensity"""
    return cell.avg_intensity > 0.05


def filter_no_eccentricity(cell: Cell) -> bool:
    """filters out cells with eccentricity < 1"""
    return bool(cell.eccentricity >= 1)


def filter_low_pixels(cell: Cell) -> bool:
    """filters out cells with too few (<9) pixels"""
    return bool(cell.n_pixels >= 16)


def filter_cells_general(cell: Cell) -> bool:
    """Filters out many types of false positives"""
    result = bool(filter_no_eccentricity(cell) and filter_low_pixels(cell))
    return result
