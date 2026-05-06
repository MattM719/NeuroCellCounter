#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utils for analysis"""

from math import sqrt
from typing import Literal

import numpy as np
from skimage.segmentation import find_boundaries


def pixels_in_radius(
    img: np.ndarray, center: tuple[int, int], r: float
) -> tuple[np.ndarray, np.ndarray]:
    """List all pixels in radius and their individual range from center

    Arguments:
        img: image array
        center: (i, j) indices
        r: radius

    Return:
        pixel intensities,
        pixel distances from center
    """
    n_rows, n_cols = np.shape(img)
    r = float(r)

    all_rows = np.arange(n_rows, dtype=np.int64)
    all_cols = np.arange(n_cols, dtype=np.int64)

    rows = all_rows[(all_rows >= center[0] - r) * (all_rows <= center[0] + r)]
    cols = all_cols[(all_cols >= center[1] - r) * (all_cols <= center[1] + r)]

    intensities = []
    dists = []

    for i in rows:
        for j in cols:
            dist = sqrt(float(i - center[0]) ** 2.0 + float(j - center[1]) ** 2.0)
            if dist <= r:
                intensities.append(img[i, j])
                dists.append(dist)

    return (np.array(intensities), np.array(dists))


def calc_weighted_intensity(intensities: np.ndarray, distances: np.ndarray) -> float:
    """Calculates distance-weighted intensity"""
    assert len(intensities.shape) == 1 and len(distances.shape) == 1

    sum_dist = np.sum(distances)

    if sum_dist == 0:
        return 0.0

    return float(np.sum(intensities * distances) / np.sum(distances))


def locs_to_row_col(locs: list) -> tuple[np.ndarray, np.ndarray]:
    """separates a list like [(i,j), (i,j), ...]
    returns ([i,i,...], [j,j,...])
    """
    rows = np.array([loc[0] for loc in locs], dtype=np.int64)
    cols = np.array([loc[1] for loc in locs], dtype=np.int64)
    return (rows, cols)


def find_edges(rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """finds edges of region"""
    r_min = np.min(rows)
    r_max = np.max(rows)
    c_min = np.min(cols)
    c_max = np.max(cols)

    img = np.zeros([r_max - r_min + 1, c_max - c_min + 1], dtype=np.int64)
    img[(rows - r_min, cols - c_min)] = 1

    labeled = find_boundaries(img, connectivity=1, mode="inner")
    edge_rows, edge_cols = labeled.nonzero()
    edge_rows += r_min
    edge_cols += c_min

    return (edge_rows, edge_cols)


def sort_regions(
    regions: list[tuple[np.ndarray, np.ndarray]],
) -> dict[
    tuple[int, int], dict[Literal["region", "edges"], tuple[np.ndarray, np.ndarray]]
]:
    """finds the center of mass for each region and saves sorted coords"""
    # ensure type is correct
    if not isinstance(regions[0], tuple):
        raise TypeError(
            "regions should be a list of tuples, with two arrays in each tuple"
        )

    def center_of_mass(rows: np.ndarray, cols: np.ndarray) -> tuple[int, int]:
        """calculates region's center of mass"""
        return (round(np.mean(rows)), round(np.mean(cols)))

    d = {}
    for region in regions:
        # rows_idx, col_idx = locs_to_row_col(region)
        rows_idx, col_idx = region
        com = center_of_mass(rows_idx, col_idx)
        edges = find_edges(rows_idx, col_idx)

        d[com] = {
            "region": region,
            "edges": edges,
        }

    return d
