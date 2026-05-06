#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Identifies cells in a preprocessed microscopy image"""

from math import sqrt

# from typing import Any, Optional
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# from skimage.io import imread, imshow
# from skimage.color import rgb2gray, label2rgb
from skimage.feature import blob_dog, blob_log  # , blob_doh

# from skimage.morphology import erosion, dilation, opening, closing
# from skimage.measure import label, regionprops

from .utils import pixels_in_radius, calc_weighted_intensity
from utils.models import Cell


def apply_blob_log_REDUNDANT(img: np.ndarray, **kwargs) -> np.ndarray:
    """Blob detection using Laplacian of Gaussian (log)

    Arguments:
        img: cleaned and preprocesed image. Should be masked to reduce false positives

    Kwargs:
        min_sigma: float, minimum sigma value, default = 2
        max_sigma: float, maximum sigma value, default = 8
        num_sigma: int, number of sigmas to test = 20
        threshold: float, absolute threshold for cell detection, default = 0.1
        overlap: float, amount cells can overlap, default = 0.5,

    Returns:
        (n x 3) array with n rows of (i, j, sigma) representing the coordinates of each
        blob and the gaussian value of sigma that detected the blob
    """
    params = {
        "min_sigma": 2,
        "max_sigma": 8,
        "num_sigma": 20,
        "threshold": 0.1,
        "threshold_rel": 0,
        "overlap": 0.6,
    }
    params.update(kwargs)

    blobs_log = blob_log(
        img,
        **params,
    )
    # blobs_log[:, 2] *= sqrt(2)

    if False:
        plt.close("all")
        fig = plt.figure()
        ax = fig.gca()
        ax.imshow(img, origin="lower", cmap="gray")
        for i in range(blobs_log.shape[0]):
            [y, x] = blobs_log[i, :2]
            r = blobs_log[i, 2]
            c = plt.Circle((x, y), r, color="r", linewidth=2, fill=False, alpha=0.6)
            ax.add_patch(c)
        plt.show()
        plt.close(fig)

    return blobs_log


def apply_blob_dog(img: np.ndarray) -> np.ndarray:
    """Blob detection using Difference of Gaussian (dog)

    More computationally efficient than Laplacian of Gaussian

    Arguments:
        img: cleaned and preprocesed image. Should be masked to reduce false positives

    Returns:
        (n x 3) array with n rows of (x, y, sigma) representing the coordinates of each
        blob and the gaussian value of sigma that detected the blob
    """
    blobs_dog = blob_dog(img, max_sigma=30, threshold=0.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    return blobs_dog


def find_blob_params(
    img: np.ndarray, blobs_info: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finds pixels

    Args:
        img: image
        blobs_info: (n x 3) array of [i, j, sigma/radius]

    Returns:
        average intensity,
        weighted intensity,
        total intensity,
    """
    n, _ = np.shape(blobs_info)
    avg_intensity = np.zeros(n, dtype=np.float64)
    weighted_intensity = np.zeros(n, dtype=np.float64)
    total_intensity = np.zeros(n, dtype=np.float64)

    for i in range(n):
        center = (blobs_info[i, 0], blobs_info[i, 1])
        intensities, dists = pixels_in_radius(img, center, blobs_info[i, 2])
        if len(intensities) > 0:
            avg_intensity[i] = np.sum(intensities)
            weighted_intensity[i] = calc_weighted_intensity(intensities, dists)
            total_intensity[i] = np.mean(intensities)

    return (avg_intensity, weighted_intensity, total_intensity)


def optimize_blob_kwargs(meta: dict, max_value: float) -> dict:
    """Optimize blob parameters"""
    x_px_width, y_px_width, _ = meta["axes_calibration"]
    x_px_width = float(x_px_width)
    y_px_width = float(y_px_width)
    if abs(x_px_width - y_px_width) > 1e-8:
        print("Confirm axes_calibration is stored as (x,y,z), not (i,j,k)")
        warnings.warn(
            f"different x, y dimensions ({x_px_width}, {y_px_width})",
            category=UserWarning,
        )

    px_width = np.mean([x_px_width, y_px_width])
    if abs(px_width - 0.8631674575031096) > 1e-3:
        warnings.warn(
            "Blob kwargs are only optimized for a pixel width of ~0.86 micrometers",
            category=UserWarning,
        )

    # All of these values are chosen semi-arbitrarily and can be adjusted.
    # The scaling factors are intended to convert pixel length (um) back to some number
    # of pixels.
    min_sigma_ = 2.2  # microns
    max_sigma_ = 4.2  # microns
    n_step_ = round((max_sigma_ - min_sigma_) / 0.1) + 1
    params = {  # expected pixel edge length is 0.8631674575031096 micrometers
        "min_sigma": float(min_sigma_ / px_width),  # 1.5 (init: 2 @ 0.86 -> 2.3)
        "max_sigma": float(max_sigma_ / px_width),  # 4.5 (init: 8 @ 0.86 -> 9.3)
        # "min_sigma": float(2 * px_width),  # 2 @ 0.86 -> 2.3
        # "max_sigma": float(8 * px_width),  # 8 @ 0.86 -> 9.3
        # "min_sigma": float(2.1 * px_width),  # 2 @ 0.86 -> 2.3
        # "max_sigma": float(9.3 * px_width),  # 8 @ 0.86 -> 9.3
        "num_sigma": n_step_,  # previously 25
        "threshold": 0.005 * max_value,  # absolute intensity a peak must exceed
        "threshold_rel": 0,  # relative intensity, between 0 and 1
        "overlap": 0.7,  # 0.6 @ 0.86 -> 0.695
    }

    return params


def filter_cells(cells_found: np.ndarray, criteria: np.ndarray) -> np.ndarray:
    """Filter cells

    Args:
        cells_found: array of (n_cells x 3)
        criteria: measure by which to filter cells

    Return:
        indices of cells to include
    """
    n_cells, _ = np.shape(cells_found)
    hist, bins = np.histogram(criteria, bins=len(np.unique(criteria)))

    x_vals = bins[:-1]

    peaks, info = find_peaks(hist)

    if len(peaks) == 0:
        thresh = 0.0
    elif len(peaks) == 1:
        thresh = float(peaks[0] / 2)
    else:
        pk0 = int(peaks[0])
        pk1 = int(peaks[1])
        thresh = x_vals[np.argmin(hist[pk0:pk1+1]) + pk0]

    indices = np.arange(n_cells, dtype=np.int64)[criteria > thresh]

    return indices


def find_cells(
    img: np.ndarray, img_norm: np.ndarray, meta: dict, include_all: bool = False
) -> list | tuple[list, dict]:
    """Finds cells within preprocessed image

    Parameters:
    ----------
    img: preprocessed image

    meta: dict of meta info for image


    Returns:
    -------
    list of Cell models

    Optionally, returns this dict:
    - all_cells: np.ndarray (n x 3) of [i, j, sigma] for every blob detected
    - all_avg_intensities: np.ndarray (n,) of average intensity within sigma for each
        blob
    - all_weighted_intensities: np.ndarray (n,) of weighted average intensity of each
        blob
    - all_total_intensities: np.ndarray (n,) of total intensity within sigma for each
        blob
    - acceptable_indices: np.ndarray (m,) of cell indices deemed true positives
    - cells: np.ndarray (m x 3) of [i, j, sigma] for "acceptable" blobs
    - avg_intensities: np.ndarray (m,) of average intensity within sigma for
        "acceptable" blobs
    - weighted_intensities: np.ndarray (m,) of weighted average intensity of
        "acceptable" blobs
    - total_intensities: np.ndarray (m,) of total intensity within sigma for
        "acceptable" blobs
    """
    # _compare_blob_methods(masked_img, mask)
    px_height, px_width, _ = meta["axes_calibration"]
    px_height = float(px_height)
    px_width = float(px_width)
    blob_kwargs = optimize_blob_kwargs(meta, np.max(img.copy().flatten()))
    cells_found = blob_log(img, **blob_kwargs)
    avg_intensity, weighted_intensity, total_intensity = find_blob_params(
        img_norm, cells_found
    )

    # filter cells to remove some false positives
    acceptable_indices = filter_cells(cells_found, total_intensity)

    cells = []
    for i, idx in enumerate(acceptable_indices):
        cells.append(
            Cell(
                uid=i,
                center=tuple(cells_found[idx, :2].astype(np.int64).tolist()),
                sigma_px=float(cells_found[idx, 2]),
                avg_intensity=float(avg_intensity[idx]),
                weighted_intensity=float(weighted_intensity[idx]),
                total_intensity=float(total_intensity[idx]),
                pixel_height=px_height,
                pixel_width=px_width,
                image_dims=tuple(meta["voxel_count"][:2]),
            )
        )

    # if False:
    #     fig, axes = plot_blobs(
    #         (img + 50),
    #         cells_found[acceptable_indices, :],
    #         side_by_side=True,
    #         show=False,
    #     )
    #     radii = cells_found[:, -1].copy().flatten()
    #     fig1, ax1 = plot_cell_histogram(
    #         radii, avg_intensity, weighted_intensity, total_intensity, show=False
    #     )
    #     plt.show()
    #     plt.close()

    if include_all:
        d = {
            "all_cells": cells_found.copy(),
            "all_avg_intensities": avg_intensity.copy(),
            "all_weighted_intensities": weighted_intensity.copy(),
            "all_total_intensities": total_intensity.copy(),
            "acceptable_indices": acceptable_indices.copy(),
            "cells": np.array(cells_found[acceptable_indices, :]),
            "avg_intensities": np.array(avg_intensity[acceptable_indices]),
            "weighted_intensities": np.array(weighted_intensity[acceptable_indices]),
            "total_intensities": np.array(total_intensity[acceptable_indices]),
        }
        return (cells, d)

    return cells
