#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Characterizes each cell"""

from typing import Optional
import warnings

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt  # noqa E402
from matplotlib.axes import Axes  # noqa E402
from matplotlib.figure import Figure  # noqa E402
from scipy import ndimage as ndi  # noqa E402
from skimage.measure import perimeter_crofton  # noqa E402
from skimage.morphology import binary_closing  # noqa E402
from sklearn.decomposition import PCA  # noqa E402
from sklearn.cluster import KMeans  # noqa E402

from utils.models import Cell  # noqa E402


def _update_mask(
    mask: np.ndarray,
    values: np.ndarray,
    bounds: Optional[float | list[float]],
    name: Optional[str] = None,
) -> np.ndarray:
    """updates the mask depending on the values and bounds, if supplied"""
    if bounds is None:
        return mask

    if isinstance(bounds, list):
        [low, high] = bounds
        low: Optional[float] = low
        high: Optional[float] = high
        if low is not None:
            mask *= values > low
        if high is not None:
            mask *= values < high

    else:
        warnings.warn(
            "bounds should be supplied as [Optional[minimum], Optional[maximum]], "
            "rather than a single float",
            category=DeprecationWarning,
        )
        if name == "total_intensity":
            mask *= values > bounds
        elif name == "avg_intensity":
            mask *= values > bounds
        elif name == "weighted_intensity":
            mask *= values > bounds
        elif name == "sigma":
            mask *= values < bounds
        elif name == "perimeter":
            mask *= values < bounds
        elif name == "area":
            mask *= values < bounds
        elif name == "eccentricity":
            mask *= values < bounds
        elif name == "ideal_radius":
            mask *= values < bounds
        else:
            raise ValueError(
                "a name must be supplied if bounds are not given as [min,max]"
            )

    return mask


def classify_pyknotic_thresholds(
    cells: list[Cell], plot_histograms: bool = False, show: bool = True, **kwargs
) -> list[Cell] | tuple[list[Cell], Figure]:
    """Labels cells as pyknotic or non-pyknotic"""
    params = {
        "n_stds": 2.5,
        "n_bins": 50,
        "total_intensity": None,
        "avg_intensity": None,
        "weighted_intensity": None,
        "sigma": None,
        "perimeter": None,
        "area": None,
        "eccentricity": None,
        "ideal_radius": None,
    }
    for key in kwargs.keys():
        if key not in params:
            raise KeyError(f"Unexpected kwarg '{key}'")
    params.update(kwargs)

    n_cells = len(cells)

    # get characteristics of dataset
    uids = np.zeros(n_cells, dtype=np.int64)
    centers = np.zeros([n_cells, 2], dtype=np.int64)
    total_intensities = np.zeros(n_cells, dtype=np.float64)
    avg_intensities = np.zeros(n_cells, dtype=np.float64)
    weighted_intensities = np.zeros(n_cells, dtype=np.float64)
    sigmas = np.zeros(n_cells, dtype=np.float64)
    perimeters = np.zeros(n_cells, dtype=np.float64)
    areas = np.zeros(n_cells, dtype=np.float64)
    ideal_radii = np.zeros(n_cells, dtype=np.float64)
    eccentricities = np.zeros(n_cells, dtype=np.float64)

    for i, cell in enumerate(cells):
        uids[i] = cell.uid
        centers[i, :] = np.array([*cell.center], dtype=np.int64)
        total_intensities[i] = cell.total_intensity
        avg_intensities[i] = cell.avg_intensity
        weighted_intensities[i] = cell.weighted_intensity
        sigmas[i] = cell.sigma_um
        perimeters[i] = cell.n_edges
        areas[i] = cell.n_pixels
        ideal_radii[i] = cell.ideal_radius
        eccentricities[i] = cell.eccentricity

    characteristics = {
        "total_intensity": total_intensities,
        "avg_intensity": avg_intensities,
        "weighted_intensity": weighted_intensities,
        "sigma_um": sigmas,
        "perimeter": perimeters,
        "area": areas,
        "eccentricity": eccentricities,
        "ideal_radius": ideal_radii,
    }

    # plot histograms
    if plot_histograms:
        mosaic = [
            ["total", "avg", "weighted", "sigma"],
            ["perim", "area", "eccentricity", "ideal_radius"],
        ]
        fig, ax = plt.subplot_mosaic(mosaic)
        fig: Figure = fig
        ax: dict[str, Axes] = ax
        fig.set_figheight(8)
        fig.set_figwidth(12)

        for key, values in characteristics.items():
            loc = str(key).replace("_intensity", "").replace("eter", "")
            ax[loc].hist(values, params["n_bins"])
            if isinstance((vals := params[key]), list):
                for v, c in zip(vals, ["b", "r"]):
                    if v is None:
                        continue
                    ax[loc].axvline(v, color=c)
            elif vals is not None:
                ax[loc].axvline(vals, color="k")

        ax["total"].set_xlabel("Total")
        ax["avg"].set_xlabel("Average")
        ax["weighted"].set_xlabel("Weighted")
        ax["sigma"].set_xlabel("Sigma")
        ax["perim"].set_xlabel("Perimeter")
        ax["area"].set_xlabel("Area")
        ax["eccentricity"].set_xlabel("Eccentricity")
        ax["ideal_radius"].set_xlabel("Ideal Radius")

        fig.tight_layout()
        if show:
            plt.show()
            plt.close()

    mask = np.ones(n_cells, dtype=bool)

    for name, values in characteristics.items():
        bounds = params[name]
        mask = _update_mask(mask, values=values, bounds=bounds, name=name)

    pyknotic_uids = uids[mask].tolist()

    for i, cell in enumerate(cells):
        cell.is_pyknotic = bool(uids[i] in pyknotic_uids)

    if plot_histograms and not show:
        return (cells, fig)

    return cells


def assign_pyknotics(
    cells: list[Cell],
    is_pyknotic: np.ndarray,
    plot_histograms: bool = False,
    show: bool = True,
    **kwargs,
) -> list[Cell] | tuple[list[Cell], Figure]:
    """Labels cells as pyknotic or non-pyknotic based on the boolean array `is_pyknotic`
    """
    params = {
        "n_stds": 2.5,
        "n_bins": 50,
    }
    for key in kwargs.keys():
        if key not in params:
            raise KeyError(f"Unexpected kwarg '{key}'")
    params.update(kwargs)

    assert isinstance(is_pyknotic, np.ndarray) and len(is_pyknotic.shape) == 1
    assert is_pyknotic.shape[0] == len(
        cells
    ), "Must explicitly supply the pyknotic status of each cell"

    n_cells = len(cells)

    # get characteristics of dataset
    uids = np.zeros(n_cells, dtype=np.int64)
    centers = np.zeros([n_cells, 2], dtype=np.int64)
    total_intensities = np.zeros(n_cells, dtype=np.float64)
    avg_intensities = np.zeros(n_cells, dtype=np.float64)
    weighted_intensities = np.zeros(n_cells, dtype=np.float64)
    sigmas = np.zeros(n_cells, dtype=np.float64)
    perimeters = np.zeros(n_cells, dtype=np.float64)
    areas = np.zeros(n_cells, dtype=np.float64)
    ideal_radii = np.zeros(n_cells, dtype=np.float64)
    eccentricities = np.zeros(n_cells, dtype=np.float64)

    for i, cell in enumerate(cells):
        uids[i] = cell.uid
        centers[i, :] = np.array([*cell.center], dtype=np.int64)
        total_intensities[i] = cell.total_intensity
        avg_intensities[i] = cell.avg_intensity
        weighted_intensities[i] = cell.weighted_intensity
        sigmas[i] = cell.sigma_um
        perimeters[i] = cell.n_edges
        areas[i] = cell.n_pixels
        ideal_radii[i] = cell.ideal_radius
        eccentricities[i] = cell.eccentricity

    characteristics = {
        "total_intensity": total_intensities,
        "avg_intensity": avg_intensities,
        "weighted_intensity": weighted_intensities,
        "sigma_um": sigmas,
        "perimeter": perimeters,
        "area": areas,
        "eccentricity": eccentricities,
        "ideal_radius": ideal_radii,
    }

    # plot histograms
    if plot_histograms:
        mosaic = [
            ["total", "avg", "weighted", "sigma_um"],
            ["perim", "area", "eccentricity", "ideal_radius"],
        ]
        fig, ax = plt.subplot_mosaic(mosaic)
        fig: Figure = fig
        ax: dict[str, Axes] = ax
        fig.set_figheight(8)
        fig.set_figwidth(12)

        for key, values in characteristics.items():
            loc = str(key).replace("_intensity", "").replace("eter", "")
            ax[loc].hist(values, params["n_bins"])

        ax["total"].set_xlabel("Total")
        ax["avg"].set_xlabel("Average")
        ax["weighted"].set_xlabel("Weighted")
        ax["sigma_um"].set_xlabel("Sigma")
        ax["perim"].set_xlabel("Perimeter")
        ax["area"].set_xlabel("Area")
        ax["eccentricity"].set_xlabel("Eccentricity")
        ax["ideal_radius"].set_xlabel("Ideal Radius")

        fig.tight_layout()
        if show:
            plt.show()
            plt.close()

    for i, cell in enumerate(cells):
        cell.is_pyknotic = bool(is_pyknotic[i])

    if plot_histograms and not show:
        return (cells, fig)

    return cells


def classify_pyknotic_clustering(
    cells: list[Cell],
    pca: PCA,
    kmeans: KMeans,
    pyknotic_class: int,
    plot_clusters: bool,
    show: bool = True,
    **kwargs,
) -> list | tuple[list, tuple[Figure, tuple[Axes, ...]]]:
    """Labels cells as pyknotic or non-pyknotic"""
    params = {
        "n_stds": 2.5,
        "n_bins": 50,
        "total_intensity": None,
        "avg_intensity": None,
        "weighted_intensity": None,
        "sigma": None,
        "perimeter": None,
        "area": None,
        "eccentricity": None,
        "ideal_radius": None,
    }
    params.update(kwargs)

    n_cells = len(cells)

    # get characteristics of dataset
    uids = np.zeros(n_cells, dtype=np.int64)
    centers = np.zeros([n_cells, 2], dtype=np.int64)
    total_intensities = np.zeros(n_cells, dtype=np.float64)
    avg_intensities = np.zeros(n_cells, dtype=np.float64)
    weighted_intensities = np.zeros(n_cells, dtype=np.float64)
    sigmas = np.zeros(n_cells, dtype=np.float64)
    perimeters = np.zeros(n_cells, dtype=np.float64)
    areas = np.zeros(n_cells, dtype=np.float64)
    ideal_radii = np.zeros(n_cells, dtype=np.float64)
    eccentricities = np.zeros(n_cells, dtype=np.float64)

    for i, cell in enumerate(cells):
        uids[i] = cell.uid
        centers[i, :] = np.array([*cell.center], dtype=np.int64)
        total_intensities[i] = cell.total_intensity
        avg_intensities[i] = cell.avg_intensity
        weighted_intensities[i] = cell.weighted_intensity
        sigmas[i] = cell.sigma_um
        perimeters[i] = cell.n_edges
        areas[i] = cell.n_pixels
        ideal_radii[i] = cell.ideal_radius
        eccentricities[i] = cell.eccentricity

    data = np.vstack(
        (
            sigmas,
            avg_intensities,
            total_intensities,
            ideal_radii,
            eccentricities,
            perimeters,
            areas,
        ),
        dtype=np.float64,
    ).T

    transformed = pca.transform(data)
    classified = kmeans.predict(transformed)

    for i, cell in enumerate(cells):
        cell.set_pyknotic(bool(classified[i] == pyknotic_class))

    if plot_clusters:
        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(
            transformed[classified != pyknotic_class, 0],
            transformed[classified != pyknotic_class, 1],
            facecolors="none",
            edgecolors="b",
            alpha=0.2,
            label="Non-Pyknotic",
        )
        ax.scatter(
            transformed[classified == pyknotic_class, 0],
            transformed[classified == pyknotic_class, 1],
            c="r",
            marker="^",
            alpha=0.8,
            label="Pyknotic",
        )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend()

        if show:
            plt.show()
            plt.close(fig)
        else:
            return (cells, (fig, ax))

    return cells


def _pixel_min_max(locs: tuple[np.ndarray, np.ndarray]) -> tuple[tuple[int, int], ...]:
    """get min and max of i and j indices"""
    [(i_min, i_max), (j_min, j_max)] = [
        (np.min(l), np.max(l)) for l in locs  # noqa E741
    ]

    return ((int(i_min), int(i_max)), (int(j_min), int(j_max)))


def fill_holes(cell: Cell) -> tuple[np.ndarray, np.ndarray]:
    """fills holes in cell body and returns updated Cell instance"""
    locs = tuple([l.copy() for l in cell.pixel_locs])  # noqa E741
    (i_min, i_max), (j_min, j_max) = _pixel_min_max(locs)
    ni = i_max - i_min + 1
    nj = j_max - j_min + 1

    i_locs, j_locs = locs
    i_locs -= i_min - 1
    j_locs -= j_min - 1

    mask = np.zeros([ni + 2, nj + 2], dtype=np.int64)
    mask[(i_locs, j_locs)] = 1

    mask_tmp = ndi.binary_fill_holes(mask)
    mask_tmp = binary_closing(mask_tmp, mode="ignore")
    i_new, j_new = mask_tmp.nonzero()
    i_new += i_min - 1
    j_new += j_min - 1

    return (i_new, j_new)


def find_perimeter(cell: Cell) -> float:
    """finds perimeter of a cell, in micrometers, assuming the cell already has no holes
    """
    locs = tuple([l.copy() for l in cell.pixel_locs])  # noqa E741
    (i_min, i_max), (j_min, j_max) = _pixel_min_max(locs)
    ni = i_max - i_min + 1
    nj = j_max - j_min + 1

    i_locs, j_locs = locs
    i_locs -= i_min - 1
    j_locs -= j_min - 1

    mask = np.zeros([ni + 2, nj + 2], dtype=np.int64)
    mask[(i_locs, j_locs)] = 1

    if not cell.is_square_pixels:
        raise ValueError(
            "Not prepared to handle non-square pixels. Will need to resample mask."
        )

    um_per_px = float(np.mean([cell.pixel_height, cell.pixel_width]))

    perimeter = perimeter_crofton(mask, directions=4)

    return um_per_px * float(perimeter)
