#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Matches cells and regions"""

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

# import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation import watershed  # , find_boundaries
from sklearn.linear_model import LogisticRegression

from utils.models import Cell, Region

# from utils.tools import retrieve_cell_region_uid
from .utils import calc_weighted_intensity, find_edges  # , sort_regions


def _test_cells(region: Region, cells: list[Cell]) -> Union[List[Cell], List[Cell]]:
    """adds cells to region and return unmatched cells"""
    region_id = region.uid
    used: List[Cell] = []
    unused: List[Cell] = []
    for cell in cells:
        if region.test_cell(cell):
            cell.region_id = region_id
            region.add_cell(cell)
            used.append(cell)
        else:
            unused.append(cell)

    assert len(cells) == len(used) + len(unused)

    return used, unused


def find_cell_regions(
    regions: List[Region], cells: List[Cell]
) -> Union[List[Region], List[Cell]]:
    """assigns each nucleus to a region. Approximates boundary of each nucleus"""
    remaining_cells = [*cells]
    for i, region in enumerate(regions):
        updated_cells, remaining_cells = _test_cells(region, remaining_cells)

        # update cells
        for updated_cell in updated_cells:
            updated_cell: Cell = updated_cell
            cells[updated_cell.uid] = updated_cell

    # drop any unused cells. They must be false positives
    for cell in sorted(remaining_cells, reverse=True):
        cell: Cell = cell
        cells.pop(cell.uid)

    return regions, cells


def _recalculate_cell_params(
    img: np.ndarray,
    locs: tuple[np.ndarray, np.ndarray],
    center: tuple[int, int],
    pixel_dims: tuple[float, float],
) -> Dict[Literal["avg_intensity", "weighted_intensity", "total_intensity"], float]:
    """Calculates total, average, and weighted average intensities of cell"""
    intensities = img[locs]
    distances = np.sqrt(
        (pixel_dims[0] * (locs[0] - center[0])) ** 2.0
        + (pixel_dims[1] * (locs[1] - center[1])) ** 2.0
    )

    area = float(locs[0].shape[0] * pixel_dims[0] * pixel_dims[1])
    avg_intensity = float(np.mean(intensities))

    params = {
        "avg_intensity": avg_intensity,
        "weighted_intensity": calc_weighted_intensity(intensities, distances),
        "total_intensity": area * avg_intensity,
    }
    return params


# function to update a cell instance once it's location has been tuned
def update_cell(
    normalized_img: np.ndarray, cell: Cell, locs: tuple[np.ndarray, ...], **kwargs
) -> Cell:
    """updates a cell"""
    cell.pixel_locs = locs
    cell.edge_locs = find_edges(*locs)
    cell.update_params(
        **_recalculate_cell_params(
            normalized_img,
            locs=locs,
            center=cell.center,
            pixel_dims=(cell.pixel_height, cell.pixel_width),
        ),
        **kwargs,
    )
    return cell


def gaussian_kernel(
    dims: Tuple[int, int], center: Tuple[int, int], sigma: float
) -> np.ndarray:
    """creates a centered, 2D Gaussian kernel with standard deviation sigma

    dims: (n_rows, n_cols)
    """
    n_rows, n_cols = dims

    Rows, Cols = np.meshgrid(
        np.arange(n_rows) - center[0],
        np.arange(n_cols) - center[1],
        indexing="ij",
    )
    d = np.sqrt(np.square(Rows) + np.square(Cols))

    return np.exp(-(d**2.0 / (2.0 * sigma**2.0)))


def _apply_gaussian_cell_distances(
    mask: np.ndarray, region: Optional[Region], cells: List[Cell]
) -> np.ndarray:
    """Imposes gaussian distances from cells"""
    # find rectangular region domain
    if region is None:
        rs, cs = [], []
        for cell in cells:
            r, c = cell.pixel_locs
            rs.append(r.flatten())
            cs.append(c.flatten())
        rows = np.vstack(tuple(rs), dtype=np.int64).flatten()
        cols = np.vstack(tuple(cs), dtype=np.int64).flatten()
    elif isinstance(region, Region):
        rows, cols = region.region_locs
    else:
        raise TypeError(
            f"region must be type Region or None. Type {type(region)} is unacceptable."
        )
    row_min = int(np.min(rows))
    row_max = int(np.max(rows))
    col_min = int(np.min(cols))
    col_max = int(np.max(cols))
    row_radius = row_max - row_min + 1
    col_radius = col_max - col_min + 1
    dims = (row_radius, col_radius)

    d = np.zeros_like(mask, dtype=np.float64)

    for cell in cells:
        di = cell.center[0] - row_min + 1
        dj = cell.center[1] - col_min + 1
        kernel = gaussian_kernel(dims=dims, center=(di, dj), sigma=cell.sigma_px)
        d[row_min:row_max+1, col_min:col_max+1] += kernel[:, :]

    return d * mask


def trim_labels(img: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Trims labeled regions"""
    center_locs: list[tuple[int, int]] = [
        (int(i), int(j)) for i, j in zip(*centers.nonzero())
    ]

    def get_radius_logistic(
        distances: np.ndarray, intensities: np.ndarray, center_intensity: float
    ) -> Optional[float]:
        """Uses a logistic regression model to find the maximum radius of the cell"""
        classes = np.where(
            intensities > 0.15 * center_intensity,
            np.ones_like(intensities, dtype=np.int64),
            np.zeros_like(intensities, dtype=np.int64),
        )

        if np.min(classes) == np.max(classes):
            return None

        model = LogisticRegression()
        model.fit(distances.reshape((-1, 1)), classes)

        coef = float(model.coef_.flatten())
        intercept = float(model.intercept_)
        x = np.unique(distances)
        y = 1.0 / (1 + np.exp(-1 * (intercept + coef * x)))

        if y[0] <= 0.5 or y[-1] >= 0.5:
            return None

        inner = x[y >= 0.5]
        return float(inner[-1])

    centers_mapped: dict[int, tuple[int, int]] = {}
    for i, j in center_locs:
        label: int = int(labels[i, j])
        centers_mapped[label] = (i, j)
        region_mask: np.ndarray = labels == label  # 2D mask
        region_is, region_js = region_mask.nonzero()
        distances: np.ndarray = np.sqrt((region_is - i) ** 2.0 + (region_js - j) ** 2.0)
        pixels = img[region_mask]  # flat array of pixel intensities)

        max_dist = int(np.max(distances) // 1)
        if len(pixels) < 7 or max_dist < 2:
            continue

        center_intensity = np.mean(pixels[distances <= 2])
        radius = get_radius_logistic(distances, pixels, center_intensity)
        if radius is None:
            continue

        # assign labels outside of intense radius to background
        labels[region_is[distances > radius], region_js[distances > radius]] = 0
        if radius <= 3:
            continue

        borderzone_index_locs = (distances <= radius) * (distances > radius - 2)
        bz_is = region_is[borderzone_index_locs]
        bz_js = region_js[borderzone_index_locs]
        bz_intensities = pixels[borderzone_index_locs]
        threshold = min([center_intensity, np.median(bz_intensities)])
        bz_drop = bz_intensities < 0.5 * threshold
        if np.any(bz_drop):
            labels[bz_is[bz_drop], bz_js[bz_drop]] = 0

    return labels


def _update_cell_locs(cell: Cell, locs: tuple[np.ndarray, np.ndarray]) -> Cell:
    """updates cell"""
    cell.pixel_locs = locs
    cell.edge_locs = find_edges(*locs)

    return cell


def fit_cells_in_regions(
    img: np.ndarray,
    norm_img: np.ndarray,
    region: Optional[Region],
    cells_by_uid: dict[int, Cell],
    mask: Optional[np.ndarray] = None,
    pixel_threshold: int = 0,
) -> List[Cell | int]:
    """updates cell boundaries within a region"""
    # get list of cells contained in the region
    cell_ids = region.cell_ids
    img_dim = img.shape

    cells: list[Cell] = [cells_by_uid[uid] for uid in cell_ids]

    # special case: no cells in region:
    if len(cells) == 0:
        return cells

    mask = np.zeros([*img_dim], dtype=np.int64)
    mask[region.region_locs] = 1
    masked_img = img * mask
    # distance = -1 * ndi.distance_transform_edt(masked_img)
    distance = -1 * _apply_gaussian_cell_distances(
        mask,
        region=region,
        cells=cells,
    )

    # define maxima as cell centers
    center_coords = np.array([list(c.center) for c in cells], dtype=np.int64)
    centers = np.zeros_like(img, dtype=bool)
    centers[tuple(center_coords.T)] = True
    markers, _ = ndi.label(centers)

    labels = watershed(distance, markers, mask=mask)
    # updater(labels)#FIXME: delete

    label_vals = labels.flatten()
    label_vals = np.unique(label_vals[label_vals > 0])

    labels = trim_labels(img=masked_img, labels=labels, centers=centers)
    for i in range(len(cells)):
        cell = cells[i]
        center_i, center_j = cell.center
        cells[i] = _update_cell_locs(
            cell, (labels == labels[center_i, center_j]).nonzero()
        )

    # assign labels (sets of locs) to cells
    untested_cells = list(range(len(cells)))
    for label_val in label_vals:
        locs = np.where(labels == label_val)
        locs_set = {(int(i), int(j)) for i, j in zip(*locs)}
        found = False
        for i in untested_cells:
            if cells[i].center in locs_set:
                if locs[0].shape[0] > pixel_threshold:
                    cells[i] = update_cell(norm_img, cells[i], locs=locs)
                else:
                    cells[i] = cells[i].uid  # marks cell for deletion
                found = True
                break
        if found:
            untested_cells.remove(i)
        else:
            raise ValueError(f"Unable to locate a cell for {len(locs)} points")

    return cells
