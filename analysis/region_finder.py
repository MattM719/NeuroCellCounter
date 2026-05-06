#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Finds distinct regions"""

from time import time
from typing import Optional

import numpy as np
from skimage.segmentation import flood
from tqdm import tqdm

from utils.decorators import timer_s
from utils.models import Region


def _loc_to_flat(loc: tuple[int, int], N: int) -> int:
    """Assigns each location an index

    M rows, N cols. Assigns indices row-wise
    """
    idx = int(loc[0] * N)
    idx += loc[1]
    return idx


def _flat_to_loc(idx: int, N: int) -> tuple[int, int]:
    """recovers location from index and N

    M rows, N cols. Assigns indices row-wise
    """
    return (int(idx // N), int(idx % N))


@timer_s
def _find_next_region(mask: np.ndarray) -> Optional[tuple[int, int]]:
    """Returns location of next region"""
    i, j = mask.nonzero()
    if len(i) == 0:
        return None
    return (int(i[0]), int(j[0]))


def _find_next_region_iter(
    mask: np.ndarray, idx: int, width: int, total: int
) -> tuple[Optional[tuple[int, int]], int]:
    """Returns location of next region"""
    for k in range(idx, total, 1):
        i, j = int(k // width), int(k % width)
        if mask[i, j] != 0:
            return ((i, j), k)
    return (None, k)


@timer_s
def _grow_region(
    mask: np.ndarray, init: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Uses simple region growing to find region

    Args:
        mask: array of 0/1 denoting regions believed to contain cells
        init: location (i,j) to start region growing

    Returns:
        mask of the selected region,
        (n x 2) array of pixels deemed part of the region
    """
    start_time = time()
    M, N = np.shape(mask)
    new_mask = np.zeros([N, M], dtype=np.int64)
    locs = []

    def find_neighbors(loc: tuple[int, int]) -> list:
        """finds new locs in mask"""
        i, j = loc
        neighbors = []

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                new_i = i + di
                new_j = j + dj
                if new_i < 0 or new_i >= M:
                    continue
                if new_j < 0 or new_j >= N:
                    continue
                neighbors.append((new_i, new_j))

        return neighbors

    def test_loc(loc: tuple[int, int]) -> bool:
        """indicates whether a location is in the mask AND is novel"""
        i, j = loc
        if mask[i, j] > 0:
            return new_mask[i, j] == 0
        return False

    # initialize
    temp_locs = [init]

    while True:
        neighbors = []
        for loc in temp_locs:
            neighbors.extend(find_neighbors(loc))
            new_mask[loc[0], loc[1]] = 1
        locs.extend(temp_locs)

        temp_locs = []
        for neighbor in neighbors:
            if test_loc(neighbor):
                temp_locs.append(neighbor)

        if len(temp_locs) == 0:
            break
        if time() - start_time > 5.0:
            print("\nTerminating after 5 seconds")
            print(f"Found {len(locs)} locs")
            print(f"Preparing to add {len(temp_locs)} new locs\n")
            break

    return (new_mask, locs)


def _build_adjacency_matrix(mask: np.ndarray, symmetric: bool = True) -> np.ndarray:
    """creates an adjacency matrix for the mask

    Args:
        mask: binarized mask of objects
        symmetric: whether to return a symmetric matrix or simply an upper triangular
            matrix
    """
    M, N = np.shape(mask)
    size = int(M * N)
    if symmetric:
        a = np.zeros([size, size], dtype=np.int64)
    else:
        a = np.empty([size, size], dtype=np.int64)

    def _positive_neighbors(loc: tuple[int, int]) -> list:
        """returns a list of the locations of positive neighbors that exist"""
        neighbors = [
            (loc[0], loc[1] + 1),
            (loc[0] + 1, loc[1] - 1),
            (loc[0] + 1, loc[1]),
            (loc[0] + 1, loc[1] + 1),
        ]

        for idx in [3, 2, 1, 0]:  # iterate in reverse order
            i, j = neighbors[idx]
            if (i >= M) or (j < 0) or (j >= N):
                neighbors.pop(idx)

        return neighbors

    # for idx in tqdm(range(size), desc="Building adjacency matrix", total=size):
    for idx in range(size):
        loc = _flat_to_loc(idx, N)

        if mask[loc[0], loc[1]] == 0:
            continue

        for nbr in _positive_neighbors(loc):
            if mask[nbr[0], nbr[1]] > 0:
                idx2 = _loc_to_flat(
                    nbr, N
                )  # idx2 > idx because only considering positive neighbors
                a[idx, idx2] = 1  # upper triangular matrix
                if symmetric:
                    a[idx2, idx] = 1  # lower triangular matrix

    return a


# @timer_s
def separate_regions_by_growing(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Uses region growing to identify unique regions"""
    regions: list[tuple[np.ndarray, np.ndarray]] = []
    temp_mask = mask.copy()
    M, N = np.shape(mask)
    size = int(M * N)

    def loc_to_flat(loc: tuple[int, int]) -> int:
        """initialized function for image size"""
        return _loc_to_flat(loc, N)

    idx = 0
    while True:
        # init = _find_next_region(temp_mask)
        init, idx = _find_next_region_iter(temp_mask, idx, N, size)
        if init is None:
            break

        if False:
            region, region_locs = _grow_region(mask, init)
            temp_mask -= region
            flat_indices = list(map(loc_to_flat, region_locs))

            regions.append(flat_indices)

        region_is, region_js = flood(temp_mask, init, connectivity=1).nonzero()
        temp_mask[(region_is, region_js)] = 0
        regions.append((region_is, region_js))

    return regions


@timer_s
def separate_regions_by_adjacency_DEPRECATED(mask: np.ndarray) -> list:
    """Uses adjacency matrix to identify unique regions"""
    known_indices = []
    regions_flat = {}
    M, N = np.shape(mask)
    size_a = int(M * N)

    a = _build_adjacency_matrix(
        mask, symmetric=False
    )  # only yields upper triangular matrix

    def find_init_idx(all_indices: list) -> int:
        """returns init idx for associated group"""
        for init in sorted(regions_flat.keys()):
            group = regions_flat[init]
            for index in all_indices:
                if index in group:
                    return init
        print("Creating an erroneous group")
        return int(min(all_indices))

    for idx in tqdm(
        range(size_a - 1), desc="Parsing adjacency matrix", total=size_a - 1
    ):
        if np.sum(a[idx, idx+1:]) == 0:
            continue
        neighbors = np.arange(idx+1, size_a, 1, dtype=np.int64)[a[idx, idx+1:] > 0]
        all_indices = [idx, *neighbors.tolist()]

        if idx in known_indices:
            init_idx = find_init_idx(all_indices)
            related: list = regions_flat.get(init_idx, [])
            related.extend(all_indices)
            regions_flat[init_idx] = related
        else:
            regions_flat[idx] = all_indices
        known_indices.extend(all_indices)

    return []


def filter_region_size(
    regions: list[tuple[np.ndarray, np.ndarray]], threshold: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """filters regions by size (number of pixels)"""
    def fxn(r: tuple[np.ndarray, np.ndarray]) -> bool:
        """Filterin function"""
        return r[0].shape[0] > threshold

    return list(filter(fxn, regions))


def paint_regions(dims: tuple[int, int], regions: list[Region]) -> np.ndarray:
    """Create a new mask from regions"""
    mask = np.zeros(list(dims), dtype=np.int64)
    for region in regions:
        mask[region.region_locs] = 1
    return mask
