#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Finds yellow dots on PNGs"""

import os
from glob import glob
from typing import Optional, Generator
from pprint import pprint

import numpy as np
import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt  # noqa E402
from matplotlib.axes import Axes  # noqa E402

try:
    from reader import png_img_reader
except ImportError:
    from ..reader import png_img_reader


def find_regions(points) -> list[set[tuple[int, int]]]:
    """sorts each point into distinct regions
    returns the center index of each region
    """
    regions: list[set[tuple[int, int]]] = []
    seen = set()

    def generate_neighbours(point) -> Generator[tuple[int, int], None, None]:
        """creates neighbors for each point"""
        neighbours = [
            (1, -1),
            (1, 0),
            (1, 1),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
        ]
        for neigh in neighbours:
            yield tuple(map(sum, zip(point, neigh)))

    for pt in points:
        nbrs = set([pt])
        for nbr in generate_neighbours(pt):
            if nbr in points:
                nbrs.add(nbr)

        if seen.isdisjoint(nbrs):
            regions.append(nbrs)
        else:
            for r in regions:
                if not nbrs.isdisjoint(r):
                    r.update(nbrs)
                    break

    n_regions = len(regions) + 1

    # check for nondisjoint sets
    while (tmp := len(regions)) < n_regions:
        n_regions = int(tmp)
        for i in sorted(range(n_regions), reverse=True):
            nondisjoint = False
            for j in range(i):
                if not regions[i].isdisjoint(regions[j]):
                    regions[j].update(regions[i])
                    nondisjoint = True
            if nondisjoint:
                regions.pop(i)

    return regions


def find_center(points: set[tuple[int, int]]) -> tuple[int, int]:
    """returns center point from a set"""
    xs = set()
    ys = set()

    for x, y in points:
        xs.add(x)
        ys.add(y)

    x = round(np.mean(sorted(xs)))
    y = round(np.mean(sorted(ys)))

    return (x, y)


def correct_rgb_image(rgb_img: np.ndarray) -> np.ndarray[np.int64]:
    """corrects an rgb image to highlight annotations"""
    rgb_img = rgb_img.astype(dtype=np.int64)
    c, dim1, dim2 = rgb_img.shape
    assert c == 3, f"expected an RGB image but image has {c} channels"

    corrected_img: np.ndarray = np.sum(rgb_img[0:2, ...], axis=0)
    corrected_img -= rgb_img[2, ...]

    corrected_img[corrected_img < 0] = 0

    return corrected_img


def create_mask(img: np.ndarray[np.int64], threshold: int) -> np.ndarray[np.int64]:
    """preprocesses image to identify the labelled points"""
    mask = np.zeros_like(img, dtype=np.int64)
    mask[img > threshold] = 1

    return mask


def locate_regions(mask: np.ndarray[np.int64]) -> list[tuple[int, int]]:
    """locates individual points in mask

    5-9 pixels should be masked for each mark,
    so this returns only the center point of each mark
    """
    all_mask_locs = set((int(x), int(y)) for x, y in zip(*np.where(mask == 1)))
    regions = find_regions(all_mask_locs)

    return regions


def find_points_in_png(
    path: str, verbose: bool = False, plot: bool = False, out_path: Optional[str] = None
) -> list[tuple[int | float, int | float]]:
    """finds distinct yellow points in a PNG of a blue-stained image

    Args:
        path: path to a PNG file
        verbose: whether to print the number of points found
        plot: whether to

    Returns:
        list of pixels clicked as (i, j) tuples
    """
    rgb_img = png_img_reader(path)
    adj_image = correct_rgb_image(rgb_img)
    mask = create_mask(adj_image, 255)
    regions = locate_regions(mask)
    centers = [find_center(r) for r in regions]

    if mask.shape == (512, 512):
        scale_factor = 1
    elif mask.shape == (1024, 1024):
        scale_factor = 2
    else:
        raise NotImplementedError()

    if verbose:
        print(f"Found {len(centers)} points")
        pprint(centers)

    if plot:
        fig, ax = plt.subplots(1, 3)
        ax: list[Axes] = ax
        ax[0].imshow(rgb_img.transpose((1, 2, 0)))
        ax[1].imshow(adj_image, cmap="gray")
        ax[2].imshow(mask, cmap="gray")

        ax[0].set_title("Original Image")
        ax[1].set_title("Adjusted Image")
        ax[2].set_title("Masked Image")
        fig.suptitle(os.path.splitext(os.path.basename(path))[0])

        for i in range(3):
            ax[i].axis("off")

        for y, x in centers:
            c = plt.Circle((x, y), 3, color="r", linewidth=1, fill=False, alpha=0.6)
            ax[2].add_patch(c)

        if isinstance(out_path, str):
            fig.savefig(out_path, dpi=600, format="png")
        else:
            plt.show()
        plt.close(fig)

    if scale_factor != 1:
        centers = [(i / scale_factor, j / scale_factor) for i, j in [*centers]]

    return centers


def main():
    """main"""
    base_name = ...  # FIXME: path to directory with annotated images
    files = glob(os.path.join(base_name, "*.png"))

    for file in files:
        find_points_in_png(file, verbose=True, plot=True)


if __name__ == "__main__":
    main()
