#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates figures to display microscopy images and processing results"""

from typing import Literal, Optional

import numpy as np
import nd2
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb

from utils.models import Cell


__all__ = [
    "plot_layers",
    "plot_blobs",
    "plot_cell_histograms",
    "plot_pyknotic",
]


def _create_cdict(
    r: int,
    g: int,
    b: int,
    a: Optional[float] = None,
    m: int = 256,
    background: Literal["white", "black"] = "black",
) -> dict:
    """creates a cdict from rgba color data"""
    if background == "black":
        bk = 0
    elif background == "white":
        bk = 1
    else:
        raise ValueError("Invalid background")

    def norm(val: float, m: int = m, den: int = 1) -> float:
        """normalize val to max (m). Can set denominator den to 2 for half max"""
        norm_val = min([float(val) / float(m), 1.0])
        if den > 1:
            return norm_val / float(den)
        return norm_val

    r_max = norm(r)
    g_max = norm(g)
    b_max = norm(b)

    cdict = {
        "red": [
            (0, bk, bk),
            (1, r_max, r_max),
        ],
        "green": [
            (0, bk, bk),
            (1, g_max, g_max),
        ],
        "blue": [
            (0, bk, bk),
            (1, b_max, b_max),
        ],
    }

    if a is not None:
        cdict["alpha"] = [
            (0, a, a),
            (1, a, a),
        ]

    return cdict


def create_cmap(
    rgba: nd2.structures.Color,
    name: Optional[str] = "tempCmap",
    N: int = 256,
    n_maps: int = 1,
) -> cm:
    """Creates a colormap"""
    cdict = _create_cdict(r=rgba.r, g=rgba.g, b=rgba.b, a=rgba.a / n_maps, m=N)
    newcmp = LinearSegmentedColormap(name, segmentdata=cdict, N=N)

    return newcmp


def plot_layers(
    meta: list[dict],
    img: np.ndarray,
    scale_bar: Optional[float] = None,
    show: bool = True,
    merge: bool = False,
) -> Optional[list[tuple[Figure, Axes]]]:
    """Plots each individual layer from output of nd2_img_reader"""
    assert isinstance(img.flatten()[0], np.float64)
    assert img.min() >= 0 and img.max() <= 1

    figs = []

    imgs: list[np.ndarray] = []

    for layer in meta:
        fig = plt.figure()
        ax = fig.gca()
        cname = layer.get("name")
        color = layer.get("color")
        idx = int(layer.get("index"))
        fig.suptitle(cname)

        patch = None
        if scale_bar is not None:
            dim1, dim2, _ = layer.get("axes_calibration")
            if dim1 != dim2:
                raise NotImplementedError("Not sure which dim is x vs y")
            else:
                dimx = float(dim1)
            x, y = 10, 10
            width = scale_bar / dimx
            height = max([width / 10, 5])
            patch = Rectangle((x, y), width=width, height=height, color="w")

            label = f"{round(scale_bar)} " + r"$\mu m$"
            text_kwargs = {
                "x": x + 1,
                "y": y + 2 * height,
                "s": label,
                "fontdict": {
                    "family": "Arial",
                    "color": "w",
                    "weight": "normal",
                    "size": 10,
                },
            }

        if merge:
            channel = gray2rgb(img[idx, ...], r=color.r, g=color.g, b=color.b)
            imgs.append(channel)
            ax.imshow(channel.transpose((1, 2, 0)), interpolation="none", origin="lower")
        else:
            cmap = create_cmap(color, cname, n_maps=1)
            ax.imshow(img[idx, ...], interpolation="none", origin="lower", cmap=cmap)

        if patch is not None:
            ax.add_patch(patch)
            ax.text(**text_kwargs)

        if show:
            plt.show()
            plt.close(fig)
        else:
            figs.append((fig, ax))

    if merge:
        fig = plt.figure()
        ax = fig.gca()
        fig.suptitle("Merged")
        merged = merge_rgb(*imgs)
        ax.imshow(merged.transpose((1, 2, 0)), interpolation="none", origin="lower")
        if scale_bar is not None:
            ax.add_patch(Rectangle((x, y), width=width, height=height, color="w"))
            ax.text(**text_kwargs)
        if show:
            plt.show()
            plt.close(fig)
        else:
            figs.append((fig, ax))

    if show:
        return None
    return figs


def plot_merged_DEPRECATED(
    meta: list[dict], img: np.ndarray, show: bool = True
) -> Optional[tuple[Figure, Axes]]:
    """Creates a figure"""
    fig = plt.figure()
    ax = fig.gca()
    fig.suptitle("Merged")

    for layer in meta:
        cname = layer.get("name")
        color = layer.get("color")
        idx = int(layer.get("index"))

        cmap = create_cmap(color, cname, n_maps=1)

        ax.imshow(img[idx, ...], interpolation="nearest", origin="lower", cmap=cmap)

    if show:
        plt.show()
        plt.close()
    else:
        return (fig, ax)
    return None


def plot_blobs(
    img: np.ndarray, blobs: np.ndarray, side_by_side: bool = True, show: bool = False
) -> Optional[tuple[Figure, tuple[Axes, ...]]]:
    """Overlays blob locations and sizes on image

    Arguments:
        img: image - array of pixels
        blobs: (n x 3) array of (x, y, radius) for each blob
        side_by_side: whether to show the original figure next to the annotated figure
        show: whether to show the figure

    Returns:
        (Figure, (Axes, ...)) only if show is set to False
    """
    if side_by_side:
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax: tuple[Axes, ...] = ax
        fig.set_figheight(5)
        fig.set_figwidth(5)
    else:
        fig = plt.figure(figsize=(5, 5))
        axis = plt.gca()
        ax: tuple[Axes, ...] = (axis,)

    ax[-1].imshow(img, interpolation="nearest", cmap="gray")
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color="y", linewidth=1, fill=False, alpha=0.4)
        ax[-1].add_patch(c)

    if side_by_side:
        ax[0].imshow(img, interpolation="nearest", cmap="gray")

    for i in range(len(ax)):
        ax[i].set_axis_off()
    fig.tight_layout()

    if show:
        plt.show()
        plt.close()
    else:
        return (fig, ax)
    return None


def plot_cell_histograms(
    cells: list[Cell], show: bool = True, **kwargs
) -> Optional[Figure]:
    """Labels cells as pyknotic or non-pyknotic"""
    params = {
        "n_bins": 50,
        "total_intensity": None,
        "avg_intensity": None,
        "weighted_intensity": None,
        "sigma": None,
        "perimeter": None,
        "area": None,
        "eccentricity": None,
        "ideal_radius": None,
        "height": 4,
        "width": 8,
        "dpi": 400,
    }
    params.update(kwargs)

    n_cells = len(cells)

    # get characteristics of dataset
    total_intensities = np.zeros(n_cells, dtype=np.float64)
    avg_intensities = np.zeros(n_cells, dtype=np.float64)
    weighted_intensities = np.zeros(n_cells, dtype=np.float64)
    sigmas = np.zeros(n_cells, dtype=np.float64)
    perimeters = np.zeros(n_cells, dtype=np.float64)
    areas = np.zeros(n_cells, dtype=np.float64)
    ideal_radii = np.zeros(n_cells, dtype=np.float64)
    eccentricities = np.zeros(n_cells, dtype=np.float64)

    for i, cell in enumerate(cells):
        total_intensities[i] = cell.total_intensity
        avg_intensities[i] = cell.avg_intensity
        weighted_intensities[i] = cell.weighted_intensity
        sigmas[i] = cell.sigma_um
        perimeters[i] = cell.n_edges
        areas[i] = cell.n_pixels
        ideal_radii[i] = cell.ideal_radius
        eccentricities[i] = cell.eccentricity

    # plot histograms
    mosaic = [
        ["total", "avg", "weighted", "sigma"],
        ["perim", "area", "eccentricity", "ideal_radius"],
    ]
    fig, ax = plt.subplot_mosaic(mosaic)
    fig: Figure = fig
    ax: dict[str, Axes] = ax
    fig.set_figheight(params["height"])
    fig.set_figwidth(params["width"])
    fig.set_dpi(params["dpi"])

    ax["total"].hist(total_intensities, params["n_bins"])
    ax["total"].set_xlabel("Total\nIntensity (AU)")
    ax["avg"].hist(avg_intensities, params["n_bins"])
    ax["avg"].set_xlabel("Average\nIntensity (AU)")
    ax["weighted"].hist(weighted_intensities, params["n_bins"])
    ax["weighted"].set_xlabel("Weighted\nIntensity (AU)")
    ax["sigma"].hist(sigmas, params["n_bins"])
    ax["sigma"].set_xlabel(r"$\sigma$" + " (" + r"$\mu m$" + ")")
    ax["perim"].hist(perimeters, params["n_bins"])
    ax["perim"].set_xlabel("Perimeter (" + r"$\mu m$" + ")")
    ax["area"].hist(areas, params["n_bins"])
    ax["area"].set_xlabel("Area (" + r"$\mu m^2$" + ")")
    ax["eccentricity"].hist(eccentricities, params["n_bins"])
    ax["eccentricity"].set_xlabel("Eccentricity")
    ax["ideal_radius"].hist(eccentricities, params["n_bins"])
    ax["ideal_radius"].set_xlabel("Ideal Radius (" + r"$\mu m$" + ")")

    for key, value in params.items():
        name = str(key).replace("_intensity", "").replace("eter", "")
        if not isinstance(value, (tuple, list)):
            value = [value]
        for val in value:
            if val is None:
                continue
            ax[name].axvline(val, c="r")

    fig.tight_layout()

    if show:
        plt.show()
        plt.close(fig)
        return None

    return fig


def plot_pyknotic(
    img: np.ndarray,
    cells: list[Cell],
    show: bool = True,
    raw_img: Optional[np.ndarray] = None,
) -> Optional[Figure]:
    """Plots pyknotic nuclei"""
    fig, ax = plt.subplots(1, 3, constrained_layout=True)
    ax: tuple[Axes, ...] = ax
    fig.set_figheight(6)
    fig.set_figwidth(18)
    fig.set_dpi(1200)

    if raw_img is None:
        raw_img = img

    ax[0].imshow(raw_img, interpolation="none", cmap="gray")
    ax[0].set_title("DAPI Image")
    ax[0].set_axis_off()
    # ax[1].imshow(raw_img, interpolation="none", cmap="gray")
    # ax[1].set_title("All Cells")
    # ax[1].set_axis_off()
    ax[2].imshow(img, interpolation="none", cmap="gray")
    ax[2].set_title("Pyknotic Cells")
    ax[2].set_axis_off()

    layers = np.zeros_like(img, dtype=np.int64)
    pyk_count = 0
    for i, cell in enumerate(cells):
        layers[cell.pixel_locs] = i + 1  # float(cell.uid) + 1.0

        if cell.is_pyknotic:
            pyk_count += 1
            y, x = cell.center
            r = cell.sigma_px
            c = plt.Circle(
                (x, y), r * 1.25, color="r", linewidth=1, fill=False, alpha=0.6
            )
            ax[2].add_patch(c)

    labeled_img = label2rgb(layers, raw_img, kind="overlay", bg_label=0, alpha=0.7)
    labeled_img = mark_boundaries(labeled_img, layers, (0.5, 0.5, 0.5), mode="inner")
    ax[1].imshow(labeled_img, interpolation="none")
    ax[1].set_title("All Cells")
    ax[1].set_axis_off()
    # ax[1].imshow(find_boundaries(layers), interpolation="none", cmap=tra)

    fig.suptitle(f"{len(cells)} Cells, {pyk_count} Pyknotic")

    if show:
        plt.show()
        plt.close()
        return None

    return fig


def gray2rgb(img_0_1: np.ndarray, r: int, g: int, b: int) -> np.ndarray:
    """Converts a gray scale image with values spanning 0-1 to RGB with the specified
    color

    Parameters:
    ----------
    img_0_1: an (M, N) array of values 0-1 representing pixel intensity

    r, g, b: maximum red, green, blue pixel intensities 0-255

    Returns:
    -------
    A (3, M, N) RGB array of the same image
    """
    # validate inputs
    r, g, b = int(r), int(g), int(b)
    assert 0 <= r <= 255
    assert 0 <= g <= 255
    assert 0 <= b <= 255

    img_0_1 = img_0_1.astype(np.float64)
    assert ((0 <= img_0_1) * (img_0_1 <= 1)).all()
    assert len(img_0_1.shape) == 2

    # Create full color RGB array
    rgb = np.zeros([3, *img_0_1.shape], dtype=np.float64)
    rgb[0, ...] = r
    rgb[1, ...] = g
    rgb[2, ...] = b

    # Create image in RGB array
    rgb *= img_0_1
    rgb_out = rgb.round().astype(np.int64)

    return rgb_out


def merge_rgb(*imgs: np.ndarray) -> np.ndarray:
    """Merges RGB images"""
    # validation and shape
    shape: Optional[np.ndarray] = None
    for img in imgs:
        assert len(img.shape) == 3
        assert img.shape[0] == 3
        assert ((0 <= img) * (img <= 255)).all()
        if shape is None:
            shape = np.array(img.shape, dtype=np.int64)
        else:
            assert (shape == np.array(img.shape, dtype=np.int64)).all()
    assert isinstance(shape, np.ndarray)

    merged = np.zeros(shape, dtype=np.int64)
    for img in imgs:
        merged += img.round()

    merged[merged > 255] = 255

    return merged
