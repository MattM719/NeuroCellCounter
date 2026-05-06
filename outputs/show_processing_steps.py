#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Various plots to display image processing progress

This is mostly for development, rather than final figures
"""

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .img_plotter import plot_layers
from reader import nd2_img_reader
from utils.types import MetaLike_ND2


def view_nd2_images(
    path: str,
    is_uint_4095: Optional[bool] = None,
    scale_bar: Optional[float] = None,
    show=True,
) -> Optional[Tuple[Tuple[Figure, ...], list[MetaLike_ND2]]]:
    """Simply plots the layers and merged image from an nd2 file"""
    meta, img = nd2_img_reader(path)

    # if isinstance(img.flatten()[0], (np.int64)):
    #     img = img.astype(np.float64) / 255
    # elif isinstance(img.flatten()[0], np.uint16) and is_uint_4095:
    #     img = img.astype(np.float64) / 4095
    # else:
    #     raise NotImplementedError()

    # normalize image
    if len(img.shape) == 2:
        img = img.astype(np.float64) / img.max()
    if len(img.shape) == 3:
        img = img.astype(np.float64)
        maxs = img.max(axis=-1).max(axis=-1)
        for i, val in enumerate(maxs):
            img[i, ...] = img[i, ...] / val

    layers = plot_layers(meta, img, show=False, scale_bar=scale_bar, merge=True)
    # merged_fig, merged_axis = plot_merged(meta, img, show=False)

    if show:
        plt.show()
        plt.close()
        return None
    figs = [f for (f, a) in layers]
    # figs.append(merged_fig)
    return tuple(figs), meta


def compare_original_masked_overlay(
    dapi_img: np.ndarray,
    masked_img: np.ndarray,
    mask: np.ndarray,
    mask_edges: np.ndarray,
    show: bool = True,
    **kwargs,
) -> Optional[Figure]:
    """Creates 3 graphs to show the initial results from filtering the image and
    applying a mask"""
    params = {
        "height": 5,
        "width": 13,
        "dpi": 400,
        "title": None,
    }
    params.update(kwargs)

    fig, ax = plt.subplots(nrows=1, ncols=3)
    fig.set_figheight(params["height"])
    fig.set_figwidth(params["width"])
    fig.set_dpi(params["dpi"])
    ax: tuple[Axes, ...] = ax

    if isinstance((title := params["title"]), str):
        fig.suptitle(title)

    ax[0].set_title("Original Image")
    ax[0].imshow(dapi_img, cmap="gray", interpolation="nearest")
    ax[0].set_axis_off()

    ax[1].set_title("Masked Image")
    ax[1].imshow(masked_img, cmap="gray", interpolation="nearest")
    ax[1].set_axis_off()

    ax[2].set_title("Mask Overlay")
    ax[2].imshow(dapi_img, cmap="gray", interpolation="nearest", alpha=1)
    ax[2].imshow(mask, cmap="Oranges", interpolation="nearest", alpha=0.3)
    # ax[2].imshow(mask_edges, cmap=red_transparent_cmap, interpolation="none")
    ax[2].set_axis_off()

    fig.tight_layout()

    if show:
        plt.show()
        plt.close()
        return None

    return fig
