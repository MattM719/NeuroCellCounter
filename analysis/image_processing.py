#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Performs basic image processing. Filters and masks images."""

from math import sqrt

import numpy as np
from skimage import morphology

from reader import nd2_img_reader, get_stain
from .filters import apply_gaussian_filter, remove_baseline, binarize
from utils.resampling import resample_image


def mask_img(
    img: np.ndarray, bin_thresh: float = 0.1, opt_thresh: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Masks image

    Parameters:
    ----------
    img: np.ndarray of image (N x M)

    bin_thresh: intensity threshold to binarize image (range: 0-1, default = 0.1)


    Returns:
    -------
    masked image containing filtered values within cell bounds

    mask used for this calculation
    """
    filtered = remove_baseline(img)
    mask = binarize(filtered, thresh=bin_thresh, opt_thresh=opt_thresh)

    masked_img = mask * filtered

    values = masked_img.copy().flatten()
    nonzero = values[values > 0]
    shift = mask * float(np.min(nonzero)) if (len(nonzero) > 0) else 0
    masked_img_shifted = masked_img - shift

    return (masked_img_shifted, mask)


def highlight_mask_edges(mask: np.ndarray) -> np.ndarray:
    """uses gaussian filter to highlight edges"""
    centers, edges = apply_gaussian_filter(mask, sigma=1, radius=2)
    edges_bin = np.zeros_like(edges, dtype=np.int64)
    edges_bin[edges > 0.5] = 1

    return edges_bin


def get_dapi_image(path: str, down_sample: bool = False) -> tuple[np.ndarray, dict]:
    """Gets DAPI-stained microscopy image from an nd2 file"""
    meta, img = nd2_img_reader(path)
    dapi_meta, dapi_img = get_stain(meta, img, "dapi")

    res = dapi_img.shape[0] * dapi_img.shape[1]
    factor = sqrt(res / 512**2)
    if down_sample and factor > 1.5:
        dapi_img, dapi_meta = resample_image(dapi_img, dapi_meta, factor=1.0 / factor)

    return (dapi_img, dapi_meta)


def get_alx568_image(path: str, down_sample: bool = False) -> tuple[np.ndarray, dict]:
    """Gets Alx568-stained microscopy image from an nd2 file"""
    meta, img = nd2_img_reader(path)
    selected_meta, selected_img = get_stain(meta, img, "Alx568")

    res = selected_img.shape[0] * selected_img.shape[1]
    factor = sqrt(res / 512**2)
    if down_sample and factor > 1.5:
        selected_img, selected_meta = resample_image(
            selected_img, selected_meta, factor=1.0 / factor
        )

    return (selected_img, selected_meta)


def preprocess_image(
    img: np.ndarray,
    bin_thresh: float = 0.1,
    opt_thresh: bool = False,
    tophat_radius: int = 0,
) -> tuple[np.ndarray, ...]:
    """Preprocesses images

    Arguments:
    ---------
    img: DAPI-stained microscopy image

    tophat_radius: recommend 2

    Results:
    -------
    preprocessed image, mask, mask_edges
    """
    # normalize
    norm_img = img.copy().astype(np.float64)
    norm_img -= norm_img.min()
    norm_img /= norm_img.max()

    # remove small objects
    if tophat_radius > 0:
        norm_img -= morphology.white_tophat(norm_img, morphology.disk(tophat_radius))
        norm_img -= norm_img.min()
        norm_img /= norm_img.max()

    # develop masks
    masked_img, mask = mask_img(norm_img, bin_thresh=bin_thresh, opt_thresh=opt_thresh)
    mask_edges = highlight_mask_edges(mask)

    return (masked_img, mask, mask_edges)
