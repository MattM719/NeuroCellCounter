#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image filters. Intended for internal use only."""

from typing import Optional
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.restoration import rolling_ball

__all__ = [
    "apply_gaussian_filter",
    "remove_baseline",
    "binarize",
]


def apply_gaussian_filter(
    img: np.ndarray, sigma: int, radius: Optional[int] = None, debug: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Applies a 2D highpass filter to remove baseline drift and detect edges, or to
    blur image

    Parameters:
    ----------
    img: image

    sigma: sigma for gaussian kernel

    radius: Optional radius for gaussian kernel

    debug: whether to display figures intended for development

    Returns:
    -------
    lowpass filtered image, highpass filtered image
    """
    # normalize and discretize pixel intensities
    img = img.copy().astype(np.float64)
    img -= np.min(img.flatten())
    img *= 256 / np.max(img.flatten())
    img = np.round(img, 0).astype(np.int64)

    # filter image
    gauss_lowpass = ndimage.gaussian_filter(img, sigma, radius=radius)
    gauss_highpass = np.array(img, dtype=np.int64) - np.array(
        gauss_lowpass, dtype=np.int64
    )
    min_highpass = np.min(np.min(gauss_highpass))
    if min_highpass < 0:
        gauss_highpass -= min_highpass

    if debug:
        # gauss_highpass = np.max(np.max(gauss_highpass)) - gauss_highpass
        print(f"original range: ({np.min(np.min(img))}, {np.max(np.max(img))})")
        print(
            f"lowpass range:  ({np.min(np.min(gauss_lowpass))}, "
            f"{np.max(np.max(gauss_lowpass))})"
        )
        print(
            f"highpass range: ({np.min(np.min(gauss_highpass))}, "
            f"{np.max(np.max(gauss_highpass))})"
        )

        fig0 = plt.figure()
        ax0 = plt.gca()
        fig0.suptitle("Original")
        ax0.imshow(img, cmap="gray")

        fig1 = plt.figure()
        ax1 = plt.gca()
        fig1.suptitle("Lowpass")
        ax1.imshow(gauss_lowpass, cmap="gray")

        fig2 = plt.figure()
        ax2 = plt.gca()
        fig2.suptitle("Highpass")
        ax2.imshow(gauss_highpass, cmap="gray")

        plt.show()
        plt.close()

    return (gauss_lowpass, gauss_highpass)


def remove_baseline(img: np.ndarray, factor: int | float = 4) -> np.ndarray:
    """uses a highpass filter to remove baseline"""
    baseline = rolling_ball(img)
    no_baseline = img - baseline

    return no_baseline


def remove_baseline_DEBUGGING(img: np.ndarray, factor: int | float = 4) -> np.ndarray:
    """uses a highpass filter to remove baseline"""
    import matplotlib.pyplot as plt

    factors = 5 * np.logspace(0, 2, num=5, endpoint=True, base=10)
    for f in [*factors, 128]:
        baseline, no_baseline = apply_gaussian_filter(img, sigma=int(f), debug=False)
        fig, ax = plt.subplots(1, 3)
        fig.suptitle(f"{f}\n{np.min(no_baseline):.3f}, {np.max(no_baseline):.3f}")
        ax[0].imshow(img)
        ax[1].imshow(baseline)
        ax[2].imshow(no_baseline)

    plt.show()

    import sys

    sys.exit()

    return no_baseline


def binarize(
    highpass_img: np.ndarray,
    opt_thresh: bool = False,
    thresh: int | float = 0.5,
    show_hist: bool = False,
) -> np.ndarray:
    """masks image

    Parameters:
    img : NDArray
        should be highpass-filtered to remove baseline and enhance edges
    opt_thresh : bool
        whether the program should decide the optimal pixel intensity threshold
    thresh : int | float
        float between 0 and 1. Sets threshold for binarizing image
    show_hist : bool
        for development only. shows histogram of intensities

    Returns:
        binarized image of same dimensions

    Raises:
        ValueError if thresh is out of range
    """
    if 0 > thresh or thresh > 1:
        raise ValueError(f"thresh must be between 0 and 1. Received value of {thresh}")

    max_val = int(np.max(np.max(highpass_img)))
    min_val = int(np.min(np.min(highpass_img)))
    thresh_val = thresh * max_val + (1 - thresh) * min_val

    if opt_thresh:
        try:
            hist, bins = np.histogram(
                highpass_img.flatten(),
                bins=round(max_val - min_val + 1),
            )
            peak = np.argmax(hist)
            hist_high = hist[peak:]
            bins_high = bins[peak:-1]
            thresh_val = np.min(bins_high[hist_high < 0.5 * hist[peak]])
            # thresh_val = bins_high[np.argmin(np.diff(hist_high))]
        except Exception:
            warnings.warn(
                "Unable to use optimal threshold. Using default threshold.",
                category=UserWarning,
            )

    if show_hist:
        plt.figure()
        plt.title("Pixel Intensities")
        plt.hist(highpass_img.copy().flatten(), bins=100)
        plt.axvline(thresh_val, color="r")
        plt.show()
        plt.close()

    bin_img = np.zeros_like(highpass_img, dtype=np.int64)
    bin_img[highpass_img > thresh_val] = 1

    return bin_img
