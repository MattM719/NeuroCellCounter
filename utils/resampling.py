#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions to resample an image. Intended to downsample images to ~512x512 px"""

from typing import Callable, List, Optional, Tuple

import numpy as np
from skimage.transform import rescale

from .types import MetaLike_ND2


def resample_image(
    image: np.ndarray, meta: Optional[MetaLike_ND2], factor: int | float = 0.5
) -> Tuple[np.ndarray, Optional[MetaLike_ND2]]:
    """Resamples an image by some ratio. A factor of 0.5 reduces the number of pixels
    along each axis by half.
    """
    image_rescaled = rescale(image, factor, anti_aliasing=False)
    real_factors = [
        float(n1) / float(n0) for n0, n1 in zip(image.shape, image_rescaled.shape)
    ]

    nd2_required = ["axes_calibrated", "axes_calibration", "voxel_count"]
    is_nd2_meta: Callable[[dict], bool] = lambda m: all([k in m for k in nd2_required])

    if isinstance(meta, dict) and is_nd2_meta(meta):
        axes_calibrated: Tuple[bool, ...] = meta["axes_calibrated"]
        axes_calibration_old: Tuple[float, ...] = meta["axes_calibration"]
        voxel_count_old: Tuple[int, ...] = meta["voxel_count"]

        axes_calibration_new: List[float] = [float(x) for x in axes_calibration_old]
        voxel_count_new: List[int] = [int(x) for x in voxel_count_old]

        img_idx = 0
        for i, is_calibrated in enumerate(axes_calibrated):
            if is_calibrated:
                axes_calibration_new[i] /= real_factors[img_idx]
                voxel_count_new[i] = image_rescaled.shape[img_idx]
                img_idx += 1

        meta["axes_calibration"] = tuple(axes_calibration_new)
        meta["voxel_count"] = tuple(voxel_count_new)

    elif meta is not None:
        raise NotImplementedError(
            f"Not yet prepared to process meta format: {list(meta.keys())}"
        )

    return (image_rescaled, meta)
