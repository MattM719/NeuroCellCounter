#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Saves processed images"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from utils.types import MetaLike_ND2


def save_image_nd2_meta(
    out_dir: Path, name: str, img: np.ndarray, meta: MetaLike_ND2
) -> None:
    """Saves an image to a PNG file and its meta data to an identically named JSON file
    """
    img_path = out_dir / f"{name}.png"
    meta_path = out_dir / f"{name}.json"

    plt.imsave(img_path, img, cmap="gray")

    with open(meta_path, mode="w", encoding="utf-8") as f:
        json.dump(meta, f)

    return None
