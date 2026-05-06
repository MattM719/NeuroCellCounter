#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compiles basic data from images"""

import csv
from copy import copy
from pathlib import Path

import numpy as np

from reader import nd2_img_reader


def save_relative_channel_intensities(img_path: Path, out_dir: Path) -> None:
    """Get the average pixel intensity for each channel

    Arguments:
    ---------
    img_path: path to a multi-channel image

    out_dir: directory to where data are being save
    """
    out_path = out_dir / "rel_channel_intensities.csv"
    mode = "a" if out_path.exists() else "w"

    ext = str(img_path.suffix).lower().replace(".", "").strip()
    data = {"Image_Path": str(img_path.stem)}

    if mode == "w":
        ch_names = []

    # read nd2 image files
    if ext == "nd2":
        metas, imgs = nd2_img_reader(path=img_path)

        for meta in metas:
            ch_name = str(meta["name"]).upper().strip()
            idx = meta["index"]

            img = imgs[idx, ...].copy().astype(np.float64).flatten()
            intensity = float(np.average(img))

            data[ch_name] = intensity
            if mode == "w":
                ch_names.append(ch_name)

    else:
        raise NotImplementedError(f"Unable to process '.{ext}' images")

    if mode == "w":
        headers = ["Image_Path", *sorted(ch_names)]
    else:
        with open(out_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f, dialect="excel", lineterminator="\n")
            headers = [*reader.fieldnames]

    for key in data.keys():
        if key not in headers:
            headers.append(key)
            val = copy(data[key])
            data[key] = f"key={val}"

    with open(out_path, mode=mode, encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=headers, dialect="excel", lineterminator="\n"
        )
        if mode == "w":
            writer.writeheader()
        writer.writerow(data)

    if mode == "w":
        out_path.chmod(0o660)

    return None
