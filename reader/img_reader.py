#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Reads nd2 images and extracts layers of interest"""

from pprint import pprint
from PIL import Image

import numpy as np
import nd2
from nd2.structures import Volume

from utils.types import MetaLike_ND2


def _process_volume(vol: Volume) -> dict:
    """Processs volume data"""
    iterables = [
        vol.axesCalibrated,
        vol.axesCalibration,
        vol.axesInterpretation,
        vol.voxelCount,
    ]

    vol_meta = {
        "axis": [],
        "calibration": [],
        "voxels": [],
    }

    for i, (calibrated, calibration, interpretation, voxels) in enumerate(
        zip(*iterables)
    ):
        if not calibrated:
            continue
        assert (
            interpretation == "distance"
        ), f"unexpected interpretation, '{interpretation}'"

        vol_meta["axis"].append(int(i))
        vol_meta["calibration"].append(int(calibration))
        vol_meta["voxels"].append(int(voxels))

    return vol_meta


def nd2_img_reader(
    path: str, print_all_meta: bool = False
) -> tuple[list[MetaLike_ND2], np.ndarray]:
    """Reads nd2 image files

    Arguments:
        path: path to .nd2 file

    Returns: tuple
        list of meta data for each channel: {name, color, index}
        image as a numpy array (N channels, x, y)
    """
    meta_data: list[MetaLike_ND2] = []
    with nd2.ND2File(path) as img_file:
        channels = img_file.metadata.channels
        if print_all_meta:
            pprint(img_file.metadata)
            print("\n\n")
        for chn in channels:
            meta_data.append(
                {
                    "name": chn.channel.name,
                    "color": chn.channel.color,
                    "index": chn.channel.index,
                    "emission_lambda": chn.channel.emissionLambdaNm,
                    "excitation_lambda": chn.channel.excitationLambdaNm,
                    "objective_magnification": chn.microscope.objectiveMagnification,
                    "zoom_magnification": chn.microscope.zoomMagnification,
                    "projective_magnification": chn.microscope.projectiveMagnification,
                    "pinhole_diameter_um": chn.microscope.pinholeDiameterUm,
                    "axes_calibrated": chn.volume.axesCalibrated,  # should be (True,True,False)  # noqa E501
                    "axes_calibration": chn.volume.axesCalibration,  # should be (float, float, 1.0) -> each num is the microns per pixel  # noqa E501
                    "axes_interpretation": chn.volume.axesInterpretation,  # should be ('distance','distance','distance')  # noqa E501
                    "voxel_count": chn.volume.voxelCount,  # should be (512, 512, 1) -> image dimensions (pixels)  # noqa E501
                    # "volume_meta": _process_volume(chn.volume),
                }
            )

    img = nd2.imread(path)

    return (meta_data, img)


def get_stain(
    meta: list[MetaLike_ND2], img: np.ndarray, target: str = "dapi"
) -> tuple[MetaLike_ND2, np.ndarray]:
    """parses meta data to only return data for specified target stain/image"""
    target = str(target).strip().lower()
    select = -1
    for layer in meta:
        name = str(layer.get("name", None)).strip().lower()
        if name == target:
            select = int(layer.get("index"))
            break

    if select == -1:
        try:
            names = [layer.get("name", None) for layer in meta]
        except Exception:
            names = []
        raise NameError(f'Could not find "{target}" in \n{names}\n')

    return (meta[select], img[select, ...])


def png_img_reader(path: str) -> np.ndarray:
    """reads a microscopy image flattened into a PNG"""
    with Image.open(path, "r", formats=("PNG",)) as image:
        img = np.array(image).transpose((2, 0, 1))

    return img
