#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compares automated vs. manual cell counts"""

import os
from glob import glob
from typing import Generator

import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")

from tqdm import tqdm  # noqa E402

from utils import Image  # noqa E402
from utils.data_pickling import read_from_pickle2, StoredData2  # noqa E402


def read_manual_counts(path: str, sheet_name: str) -> pd.DataFrame:
    """Reads manual cell counts for each file and returns a clean DataFrame"""
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.loc[:, ["Slice", "Region", "Image #", "Pyk Nuc", "All DAPI"]]
    df.dropna(axis=0, how="any", inplace=True, ignore_index=True)
    df["Region"] = df["Region"].str.upper()

    return df


def read_auto_counts_from_pkl(path: str) -> Generator[Image, None, None]:
    """Reads data from annotations.pkl files"""
    pkl_files = glob(os.path.join(path, "*", "annotations.pkl"))
    for pkl_file in tqdm(pkl_files, total=len(pkl_files), desc="Reading annotations"):
        annotations: StoredData2 = read_from_pickle2(pkl_file)

        image = Image(annotations.image_path, annotations.image_meta)
        image.set_cells(annotations.cells)
        image.set_regions(annotations.regions)

        yield image


def align_auto_manual_counts(
    manual_counts_path: str, manual_counts_sheet: str, auto_counts_path: str
) -> Generator[Image, None, None]:
    """Creates a list of aligned Image classes with aligned manual and automatic counts
    """
    manual_counts = read_manual_counts(
        manual_counts_path, sheet_name=manual_counts_sheet
    )

    for image in read_auto_counts_from_pkl(auto_counts_path):
        slice_num = image.slice_num
        brain_region = image.brain_region
        image_num = image.image_num
        manual_tmp: pd.DataFrame = (
            manual_counts.where(manual_counts["Slice"] == slice_num, inplace=False)
            .where(manual_counts["Region"] == brain_region, inplace=False)
            .where(manual_counts["Image #"] == image_num, inplace=False)
            .dropna(axis=0, how="any", inplace=False, ignore_index=True)
        )

        if manual_tmp.shape[0] != 1:
            continue
        try:
            all_nuclei = int(manual_tmp.iloc[0, 4])
            pyk_nuclei = int(manual_tmp.iloc[0, 3])
        except Exception:
            continue

        image.set_manual_count(
            all_nuclei=all_nuclei,
            pyk_nuclei=pyk_nuclei,
        )
        yield image
