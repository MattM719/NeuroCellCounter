#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Applies a previously trained RdF model to a properties.csv file"""

import csv
import shutil
from typing import List, Tuple, Literal
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt  # noqa F402
from matplotlib.axes import Axes  # noqa F402
from sklearn.ensemble import RandomForestClassifier  # noqa F402
from tqdm import tqdm  # noqa F402
from skimage.color import rgb2gray  # noqa F402

from classifiers import get_features_multiple_cells  # noqa F402
from utils import Cell, OnnxAsSklean  # noqa F402
from utils.data_pickling import (  # noqa F402
    StoredData2,
    read_from_pickle2,
    StoredAnnotations1,
    StoredModel1,
)
from utils.color_maps import dapi_cmap  # noqa F402


BASE_DIR = Path(...)  # FIXME: path to data directory

OUT_DIR = BASE_DIR / "results"

MODEL_BATCH_NAME: str = "batch_Apr24v4_040725_164950"
TARGET_BATCH_DIR = OUT_DIR / "batch_Apr23_Caff_052425_001539"

IMAGE_DIR_512 = TARGET_BATCH_DIR / "downsampled_images"

PROPERTY_NAMES = [
    "sigma",
    "avg_intensity",
    "weighted_intensity",
    "total_intensity",
    "ideal_radius",
    "eccentricity",
    "perimeter",
    "area",
]


def get_image_data(
    data: pd.DataFrame, name: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """compiles pre-gathered data for an image"""
    props: pd.DataFrame = data.where(data["file_name"] == name, inplace=False)
    props.dropna(axis=0, how="all", inplace=True)

    all_centers = props.loc[:, ["i", "j"]]
    identifiers = props.loc[:, ["file_name", "uid", "region_id"]]
    all_prop_vals = props.loc[:, PROPERTY_NAMES]

    return (all_centers, identifiers, all_prop_vals)


def update_pyknotic_counts(
    file_classifications: pd.DataFrame, file_names: list[str], new_dir: Path
) -> None:
    """calculate and save updated pyknotic cell counts"""
    new_counts: list[list[str | int]] = [
        ["file_name", "all_nuclei", "pyknotic", "non-pyknotic"]
    ]

    for name in file_names:
        clfs = (
            file_classifications.loc[
                file_classifications["file_name"] == name, ["classified_pyknotic"]
            ]
            .to_numpy(bool)
            .flatten()
        )
        total = int(clfs.shape[0])
        pyk = int(np.sum(clfs))
        new_counts.append([name, total, pyk, total - pyk])

    with open(new_dir / "counts_updated.csv", mode="w", encoding="utf-8") as f:
        writer = csv.writer(f, dialect="excel", lineterminator="\n")
        writer.writerows(new_counts)

    return None


def get_pretrained_rdf_clsf(
    clf_path: Path,
) -> Tuple[RandomForestClassifier | OnnxAsSklean, List[str]]:
    """Classifies cells as pyknotic or non-pyknotic from a pre-trained random forest
    classifier.
    """

    def is_StoredData2(data: StoredData2 | StoredModel1 | StoredAnnotations1) -> bool:
        """True if data is a valid StoredData2 model"""
        tests = [
            data.class_name == "StoredData2",
            hasattr(data, "classifier"),
            hasattr(data, "features"),
        ]
        return all(tests)

    # validation
    clf_type: Literal["pkl", "onnx", None] = None
    clf_path: Path = Path(clf_path) if isinstance(clf_path, str) else clf_path
    if not isinstance(clf_path, Path):
        raise TypeError("clf_path must be a Path")
    if clf_path.is_file() and clf_path.suffix == ".pkl":
        clf_type = "pkl"
    elif clf_path.is_dir() and "onnx" in clf_path.name:
        clf_type = "onnx"
    else:
        raise FileNotFoundError(f"Could not find file {clf_path.name}")

    # read and validate model
    if clf_type == "pkl":
        stored_model: StoredData2 = read_from_pickle2(pkl_path=clf_path)

        if not is_StoredData2(stored_model):
            raise NotImplementedError(f"Not prepared to read {stored_model=}")

        clsf = stored_model.classifier
        features: List[str] = stored_model.features

    elif clf_type == "onnx":
        clsf = OnnxAsSklean(clf_path)
        features = clsf.features

    else:
        raise NotImplementedError(f"Unexpected {clf_type=}")

    return (clsf, features)


def read_onnx_rdf_clsf(onnx_path: Path):
    """Like `get_pretrained_rdf_clsf()` but for classifiers saved as ONNX types"""


def _classify_cells_rdf_by_props(
    properties: pd.DataFrame,
    clsf: RandomForestClassifier | OnnxAsSklean,
    features: list[str],
) -> np.ndarray:
    """Classifies cells in a simple properties dataframe"""
    props = properties.loc[:, features]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X has feature names, but RandomForestClassifier was fitted "
            "without feature names",
        )
        clsfs = clsf.predict(props)
    return clsfs


# NOTE: Called by cell_counter.py
def classify_cells_rdf(
    rdf_path: Path, cells: List[Cell]
) -> Tuple[List[Cell], np.ndarray]:
    """Classifies cells using a pre-trained RdF classifier"""
    clsf, features = get_pretrained_rdf_clsf(rdf_path)

    properties = get_features_multiple_cells(cells, features)
    properties_df = pd.DataFrame(properties, columns=features)

    clsfs = _classify_cells_rdf_by_props(properties_df, clsf, features)
    clsfs = clsfs.astype(dtype=bool)

    for cell, is_pyk in zip(cells, clsfs):
        cell.is_pyknotic = bool(is_pyk)

    return (cells, clsfs)


def main():
    """main"""
    # create results directory
    testing_dir = TARGET_BATCH_DIR / f"classifier_test_results_{MODEL_BATCH_NAME}"
    if not testing_dir.exists():
        testing_dir.mkdir(0o751)

    # read cell properties
    properties_path = TARGET_BATCH_DIR / "properties.csv"
    all_cell_properties = pd.read_csv(properties_path)

    # merge properties with pyknotic cell annotations, for annotated cells

    # ================================================================================ #
    # get classifier                                                                   #
    # ================================================================================ #

    # is_StoredData2: Callable[[StoredData2|StoredModel1|StoredAnnotations1],bool] = (
    #     lambda x : x.class_name == "StoredData2" and hasattr(x, "classifier") and
    #       hasattr(x, "features")
    # )
    # stored_model: StoredData2 = read_from_pickle2(pkl_path=OUT_DIR / MODEL_BATCH_NAME
    #   / "training_data/rdf_model.pkl")

    # if is_StoredData2:
    #     clsf: RandomForestClassifier = stored_model.classifier
    #     features: List[str] = stored_model.features
    # else:
    #     raise NotImplementedError(f"Not prepared to read {stored_model=}")

    # # classify cells
    # rdf_data = all_cell_properties.loc[:,features].to_numpy(dtype=np.float64)
    # classifications = clsf.predict(rdf_data)
    # classifications = classifications.astype(dtype=bool)
    # all_cell_properties["classified_pyknotic"] = classifications
    # all_cell_properties.to_csv(testing_dir / "properties_updated.csv", index=False)

    # find RdF models
    model_folders = [
        "training_data",
        "training_data_balanced",
        "training_data_unbalanced",
    ]
    model_paths: List[Path] = [OUT_DIR / MODEL_BATCH_NAME / x for x in model_folders]
    rdf_parent = None
    for p in model_paths:
        if p.exists():
            rdf_parent = p
            break
    if rdf_parent is None:
        raise FileNotFoundError("Could not find a valid model path")

    # choose ONNX if it exists, otherwise PKL file
    onnx_path = rdf_parent / "onnx_rdf_classifier"
    if onnx_path.exists():
        rdf_path = onnx_path
    else:
        rdf_path = rdf_parent / "rdf_model.pkl"
    clsf, features = get_pretrained_rdf_clsf(rdf_path)

    is_pyknotics = _classify_cells_rdf_by_props(all_cell_properties, clsf, features)
    is_pyknotics = is_pyknotics.astype(dtype=bool)

    all_cell_properties["classified_pyknotic"] = is_pyknotics
    all_cell_properties.to_csv(testing_dir / "properties_updated.csv", index=False)

    # get list of unique file names
    file_names_tmp = set()
    for name in all_cell_properties["file_name"]:
        file_names_tmp.add(str(name))
    file_names: list[str] = sorted(file_names_tmp)
    del file_names_tmp

    update_pyknotic_counts(
        file_classifications=all_cell_properties,
        file_names=file_names,
        new_dir=testing_dir,
    )

    # ================================================================================ #
    # annotate images                                                                  #
    # ================================================================================ #

    images = [
        (im, IMAGE_DIR_512 / f"{im.stem}.json") for im in IMAGE_DIR_512.glob("*.png")
    ]

    annotated_images = TARGET_BATCH_DIR / "rdf_annotations"
    if annotated_images.exists():
        shutil.rmtree(annotated_images)
    annotated_images.mkdir(0o751)

    for image_path, meta_path in tqdm(
        images, desc="annotating images", total=len(images)
    ):
        name = image_path.stem

        # get image
        img = plt.imread(image_path, format="png")
        if not meta_path.exists():
            print(f"Could not find {meta_path=}")
            continue

        if len(img.shape) != 2:
            img = rgb2gray(img[..., :3])

        cell_props_tmp = all_cell_properties.where(
            all_cell_properties["file_name"] == name
        ).copy()
        cell_props_tmp.dropna(axis=0, how="all", inplace=True)
        cell_locs_tmp = cell_props_tmp.loc[:, ["i", "j"]].to_numpy(dtype=np.int64)
        cell_sigmas = cell_props_tmp.loc[:, ["sigma"]].to_numpy(dtype=np.int64).flatten()
        outcomes_tmp = (
            cell_props_tmp["classified_pyknotic"].to_numpy(dtype=bool).flatten()
        )

        fig, ax = plt.subplots(1, 2, layout="constrained")
        ax: list[Axes] = ax
        fig.set_figheight(4)
        fig.set_figwidth(8)
        fig.set_dpi(800)
        fig.suptitle(name, fontsize=14)
        ax[0].imshow(img, cmap=dapi_cmap, interpolation=None)
        ax[0].axis("off")
        ax[0].set_title("Original Image", fontsize=14)
        ax[1].imshow(img, cmap=dapi_cmap, interpolation=None)
        ax[1].set_title("Annotated Image", fontsize=14)
        ax[1].axis("off")
        for k, outcome in enumerate(outcomes_tmp):
            if outcome:
                [y, x] = cell_locs_tmp[k, :]
                s = cell_sigmas[k]
                c = plt.Circle(
                    (x, y), 2 * s, color="r", linewidth=1, fill=False, alpha=0.4
                )
                ax[1].add_patch(c)

        fig.savefig(annotated_images / f"{name}.png", dpi=800, format="png")
        plt.close(fig)


if __name__ == "__main__":
    main()
