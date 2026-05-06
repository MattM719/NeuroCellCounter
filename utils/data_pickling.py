#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classes to organize information for runML.py"""

# import packages
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


@dataclass
class StoredModel1:
    """Simplified and more compatible dataclass"""

    pca: PCA
    kmeans: KMeans
    target_class: int
    training_data: np.ndarray
    full_data: pd.DataFrame
    guessed_classifications: np.ndarray
    features: list
    training_data_path: str  # location data were accessed from
    training_data_sheet_name: Optional[str] = None
    class_name: str = "StoredModel1"


@dataclass
class StoredData2:
    """Simplified and more compatible dataclass"""

    classifier: RandomForestClassifier
    data: np.ndarray
    outcomes: np.ndarray
    ids: list
    features: list
    training_data_path: str  # location data were accessed from
    training_data_sheet_name: Optional[str]
    class_name: str = "StoredData2"


@dataclass
class StoredAnnotations1:
    """Simplified and more compatible dataclass"""

    image_path: str
    regions: list  # list of regions
    cells: list  # list of cells
    image: np.ndarray
    image_meta: dict
    image_unprocessed: np.ndarray
    class_name: str = "StoredAnnotations1"


def save_with_dataclass_pca(
    pkl_path: str,
    data: StoredModel1 | StoredAnnotations1,
) -> None:
    """Save values to a dataclass and store in a pkl file"""

    # re-initiate model
    if isinstance(data, StoredModel1):
        store = StoredModel1(**asdict(data))
    elif isinstance(data, StoredAnnotations1):
        store = StoredAnnotations1(**asdict(data))
    else:
        raise TypeError(f"Unrecognized dataclass type {type(data)}")

    # save to pickle file
    with open(pkl_path, "wb") as pkl:
        pickle.dump(store, pkl, pickle.HIGHEST_PROTOCOL)

    return None


def save_with_dataclass_rdf(
    pkl_path: str,
    classifier: RandomForestClassifier,
    training_data: np.ndarray,
    outcomes: list,
    ids: list,
    features: list,
    training_data_path: str,  # location data were accessed from
    training_data_sheet_name: Optional[str] = None,
) -> None:
    """Save values to a dataclass and store in a pkl file"""

    store = StoredData2(
        classifier=classifier,
        data=np.array(training_data),
        outcomes=np.array(outcomes),
        ids=[*ids],
        features=[*features],
        training_data_path=str(training_data_path),
        training_data_sheet_name=(
            str(training_data_sheet_name)
            if isinstance(training_data_sheet_name, str)
            else None
        ),
    )

    with open(pkl_path, "wb") as pkl:
        pickle.dump(store, pkl, pickle.HIGHEST_PROTOCOL)


def read_from_pickle2(pkl_path: str) -> StoredData2 | StoredModel1 | StoredAnnotations1:
    """Reads pickle file and returns values"""
    # with open(pkl_path, "rb") as pkl:
    #    stored_data = pickle.load(pkl_path, fix_imports=True)
    objects = []
    with open(pkl_path, "rb") as pkl:
        while True:
            try:
                objects.append(pickle.load(pkl))
            except EOFError:
                break

    return objects[0]
