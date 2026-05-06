#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Transforms data, for subsequent classification"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from utils.models import Cell


def get_features_one_cell(cell: Cell, features: List[str]) -> np.ndarray:
    """Gets features from cell instance

    sigma	avg_intensity	weighted_intensity	ideal_radius	eccentricity	perimeter	area
    """
    features = [getattr(cell, ft, np.nan) for ft in features]
    return np.array(features, dtype=np.float64)


def get_features_multiple_cells(cells: List[Cell], features: List[str]) -> np.ndarray:
    """Get array of features from list of cells

    sigma	avg_intensity	weighted_intensity	ideal_radius	eccentricity	perimeter	area
    """
    feature_vals = np.zeros([len(cells), len(features)], dtype=np.float64)
    for i, cell in enumerate(cells):
        feature_vals[i, :] = get_features_one_cell(cell, features)[:]
    return feature_vals


class LinearTransformer(object):
    """Reproduces a linear transformation, especially for PCA"""

    def __init__(self) -> None:
        """initializes LinearTransformer instance"""
        self.features: Optional[List[str]] = None

        # standard scaler
        self.standard_scaler_means: Optional[np.ndarray] = None
        self.standard_scaler_stds: Optional[np.ndarray] = None

        # PCA
        self.pca_vT: Optional[np.ndarray] = None

        # data
        self.original_data: Optional[pd.DataFrame | np.ndarray] = None
        self.transformed_data: Optional[pd.DataFrame | np.ndarray] = None

    def load_pca_transformation_data(self, path: Path) -> None:
        """reads an excel spreadsheet with info for transforming data"""
        if not isinstance(path, Path):
            path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Could not find file {path.name}")

        try:
            scaling_data = pd.read_excel(path, "scaling_params", index_col=0)
            vt_df = pd.read_excel(path, sheet_name="VT", header=None)

        except Exception as exc:
            print("The following Exception caused a FileNotFoundError")
            print(str(exc))
            raise FileNotFoundError(
                f"Unable to load a valid PCA transformation data from {path.name}"
            )

        # feature names
        self.features = scaling_data.columns.to_list()

        # scaling data
        means = scaling_data.loc[["Means"], :].to_numpy(dtype=np.float64).flatten()
        stds = scaling_data.loc[["StdDevs"], :].to_numpy(dtype=np.float64).flatten()
        self.standard_scaler_means = means
        self.standard_scaler_stds = stds

        # PCA transformation matrix
        vt = vt_df.to_numpy(dtype=np.float64)
        assert vt.shape[0] == vt.shape[1] == len(self.features)
        self.pca_vT = vt

    def transform_to_principal_components(
        self, data: pd.DataFrame | np.ndarray, n_pcs: Optional[int] = None
    ) -> pd.DataFrame | np.ndarray:
        """Transform data into its principal components"""
        if not self.has_standard_scaler or not self.has_pca_matrix:
            raise AttributeError("Data not loaded for standard scaler or PCA matrix")

        scaled_data = self._convert_to_zscore(data)
        all_pcs = self._convert_to_pcs(scaled_data)

        self.transformed_data = all_pcs

        if isinstance(all_pcs, pd.DataFrame):
            return all_pcs.iloc[:, :n_pcs]
        return all_pcs[:, :n_pcs]

    def transform_cells_to_pcs(
        self, cells: List[Cell], n_pcs: Optional[int] = None
    ) -> np.ndarray:
        """Transforms a list of cell instances into an np.NDArray of principal
        components.

        Cells become rows in the same order.
        """
        features = self.get_features_multiple_cells(cells)
        pcs = self.transform_to_principal_components(features, n_pcs)
        return pcs

    def get_features_one_cell(self, cell: Cell) -> np.ndarray:
        """Gets features from cell instance

        sigma	avg_intensity	weighted_intensity	ideal_radius	eccentricity	perimeter	area
        """
        return get_features_one_cell(cell, features=self.features)

    def get_features_multiple_cells(self, cells: List[Cell]) -> np.ndarray:
        """Get array of features from list of cells

        sigma	avg_intensity	weighted_intensity	ideal_radius	eccentricity	perimeter	area
        """
        return get_features_multiple_cells(cells, self.features)

    def _convert_to_zscore(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """rescales data to Z scores"""
        if not self.has_standard_scaler:
            raise AttributeError("Data not loaded for standard scaler")

        # validate types
        if is_df := isinstance(data, pd.DataFrame):
            indices = data.index.to_list()
            data = data.loc[:, self.features].to_numpy(dtype=np.float64)
        elif isinstance(data, np.ndarray):
            assert data.shape[1] == len(self.features)
        else:
            raise TypeError("data must be an NDArray or DataFrame")

        # convert data to Z score
        z_score = data - self.standard_scaler_means.reshape((1, -1))
        z_score /= self.standard_scaler_stds.reshape((1, -1))

        if is_df:
            z_score_df = pd.DataFrame(z_score, columns=self.features, index=indices)
            return z_score_df
        return z_score

    def _convert_to_pcs(
        self, data: pd.DataFrame | np.ndarray
    ) -> pd.DataFrame | np.ndarray:
        """Converts data to principal components"""
        if not self.has_pca_matrix:
            raise AttributeError("Data not loaded for PCA matrix")

        # validate types
        if is_df := isinstance(data, pd.DataFrame):
            indices = data.index.to_list()
            data = data.loc[:, self.features].to_numpy(dtype=np.float64)
        elif isinstance(data, np.ndarray):
            assert data.shape[1] == len(self.features)
        else:
            raise TypeError("data must be an NDArray or DataFrame")

        # conversion
        pcs = data @ self.pca_vT.T

        if is_df:
            pcs_df = pd.DataFrame(
                pcs,
                columns=[f"PC{i}" for i in range(pcs.shape[1])],
                index=indices,
            )
            return pcs_df
        return pcs

    @property
    def has_standard_scaler(self) -> bool:
        """True if able to scale data"""
        features: bool = self.features is not None
        scaler_means: bool = self.standard_scaler_means is not None
        scaler_stds: bool = self.standard_scaler_stds is not None
        return all([features, scaler_means, scaler_stds])

    @property
    def has_pca_matrix(self) -> bool:
        """True if able to scale data"""
        return (self.features is not None) and (self.pca_vT is not None)
