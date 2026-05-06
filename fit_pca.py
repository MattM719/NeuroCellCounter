#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training PCA and linear models.

@author: Matthew Magoon
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg

# data info
DATA_PATH = (...)  # FIXME: Provide path to properties.csv or properties_updated.csv

INSTRUCTIONS = """X_(m x n) is the original values, m samples with n features

A_(m x n) is the Z scores for the data in X

SVD:
A = U S Vt, Vt is already transposed as written
    dimensions:
    U_(m x m)
    S_(m x n) is a diagonal matrix of eigenvalues, with zeroes everwhere else
        linalg.svd gives "s", the flat array of values on the diagonal
    Vt_(n x n) is the same as V.T

V (= Vt.T) is the column matrix of PC axes (which are orthonormal)
    PC0 describes the axis given by V[:,0] = Vt[0,:] in R^n space
    Definitionally, V.T = V^-1

PCA:
Let P be the (m x n) matrix of all n principal components for each sample
P = A V = A Vt.T
    Note, this also means P = U S and A = P Vt

Let K_(m x k), k <= n, be the matrix of the first k PCs.
    K = P[:,:k]

Let J_(m x n-k) contain the mean value of each column in P that isn't in
the k principal components. J has the same set of values in each row.

Thus, [K J] ~ P and A ~ [K J] Vt


*** Takeaways ***

To get the k principal components from A:
P = A Vt.T   and   K = P[:,:k]

To approximate the values of A from K:
A ~ [K J] Vt
"""


"""# Z-scores
scaler = StandardScaler()
A = scaler.fit_transform(X)
z_mean = scaler.mean_
z_var = scaler.var_

# do SVD
U, s, Vt = linalg.svd(A, full_matrices=True, compute_uv=True)
print(f"{U.shape=}") # 100 x 100
print(f"{s=}") # len = r
print(f"{Vt.shape=}") # 8 x 8
S = np.zeros(A.shape) # 100 x 8
S[:len(s),:len(s)] = np.diag(s) # create S (sparse diag matrix of eig vals)

print(np.allclose(A, U @ S @ Vt)) # True
"""


def save_transformation_data(
    ident_info: pd.DataFrame,
    original: pd.DataFrame,
    transformed: pd.DataFrame,
    outcomes: np.ndarray,
    z_means: np.ndarray,
    z_stds: np.ndarray,
    U: Optional[np.ndarray],
    S: np.ndarray,
    Vt: np.ndarray,
    features: List[str],
) -> None:
    """Saves data for recreating or reversing transformation"""
    # save instructions
    instructions_path = DATA_PATH.parent / "pca_transformation_instructions.txt"
    if not instructions_path.exists():
        with open(instructions_path, mode="w", encoding="utf-8") as f:
            f.write(INSTRUCTIONS)
        instructions_path.chmod(0o660)

    # scaling info
    scaling_df = pd.DataFrame(
        np.vstack((z_means.flatten(), z_stds.flatten())),
        columns=features,
        index=["Means", "StdDevs"],
    )

    # SVD matrices
    u_df = None if U is None else pd.DataFrame(U)
    if len(S.shape) == 1:
        sigma_df = pd.DataFrame(S.reshape((1, -1)))
    else:
        sigma_df = pd.DataFrame(S)
    vt_df = pd.DataFrame(Vt)
    v_df = pd.DataFrame(Vt.T)

    # save transformation data
    output_path = DATA_PATH.parent / "pca_transformation.xlsx"
    with pd.ExcelWriter(output_path, engine="xlsxwriter", mode="w") as xl:
        scaling_df.to_excel(excel_writer=xl, sheet_name="scaling_params")
        if u_df is not None:
            u_df.to_excel(excel_writer=xl, sheet_name="U", header=False, index=False)
        sigma_df.to_excel(excel_writer=xl, sheet_name="Sigma", header=False, index=False)
        vt_df.to_excel(excel_writer=xl, sheet_name="VT", header=False, index=False)
        v_df.to_excel(excel_writer=xl, sheet_name="V", header=False, index=False)
        original.to_excel(excel_writer=xl, sheet_name="Original_data", index=False)
        transformed.to_excel(excel_writer=xl, sheet_name="Transformed_data", index=False)
        outcomes.to_excel(excel_writer=xl, sheet_name="Outcomes", index=False)
        ident_info.to_excel(excel_writer=xl, sheet_name="Identities", index=False)
    output_path.chmod(0o660)

    return None


def main():
    """main"""
    # read data
    full_data = pd.read_csv(DATA_PATH)
    ident_info = full_data.loc[:, ["file_name", "uid", "region_id", "i", "j"]]

    # organize data
    properties_df = full_data.loc[
        :,
        [
            "sigma",
            "avg_intensity",
            "weighted_intensity",
            "ideal_radius",
            "eccentricity",
            "perimeter",
            "area",
        ],
    ]
    properties = properties_df.to_numpy(dtype=np.float64)
    outcomes_df = full_data["known_pyknotic"]
    outcomes = outcomes_df.to_numpy(dtype=np.int64).flatten()

    # rescale data
    z_means = np.average(properties, axis=0)
    z_stds = np.std(properties, axis=0)
    scaled_data = (properties - z_means.reshape((1, -1))) / z_stds.reshape((1, -1))

    # transform data
    try:
        U, s, Vt = linalg.svd(
            scaled_data, full_matrices=True, compute_uv=True, lapack_driver="gesdd"
        )
    except ValueError:
        U, s, Vt = linalg.svd(
            scaled_data, full_matrices=False, compute_uv=True, lapack_driver="gesdd"
        )
    # print(f"{U.shape=}")  # m x m
    # print(f"{s=}")  # len = rank(data)
    # print(f"{Vt.shape=}")  # n x n
    S = np.zeros(scaled_data.shape)  # m x n
    S[: len(s), : len(s)] = np.diag(s)  # create S (sparse diag matrix of eig vals)
    # print(np.allclose(scaled_data, U @ S @ Vt)) # True

    transformed = scaled_data @ Vt.T
    transformed_df = pd.DataFrame(
        transformed,
        columns=[f"PC{i}" for i in range(transformed.shape[1])],
    )

    # save data
    save_transformation_data(
        ident_info=ident_info,
        original=properties_df,
        transformed=transformed_df,
        outcomes=outcomes_df,
        z_means=z_means,
        z_stds=z_stds,
        U=None,
        S=s,
        Vt=Vt,
        features=properties_df.columns.to_list(),
    )

    # plot
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(
        transformed[outcomes == 0, 0],
        transformed[outcomes == 0, 1],
        c="b",
        s=0.8,
        marker="o",
        facecolor="none",
        alpha=0.6,
    )
    ax.scatter(
        transformed[outcomes == 1, 0],
        transformed[outcomes == 1, 1],
        c="r",
        s=0.8,
        marker="o",
        facecolor="none",
        alpha=0.6,
    )

    ax.set_xlabel("PC0")
    ax.set_ylabel("PC1")
    # ax.set_zlabel('PC2')

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
