#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to help optimize cell parameters"""

import os
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt  # noqa E402
from sklearn.cluster import KMeans  # noqa E402
from sklearn.decomposition import PCA  # noqa E402

from utils.decorators import timer_s  # noqa E402
from utils.data_pickling import StoredModel1  # noqa E402


BASE_DIR = ...  # FIXME: path to data directory
CELL_PROPS = os.path.join(BASE_DIR, "results", "properties.csv")


@timer_s
def optimize_with_pca(
    training_data_path: str,
    balance_classes: bool = True,
    n_components: int = 2,
    n_clusters: int = 2,
) -> tuple[PCA, KMeans, int, StoredModel1]:
    """Uses unsupervised ML (PCA and k-means clustering) to classify cells as pyknotic or
    non-pyknotic
    """
    # read data
    df = pd.read_csv(training_data_path)
    columns = [
        "sigma",
        "avg_intensity",
        "total_intensity",
        "ideal_radius",
        "eccentricity",
        "perimeter",
        "area",
    ]
    data_orig = df.loc[:, columns].to_numpy(dtype=np.float64)
    data = data_orig.copy()
    classified_pyknotic = df.classified_pyknotic.to_numpy(
        dtype=bool
    )  # which cells were labelled pyknotic
    n_samples, n_features = np.shape(data)
    n_cases = int(np.sum(classified_pyknotic))

    if balance_classes:
        control_indices = np.arange(n_samples)[~classified_pyknotic]
        np.random.shuffle(control_indices)
        keep = np.sort(control_indices[:n_cases])
        data = np.vstack((data[keep, :], data[control_indices, :]))

    # run PCA
    pca = PCA(n_components=n_components)
    balanced_reduced_data = pca.fit_transform(data)
    reduced_data = pca.transform(data_orig)

    if False:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            reduced_data[~classified_pyknotic, 0],
            reduced_data[~classified_pyknotic, 1],
            reduced_data[~classified_pyknotic, 2],
            facecolors="none",
            edgecolors="b",
            alpha=0.2,
        )
        ax.scatter(
            reduced_data[classified_pyknotic, 0],
            reduced_data[classified_pyknotic, 1],
            reduced_data[classified_pyknotic, 2],
            c="r",
            marker="^",
            alpha=0.8,
        )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.show()
        plt.close()

    # cluster
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init="auto",
        random_state=None,
        algorithm="lloyd",
    )
    kmeans.fit(balanced_reduced_data)
    categories = kmeans.predict(reduced_data)

    correct = np.zeros(2, dtype=np.float64)
    for i, category in enumerate(np.unique(categories)):
        n = np.sum(categories[classified_pyknotic] == category)
        correct[i] = 100 * n / n_cases
        if False:
            print(f"category {category}: {int(n)}/{n_cases} ({round(correct[i])}%)")

    # get centers
    centers_pca = kmeans.cluster_centers_
    midpoint_pca = np.mean(centers_pca, axis=0).reshape((1, -1))
    centers_orig = pca.inverse_transform(centers_pca)
    midpoint_orig = pca.inverse_transform(midpoint_pca).flatten()
    pyknotic_class = int(np.argmax(correct))

    thresholds = {c: float(v) for c, v in zip(columns, midpoint_orig)}
    pyk_centers = {c: float(v) for c, v in zip(columns, centers_orig[pyknotic_class, :])}
    if False:
        print("\nThresholds:")
        pprint(thresholds)
        print("\nPyknotic centers:")
        pprint(pyk_centers)

    # build dataframes
    if False:
        pcs_out = np.hstack((data, pcs))  # noqa F821
        pcs_df = pd.DataFrame(pcs_out, columns=[*columns, *pc_names])  # noqa F821

        pc_feature_df = pd.DataFrame(
            map_pc_to_feature, index=pc_names, columns=columns  # noqa F821
        )
        feature_pc_df = pd.DataFrame(
            map_feature_to_pc, index=columns, columns=pc_names  # noqa F821
        )

        sigma_df = pd.DataFrame(sigma.reshape((1, -1)), columns=pc_names)  # noqa F821

        output_path = os.path.join(BASE_DIR, "results", "pca.xlsx")
        with pd.ExcelWriter(output_path, "xlsxwriter", mode="w") as xl:
            df.to_excel(xl, sheet_name="features", index=False)
            pcs_df.to_excel(xl, sheet_name="pcs", index=False)
            pc_feature_df.to_excel(xl, sheet_name="pcs_to_features")
            feature_pc_df.to_excel(xl, sheet_name="features_to_pcs")
            sigma_df.to_excel(xl, sheet_name="eigenvalues", index=False)
            df.classified_pyknotic.to_excel(
                xl, sheet_name="classifications", index=False
            )

    # save to dataclass
    data_class = StoredModel1(
        pca=pca,
        kmeans=kmeans,
        target_class=pyknotic_class,
        training_data=data,
        full_data=df,
        guessed_classifications=classified_pyknotic,
        features=columns,
        training_data_path=training_data_path,
    )

    return (pca, kmeans, pyknotic_class, data_class)


if __name__ == "__main__":
    optimize_with_pca(CELL_PROPS)
