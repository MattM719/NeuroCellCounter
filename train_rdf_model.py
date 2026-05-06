#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Develop an RdF model to classify cells as pyknotic"""

import csv
import os
import shutil
from copy import deepcopy
from typing import Optional, Any
from pathlib import Path
import math
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt  # noqa F402
from matplotlib.axes import Axes  # noqa F402
from sklearn import metrics  # noqa F402
from sklearn.ensemble import RandomForestClassifier  # noqa F402
from sklearn.model_selection import StratifiedKFold  # noqa F402
from sklearn.metrics import precision_recall_curve, classification_report  # noqa F402
import shap  # noqa F402
from tqdm import tqdm  # noqa F402
from shap import Explanation, TreeExplainer  # noqa F402
from skimage.color import rgb2gray  # noqa F402

from utils.data_pickling import save_with_dataclass_rdf  # noqa F402
from analysis.find_annotations import find_points_in_png  # noqa F402
from utils.color_maps import dapi_cmap  # noqa F402
from utils import write_rdf_to_onnx  # noqa F402

SCALING_FACTOR = 2
BALANCE = True
BALANCE_SEED: Optional[int] = 710
SAVE_PKL: bool = True
SAVE_ONNX: bool = True

BASE_DIR = Path("/Volumes/Expansion_Usable/data/cell_counting")
ANNOTATIONS_DIR = BASE_DIR / "Aug24 Caffeine Pyk Annotations"

OUT_DIR = BASE_DIR / "results"

BATCH_DIR = OUT_DIR / "batch_CaffeineAug24TrainRDF_sigma22_filter_small_070825_115728"
IMAGE_DIR = BATCH_DIR / "downsampled_images"

PROPERTY_NAMES = [
    "sigma_um",
    "avg_intensity",
    "weighted_intensity",
    "total_intensity",
    "ideal_radius",
    "eccentricity",
    "perimeter",
    "area",
]


def match_images_annotations() -> list[tuple[Path, Path]]:
    """matches original images with their annotated PNG images

    Returns:
        list of tuples, (basename, ND2 image path, annotated PNG path)
    """
    # find annotations
    annotated_image_paths = ANNOTATIONS_DIR.glob("*.png")

    matched = []
    for annotated_path in annotated_image_paths:
        name = (
            annotated_path.stem.replace("_counted", "")
            .replace("-1", "")
            .replace(".nd2 (RGB)", "")
            .replace(" (1)", "")
        )

        img_path = IMAGE_DIR / f"{name}.png"
        if not img_path.exists():
            print(f"Could not find {img_path}")
            continue

        matched.append((name, img_path, annotated_path))

    return matched


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


def get_true_annotations(
    all_centers: np.ndarray, radii: np.ndarray, annotation_path: Path | str
) -> np.ndarray:
    """Processes annotated image and matches selected points to cell centers"""
    n_cells = all_centers.shape[0]
    pyk = np.zeros(n_cells, dtype=np.int64)
    cell_indices = np.arange(n_cells, dtype=np.int64)

    radii *= 1.2  # give some buffer
    radii /= 0.86  # convert to px

    # get the center points for true pyknotic cells.
    # These numbers are automatically rescaled to fit a 512x512 image, if needed.
    true_pyk_centers = find_points_in_png(
        str(annotation_path), verbose=False, plot=False
    )

    for i, j in true_pyk_centers:
        deltas = all_centers - np.array([[i, j]] * n_cells, dtype=np.float64)
        dists = np.sum(deltas**2.0, axis=1) ** 0.5  # units: px, shape: flat

        is_valid = dists < radii
        if not np.any(is_valid):
            continue

        valid_indices = cell_indices[is_valid]
        min_dist_idx = valid_indices[np.nanargmin(dists[valid_indices])]

        if dists[min_dist_idx] > 25:  # sanity check, px
            continue

        pyk[min_dist_idx] = 1

    return pyk


def plot_roc(
    fprs: list,
    tprs: list,
    tprs_interp: list,
    roc_aucs: list,
    fpr_axis: np.ndarray,
    out_dir: str,
    **kwargs,
) -> None:
    """Plots ROC curve

    Expects data_dict to include:
        tpr: list of True Positive Rates
        fpr: list of False Positive Rates

    Args:
        data_dict: dict of relevant values
        out_dir: directory to save file to
    """
    # set variables that may be adjusted by kwargs
    params = {
        "dpi": 600,
        "figsize": (10, 10),
        "filename": "ROC",
        "title": "Cross-Validation ROC of Random Forest",
        "title_font": 18,
        "img_format": "png",
    }
    params.update(kwargs)

    # initialize figure
    plt.figure(figsize=params["figsize"])

    for i in range(len(tprs)):
        tpr = tprs[i]
        fpr = fprs[i]
        roc_auc = roc_aucs[i]
        plt.plot(
            fpr,
            tpr,
            lw=1,
            alpha=0.3,
            label="ROC fold %d (AUC = %0.2f)" % (i, roc_auc),
        )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        lw=2,
        color="r",
        label="Chance",
        alpha=0.8,
    )

    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(fpr_axis, mean_tpr)
    std_auc = np.std(roc_aucs)
    plt.plot(
        fpr_axis,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs_interp, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(
        fpr_axis,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)
    plt.title(params["title"], fontsize=params["title_font"])
    plt.legend(loc="lower right", prop={"size": 15})
    path = os.path.join(out_dir, f"{params['filename']}.{params['img_format']}")
    plt.tight_layout()
    plt.savefig(
        path, format=params["img_format"], dpi=params["dpi"]
    )  # outputs ROC curve
    plt.close()
    os.chmod(path, 0o600)


def plot_pr(
    precisionFolds: list,
    recallFolds: list,
    reversed_mean_precision: np.ndarray,
    pr_aucs: list,
    mean_recall_axis: np.ndarray,
    precision_array: list,
    recall_array: np.ndarray,
    out_dir: str,
    **kwargs,
) -> None:
    """Plots PR curve.

    Args:
        precisionFolds: list of precisions from cross-validation
        recallFolds: list of recalls from cross-validation
        reversed_mean_precision: np.ndarray, calculated during cross-validation
        pr_aucs: list, also calculated during cross-validation
        mean_recall_axis: np.ndarray, recall values for mean
        precision_array: list, precision values
        recall_array: np.ndarray, recalls
        out_dir: str, directory to save image to
        **kwargs: updates params

    Returns: None
    """
    # set variables that may be adjusted by kwargs
    params = {
        "dpi": 600,
        "figsize": (10, 10),
        "filename": "PRCurve",
        "title": "Precision-Recall Curve",
        "title_font": 32,
        "img_format": "svg",
    }
    params.update(kwargs)

    # initialize figure
    plt.figure(figsize=params["figsize"])

    for i, (precisionFold, recallFold, pr_auc) in enumerate(
        zip(precisionFolds, recallFolds, pr_aucs)
    ):
        lab_fold = "PR fold %d AUPR=%.4f" % (i + 1, pr_auc)
        plt.plot(recallFold, precisionFold, alpha=0.3, label=lab_fold)

    reversed_mean_precision_temp = np.array(
        reversed_mean_precision, dtype=np.float64
    ) / len(precisionFolds)
    reversed_mean_precision_temp[0] = 1
    mean_auc_pr = metrics.auc(mean_recall_axis, reversed_mean_precision_temp)
    mean_precision = np.mean(precision_array, axis=0)
    std_precision = np.std(precision_array, axis=0)
    plt.fill_between(
        recall_array,
        mean_precision + std_precision,
        mean_precision - std_precision,
        alpha=0.3,
        linewidth=0,
        color="grey",
    )
    plt.plot(
        mean_recall_axis,
        ([reversed_mean_precision_temp])[0],
        label="Mean PR (AUC = %0.2f  $\\pm$ %0.2f)"
        % (mean_auc_pr, np.mean(std_precision)),
        lw=2,
        color="blue",
    )
    plt.xlabel("Recall (Sensitivity)", fontsize=28)
    plt.ylabel("Precision (PPV)", fontsize=28)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])

    plt.legend(loc="lower right", prop={"size": 24})
    plt.title(params["title"], fontsize=params["title_font"])
    path = os.path.join(out_dir, f"{params['filename']}.{params['img_format']}")
    plt.tight_layout()
    plt.savefig(path, format=params["img_format"], dpi=params["dpi"])
    plt.close()
    os.chmod(path, 0o600)


def train_test_n_classifiers(
    data: np.ndarray[np.float64],
    outcomes: np.ndarray[np.float64],
) -> dict[str, Any]:
    """train and cross validate models

    Number of folds is given by self.n_splits
    """
    # ROC data
    fprs = []  # actual calculated values (list of arrays)
    tprs = []  # actual calculated values (list of arrays)
    # interpolated True Positive Rates to match number of False Positive Rate datapoints
    tprs_interp = []
    rocThresholdss = []
    roc_aucs = []
    precision_axis = np.linspace(0, 1, 100)  # noqa F841
    fpr_axis = np.linspace(0, 1, 100)
    recall_array = np.linspace(0, 1, 100)
    mean_recall_axis = np.linspace(0, 1, 100)

    # PR data
    precisions = []
    recalls = []
    prThresholdss = []
    precision_array = []
    pr_aucs = []
    precisionFolds = []
    recallFolds = []

    # Other metrics
    y_real = []  # noqa F841
    y_proba = []  # noqa F841
    list_shap_values = []  # noqa F841
    list_test_sets = []  # noqa F841
    explainers = []  # noqa F841
    trains = []
    tests = []
    train_outcomes = []
    test_outcomes = []
    train_indices = []
    test_indices = []

    reversed_mean_precision = 0.0

    # =====================================
    # classifier                      #
    # =====================================

    # 5-fold cross validation to ensure ~70/30 T/T split
    # Stratified to ensure equal distrubution of +/- samples
    cv = StratifiedKFold(n_splits=5, shuffle=True)

    # Random Forest Classifier
    classifiers = [
        RandomForestClassifier(bootstrap=True, max_features="sqrt") for _ in range(5)
    ]

    # balance
    if True:
        pos = np.where(outcomes == 1, 1, 0)
        neg = np.where(outcomes == 0, 1, 0)
        nPos = np.sum(pos)
        nNeg = np.sum(neg)
        gcd = math.gcd(nPos, nNeg)
        posWeight, negWeight = (
            round(nNeg / gcd),
            round(nPos / gcd),
        )
        balanced = posWeight * pos + negWeight * neg
        balanced = np.array(balanced, dtype=np.int64).flatten()

    # =====================================
    # train and test                      #
    # =====================================

    for i, (train, test) in enumerate(cv.split(data, outcomes)):
        # train classifier
        balancedTraining = balanced[train]
        probas_ = (
            classifiers[i]
            .fit(
                data[train],
                outcomes[train],
                sample_weight=balancedTraining,
            )
            .predict_proba(data[test])
        )

        trains.append(deepcopy(data[train]))
        tests.append(deepcopy(data[test]))
        train_outcomes.append(deepcopy(outcomes[train]))
        test_outcomes.append(deepcopy(outcomes[test]))
        train_indices.append(deepcopy(train))
        test_indices.append(deepcopy(test))

        # Compute ROC curve
        fpr, tpr, rocThresholds = metrics.roc_curve(
            outcomes[test], probas_[:, 1], pos_label=1.0
        )  # arrays of True/False Positive Rates and Thresholds
        fprs.append(fpr)
        tprs.append(tpr)
        tprs_interp.append(np.interp(fpr_axis, fpr, tpr))
        rocThresholdss.append(rocThresholds)

        # compute PR curve
        precision, recall, prThresholds = precision_recall_curve(
            outcomes[test], probas_[:, 1], pos_label=1.0
        )
        precisions.append(precision)
        recalls.append(recall)
        prThresholdss.append(prThresholds)

        # backend calculations/corrections
        tprs_interp[-1][0] = 0.0  # inf --> 0
        roc_auc = metrics.auc(fpr, tpr)
        roc_aucs.append(roc_auc)

        precision_fold, recall_fold = (
            precision[::-1],
            recall[::-1],
        )  # reverse order of results
        prec_array = np.interp(recall_array, recall_fold, precision_fold)
        pr_auc = metrics.auc(recall_array, prec_array)
        precisionFolds.append(precision_fold)
        recallFolds.append(recall_fold)
        precision_array.append(prec_array)
        pr_aucs.append(pr_auc)

        reversed_mean_precision += np.interp(
            mean_recall_axis, recall_fold, precision_fold
        )
        reversed_mean_precision[0] = 0.0

    # =====================================
    # save values                         #
    # =====================================

    chosen_classifier = int(np.argmax(roc_aucs))

    # classifier and explainability mechanisms
    results = {
        "chosen_classifier": chosen_classifier,
        "classifier": classifiers[chosen_classifier],
        "roc_auc": roc_aucs[chosen_classifier],
        "pr_auc": pr_aucs[chosen_classifier],
        "roc_plotting": {
            "all_roc_aucs": roc_aucs,
            "fprs": fprs,
            "tprs": tprs,
            "tprs_interp": tprs_interp,
            "roc_thresholds": rocThresholdss,
        },
        "pr_plotting": {
            "all_pr_aucs": pr_aucs,
            "precisions": precisionFolds,
            "recalls": recallFolds,
            "reversed_mean_precision": reversed_mean_precision,
            "mean_recall_axis": mean_recall_axis,
            "precision_array": precision_array,
            "recall_array": recall_array,
        },
        "all_classifiers": classifiers,
        "training_data": trains,
        "testing_data": tests,
        "training_outcomes": train_outcomes,
        "testing_outcomes": test_outcomes,
        "training_indices": train_indices,
        "testing_indices": test_indices,
    }

    return results


def create_classification_report(
    classifier: RandomForestClassifier,
    test_data: np.ndarray,
    ground_truth: np.ndarray | list,
    path: str,
) -> None:
    """Builds a classification report for the model"""
    integers = False
    if isinstance(classifier, RandomForestClassifier):
        predictions = classifier.predict(test_data)
        integers = True
    else:
        raise TypeError(f"Unrecognized classifier type {type(classifier)}")

    # preprocess ground truth values
    ground_truth = np.array(list(ground_truth)).flatten()

    # preprocess outcomes
    predictions = predictions.flatten()
    ground_truth = ground_truth.flatten()
    if integers:
        predictions = predictions.astype(np.int64)
        ground_truth = ground_truth.astype(np.int64)

    assert len(ground_truth) == len(predictions), (
        "Must have a ground truth value for each prediction. "
        + f"Currently have {len(predictions)} predictions and "
        + f"{len(ground_truth)} ground truths."
    )

    # build classification report
    report: dict = classification_report(
        ground_truth,
        predictions,
        # labels=outcome_labels,
        output_dict=True,
        zero_division=np.nan,  # should this be 1.0?
    )
    reportdf = pd.DataFrame(report)

    if (ext := os.path.splitext(path)[1]) == ".csv":
        reportdf.to_csv(path)
    elif ext in [".xlsx", ".xls"]:
        reportdf.to_excel(path)
    else:
        raise TypeError(f"Unable to create a spreadsheet of filetype '{ext}'")
    os.chmod(path, 0o600)


def update_pyknotic_counts(
    file_classifications: pd.DataFrame, file_names: list[str]
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

    with open(BATCH_DIR / "counts_updated.csv", mode="w", encoding="utf-8") as f:
        writer = csv.writer(f, dialect="excel", lineterminator="\n")
        writer.writerows(new_counts)

    return None


def merge_annotated_cells_with_properties(
    all_cell_properties: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads annotated images and merges them with cell properties"""
    rdf_data: pd.DataFrame = pd.DataFrame()
    rdf_annotations: pd.DataFrame = pd.DataFrame()

    for name, image_path, annotation_path in match_images_annotations():
        centers, identifiers, cell_properties = get_image_data(all_cell_properties, name)
        rows = identifiers.index.to_list()

        # NOTE: confirms images are being read correctly
        ############################################
        # import sys
        # from reader import png_img_reader
        # rgb_img_actual = png_img_reader(image_path)
        # rgb_img_ann = png_img_reader(annotation_path)
        # img_actual = rgb_img_actual[2,...]
        # img_ann = rgb_img_ann[2,...]
        # fig, axs = plt.subplots(1,2)
        # [ax0, ax1] = axs.ravel()
        # ax0: Axes = ax0; ax1: Axes = ax1
        # ax0.imshow(img_actual, origin="upper", cmap='gray')
        # ax1.imshow(img_ann, origin="upper", cmap='gray')
        # ax0.set_title("Actual")
        # ax1.set_title("Annotated")
        # plt.show()
        # plt.close(fig)
        # sys.exit()
        ############################################

        # update dataframes
        if rdf_data.empty:
            rdf_data = cell_properties.copy()
        else:
            rdf_data = pd.concat([rdf_data, cell_properties], ignore_index=False)

        radii = cell_properties["ideal_radius"].to_numpy(dtype=np.float64).flatten()
        pyks = get_true_annotations(
            centers.to_numpy(dtype=np.int64), radii, annotation_path
        )
        all_cell_properties.loc[
            all_cell_properties["file_name"] == name, ["known_pyknotic"]
        ] = pyks

        rdf_annotations_tmp = pd.DataFrame(
            pyks.reshape((-1, 1)), columns=["pyknotic"], index=rows
        )
        if rdf_annotations.empty:
            rdf_annotations = rdf_annotations_tmp.copy()
        else:
            rdf_annotations = pd.concat(
                [rdf_annotations, rdf_annotations_tmp], ignore_index=False
            )

    rdf_annotated_data = rdf_data.copy()
    rdf_annotated_data.insert(
        loc=0, column="pyknotic", value=rdf_annotations["pyknotic"]
    )

    return (all_cell_properties, rdf_annotated_data)


def main():
    """main"""
    # create results directory
    training_dir = BATCH_DIR / f"training_data_{'balanced' if BALANCE else 'unbalanced'}"
    if not training_dir.exists():
        training_dir.mkdir(0o751)

    # read cell properties
    properties_path = BATCH_DIR / "properties.csv"
    if not properties_path.is_file():
        raise FileNotFoundError("Could not find properties.csv")

    all_cell_properties = pd.read_csv(properties_path)
    all_cell_properties = all_cell_properties.astype({"i": int, "j": int}, copy=True)

    # merge properties with pyknotic cell annotations, for annotated cells
    all_cell_properties, rdf_annotated_data = merge_annotated_cells_with_properties(
        all_cell_properties
    )
    indices_of_annotated_cells = np.array(rdf_annotated_data.index.to_list()).flatten()

    # rdf_annotated_data is a dataframe with the classification and properties of every
    # cell in each annotated image
    rdf_annotation_arr = rdf_annotated_data.iloc[:, 0].to_numpy(dtype=np.int64)
    rdf_data_arr = rdf_annotated_data.iloc[:, 1:].to_numpy(dtype=np.float64)

    # from the rdf_data_arr, include all TP cells & random sample of 3x as many TN cells
    rdf_data_arr_indices = np.arange(rdf_data_arr.shape[0], dtype=np.int64)
    if BALANCE:
        tp_indices = rdf_data_arr_indices[rdf_annotation_arr == 1]
        tn_indices = rdf_data_arr_indices[rdf_annotation_arr == 0]
        n_tp = tp_indices.shape[0]

        rng = np.random.default_rng(seed=BALANCE_SEED)
        rng.shuffle(tn_indices)
        balanced_indices = np.array(
            [*tp_indices, *tn_indices[: n_tp * 3]], dtype=np.int64
        )
        balanced_indices.sort()
    else:
        balanced_indices = rdf_data_arr_indices

    # ================================================================================ #
    # train classifier                                                                 #
    # ================================================================================ #

    classifier_results = train_test_n_classifiers(
        rdf_data_arr[balanced_indices, :],
        rdf_annotation_arr[balanced_indices],
    )

    plot_roc(
        fprs=classifier_results["roc_plotting"]["fprs"],
        tprs=classifier_results["roc_plotting"]["tprs"],
        tprs_interp=classifier_results["roc_plotting"]["tprs_interp"],
        roc_aucs=classifier_results["roc_plotting"]["all_roc_aucs"],
        fpr_axis=np.linspace(0, 1, 100),
        out_dir=training_dir,
        img_format="png",
    )
    plot_pr(
        precisionFolds=classifier_results["pr_plotting"]["precisions"],
        recallFolds=classifier_results["pr_plotting"]["recalls"],
        reversed_mean_precision=classifier_results["pr_plotting"][
            "reversed_mean_precision"
        ],
        pr_aucs=classifier_results["pr_plotting"]["all_pr_aucs"],
        mean_recall_axis=classifier_results["pr_plotting"]["mean_recall_axis"],
        precision_array=classifier_results["pr_plotting"]["precision_array"],
        recall_array=classifier_results["pr_plotting"]["recall_array"],
        out_dir=training_dir,
        img_format="png",
    )

    classifier: RandomForestClassifier = classifier_results["classifier"]

    training_data = classifier_results["training_data"][
        classifier_results["chosen_classifier"]
    ]
    testing_data = classifier_results["testing_data"][
        classifier_results["chosen_classifier"]
    ]
    training_outcomes = classifier_results["training_outcomes"][
        classifier_results["chosen_classifier"]
    ]
    testing_outcomes = classifier_results["testing_outcomes"][
        classifier_results["chosen_classifier"]
    ]
    training_indices = classifier_results["training_indices"][
        classifier_results["chosen_classifier"]
    ]

    create_classification_report(
        classifier=classifier,
        test_data=testing_data,
        ground_truth=np.array(testing_outcomes).flatten(),
        path=training_dir
        / (
            "classification_report_balanced_subset.csv"
            if BALANCE
            else "classification_report_all_candidates.csv"
        ),
    )

    # save classifier - pickle
    save_with_dataclass_rdf(
        pkl_path=training_dir / "rdf_model.pkl",
        classifier=classifier,
        training_data=training_data,
        outcomes=training_outcomes.tolist(),
        features=PROPERTY_NAMES,
        ids=indices_of_annotated_cells[balanced_indices][training_indices].tolist(),
        training_data_path=str(properties_path),
    )

    write_rdf_to_onnx(
        training_dir,
        classifier=classifier,
        training_data=training_data,
        outcomes=training_outcomes,
        features=PROPERTY_NAMES,
        ids=indices_of_annotated_cells[balanced_indices][training_indices].tolist(),
        outcome_names=None,
        predicted_outcomes=classifier.predict(training_data),
        save_training_data=True,
        warn_mismatched_predictions=True,
    )

    # reclassify all cells
    classifications = classifier.predict(
        all_cell_properties.loc[:, PROPERTY_NAMES].to_numpy(dtype=np.float64)
    )
    classifications = classifications.astype(dtype=bool)
    all_cell_properties["classified_pyknotic"] = classifications
    all_cell_properties.to_csv(BATCH_DIR / "properties_updated.csv", index=False)

    # trying to reduce memory
    file_classifications = all_cell_properties.loc[
        :, ["file_name", "classified_pyknotic"]
    ].copy()
    # del all_cell_properties, classifications

    # get list of unique file names
    file_names = set()
    for name in file_classifications["file_name"]:
        file_names.add(str(name))
    file_names: list[str] = sorted(file_names)

    update_pyknotic_counts(
        file_classifications=file_classifications, file_names=file_names
    )

    # ================================================================================ #
    # explain classifier                                                               #
    # ================================================================================ #

    if True:
        explainer = TreeExplainer(
            classifier,
            data=rdf_data_arr[balanced_indices, :],
            feature_names=PROPERTY_NAMES,
        )
        try:
            explanation: Explanation = explainer(
                rdf_data_arr[balanced_indices, :], check_additivity=True
            )
        except Exception:
            # setting check additivity to False
            explanation: Explanation = explainer(
                rdf_data_arr[balanced_indices, :], check_additivity=False
            )
        shap_values: np.ndarray = np.array(explanation.values[:, :, 1])

        # create explanations file
        explanations_path = training_dir / "explanations.xlsx"

        features_df = pd.DataFrame(
            rdf_data_arr[balanced_indices, :],
            columns=PROPERTY_NAMES,
            index=indices_of_annotated_cells[balanced_indices],
        )
        shap_df = pd.DataFrame(
            shap_values,
            columns=PROPERTY_NAMES,
            index=indices_of_annotated_cells[balanced_indices],
        )
        expected_vals_df = pd.DataFrame(
            np.array(explainer.expected_value).reshape((1, 2)),
            columns=["Non-Pyknotic", "Pyknotic"],
        )

        outcomes = np.zeros([shap_values.shape[0], 2], dtype=np.int64)
        outcomes[:, 0] = rdf_annotation_arr.flatten()[balanced_indices]
        outcomes[:, 1] = classifier.predict(rdf_data_arr[balanced_indices, :]).flatten()[
            :
        ]
        outcomes_df = pd.DataFrame(
            outcomes,
            columns=["True", "Predicted"],
            index=indices_of_annotated_cells[balanced_indices],
        )

        with pd.ExcelWriter(explanations_path, engine="xlsxwriter", mode="w") as xl:
            features_df.to_excel(xl, sheet_name="Features", index=False)
            shap_df.to_excel(xl, sheet_name="SHAP", index=False)
            expected_vals_df.to_excel(xl, sheet_name="Expect", index=False)
            outcomes_df.to_excel(xl, sheet_name="Outcomes", index_label="Cell UID")
        explanations_path.chmod(0o660)
        del features_df, shap_df, expected_vals_df, outcomes, outcomes_df

        # create summary plot
        try:
            plt.figure()
            shap.summary_plot(
                shap_values=shap_values,
                features=rdf_data_arr[balanced_indices, :],
                feature_names=PROPERTY_NAMES,
                plot_type="violin",
                # max_display=50,
                show=False,
            )
            fig = plt.gcf()
            fig.tight_layout()
            path_summary_plot = training_dir / "summary_plot.svg"
            # plt.show()
            fig.savefig(path_summary_plot, format="svg")
            plt.close(fig)
            path_summary_plot.chmod(0o660)
        except Exception:
            print(traceback.format_exc())
            print("Failed to create summary plot. Moving on.")

    # ================================================================================ #
    # annotate images                                                                  #
    # ================================================================================ #

    images = [(im, IMAGE_DIR / f"{im.stem}.json") for im in IMAGE_DIR.glob("*.png")]

    annotated_images = (
        BATCH_DIR / f"rdf_annotations_{'balanced' if BALANCE else 'unbalanced'}"
    )
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

        # with open(meta_path, mode="r", encoding="utf-8") as f:
        #     meta_data = json.load(f)

        if len(img.shape) != 2:
            img = rgb2gray(img[..., :3])

        cell_props_tmp = all_cell_properties.where(
            all_cell_properties["file_name"] == name
        ).copy()
        cell_props_tmp.dropna(axis=0, how="all", inplace=True)
        cell_locs_tmp = cell_props_tmp.loc[:, ["i", "j"]].to_numpy(dtype=np.int64)
        cell_sigmas = (
            cell_props_tmp.loc[:, ["sigma_um"]].to_numpy(dtype=np.int64).flatten()
        )
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
