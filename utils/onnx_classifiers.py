#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""methods to more reliably save random forest classifiers with ONNX

Data are stored with the following scheme

 - onnx_rdf_classifier/
 | - onnx_rdf_classifier_meta_data.json
 | - onnx_rdf_classifier.onnx
 | - onnx_rdf_classifier_training_data.csv
 | - onnx_rdf_classifier_outcomes.csv
"""

import json
import shutil
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, List, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnx.onnx_ml_pb2 import ModelProto

# paths
ONNX_RDF_CLF_DIR: str = "onnx_rdf_classifier"
ONNX_RDF_CLF_FILE: str = "onnx_rdf_classifier.onnx"
ONNX_RDF_CLF_META: str = "onnx_rdf_classifier_meta_data.json"
ONNX_RDF_CLF_TRAINING_DATA: str = "onnx_rdf_classifier_training_data.csv"
ONNX_RDF_CLF_TRAINING_OUTCOMES: str = "onnx_rdf_classifier_outcomes.csv"

# params
ONNX_TARGET_OPSET: int = 21


# ==================================================================================== #
# Classes                                                                              #
# ==================================================================================== #


@dataclass
class RDFMetaData:
    """class to organize meta data for RandomForestClassifier"""

    classifier_path: str  # relative path to classifier
    training_data_path: Optional[str]  # relative path to data can be accessed from
    training_outcomes_path: Optional[str]  #
    ids: Optional[List[int | str]]
    features: List[str]
    outcome_names: Optional[List[str]]
    class_name: str = "RDFMetaData"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        return cls(
            classifier_path=data.get("classifier_path"),
            training_data_path=data.get("training_data_path"),
            training_outcomes_path=data.get("training_outcomes_path"),
            ids=data.get("ids"),
            features=data.get("features"),
            outcome_names=data.get("outcome_names"),
            class_name=data.get("class_name"),
        )


class OnnxAsSklean(object):
    """Reads ONNX model and associated meta data to act like an sk-learn model for
    predicting outcomes
    """

    def __init__(self, path: Optional[Path] = None, **kwargs) -> None:
        """Creates an OnnxAsSklean instance for treating an ONNX classifier like an
        sklearn classifier

        Parameters:
        ----------
        path: If a path to a `ONNX_RDF_CLF_DIR` or 'onnx_rdf_classifier/' directory is
        provided, reads data from the expected file structure.
        """
        # immutable initial values - ONNX
        self.onnx_file_path: Optional[Path] = None
        self._sess: Optional[rt.InferenceSession] = None
        self._sess_input_name: Optional[str] = None  # probably: 'float_input'
        self._sess_label_name: Optional[str] = None  # default: 'output_label'
        # self.onnx_model: Optional[ModelProto] = None

        # immutable initial values - meta
        self.meta_path: Optional[Path] = None
        self._meta: Optional[RDFMetaData] = None

        # set kwargs
        self.valid_onnx_dir_name: str = ONNX_RDF_CLF_DIR
        self.onnx_model_name: str = ONNX_RDF_CLF_FILE
        self.onnx_meta_name: str = ONNX_RDF_CLF_META
        self.pred_class: int = 1
        valid_kwargs = [
            "valid_onnx_dir_name",
            "onnx_model_name",
            "onnx_meta_name",
            "pred_class",
        ]
        for kw in kwargs.keys():
            if kw not in valid_kwargs:
                raise KeyError(f"Invalid kwarg: '{kw}'")
        self.__dict__.update(kwargs)

        # set path (None is acceptable) and read data if available
        self.onnx_dir = path
        if isinstance(self.onnx_dir, Path):
            self.add_directory()

        return None

    def _build_sess(self):
        """builds a session"""
        if not isinstance(self.onnx_file_path, Path):
            raise ValueError("`onnx_file_path` not specified")
        if not self.onnx_file_path.is_file():
            raise FileNotFoundError(f"Could not find file {self.onnx_file_path=}")
        if self.onnx_file_path.suffix != ".onnx":
            raise FileNotFoundError(f"Invalid ONNX file path '{self.onnx_file_path}'")

        self._sess = rt.InferenceSession(
            self.onnx_file_path, providers=["CPUExecutionProvider"]
        )
        self._sess_input_name = self._sess.get_inputs()[0].name
        self._sess_label_name = self._sess.get_outputs()[0].name

        assert isinstance(self._sess, rt.InferenceSession)
        assert isinstance(self._sess_input_name, str)
        assert isinstance(self._sess_label_name, str)

    def _predict(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """actual prediction"""
        if self._sess is None:
            self._build_sess()
        sess: rt.InferenceSession = self._sess
        label_name: str = self._sess_label_name
        input_name: str = self._sess_input_name
        if sess is None or label_name is None or input_name is None:
            raise ValueError(
                f"session was not properly loaded {sess=}, {label_name=}, {input_name=}"
            )

        # preds = sess.run([label_name], {input_name: data.astype(np.float32)})[0]
        [preds, _probas] = sess.run(None, {input_name: data.astype(np.float32)})

        preds: np.ndarray = preds.astype(dtype=np.int64)
        probas = np.array([d[self.pred_class] for d in _probas], dtype=np.float64)

        return (preds, probas)

    def predict(
        self, data: pd.DataFrame | np.ndarray, ignored_outcomes: Optional[Any] = None
    ) -> np.ndarray:
        """Acts like predict() in an SKLearn model"""
        x = self._clean_data(data)
        preds, probas = self._predict(x)
        return preds

    def predict_probas(
        self, data: pd.DataFrame | np.ndarray, ignored_outcomes: Optional[Any] = None
    ) -> np.ndarray:
        """Acts like predict_probas() in an SKLearn model"""
        x = self._clean_data(data)
        preds, probas = self._predict(x)
        return probas

    def _clean_data(self, data: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Cleans data to match expected input type"""
        if not isinstance(self._meta, RDFMetaData):
            raise ValueError("Must read meta before receiving data")

        features: Optional[List[str]] = self._meta.features
        if not isinstance(features, list):
            msg = "No features provided, unable to validate data format."
            warnings.warn(msg, category=UserWarning)

        if isinstance(data, np.ndarray) and isinstance(features, list):
            if data.shape[1] == len(features):
                return data
            raise IndexError(
                "Received data as an NDArray with incorrect number of features. "
                + "Unable to clean improperly formatted data."
            )
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, pd.DataFrame):
            if isinstance(features, list):
                data = data.loc[:, features]
            return data.to_numpy(dtype=np.float64)

        raise TypeError(f"Could not process data type {type(data)}")

    def _read_meta(self, path: Optional[Path | str] = None) -> None:
        """reads meta data into an RDFMetaData instance"""
        if isinstance(path, str):
            path = Path(path)

        if isinstance(path, Path) and path.is_file() and path.suffix == ".json":
            pass
        elif self.meta_path is not None:
            path = self.meta_path
        elif self.onnx_dir is not None:
            path = self.onnx_dir / self.onnx_meta_name
            if not path.is_file():
                raise FileNotFoundError(f"Could not find meta: '{path}'")
        else:
            raise FileNotFoundError("No path information provided")

        with open(path, mode="r", encoding="utf-8") as f:
            txt = f.read()

        meta_dict = json.loads(txt)

        self.meta_path = path
        self._meta = RDFMetaData.from_dict(meta_dict)

    @property
    def meta(self) -> Optional[RDFMetaData]:
        return self._meta

    @property
    def features(self) -> Optional[List[str]]:
        if self.meta is None:
            return None
        return self.meta.features

    def add_directory(self, path: Optional[Path] = None) -> None:
        """Reads data from an ONNX directory into the class instance"""
        if path is not None:
            self.onnx_dir = path

        self.meta_path = self.onnx_dir / self.onnx_meta_name
        self._read_meta()

        self.onnx_file_path = self.onnx_dir / self.onnx_model_name
        self._build_sess()

    def _is_onnx_dir_valid(self, path: Any) -> bool:
        """True if path is a valid ONNX dir"""
        if path is None:
            return True
        elif isinstance(path, Path):
            pass
        else:
            return False

        if not path.is_dir():
            return False
        if path.name != self.valid_onnx_dir_name:
            return False

        expected_suffixes = [
            f".{self.onnx_meta_name.split('.')[-1]}",  # .json
            f".{self.onnx_model_name.split('.')[-1]}",  # .onnx
        ]

        found_suffixes = [x.suffix for x in path.glob("*")]

        for suf in expected_suffixes:
            if suf not in found_suffixes:
                print(f"Could not find a file with '{suf}' in {path}")
                return False

        return True

    @property
    def onnx_dir(self) -> Optional[Path]:
        path = self._onnx_dir
        assert self._is_onnx_dir_valid(path)
        return path

    @onnx_dir.setter
    def onnx_dir(self, path: Optional[Path] | Any):
        # bail out None type
        if path is None:
            self._onnx_dir: Optional[Path] = None
            return None

        # fix str and validate path
        if isinstance(path, str) and path != "":
            path = Path(path)
        if not isinstance(path, Path):
            raise TypeError("If not None, onnx_dir must be Path-like")

        path = path.absolute()
        if not self._is_onnx_dir_valid:
            raise FileNotFoundError(
                f"Could not find '{path}' matching '.../{self.valid_onnx_dir_name}/'"
            )

        self._onnx_dir: Optional[Path] = path


# ==================================================================================== #
# Save classifier                                                                      #
# ==================================================================================== #


def _save_onnx_rdf_classifier(
    path: Path, classifier: RandomForestClassifier, features: List[str]
) -> None:
    # save classifier - ONNX
    initial_type = [("float_input", FloatTensorType([None, len(features)]))]
    # final_type = [("output_label", Int64TensorType([None, 1]))]  # output_probability
    doc_string = f"RandomForestClassifier with inputs: {features}"
    onx: ModelProto = convert_sklearn(
        classifier,
        doc_string=doc_string,
        initial_types=initial_type,
        target_opset=ONNX_TARGET_OPSET,
        # final_types=final_type,
    )

    with open(path, "wb") as f:
        f.write(onx.SerializeToString())

    return None


def write_rdf_to_onnx(
    out_dir: Path,
    classifier: RandomForestClassifier,
    training_data: Optional[pd.DataFrame | np.ndarray],
    outcomes: Optional[np.ndarray | pd.DataFrame | pd.Series],
    features: Optional[List[str]] = None,
    ids: Optional[List[int | str]] = None,
    outcome_names: Optional[List[str]] = None,
    predicted_outcomes: Optional[np.ndarray] = None,
    save_training_data: bool = True,
    warn_mismatched_predictions: bool = False,
) -> None:
    """Saves RdF classifier and related meta data using ONNX for enhanced
    reproducibility.

    Parameters:
    ----------
    out_dir: directory that a subdirectory will be saved to.

    classifier: the RandomForestClassifier instance being saved.

    training_data: the data used to train the random forest classifier.

    outcomes: actual outcomes for training data.

    features: required if training_data is None or is an NDArray.

    ids: required only if training data is an NDArray and will be saved.

    predicted_outcomes: it is recommended to supply the predicted outcomes to confirm
    the ONNX classifier gives the same predictions.

    save_training_data: can be set to False to prevent files from being saved, for
    example if they contain protected data but the user wants to share the classifier.

    warn_mismatched_predictions: If False, mismatch raises an error. If True, mismatch
    only raises a warning.
    """
    # validate out_dir
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)
    elif not isinstance(out_dir, Path):
        raise TypeError("out_dir must be a Path")

    # confirm out_dir
    if not out_dir.is_dir():
        raise FileNotFoundError(
            "out_dir must be a directory that already exists. "
            + "This will be the parent for the directory that will be created "
            + "to contain the classifier ONNX file and associated meta data."
        )

    # validate other types
    assert isinstance(classifier, RandomForestClassifier)
    assert training_data is None or isinstance(training_data, (pd.DataFrame, np.ndarray))
    assert outcomes is None or isinstance(
        outcomes, (pd.Series, pd.DataFrame, np.ndarray)
    )
    assert ids is None or isinstance(ids, list)
    assert outcome_names is None or (
        isinstance(outcome_names, list) and len(outcome_names) == 2
    )
    assert predicted_outcomes is None or isinstance(predicted_outcomes, np.ndarray)
    assert isinstance(save_training_data, bool)

    # validate features
    if isinstance(training_data, pd.DataFrame) and isinstance(features, list):
        assert features == training_data.columns.to_list()
    elif isinstance(training_data, pd.DataFrame):
        features = training_data.columns.to_list()
    elif isinstance(features, list):
        pass
    else:
        raise ValueError(
            "If features cannot be obtained from a training_data DataFrame, "
            + "they must be explicitly supplied as a list."
        )

    if isinstance(training_data, (pd.DataFrame, np.ndarray)):
        assert len(features) == training_data.shape[1]

    # create paths
    onnx_dir = out_dir.absolute() / ONNX_RDF_CLF_DIR
    onnx_file_path = onnx_dir / ONNX_RDF_CLF_FILE
    meta_path = onnx_dir / ONNX_RDF_CLF_META
    training_data_path = onnx_dir / ONNX_RDF_CLF_TRAINING_DATA
    outcomes_path = onnx_dir / ONNX_RDF_CLF_TRAINING_OUTCOMES

    if onnx_dir.exists():
        shutil.rmtree(onnx_dir)
    onnx_dir.mkdir(0o751)

    # save classifier
    _save_onnx_rdf_classifier(
        path=onnx_file_path, classifier=classifier, features=features
    )

    # save training and outcomes data, if applicable
    training_data_df: Optional[pd.DataFrame] = None
    outcomes_df: Optional[pd.DataFrame] = None
    if save_training_data:
        # format training data as a dataframe
        if isinstance(training_data, pd.DataFrame):
            training_data_df = training_data
            if ids is not None and training_data.index.to_list() != ids:
                training_data_df.set_index(ids, inplace=True)
        elif isinstance(training_data, np.ndarray):
            training_data_df = pd.DataFrame(training_data, columns=features, index=ids)

        # save training data
        if isinstance(training_data_df, pd.DataFrame):
            training_data_df.to_csv(
                training_data_path,
                index=True,
                index_label="Identities",
                encoding="utf-8",
            )
            training_data_path.chmod(0o660)

        # format outcomes, if applicable AND training data were saved
        if isinstance(training_data_df, pd.DataFrame) and isinstance(
            outcomes, np.ndarray
        ):
            outcomes_df = pd.DataFrame(
                outcomes.reshape((-1, 1)),
                columns=(["outcome"] if outcome_names is None else [outcome_names[-1]]),
                index=ids,
            )

        # save outcomes
        if isinstance(outcomes_df, pd.DataFrame):
            outcomes_df.to_csv(
                outcomes_path, index=True, index_label="Identities", encoding="utf-8"
            )
            outcomes_path.chmod(0o660)

    # test classifier
    if isinstance(training_data_df, pd.DataFrame) and isinstance(outcomes, np.ndarray):
        sess = rt.InferenceSession(onnx_file_path, providers=["CPUExecutionProvider"])
        input_name: str = sess.get_inputs()[0].name  # 'float_input'
        label_name = sess.get_outputs()[0].name  # 'output_label'
        pred_onx = sess.run(
            [label_name], {input_name: training_data.astype(np.float32)}
        )[0]
        if np.all(predicted_outcomes == np.array(pred_onx)):
            pass
        elif warn_mismatched_predictions:
            _outcomes = np.vstack((outcomes, predicted_outcomes, np.array(pred_onx))).T
            _outcomes_df = pd.DataFrame(
                _outcomes, columns=["GroundTruth", "sk-learn", "onnx"]
            )
            _outcomes_df.index = training_data_df.index
            _outcomes_df.to_csv(onnx_dir / "prediction_mismatch.csv", index=True)
            warnings.warn(
                "Failed test, ONNX and sklearn predictions did not match.",
                category=UserWarning,
            )
        else:
            raise AssertionError(
                "Failed test, ONNX and sklearn predictions did not match."
            )
        print("ONNX file passed test!")

    # save meta
    onnx_file_path_str = str(onnx_file_path.relative_to(onnx_dir))
    training_data_path_str = (
        None
        if training_data_df is None
        else str(training_data_path.relative_to(onnx_dir))
    )
    outcomes_path_str = (
        None if outcomes_df is None else str(outcomes_path.relative_to(onnx_dir))
    )

    meta = RDFMetaData(
        classifier_path=onnx_file_path_str,
        training_data_path=(training_data_path_str if save_training_data else None),
        training_outcomes_path=(outcomes_path_str if save_training_data else None),
        ids=(ids if save_training_data else None),
        features=features,
        outcome_names=outcome_names,
    )
    meta_json = json.dumps(asdict(meta))
    with open(meta_path, mode="w", encoding="utf-8") as f:
        f.write(meta_json)
    meta_path.chmod(0o660)
