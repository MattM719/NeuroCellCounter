#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Logs information and errors from analysis"""

import os
import datetime
from pathlib import Path
from pprint import pformat
from typing import Optional, Literal

from utils.data_pickling import StoredData2


def save_info(
    path: str | Path,
    source_dir: str | Path,
    now: datetime.datetime,
    pca_path: Optional[str],
    rdf_path: Optional[str],
    model: StoredData2 | dict,
    model_type: Literal["pca_rdf_combo", "new_pca", "old_pca", "thresholds"],
) -> None:
    """Saves info about model"""
    date = now.strftime("%m/%d/%y")
    time = now.strftime("%H:%M:%S")

    # introduce model
    if model_type == "new_pca":
        model_intro = f"Trained new model at {rdf_path}"
    elif model_type == "old_pca":
        model_intro = f"Used previous model from {rdf_path}"
    elif model_type == "thresholds":
        model_intro = "Used thresholds, as shown below:"
    elif model_type == "pca_rdf_combo":
        model_intro = "Used the PCA thresholds below to preprocess cells.\n"
        model_intro += f"PCA transformation file: {pca_path}\n"
        model_intro += f"Then applied RdF model: {rdf_path}\n"
    else:
        model_intro = ""

    # create file
    if not isinstance(path, Path):
        path = Path(path)
    file_path = path / "info.txt"
    with open(file_path, mode="w", encoding="utf-8") as f:
        f.write(f"Info for cell counting, started at {time} on {date}.\n")
        f.write(f"Source directory: `{str(source_dir)}/`.\n")
        f.write(model_intro + "\n\n")
        f.write(pformat(model))
    file_path.chmod(0o660)


class Log(object):
    """saves cell counts"""

    fname = "log"
    extension = "txt"
    encoding = "utf-8"
    lineterminator = "\n"
    sep = 50 * "-"

    def __init__(self, out_dir: str, **kwargs) -> None:
        """setup instance to save cell counts"""
        self.out_dir: str = out_dir
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir, mode=0o751)

        self.__dict__.update(kwargs)

        self.start_datetime = datetime.datetime.now()
        day = self.start_datetime.strftime("%m/%d/%y")
        time = self.start_datetime.strftime("%H:%M:%S")

        # initialize file
        self.path = os.path.join(self.out_dir, f"{self.fname}.{self.extension}")
        with open(self.path, mode="w", encoding=self.encoding) as f:
            f.write(f"*** Initializing log at {time} on {day} ***\n")

    @property
    def time_stamp(self) -> str:
        """creates a time stamp"""
        now = datetime.datetime.now()
        day = now.strftime("%m/%d/%y")
        time = now.strftime("%H:%M:%S")
        return f"[{day}--{time}]"

    def log_err(self, file_path: str, error_message: str) -> None:
        """Logs an error message"""
        with open(self.path, mode="a", encoding=self.encoding) as f:
            f.write(f"\n{self.time_stamp}: Exception occurred while reading file:")
            f.write(f"\n{file_path}\n\n")
            f.write(error_message)
            f.write(f"\n{self.sep}\n")
