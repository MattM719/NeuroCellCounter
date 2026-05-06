#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Saves results of analysis"""

import os
import csv
from typing import Optional

from utils.models import Cell


class SaveCounts(object):
    """saves cell counts"""

    fname = "counts"
    extension = "csv"
    encoding = "utf-8"
    delimiter = ","
    lineterminator = "\n"

    headers = ["file_name", "all_nuclei", "pyknotic", "non-pyknotic"]

    def __init__(self, out_dir: str, **kwargs) -> None:
        """setup instance to save cell counts"""
        self.out_dir: str = out_dir
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir, mode=0o751)

        self.__dict__.update(kwargs)

        if self.extension != "csv":
            raise ValueError(
                f"Unsure how to process a file with extension '{self.extension}'"
            )

        # initialize file
        self.path = os.path.join(self.out_dir, f"{self.fname}.{self.extension}")
        with open(self.path, mode="w", encoding=self.encoding) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.headers,
                dialect="excel",
                delimiter=self.delimiter,
                lineterminator=self.lineterminator,
            )
            writer.writeheader()

    def add_counts(self, img_name: str, cell_count: int, pyknotic: int) -> None:
        """Adds a line to the count"""
        with open(self.path, mode="a", encoding=self.encoding) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.headers,
                dialect="excel",
                delimiter=self.delimiter,
                lineterminator=self.lineterminator,
            )
            writer.writerow(
                {
                    "file_name": img_name,
                    "all_nuclei": cell_count,
                    "pyknotic": pyknotic,
                    "non-pyknotic": cell_count - pyknotic,
                }
            )

    def cells_to_entry(self, img_name: str, cells: list) -> None:
        """builds a row entry from cells"""

        def test_pyknotic(cell: Cell) -> bool:
            return cell.is_pyknotic

        pyknotic = list(filter(test_pyknotic, cells))

        self.add_counts(img_name, len(cells), len(pyknotic))


class SaveCellProperties(object):
    """saves cell properties"""

    fname = "properties"
    extension = "csv"
    encoding = "utf-8"
    delimiter = ","
    lineterminator = "\n"

    general_headers = ["file_name", "known_pyknotic"]
    cell_headers = [
        "uid",
        "region_id",
        "i",
        "j",
        "sigma_um",
        "avg_intensity",
        "weighted_intensity",
        "total_intensity",
        "ideal_radius",
        "eccentricity",
        "perimeter",
        "area",
        "mean_alx568_internal_intensity",
        "classified_pyknotic",
    ]
    headers = [*general_headers, *cell_headers]

    def __init__(self, out_dir: str, **kwargs) -> None:
        """setup instance to save cell counts"""
        self.out_dir: str = out_dir
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir, mode=0o751)

        self.__dict__.update(kwargs)

        if self.extension != "csv":
            raise ValueError(
                f"Unsure how to process a file with extension '{self.extension}'"
            )

        # initialize file
        self.path = os.path.join(self.out_dir, f"{self.fname}.{self.extension}")
        with open(self.path, mode="w", encoding=self.encoding) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.headers,
                dialect="excel",
                delimiter=self.delimiter,
                lineterminator=self.lineterminator,
            )
            writer.writeheader()

    def _create_row(
        self, img_name: str, cell: Cell, pyknotic: Optional[bool] = None
    ) -> dict:
        """creates a dict of cell info for spreadsheet"""
        row = {
            "file_name": img_name,
            "known_pyknotic": "",
            "uid": cell.uid,
            "region_id": cell.region_id,
            "i": str(cell.center[0]),
            "j": str(cell.center[1]),
            "sigma_um": cell.sigma_um,
            "avg_intensity": cell.avg_intensity,
            "weighted_intensity": cell.weighted_intensity,
            "total_intensity": cell.total_intensity,
            "ideal_radius": cell.ideal_radius,
            "eccentricity": cell.eccentricity,
            "perimeter": cell.perimeter,
            "area": cell.area,
            "mean_alx568_internal_intensity": cell.mean_alx568_internal_intensity,
            "classified_pyknotic": str(cell.is_pyknotic).upper(),
        }
        return row

    def save_cells(self, img_name: str, cells: list | Cell) -> None:
        """Adds a line to the count"""
        if isinstance(cells, Cell):
            cells = [cells]
        if not isinstance(cells, (list, tuple)):
            raise TypeError(
                "cells must be of type Cell, list, or tuple. " +
                f"Received type {type(cells)}"
            )

        rows = [self._create_row(img_name, cell) for cell in cells]

        with open(self.path, mode="a", encoding=self.encoding) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.headers,
                dialect="excel",
                delimiter=self.delimiter,
                lineterminator=self.lineterminator,
            )
            writer.writerows(rows)
