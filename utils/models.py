#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Models"""

import os
import datetime
import warnings
from pprint import pprint
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np

__all__ = ["Cell", "Region", "Image"]


BRAIN_REGION_OPTIONS = ["BG", "CC", "CORTEX", "SWM", "THALAMUS", "HIPPO"]
BRAIN_REGION_TYPE = Literal["BG", "CC", "CORTEX", "SWM", "THALAMUS", "HIPPO"]


def _format_location_indices(
    value: Tuple[np.ndarray, np.ndarray] | np.ndarray | List[Tuple[int, int]] | Any,
) -> Union[np.ndarray, np.ndarray]:
    """Converts several formats of locs to a consistent format"""
    if isinstance(value, tuple):
        i, j = value
        if not isinstance(i, np.ndarray) or not isinstance(j, np.ndarray):
            raise TypeError("Improperly formatted pixel indices")
        return (i.copy().flatten(), j.copy().flatten())

    elif isinstance(value, np.ndarray):
        assert len(value.shape) == 2
        if value.shape[1] == 2:
            return (value[:, 0].copy().flatten(), value[:, 0].copy().flatten())
        elif value.shape[0] == 2:
            return (value[0, :].copy().flatten(), value[1, :].copy().flatten())
        raise IndexError("if pixel indices must be a 2 x n array")

    elif isinstance(value, list) and len(value == 0):
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    elif isinstance(value, list) and isinstance(value[0], tuple):
        idx0 = np.zeros(len(value), dtype=np.int64)
        idx1 = np.zeros(len(value), dtype=np.int64)
        for idx, (i, j) in enumerate(value):
            idx0[idx] = int(i)
            idx1[idx] = int(j)
        return (idx0, idx1)

    raise TypeError(f"Unrecognized value type {type(value)}")


class CompareUIDs(object):
    uid: int

    def __eq__(self, value: object) -> bool:
        if isinstance(value, type(self)):
            return self.uid == value.uid
        return False

    def __lt__(self, value: object) -> bool:
        if isinstance(value, type(self)):
            return self.uid < value.uid
        return False

    def __gt__(self, value: object) -> bool:
        if isinstance(value, type(self)):
            return self.uid > value.uid
        return False

    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value)

    def __le__(self, value: object) -> bool:
        return self.__eq__(value) or self.__lt__(value)

    def __ge__(self, value: object) -> bool:
        return self.__eq__(value) or self.__gt__(value)

    def __hash__(self):
        return hash(self.__repr__())


class Cell(CompareUIDs):
    """Class to store information on each cell"""

    def __init__(
        self,
        uid: int,
        center: Tuple[int, int],
        sigma_px: float,
        avg_intensity: float,
        weighted_intensity: float,
        total_intensity: float,
        pixel_height: float,
        pixel_width: float,
        image_dims: Tuple[int, int],
    ) -> None:
        """initialize cell class

        Args:
            uid: unique number for cell in image
            center: (i,j) coordinates of center of cell, from blob detection
            sigma: of gaussian distribution, from blob detection (units = pixels [count])
            *_intensity: intensity of cell's pixels
            pixel_height: height (dy) of each pixel, in micrometers
            pixel_width: width (dx) of each pixel, in micrometers
        """
        self.uid = uid
        self.pixel_height = float(pixel_height)
        self.pixel_width = float(pixel_width)

        self.center = center
        self.sigma_px = sigma_px
        self.avg_intensity = avg_intensity
        self.weighted_intensity = weighted_intensity
        self.total_intensity = total_intensity
        self.image_dims = (int(image_dims[0]), int(image_dims[1]))

        self.region_id: Optional[int] = None
        self.perimeter: float = float("nan")
        self._is_pyknotic: Optional[bool] = None
        self._alx568_internal_intensities: Optional[np.ndarray] = None

    @property
    def sigma(self) -> float:  # microns
        warnings.warn(
            "Must explicitly call `sigma_um` instead of `sigma` (ambiguous)",
            category=DeprecationWarning,
        )
        if not self.is_square_pixels:
            raise NotImplementedError("Require square pixels")
        return self.sigma_px * self.pixel_width

    @sigma.setter
    def sigma(self, value: float) -> None:  # pixels (count)
        """catches old use of `sigma` instead of `sigma_px`"""
        warnings.warn(
            "Must explicitly set `sigma_px` instead of `sigma` (ambiguous)",
            category=DeprecationWarning,
        )
        self._sigma_px = value

    @property
    def sigma_um(self) -> float:  # microns
        if not self.is_square_pixels:
            raise NotImplementedError("Require square pixels")
        return self.sigma_px * self.pixel_width

    @property
    def sigma_px(self) -> float:  # pixels (count)
        return float(self._sigma_px)

    @sigma_px.setter
    def sigma_px(self, value: float) -> None:  # pixels (count)
        self._sigma_px = value

    @property
    def n_pixels(self) -> int:
        return int(self.pixel_locs[0].shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edge_locs[0].shape[0])

    @property
    def pixel_locs(self) -> Tuple[np.ndarray, np.ndarray]:
        """(i_indices, j_indices)"""
        return self._pixel_locs

    @pixel_locs.setter
    def pixel_locs(
        self,
        value: Tuple[np.ndarray, np.ndarray] | np.ndarray | List[Tuple[int, int]] | Any,
    ):
        self._pixel_locs: tuple[np.ndarray, np.ndarray] = _format_location_indices(value)
        self._pixel_locs_set: set[tuple[int, int]] = {
            (int(i), int(j)) for i, j in zip(*self._pixel_locs)
        }

    @property
    def edge_locs(self) -> Tuple[np.ndarray, np.ndarray]:
        """list of edge locs"""
        return self._edge_locs

    @edge_locs.setter
    def edge_locs(
        self,
        value: Tuple[np.ndarray, np.ndarray] | np.ndarray | List[Tuple[int, int]] | Any,
    ) -> None:
        self._edge_locs: tuple[np.ndarray, np.ndarray] = _format_location_indices(value)
        self._edge_locs_set: set[tuple[int, int]] = {
            (int(i), int(j)) for i, j in zip(*self._edge_locs)
        }

    @property
    def alx568_internal_intensities(self) -> Optional[np.ndarray]:
        """Intensity of Alx568 at each internal pixel"""
        return self._alx568_internal_intensities

    def set_alx568_internal_intensities(self, alx568_image: np.ndarray):
        """determines intensities of cell at each internal location"""
        # get pixel locs
        i_idx, j_idx = self.pixel_locs

        # validation
        assert len(alx568_image.shape) == 2
        assert i_idx.max() < alx568_image.shape[0]
        assert j_idx.max() < alx568_image.shape[1]

        # get intensities
        self._alx568_internal_intensities = alx568_image[(i_idx, j_idx)]

    @property
    def mean_alx568_internal_intensity(self) -> float:
        """Mean internal intensity of Alx568 (AU / micron^2)"""
        if self.alx568_internal_intensities is None:
            return float("nan")
        total = float(np.sum(self.alx568_internal_intensities))
        return total / self.area

    @property
    def region_id(self) -> Optional[int]:
        """returns region ID, or -1 if no region ID"""
        return self._region_id

    @region_id.setter
    def region_id(self, value: Optional[int]):
        if value is None:
            self._region_id = None
        else:
            self._region_id = int(value)

    @property
    def is_pyknotic(self) -> bool:
        """indicates whether the cell is labeled as pyknotic"""
        if not isinstance(self._is_pyknotic, bool):
            raise AttributeError("is_pyknotic is not yet assigned")
        return self._is_pyknotic

    @is_pyknotic.setter
    def is_pyknotic(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("is_pyknotic must be a bool or None")
        self._is_pyknotic = value

    @property
    def area(self) -> float:
        """area in square micrometers"""
        return float(self.n_pixels) * (self.pixel_height * self.pixel_width)

    @property
    def perimeter(self) -> float:
        """returns perimeter of cell, in micrometers"""
        if self._perimeter != self._perimeter:
            raise AttributeError("Perimeter has not yet been supplied!")
        return self._perimeter

    @perimeter.setter
    def perimeter(self, value: float):
        self._perimeter = float(value)

    @property
    def ideal_radius(self) -> float:
        """radius (um) of a perfect circle with the same area"""
        return float(np.sqrt(self.area / np.pi))

    @property
    def eccentricity(self) -> float:
        """eccentricity of circle"""
        return self.perimeter / float(2.0 * np.pi * self.ideal_radius)

    @property
    def is_square_pixels(self) -> bool:
        """indicates whether pixels are square (within error)"""
        return abs(self.pixel_height - self.pixel_width) < 1e-3

    @property
    def is_touching_edge(self) -> bool:
        """True if at least one pixel touches an edge"""
        i_min = int(np.min(self.pixel_locs[0]))
        i_max = int(np.max(self.pixel_locs[0]))
        j_min = int(np.min(self.pixel_locs[1]))
        j_max = int(np.max(self.pixel_locs[1]))

        m = int(self.image_dims[0]) - 1
        n = int(self.image_dims[1]) - 1

        tl = (i_min == 0) or (j_min == 0)
        br = (i_max == m) or (j_max == n)

        return bool(tl or br)

    def update_params(self, **params) -> None:
        """updates parameters when specific cell characteristics are known"""
        allowed = ["avg_intensity", "weighted_intensity", "total_intensity"]

        # check keys being updated
        keys = list(params.keys())
        for key in keys:
            if key not in allowed:
                raise IndexError(
                    f"Cell cannot update {key} through Cell.update_params. "
                    f"Only the following may be updated:\n{allowed}"
                )

        self.__dict__.update(params)

    def __repr__(self) -> str:
        return f"Cell(uid={self.uid})"

    def __str__(self) -> str:
        return f"Cell {self.uid} at {self.center}"


class Region(CompareUIDs):
    """Class to store information on each region and each cell within that region"""

    def __init__(
        self,
        uid: int,
        center_of_mass: tuple[int, int],
        region_info: dict[Literal["region", "edges"], tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Initialize a new class for each region of an image"""
        self.uid: int = uid  # unique identifier for region
        self.center_of_mass: tuple[int, int] = center_of_mass
        self.region_locs = region_info["region"]
        self.edge_locs = region_info["edges"]

        self._n_cells: int = 0  # counts number of cells
        self._cell_ids = set()

    @property
    def region_locs(self) -> tuple[np.ndarray, np.ndarray]:
        """list of region locs"""
        return self._region_locs

    @region_locs.setter
    def region_locs(
        self,
        value: tuple[np.ndarray, np.ndarray] | np.ndarray | list[tuple[int, int]] | Any,
    ):
        self._region_locs: tuple[np.ndarray, np.ndarray] = _format_location_indices(
            value
        )
        self._region_locs_set: set[tuple[int, int]] = {
            (int(i), int(j)) for i, j in zip(*self._region_locs)
        }

    @property
    def edge_locs(self) -> tuple[np.ndarray, np.ndarray]:
        """list of edge locations"""
        return self._edge_locs

    @edge_locs.setter
    def edge_locs(
        self,
        value: tuple[np.ndarray, np.ndarray] | np.ndarray | list[tuple[int, int]] | Any,
    ):
        self._edge_locs = _format_location_indices(value)
        self._edge_locs_set: set[tuple[int, int]] = {
            (int(i), int(j)) for i, j in zip(*self._edge_locs)
        }

    @property
    def cell_ids(self) -> set[int]:
        return self._cell_ids

    @property
    def n_cells(self) -> int:
        return len(self.cell_ids)

    @property
    def n_pixels(self) -> int:
        return len(self.region_locs[0])

    def test_cell(self, cell: Cell) -> bool:
        """tests whether a cell's center is within the region"""
        cell_loc: tuple[int, int] = cell.center
        return cell_loc in self._region_locs_set

    def intersecting_cells(
        self, cell_centers: set[tuple[int, int]]
    ) -> set[tuple[int, int]]:
        """returns a set of intersecting cells"""
        return self._region_locs_set.intersection(cell_centers)

    def add_cell(self, cell: Cell) -> None:
        """adds a cell to this region"""
        self._cell_ids.add(int(cell.uid))

    def __repr__(self) -> str:
        return f"Region(uid={self.uid})"

    def __str__(self) -> str:
        return f"Region {self.uid} at {self.center_of_mass}"


def get_single_num(txt: str) -> Optional[int]:
    """gets a single integer from a string"""
    nums = [x for x in txt if x.isdigit()]

    if len(nums) == 0:
        return None

    num = "".join(nums)

    return int(num)


def fix_image_region_abbrevs(region_name: str) -> str:
    """Fixes abbreviations
    ['BG','CC','CORTEX','SWM','THALAMUS','HIPPO']
    """
    abbrevs = {"THAL": "THALAMUS", "HIP": "HIPPO"}
    if region_name in BRAIN_REGION_OPTIONS:
        return region_name

    out = abbrevs.get(region_name, None)
    if isinstance(out, str):
        return out

    raise ValueError(f"Unexpected abbreviation: {region_name}")


class Image(object):
    """Stores cumulative data about an image"""

    def __init__(self, path: str, meta: dict[str, Any]) -> None:
        """Initialize an Image object to orgaize regions, cells, counts, and meta data"""
        # load image data
        self.path = path
        self.basename: str = os.path.splitext(os.path.basename(path))[0]
        self.meta: dict[str, Any] = meta

        # process basename
        # basename format: "40X_bg_slice13_plate2_image1_062823" (removed .nd2)
        try:
            [_, region, slice_num, plate_num, image_num, date] = self.basename.split("_")
        except Exception:
            print(
                f"*** Difficulty parsing the name '{self.basename}' "
                f"into 6 categories:\n{self.basename.split('_')}"
            )
            pprint(self.__dict__)
            raise

        self.brain_region: BRAIN_REGION_TYPE = fix_image_region_abbrevs(
            str(region).upper()
        )
        self.slice_num: int = get_single_num(slice_num)
        self.plate_num: int = get_single_num(plate_num)
        self.image_num: int = get_single_num(image_num)

        try:
            date_tmp = datetime.datetime.strptime(date, "%m%d%y")
        except Exception:
            print(f"*** Unable to parse date '{date}'")
            raise
        self.date: datetime.date = datetime.date(
            year=date_tmp.year, month=date_tmp.month, day=date_tmp.day
        )

        assert isinstance(
            self.slice_num, int
        ), f"Unable to find slice number: '{slice_num}'"
        assert isinstance(
            self.plate_num, int
        ), f"Unable to find plate number: '{plate_num}'"
        assert isinstance(
            self.image_num, int
        ), f"Unable to find image number: '{image_num}'"

        # initialize empty attributes
        self.dim: tuple[int, int] = tuple([])
        self.all_cells: set[Cell] = set()
        self.pyk_cells: set[Cell] = set()
        self.regions: set[Region] = set()
        self.auto_all: int = -1
        self.auto_pyk: int = -1
        self.manual_all: int = -1
        self.manual_pyk: int = -1

    def set_cells(self, cells: list[Cell]) -> None:
        """Sets image's cells"""
        del self.all_cells
        self.all_cells = set(cells)
        for cell in cells:
            if cell.is_pyknotic:
                self.pyk_cells.add(cell)

        assert self.pyk_cells.issubset(self.all_cells)

        self.auto_all = len(self.all_cells)
        self.auto_pyk = len(self.pyk_cells)

    def set_regions(self, regions: list[Region]) -> None:
        """Sets image's regions"""
        del self.regions
        self.regions = set(regions)

    def set_manual_count(self, all_nuclei: int, pyk_nuclei: int) -> None:
        """Sets manual counts"""
        self.manual_all = int(all_nuclei)
        self.manual_pyk = int(pyk_nuclei)


#########################################################################################
#########################################################################################


def test_get_single_num() -> None:
    """Tests get_single_num function"""
    assert 7 == get_single_num("only_give7out")
    assert 42 == get_single_num("the4answer2everything")


if __name__ == "__main__":
    test_get_single_num()
    print("success")
