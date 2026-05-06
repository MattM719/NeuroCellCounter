#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Objects to classify nuclei"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class PolygonClassifier(object):
    """Classifies data"""

    def __init__(self) -> None:
        """initialize PolygonClassifier instance"""
        self.polygon: Optional[Polygon] = None

    def set_polygon(self, *points: Tuple[float, ...]) -> None:
        """creates a polygon"""
        if len(points) < 3:
            return None

        pts = list(points)
        if pts[0] != pts[-1]:
            pt_last = [x for x in pts[0]]
            pts.append(tuple(pt_last))

        self.polygon = Polygon(tuple(pts))

    def is_point_in_polygon(self, point: Point | Tuple[float, float]) -> bool:
        """classifies points as in (1) or out (0) of the polygon"""
        if self.polygon is None:
            raise ValueError("Polygon was not set")

        if isinstance(point, Point):
            pt: Point = point
        elif isinstance(point, tuple):
            pt: Point = Point(*point)
        else:
            raise TypeError("point must be a Point or a tuple")

        return pt.within(self.polygon)

    def classify_points_in_polygon(
        self, points: np.ndarray | pd.DataFrame
    ) -> np.ndarray:
        """Classifies points in polygon

        points should be an array of points (n_points, n_dimensions 2 or 3)

        returns a flat array of len n_points, True where point is in the polygon
        """
        if self.polygon is None:
            raise ValueError("Polygon was not set")
        if isinstance(points, np.ndarray):
            pass
        elif isinstance(points, pd.DataFrame):
            points = points.to_numpy(dtype=np.float64)
        else:
            raise TypeError("points must be a numpy array or dataframe")
        assert points.shape[1] == 2 or points.shape[1] == 3

        # classifies whether points are in the polygon
        pts = [tuple([float(j) for j in k]) for k in tuple(points)]
        outcomes = [self.is_point_in_polygon(pt) for pt in pts]

        return np.array(outcomes, dtype=bool)


if __name__ == "__main__":
    classifier = PolygonClassifier()
    classifier.set_polygon((0, 0), (10, 47), (20, 5), (6, 0))
    arr0 = np.arange(start=-10, stop=30, step=4)
    arr1 = arr0 + 1
    arr = np.vstack((arr0, arr1), dtype=np.float64).T
    print(classifier.classify_points_in_polygon(arr))
