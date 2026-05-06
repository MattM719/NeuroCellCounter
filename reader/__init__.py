#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Reads microscopy image files"""

try:
    import numpy as _  # noqa F401 F811
    import nd2 as _  # noqa F401 F811
except ImportError:
    raise ImportError("Ensure numpy and nd2 are installed")

from .img_reader import nd2_img_reader, get_stain, png_img_reader  # noqa F401

from .validation_reader import (  # noqa F401
    read_manual_counts,
    read_auto_counts_from_pkl,
    align_auto_manual_counts,
)
