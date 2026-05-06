#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""analyzes images of microscope slides"""

try:
    import numpy as _
    from scipy import ndimage as _  # noqa F811
    import matplotlib as _  # noqa F811
    import skimage as _  # noqa F401 F811
except ImportError:
    raise ImportError("numpy, scipy, matplotlib, and scikit-image must be installed")
