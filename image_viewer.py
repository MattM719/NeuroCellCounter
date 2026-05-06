#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Views images"""

import os
from pprint import pprint

import numpy as np

from reader import nd2_img_reader
from outputs.show_processing_steps import view_nd2_images

PATH = os.path.join(
    ...,  # FIXME: Path to folder with ND2-formatted images
    "*.nd2",
)


def main():
    """main"""
    if PATH.is_dir():
        paths = PATH.glob("*.nd2")
    else:
        paths = [PATH]
    # paths = [
    #     BASE_DIR / "images" / "40X_cortex_slice43_plate3_image3_062923.nd2"
    # ]

    known_4095_uint: bool = False

    for i, path in enumerate(paths):
        if "Tilescan" in path:
            continue
        meta, img = nd2_img_reader(path, print_all_meta=False)

        if len(img.flatten()) == 0:
            pprint(meta)
            print("\n*****  SKIPPING EMPTY IMAGE  *****\n")

        # test
        if known_4095_uint:
            pass
        elif isinstance(img.flatten()[0], np.uint16) and img.max() == 4095:
            known_4095_uint = True
        elif i == 0 and len(paths) > 1 and isinstance(img.flatten()[0], np.uint16):
            meta, img = nd2_img_reader(path, print_all_meta=False)
            assert isinstance(img.flatten()[0], np.uint16) and img.max() == 4095
            known_4095_uint = True

        print("\n\n")
        pprint(meta)
        print(os.path.basename(path))
        figs, _metas = view_nd2_images(path, is_uint_4095=known_4095_uint, show=True)


if __name__ == "__main__":
    main()
