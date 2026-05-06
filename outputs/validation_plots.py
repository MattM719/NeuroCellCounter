#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Visually compares automated vs. manual cell counts"""

import os
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt  # noqa E402
from matplotlib.axes import Axes  # noqa E402

from utils import Image  # noqa E402


matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["font.family"] = "Arial"


def plot_bland_altman(images: list[Image], path: Optional[str], **kwargs) -> None:
    """Creates a Bland-Altman plot to show agreement between manual and pyknotic cell
    counts
    """
    params = {
        "filename": "bland_altman",
        "ext": "png",
        "dpi": 400,
        "height": 4,
        "width": 6,
    }
    params.update(kwargs)
    n_aligned = len(images)
    all_cells = np.zeros([n_aligned, 2], dtype=np.int64)
    pyk_cells = np.zeros([n_aligned, 2], dtype=np.int64)
    for i, image in enumerate(images):
        all_cells[i, :] = [image.manual_all, image.auto_all]
        pyk_cells[i, :] = [image.manual_pyk, image.auto_pyk]

    all_cells_diff = np.diff(all_cells, axis=1).flatten()
    pyk_cells_diff = np.diff(pyk_cells, axis=1).flatten()

    # stats
    all_mean = float(np.mean(all_cells_diff))
    pyk_mean = float(np.mean(pyk_cells_diff))
    all_sd = float(np.std(all_cells_diff))
    pyk_sd = float(np.std(pyk_cells_diff))

    fig, ax = plt.subplots(2, 1, constrained_layout=True)
    fig.set_figheight(params["height"])
    fig.set_figwidth(params["width"])
    fig.set_dpi(params["dpi"])
    ax: tuple[Axes, ...] = ax

    ax[0].scatter(all_cells[:, 0], all_cells_diff, c="k", s=2.5, alpha=0.8)
    ax[0].axhline(
        all_mean + 2 * all_sd,
        linewidth=1.2,
        color="r",
        linestyle="--",
        label=f"Mean +2 SD = {round(all_mean+2*all_sd,1)}",
    )
    ax[0].axhline(
        all_mean,
        linewidth=1.2,
        color="b",
        linestyle="-",
        label=f"Mean    =    {round(all_mean,1)}",
    )
    ax[0].axhline(
        all_mean - 2 * all_sd,
        linewidth=1.2,
        color="r",
        linestyle="--",
        label=f"Mean -2 SD = {round(all_mean-2*all_sd,1)}",
    )
    ax[0].set_xlabel("Manual Count")
    ax[0].set_ylabel("Differences in\nAll Nuclei")
    ax[0].legend()

    ax[1].scatter(pyk_cells[:, 0], pyk_cells_diff, c="k", s=2.5, alpha=0.8)
    ax[1].axhline(
        pyk_mean + 2 * pyk_sd,
        linewidth=1,
        color="r",
        linestyle="--",
        label=f"Mean +2 SD = {round(pyk_mean+2*pyk_sd,1)}",
    )
    ax[1].axhline(
        pyk_mean,
        linewidth=1.1,
        color="b",
        linestyle="-",
        label=f"Mean    =    {round(pyk_mean,1)}",
    )
    ax[1].axhline(
        pyk_mean - 2 * pyk_sd,
        linewidth=1,
        color="r",
        linestyle="--",
        label=f"Mean -2 SD = {round(pyk_mean-2*pyk_sd,1)}",
    )
    ax[1].set_xlabel("Manual Count")
    ax[1].set_ylabel("Differences in\nPyknotic Nuclei")
    ax[1].legend()

    if isinstance(path, str):
        fig.savefig(
            os.path.join(path, f"{params['filename']}.{params['ext']}"),
            format=params["ext"],
        )
    else:
        plt.show()

    plt.close(fig)
