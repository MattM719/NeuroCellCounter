#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Counts cells"""

import os
import traceback
from pathlib import Path
from datetime import datetime
from typing import Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm

from apply_prev_rdf_model import classify_cells_rdf
from analysis.image_processing import get_dapi_image, get_alx568_image, preprocess_image
from analysis.region_finder import separate_regions_by_growing, paint_regions
from analysis.cell_counter import find_cells
from analysis.match_cell_regions import (
    find_cell_regions,
    fit_cells_in_regions,
    update_cell,
)
from analysis.characterize_cells import (
    assign_pyknotics,
    fill_holes,
    find_perimeter,
)
from analysis.utils import sort_regions
from classifiers import LinearTransformer
from outputs import (
    SaveCounts,
    SaveCellProperties,
    Log,
    save_info,
    plot_pyknotic,
)
from outputs.show_processing_steps import (
    compare_original_masked_overlay,
    view_nd2_images,
)
from outputs.basic_analyses import save_relative_channel_intensities
from outputs.save_processed_images import save_image_nd2_meta
from utils import (
    Cell,
    Region,
    validate_meta,
    filter_tile_scan,
    filter_edge,
    filter_low_intensity,
    filter_low_pixels,
    filter_small_radius,
    filter_none,
)
from utils.data_pickling import save_with_dataclass_pca, StoredAnnotations1

BASE_DIR = Path("/Volumes/Expansion_Usable/data/cell_counting")
FILES_PARENT = BASE_DIR / "Dec22_Caffeine" / "images"
TAG = "CaffeineDec22_TEST"

OUT_DIR = BASE_DIR / "results"
TRANSFORMER_PATH: Optional[Path] = None

# choose pkl file for the pre-existing model you'd like to use
RDF_PATH: Optional[Path] = (
    OUT_DIR / "batch_CaffeineAug24TrainRDF_sigma22_filter_small_070825_115728"
)
RDF_PATH_SUFFIX: str = (
    "training_data_balanced/onnx_rdf_classifier"  # "training_data/rdf_model.pkl"
)


IMG_FMT: Literal["png", "svg"] = "png"
SAVE_FIGURES: bool = True
SHOW_FIGURES: bool = False
THRESHOLDS = {
    # all cell thresholds
    "pc0_all_thresh_high": 2.5,  # 2.5, # minimum size
    "pc0_all_thresh_low": None,  # maximum size
    "pc1_all_thresh_high": None,  # maximum intensity
    "pc1_all_thresh_low": -2,  # -2, # minimum intensity
    # pyknotic cell thresholds
    "pc0_pyk_thresh_high": 2.5,  # 2.5, # minimum size
    "pc0_pyk_thresh_low": -0.5,  # -0.5, # maximum size
    "pc1_pyk_thresh_high": None,  # minimum intensity
    "pc1_pyk_thresh_low": 0,  # 0, # minimum intensity
}

SAVE_IMAGE_PICKLE: bool = False


def main():
    """main"""
    OUT_DIR.mkdir(0o771, exist_ok=True)

    files: list[Path] = sorted(filter(filter_tile_scan, FILES_PARENT.glob("*.nd2")))

    now = datetime.now()
    time_stamp = now.strftime("%m%d%y_%H%M%S")

    # create/empty batch folder
    batch_dir = OUT_DIR / f"batch_{TAG}_{time_stamp}"
    batch_dir.mkdir(mode=0o751)

    # folder to store preprocessed images
    downsampled_img_path = batch_dir / "downsampled_images"
    downsampled_img_path.mkdir(mode=0o750)
    downsampled_img_path_alx568 = batch_dir / "downsampled_images_Alx568"
    downsampled_img_path_alx568.mkdir(mode=0o750)

    rdf_path: Optional[Path] = RDF_PATH
    if RDF_PATH is not None:
        rdf_path /= RDF_PATH_SUFFIX
        assert rdf_path.exists()

    save_info(
        path=batch_dir,
        source_dir=str(FILES_PARENT.relative_to(BASE_DIR)),
        now=now,
        pca_path=str(TRANSFORMER_PATH),
        rdf_path=str(rdf_path),
        model=THRESHOLDS,
        model_type="pca_rdf_combo",
    )

    savecounts = SaveCounts(str(batch_dir))
    cellprops = SaveCellProperties(str(batch_dir))
    log = Log(str(batch_dir))

    fig0: Optional[Figure] = None
    fig1: Optional[Figure] = None
    fig2: Optional[Figure] = None

    # process files
    for file in tqdm(files, total=len(files), desc="Processing images"):
        plt.close("all")
        fname = file.stem

        # output directory
        out_dir = batch_dir / fname
        out_dir.mkdir(mode=0o751, exist_ok=True)

        try:
            save_relative_channel_intensities(file, out_dir.parent)
            raw_img, meta = get_dapi_image(str(file), down_sample=True)

            validate_meta(str(file), meta, raw_img)
            save_image_nd2_meta(
                out_dir=downsampled_img_path, name=fname, img=raw_img, meta=meta
            )

            pixel_h, pixel_w, _ = meta.get("axes_calibration", None)
            pixel_threshold = round(10 * 0.86**2.0 / (float(pixel_h) * float(pixel_w)))

            # normalize
            clean_img, mask, mask_edges = preprocess_image(
                raw_img,
                bin_thresh=0.1,
                opt_thresh=False,
                tophat_radius=1,
            )

            # get Alx568 image
            has_alx568 = False
            try:
                alx_raw_img, alx_meta = get_alx568_image(str(file), down_sample=True)
                validate_meta(str(file), alx_meta, alx_raw_img)
                assert (np.array(raw_img.shape) == np.array(alx_raw_img.shape)).all()
                save_image_nd2_meta(
                    out_dir=downsampled_img_path_alx568,
                    name=fname,
                    img=alx_raw_img,
                    meta=alx_meta,
                )
                alx_clean_img, alx_mask, mask_edges = preprocess_image(
                    alx_raw_img,
                    bin_thresh=0.1,
                    opt_thresh=False,
                )
                has_alx568 = True
            except Exception:
                has_alx568 = False

            # identify regions in the image. Further optimize the mask
            all_regions = separate_regions_by_growing(mask)
            regions_info = sort_regions(all_regions)
            regions = [
                Region(i, com, info)
                for i, (com, info) in enumerate(regions_info.items())
            ]

            region_mask = paint_regions(clean_img.shape, regions=regions)
            clean_img *= region_mask
            norm_img = clean_img / float(clean_img.max())

            fig0 = compare_original_masked_overlay(
                dapi_img=raw_img,
                masked_img=clean_img,
                mask=region_mask,
                mask_edges=mask_edges,
                show=False,
            )

            cells: list[Cell] = find_cells(clean_img, norm_img, meta)

            regions, cells = find_cell_regions(regions, cells)

            cells_by_uid = {cell.uid: cell for cell in cells}
            cells_mapping = {cell.uid: i for i, cell in enumerate(cells)}
            for region in regions:
                cells_tmp = fit_cells_in_regions(
                    clean_img,
                    norm_img,
                    region,
                    cells_by_uid,
                    pixel_threshold=pixel_threshold,
                )
                for cell in cells_tmp:
                    if isinstance(cell, int):
                        uid = int(cell)
                        cell = None
                    else:
                        uid = int(cell.uid)
                    cells[cells_mapping[uid]] = cell
                    cells_by_uid.pop(uid)
                    cells_mapping.pop(uid)

            # remove cells that are touching an edge or have invalid properties
            filts = [
                filter_none,  # must be first
                filter_small_radius,
                filter_edge,
                filter_low_pixels,
                filter_low_intensity,
            ]
            for filt in filts:
                cells = filter(filt, cells)
            cells: list[Cell] = list(cells)

            # clean up cells - fill holes, update intensity calculations,
            # and then calculate actual perimeter
            for i, cell in enumerate(cells):
                locs = fill_holes(cell)
                cell = update_cell(norm_img, cell=cell, locs=locs)
                cell.perimeter = find_perimeter(cell)
                if has_alx568:
                    cell.set_alx568_internal_intensities(alx_clean_img)
                cells[i] = cell

            # further filter cells by PCA results
            if TRANSFORMER_PATH is not None:
                linear_transformer = LinearTransformer()
                linear_transformer.load_pca_transformation_data(TRANSFORMER_PATH)
                transformed_data = linear_transformer.transform_cells_to_pcs(
                    cells, n_pcs=2
                )
                pc0 = transformed_data[:, 0].flatten()
                pc1 = transformed_data[:, 1].flatten()

                pc0_all_thresh_high: Optional[float] = THRESHOLDS["pc0_all_thresh_high"]
                pc0_all_thresh_low: Optional[float] = THRESHOLDS["pc0_all_thresh_low"]
                pc1_all_thresh_high: Optional[float] = THRESHOLDS["pc1_all_thresh_high"]
                pc1_all_thresh_low: Optional[float] = THRESHOLDS["pc1_all_thresh_low"]

                is_valid_linear = np.ones_like(pc0, dtype=bool)
                is_valid_linear *= (
                    True if pc0_all_thresh_high is None else pc0 < pc0_all_thresh_high
                )
                is_valid_linear *= (
                    True if pc0_all_thresh_low is None else pc0 > pc0_all_thresh_low
                )
                is_valid_linear *= (
                    True if pc1_all_thresh_high is None else pc1 < pc1_all_thresh_high
                )
                is_valid_linear *= (
                    True if pc1_all_thresh_low is None else pc1 > pc1_all_thresh_low
                )

                pc0 = pc0[is_valid_linear]
                pc1 = pc1[is_valid_linear]

                cells_cleaned = [None] * np.sum(is_valid_linear)
                is_valid_idx = np.arange(is_valid_linear.shape[0], dtype=np.int64)[
                    is_valid_linear
                ]
                for i, idx in enumerate(is_valid_idx):
                    cells_cleaned[i] = cells[idx]
                cells = [*cells_cleaned]
                del cells_cleaned

                # classify pyknotic cells
                # cells, fig1 = classify_pyknotic_thresholds(
                #     cells, plot_histograms=True, show=False, **THRESHOLDS
                # )
                pc0_pyk_thresh_high: Optional[float] = THRESHOLDS["pc0_pyk_thresh_high"]
                pc0_pyk_thresh_low: Optional[float] = THRESHOLDS["pc0_pyk_thresh_low"]
                pc1_pyk_thresh_high: Optional[float] = THRESHOLDS["pc1_pyk_thresh_high"]
                pc1_pyk_thresh_low: Optional[float] = THRESHOLDS["pc1_pyk_thresh_low"]

                is_pyk_candidate = np.ones_like(pc0, dtype=bool)
                is_pyk_candidate *= (
                    True if pc0_pyk_thresh_high is None else pc0 < pc0_pyk_thresh_high
                )
                is_pyk_candidate *= (
                    True if pc0_pyk_thresh_low is None else pc0 > pc0_pyk_thresh_low
                )
                is_pyk_candidate *= (
                    True if pc1_pyk_thresh_high is None else pc1 < pc1_pyk_thresh_high
                )
                is_pyk_candidate *= (
                    True if pc1_pyk_thresh_low is None else pc1 > pc1_pyk_thresh_low
                )

                cells, fig1 = assign_pyknotics(
                    cells, is_pyk_candidate, plot_histograms=True, show=False
                )

            if rdf_path is not None:
                if TRANSFORMER_PATH is None:
                    candidate_indices: Optional[list[int]] = None
                    candidates: list[Cell] = cells
                else:
                    filt_is_candidate: callable[[int, Cell], bool] = lambda x: x[
                        1
                    ].is_pyknotic

                    candidate_indices: Optional[list[int]] = []
                    candidates: list[Cell] = []
                    for c_idx, cell in filter(filt_is_candidate, enumerate(cells)):
                        candidate_indices.append(int(c_idx))
                        candidates.append(cell)

                candidates, clsfs = classify_cells_rdf(rdf_path, cells=candidates)
                is_pyk_all = np.zeros(len(cells), dtype=bool)

                # stitch candidates back together with all cells
                for i, cell in enumerate(candidates):
                    j = i if candidate_indices is None else candidate_indices[i]
                    cells[j] = cell
                    is_pyk_all[j] = clsfs[i]

                # creates fig1
                cells, fig1 = assign_pyknotics(
                    cells, is_pyk_all, plot_histograms=True, show=False
                )

            elif RDF_PATH is None and TRANSFORMER_PATH is None:
                for cell in cells:
                    cell.is_pyknotic = False
                # FIXME: Optimizing cell params
                sigmas = np.array([cell.sigma_um for cell in cells])
                ints = np.array([cell.avg_intensity for cell in cells])
                sigma_thrsh = np.percentile(sigmas, 10)
                ints_thrsh = np.percentile(ints, 90)
                for cell in cells:
                    test = [
                        cell.sigma_um <= sigma_thrsh,
                        cell.avg_intensity >= ints_thrsh,
                    ]
                    cell.is_pyknotic = all(test)

            if TRANSFORMER_PATH is not None or rdf_path is not None:
                fig2: Optional[Figure] = plot_pyknotic(
                    clean_img, cells, raw_img=raw_img, show=False
                )
            else:
                fig2: Optional[Figure] = None
                # #FIXME: Optimizing cell params
                fig2: Optional[Figure] = plot_pyknotic(
                    clean_img, cells, raw_img=raw_img, show=False
                )

            # save cells and regions in a pickle file
            if SAVE_IMAGE_PICKLE:
                save_with_dataclass_pca(
                    pkl_path=os.path.join(out_dir, "annotations.pkl"),
                    data=StoredAnnotations1(
                        image_path=file,
                        regions=regions,
                        cells=cells,
                        image=clean_img,
                        image_meta=meta,
                        image_unprocessed=raw_img,
                    ),
                )

            if has_alx568:
                channels_merged, _ = view_nd2_images(path=file, scale_bar=50, show=False)

                if SAVE_FIGURES:
                    for fig in channels_merged:
                        title = fig.get_suptitle()
                        fig.savefig(
                            out_dir / f"{title.lower()}.svg", format="svg", dpi=300
                        )
                        plt.close(fig)

            if SAVE_FIGURES:
                if isinstance(fig0, Figure):
                    fig0.savefig(
                        out_dir / f"image_processing.{IMG_FMT}", format=IMG_FMT, dpi=600
                    )
                    plt.close(fig0)
                if isinstance(fig1, Figure):
                    fig1.savefig(
                        out_dir / f"histograms.{IMG_FMT}", format=IMG_FMT, dpi=400
                    )
                    plt.close(fig1)
                if isinstance(fig2, Figure):
                    fig2.savefig(
                        out_dir / f"cell_labels.{IMG_FMT}", format=IMG_FMT, dpi=600
                    )
            elif SHOW_FIGURES:
                plt.show()

        except Exception:
            log.log_err(file, str(traceback.format_exc()))
            continue

        cellprops.save_cells(
            file.stem, cells
        )  # saves cell properties for parameter tuning
        savecounts.cells_to_entry(file.name, cells)  # saves cell counts


if __name__ == "__main__":
    main()
