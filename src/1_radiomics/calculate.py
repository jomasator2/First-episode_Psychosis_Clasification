# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Radiomics feature extraction with PyRadiomics using multiprocessing.

- Reads an input CSV with columns: ID, Image, Mask
- Optionally loads parameters from a PyRadiomics YAML file
- Extracts features per label in the mask for each case
- Writes a wide CSV with one row per case (and per-label prefixes)

Original authorship and adaptations preserved in header comments below.
"""

# author: Adolfo López Cerdán
# email: adlpecer@gmail.com
# Description: Radiomics features extraction with PyRadiomics using multiprocessing
# and pandas formatting. Strongly based on PyRadiomics examples (http://radiomics.io).
# Adapted by: Hector Carceller, Joaquim Montell and Jesus Alejandro Alzate
# Adapted by: Elena Oliver-Garcia 20230522
# Adapted by: Alejandro Mora-Rubio 20240125
# Adapted by: Jose Manuel Saborit-Torres 20250922

from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

import radiomics
from radiomics import featureextractor, imageoperations


# -----------------------------
# Configuration flags (can be overridden here if needed)
# -----------------------------
SHIFT_MASK = False
EROSION = False
DILATION = False
CENTER_SLICE = False
REORIENT = True


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class Case:
    """Single case containing identifiers and paths."""
    id: str
    image_path: Path
    mask_path: Path


# -----------------------------
# Utilities
# -----------------------------
def check_mask_volume(image: sitk.Image, mask: sitk.Image, label: int) -> int | None:
    """Validate that the mask for a given label has sufficient volume."""
    try:
        imageoperations.checkMask(
            image, mask, minimumROIDimensions=3, minimumROISize=1000, label=label
        )
        return label
    except Exception:
        return None


def configure_extractor(params_path: Path | None) -> featureextractor.RadiomicsFeatureExtractor:
    """Create a PyRadiomics extractor from YAML params or fallback settings."""
    if params_path and params_path.exists():
        return featureextractor.RadiomicsFeatureExtractor(str(params_path))

    # Fallback defaults if no params provided
    settings = {
        "binWidth": 25,
        "resampledPixelSpacing": None,
        "interpolator": sitk.sitkBSpline,
        "enableCExtensions": True,
    }
    return featureextractor.RadiomicsFeatureExtractor(**settings)


def load_images(image_path: Path, mask_path: Path) -> Tuple[sitk.Image, sitk.Image]:
    """Read image and mask from disk."""
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    return image, mask


def maybe_reorient(image: sitk.Image, mask: sitk.Image) -> Tuple[sitk.Image, sitk.Image]:
    """Reorient both image and mask to RAS if enabled."""
    if not REORIENT:
        return image, mask
    orient = sitk.DICOMOrientImageFilter()
    orient.SetDesiredCoordinateOrientation("RAS")
    return orient.Execute(image), orient.Execute(mask)


def maybe_center_slice(image: sitk.Image, mask: sitk.Image) -> Tuple[sitk.Image, sitk.Image]:
    """Reduce to center axial slice if enabled (debug/experimentation)."""
    if not CENTER_SLICE:
        return image, mask
    center = image.GetSize()[2] // 2
    return image[:, :, center], mask[:, :, center]


def maybe_shift_mask(mask: sitk.Image) -> sitk.Image:
    """Apply a small random translation to the mask if enabled (perturbation)."""
    if not SHIFT_MASK:
        return mask
    max_shift_x, max_shift_y = 3, 5
    dx = int(np.random.randint(-max_shift_x, max_shift_x + 1))
    dy = int(np.random.randint(-max_shift_y, max_shift_y + 1))
    translation = sitk.TranslationTransform(3, [dx, dy, 0])
    return sitk.Resample(mask, translation)


def maybe_morphology(mask: sitk.Image, labels: Iterable[int]) -> sitk.Image:
    """Optionally erode/dilate mask per label."""
    if not (EROSION or DILATION):
        return mask

    working = mask
    label_list = list(labels)
    if EROSION:
        for lab in label_list:
            working = sitk.BinaryErode(working, foregroundValue=float(lab))
    if DILATION:
        for lab in label_list:
            working = sitk.BinaryDilate(working, foregroundValue=float(lab))
    return working


# -----------------------------
# Core processing
# -----------------------------
def run(row: tuple[int, pd.Series]) -> Tuple[pd.Series | pd.DataFrame, float]:
    """
    Process a single case row:
    - Load image/mask
    - (Optional) reorient/perturb
    - Extract features for each label present in mask
    - Return a pandas Series (or wide DataFrame if multiple labels) and execution time
    """
    start = time.time()

    _, case_row = row
    case_id = str(case_row["ID"])
    image_path = Path(case_row["Image"])
    mask_path = Path(case_row["Mask"])

    logger = rLogger.getChild(case_id)
    threading.current_thread().name = case_id

    extractor = configure_extractor(params)

    logger.info("Processing case %s (Image: %s, Mask: %s)", case_id, image_path, mask_path)

    # Load inputs
    try:
        image, mask = load_images(image_path, mask_path)
    except Exception:
        logger.error("Failed to read image/mask from disk.", exc_info=True)
        return pd.Series(name=case_id), time.time() - start

    # Pre-processing (optional)
    image, mask = maybe_reorient(image, mask)
    image, mask = maybe_center_slice(image, mask)
    mask = maybe_shift_mask(mask)

    # Labels present in the mask
    labels = np.unique(sitk.GetArrayFromImage(mask).ravel())
    labels = [int(l) for l in labels]  # keep original behavior (includes 0 if present)

    # Optional morphology
    mask = maybe_morphology(mask, labels)

    # Extract features per label
    per_label_results: list[pd.Series] = []
    for label in labels:
        logger.info("Case %s -> Label %s", case_id, label)
        try:
            series = pd.Series(extractor.execute(image, mask, label), dtype="object")
        except Exception:
            logger.error("Feature extraction failed for label %s.", label, exc_info=True)
            series = pd.Series(dtype="object")

        # Prefix by label to avoid collisions across multi-label masks
        series.name = case_id
        series = series.add_prefix(f"label{label}_")
        per_label_results.append(series)

    # Consolidate per-case
    if not per_label_results:
        logger.error("No features extracted: %s", case_id)
        result: pd.Series | pd.DataFrame = pd.Series(name=case_id, dtype="object")
    elif len(per_label_results) == 1:
        result = per_label_results[0]
    else:
        # Multiple labels: concatenate into one long Series (still one row after transpose)
        result = pd.concat(per_label_results, axis=0)

    elapsed = time.time() - start
    return result, elapsed


def run_batch(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[float]]:
    """Run multiprocessing over the input DataFrame; merge and return results."""
    with Pool(processes=cpu_count()) as pool:
        pool_results = pool.map(run, df.iterrows())

    rows = [r for r, _ in pool_results]
    times = [t for _, t in pool_results]

    # Merge all results efficiently
    # Each item in `rows` is a Series with name=case_id (or an empty Series).
    results = pd.concat(rows, axis=1) if rows else pd.DataFrame()
    results = results.T  # one row per case
    return results, times


# -----------------------------
# Entry point
# -----------------------------
def main() -> None:
    # Batch logger
    logger = rLogger.getChild("batch")

    # Increase verbosity of PyRadiomics (and ensure consistent formatting)
    radiomics.setVerbosity(logging.INFO)
    logger.info("PyRadiomics version: %s", radiomics.__version__)
    # --- Read input list of cases (CSV or TSV, auto-detect separator) ---
    try:
        # sep=None + engine='python' -> sniffear separador (coma, tab, etc.)
        df = pd.read_csv(input_csv, sep=None, engine="python")

        # Si viene una columna índice accidental, elimínala
        for junk in ("Unnamed: 0", "index"):
            if junk in df.columns:
                df = df.drop(columns=[junk])

    except Exception:
        logger.error("Failed to read CSV/TSV: %s", input_csv, exc_info=True)
        raise SystemExit(1)

    # Basic validation
    required_cols = {"ID", "Image", "Mask"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error(
            "CSV missing required columns: %s. Found columns: %s",
            ", ".join(sorted(missing)),
            ", ".join(df.columns.astype(str)),
        )
        raise SystemExit(1)

    logger.info("Loaded %d cases. Columns: %s", df.shape[0], ", ".join(df.columns))


    # Process
    results, times = run_batch(df)

    # Persist
    logger.info("Writing features to: %s", output_csv)
    results.to_csv(output_csv, na_rep="NaN", sep="\t", index=False)

    if times:
        logger.info(
            "Average execution time: %.2f ± %.2f seconds",
            float(np.mean(times)),
            float(np.std(times)),
        )
    logger.info("Done.")


if __name__ == "__main__":
    # Thread label for main
    threading.current_thread().name = "Main"

    # Force ITK single-thread filters (since we parallelize at the case level)
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)

    # -----------------------------
    # CLI
    # -----------------------------
    parser = argparse.ArgumentParser(
        prog="Radiomics Feature Extractor",
        description=(
            "Read a CSV with columns [ID, Image, Mask] and extract PyRadiomics features "
            "per label, using an optional parameter YAML file."
        ),
    )
    parser.add_argument("input_csv", type=Path, help="Path to the input TSV.")
    parser.add_argument("output_csv", type=Path, help="Path to the output TSV.")
    parser.add_argument(
        "--param",
        type=Path,
        default=Path(__file__).with_name("Params.yaml"),
        help="Path to the PyRadiomics parameter file (default: ./Params.yaml).",
    )
    parser.add_argument(
        "--logfile",
        type=Path,
        default=None,
        help="Path to the log file (default: None -> log to console).",
    )
    args = parser.parse_args()

    # Bind global-like variables used by worker function
    input_csv: Path = args.input_csv
    output_csv: Path = args.output_csv
    params: Path | None = args.param if args.param is not None else None

    # Configure logging once (root radiomics logger is reused in workers)
    rLogger = radiomics.logger
    rLogger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(levelname)s: (%(threadName)s) %(name)s: %(message)s")

    if args.logfile:
        file_handler = logging.FileHandler(filename=str(args.logfile), mode="a")
        file_handler.setFormatter(fmt)
        rLogger.addHandler(file_handler)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        rLogger.addHandler(stream_handler)

    main()
