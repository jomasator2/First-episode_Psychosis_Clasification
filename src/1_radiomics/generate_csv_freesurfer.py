#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a CSV of FreeSurfer outputs and convert missing .mgz -> .nii with mri_convert.

- Scans first-level subfolders of ROOT for a `mri/` directory.
- Looks for `brain.mgz` and `aparc+aseg.mgz`; creates `brain.nii` and `aparc+aseg.nii` if absent.
- Emits a CSV with columns: [ID, image_mgz, aparc_mgz, Image, Mask].

Requirements:
- FreeSurfer must be available (mri_convert in PATH) for on-the-fly conversions.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd
from shutil import which


# -----------------------------
# Logging
# -----------------------------
LOGGER = logging.getLogger("freesurfer_table")


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    LOGGER.setLevel(level)
    LOGGER.handlers[:] = [handler]


# -----------------------------
# Core utilities
# -----------------------------
def convert_mgz_to_nii(src_mgz: Path, dst_nii: Path) -> bool:
    """
    Run FreeSurfer's mri_convert to create a NIfTI from an MGZ.
    Returns True on success, False otherwise.
    """
    if which("mri_convert") is None:
        LOGGER.error("`mri_convert` not found. Ensure FreeSurfer is correctly set in your PATH.")
        return False

    dst_nii.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["mri_convert", str(src_mgz), str(dst_nii)]
    LOGGER.debug("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        LOGGER.error("`mri_convert` is not available on this system.")
        return False

    if result.returncode != 0:
        LOGGER.error(
            "Conversion failed: '%s' -> '%s' (exit %s)\nstderr: %s",
            src_mgz, dst_nii, result.returncode, (result.stderr or "").strip(),
        )
        return False

    LOGGER.info("Created NIfTI: %s", dst_nii)
    return True


def scan_subject(root: Path, subject_dir: Path, do_convert: bool) -> Dict[str, str]:
    """
    Inspect a single subject directory:
    - Check `<subject>/mri/brain.mgz` and `<subject>/mri/aparc+aseg.mgz`
    - Create .nii counterparts if missing and `do_convert=True`
    - Return a dict row for the output table
    """
    mri_dir = subject_dir / "mri"
    if not mri_dir.is_dir():
        LOGGER.warning("Missing 'mri/' in '%s' — skipping.", subject_dir)
        return {
            "ID": subject_dir.name,
            "image_mgz": "",
            "aparc_mgz": "",
            "Image": "",
            "Mask": "",
        }

    brain_mgz = mri_dir / "brain.mgz"
    aparc_mgz = mri_dir / "aparc+aseg.mgz"
    brain_nii = mri_dir / "brain.nii"
    aparc_nii = mri_dir / "aparc+aseg.nii"

    image_mgz = str(brain_mgz) if brain_mgz.is_file() else ""
    aparc_path_mgz = str(aparc_mgz) if aparc_mgz.is_file() else ""
    image_nii = ""
    aparc_path_nii = ""

    # brain.mgz -> brain.nii
    if brain_mgz.is_file():
        if brain_nii.is_file():
            image_nii = str(brain_nii)
        elif do_convert:
            LOGGER.info("Missing %s — converting from %s", brain_nii, brain_mgz)
            if convert_mgz_to_nii(brain_mgz, brain_nii):
                image_nii = str(brain_nii)
        else:
            LOGGER.info("Missing %s (conversion disabled)", brain_nii)
    else:
        LOGGER.warning("Not found: %s", brain_mgz)

    # aparc+aseg.mgz -> aparc+aseg.nii
    if aparc_mgz.is_file():
        if aparc_nii.is_file():
            aparc_path_nii = str(aparc_nii)
        elif do_convert:
            LOGGER.info("Missing %s — converting from %s", aparc_nii, aparc_mgz)
            if convert_mgz_to_nii(aparc_mgz, aparc_nii):
                aparc_path_nii = str(aparc_nii)
        else:
            LOGGER.info("Missing %s (conversion disabled)", aparc_nii)
    else:
        LOGGER.warning("Not found: %s", aparc_mgz)

    return {
        "ID": subject_dir.name,
        "image_mgz": image_mgz,
        "aparc_mgz": aparc_path_mgz,
        "Image": image_nii,
        "Mask": aparc_path_nii,
    }


def build_table(root_path: Path, convert_missing: bool = True) -> pd.DataFrame:
    """
    Walk first-level subdirectories of `root_path`, gather MGZ/NII paths,
    convert MGZ->NII if missing and `convert_missing=True`, and return a DataFrame.
    """
    rows: List[Dict[str, str]] = []

    for entry in sorted(root_path.iterdir()):
        if entry.is_dir():
            rows.append(scan_subject(root_path, entry, convert_missing))

    return pd.DataFrame(rows, columns=["ID", "image_mgz", "aparc_mgz", "Image", "Mask"])


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate a CSV with FreeSurfer paths. Converts .mgz to .nii with mri_convert "
            "when needed (unless --no-convert)."
        )
    )
    p.add_argument("root_path", type=Path, help="Root directory containing subject subfolders.")
    p.add_argument(
        "-o", "--output", type=Path, default=Path("freesurfer_table.tsv"),
        help="Output CSV filename (default: freesurfer_table.tsv).",
    )
    p.add_argument(
        "--no-convert", action="store_true",
        help="Do not attempt .mgz -> .nii conversion; only report existing files.",
    )
    p.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    root = args.root_path
    if not root.is_dir():
        LOGGER.error("Provided path is not a directory or does not exist: %s", root)
        raise SystemExit(1)

    df = build_table(root, convert_missing=not args.no_convert)

    # Save CSV (with index label blank to match your original behavior)
    df.to_csv(args.output, sep="\t", index=False)
    LOGGER.info("CSV written: %s", args.output)


if __name__ == "__main__":
    main()
