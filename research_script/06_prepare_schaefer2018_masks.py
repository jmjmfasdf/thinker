#!/usr/bin/env python3
"""
Prepare TemplateFlow Schaefer2018 atlas masks for the Atari ANTs-MNI fMRI grid.

The TemplateFlow atlas used in the visualisation notebook is a 1 mm
MNI152NLin2009cAsym label image. The Atari fMRI runs in this project are on a
2.5 mm ANTs-MNI grid, so the atlas must be nearest-neighbor resampled before it
can be passed to 06_representational_mechanism.py as an ROI mask.
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ATLAS_DIR = (
    ROOT
    / "research_script"
    / "outputs"
    / "06_representational_mechanism"
    / "atlas"
    / "templateflow_schaefer2018"
)
DEFAULT_ATLAS = DEFAULT_ATLAS_DIR / (
    "tpl-MNI152NLin2009cAsym_res-01_atlas-Schaefer2018_"
    "desc-100Parcels7Networks_dseg.nii.gz"
)
DEFAULT_LABELS = DEFAULT_ATLAS_DIR / (
    "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018_"
    "desc-100Parcels7Networks_dseg.tsv"
)
DEFAULT_REFERENCE = Path(
    "/home/jeongmin/fmri/atari/derivatives/ants_mni/"
    "sub001-1/Session3/s5_wfiltered_func_data.nii"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ATLAS_DIR / "ants_mni_2p5mm_masks"


@dataclass(frozen=True)
class ParcelLabel:
    index: int
    name: str
    color: str

    @property
    def hemisphere(self) -> str:
        return self.name.split("_")[1]

    @property
    def network(self) -> str:
        return self.name.split("_")[2]


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")


def load_labels(path: Path) -> List[ParcelLabel]:
    labels: List[ParcelLabel] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            labels.append(
                ParcelLabel(
                    index=int(row["index"]),
                    name=str(row["name"]),
                    color=str(row.get("color", "")),
                )
            )
    return labels


def resample_labels_to_reference(atlas_img: nib.spatialimages.SpatialImage, ref_img: nib.spatialimages.SpatialImage) -> np.ndarray:
    atlas_data = np.rint(np.asanyarray(atlas_img.dataobj)).astype(np.int16)
    ref_shape = tuple(int(x) for x in ref_img.shape[:3])

    source_from_target = np.linalg.inv(atlas_img.affine) @ ref_img.affine
    matrix = source_from_target[:3, :3]
    offset = source_from_target[:3, 3]
    resampled = affine_transform(
        atlas_data,
        matrix=matrix,
        offset=offset,
        output_shape=ref_shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    return np.rint(resampled).astype(np.int16)


def save_nifti(data: np.ndarray, ref_img: nib.spatialimages.SpatialImage, path: Path, dtype: np.dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = ref_img.header.copy()
    hdr.set_data_shape(data.shape)
    hdr.set_data_dtype(dtype)
    img = nib.Nifti1Image(data.astype(dtype), ref_img.affine, hdr)
    nib.save(img, str(path))


def save_mask(
    label_img: np.ndarray,
    ref_img: nib.spatialimages.SpatialImage,
    labels: Iterable[int],
    path: Path,
) -> int:
    ids = np.array(list(labels), dtype=np.int16)
    mask = np.isin(label_img, ids).astype(np.uint8)
    save_nifti(mask, ref_img, path, np.uint8)
    return int(mask.sum())


def write_summary(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mask_type",
        "mask_name",
        "label_indices",
        "n_voxels",
        "path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--reference-fmri", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-parcel-masks",
        action="store_true",
        help="Only write network and hemisphere-network masks, not all 100 parcel masks.",
    )
    args = parser.parse_args()

    if not args.atlas.exists():
        raise FileNotFoundError(f"Atlas NIfTI not found: {args.atlas}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Atlas TSV not found: {args.labels}")
    if not args.reference_fmri.exists():
        raise FileNotFoundError(f"Reference fMRI not found: {args.reference_fmri}")

    labels = load_labels(args.labels)
    atlas_img = nib.load(str(args.atlas))
    ref_img = nib.load(str(args.reference_fmri))
    label_img = resample_labels_to_reference(atlas_img, ref_img)

    out = args.output_dir
    resampled_path = out / "resampled" / (
        "tpl-MNI152NLin2009cAsym_space-antsmni_res-2p5_"
        "atlas-Schaefer2018_desc-100Parcels7Networks_dseg.nii.gz"
    )
    save_nifti(label_img, ref_img, resampled_path, np.int16)

    summary_rows: List[Dict[str, object]] = [
        {
            "mask_type": "dseg",
            "mask_name": "Schaefer2018_100Parcels7Networks_resampled",
            "label_indices": ";".join(str(label.index) for label in labels),
            "n_voxels": int(np.sum(label_img > 0)),
            "path": str(resampled_path),
        }
    ]

    by_network: Dict[str, List[int]] = {}
    by_hemi_network: Dict[str, List[int]] = {}
    for label in labels:
        by_network.setdefault(label.network, []).append(label.index)
        by_hemi_network.setdefault(f"{label.hemisphere}_{label.network}", []).append(label.index)

    for network, indices in sorted(by_network.items()):
        path = out / "masks" / "network" / f"roi-network-{sanitize_name(network)}_mask.nii.gz"
        n_voxels = save_mask(label_img, ref_img, indices, path)
        summary_rows.append(
            {
                "mask_type": "network",
                "mask_name": network,
                "label_indices": ";".join(str(i) for i in indices),
                "n_voxels": n_voxels,
                "path": str(path),
            }
        )

    for name, indices in sorted(by_hemi_network.items()):
        path = out / "masks" / "hemisphere_network" / f"roi-{sanitize_name(name)}_mask.nii.gz"
        n_voxels = save_mask(label_img, ref_img, indices, path)
        summary_rows.append(
            {
                "mask_type": "hemisphere_network",
                "mask_name": name,
                "label_indices": ";".join(str(i) for i in indices),
                "n_voxels": n_voxels,
                "path": str(path),
            }
        )

    if not args.skip_parcel_masks:
        for label in labels:
            path = out / "masks" / "parcel" / f"roi-{label.index:03d}_{sanitize_name(label.name)}_mask.nii.gz"
            n_voxels = save_mask(label_img, ref_img, [label.index], path)
            summary_rows.append(
                {
                    "mask_type": "parcel",
                    "mask_name": label.name,
                    "label_indices": str(label.index),
                    "n_voxels": n_voxels,
                    "path": str(path),
                }
            )

    summary_path = out / "schaefer2018_100parcels7networks_mask_summary.csv"
    write_summary(summary_rows, summary_path)

    print(f"[atlas] {args.atlas}")
    print(f"[labels] {args.labels}")
    print(f"[reference] {args.reference_fmri}")
    print(f"[resampled] {resampled_path}")
    print(f"[summary] {summary_path}")
    print(f"[done] wrote {len(summary_rows)} mask entries")


if __name__ == "__main__":
    main()
