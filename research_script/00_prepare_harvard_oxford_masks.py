#!/usr/bin/env python3
"""
Prepare Harvard-Oxford atlas masks for the Atari ANTs-MNI fMRI grid.

The Atari fMRI runs are normalized to an FSL MNI152-derived 2.5 mm grid using
ANTs. Harvard-Oxford max-probability atlases are FSL MNI atlases, but they are
distributed on 1 mm or 2 mm grids, so the label images still need nearest
neighbor resampling before use as ROI masks in the RSA/RDM scripts.
"""
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
    / "harvard_oxford"
)
DEFAULT_REFERENCE = Path(
    "/home/jeongmin/fmri/atari/derivatives/ants_mni/"
    "sub001-1/Session2/MNI152_T1_2.5x2.5x2.5mm_brain.nii"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ATLAS_DIR / "ants_mni_2p5mm_masks"

PFC_LABEL_PATTERNS = (
    "Frontal Pole",
    "Superior Frontal Gyrus",
    "Middle Frontal Gyrus",
    "Inferior Frontal Gyrus",
    "Frontal Medial Cortex",
    "Frontal Orbital Cortex",
)


@dataclass(frozen=True)
class AtlasSpec:
    scope: str
    atlas_name: str


@dataclass(frozen=True)
class RegionLabel:
    index: int
    name: str

    @property
    def is_background(self) -> bool:
        return self.index == 0 or self.name.lower() == "background"

    @property
    def side_and_base_name(self) -> Tuple[str | None, str]:
        for side in ("Left", "Right"):
            prefix = f"{side} "
            if self.name.startswith(prefix):
                return side, self.name[len(prefix) :]
        return None, self.name


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")


def atlas_specs(threshold: str, resolution: str) -> List[AtlasSpec]:
    return [
        AtlasSpec("cortical", f"cort-maxprob-thr{threshold}-{resolution}"),
        AtlasSpec("subcortical", f"sub-maxprob-thr{threshold}-{resolution}"),
    ]


def fetch_harvard_oxford(
    spec: AtlasSpec,
    data_dir: Path,
    symmetric_split: bool,
    verbose: int,
) -> Tuple[nib.spatialimages.SpatialImage, List[RegionLabel]]:
    try:
        from nilearn.datasets import fetch_atlas_harvard_oxford
    except ImportError as exc:
        raise ImportError(
            "nilearn is required to fetch and symmetric-split Harvard-Oxford atlases. "
            "Activate the thinker conda environment first."
        ) from exc

    atlas = fetch_atlas_harvard_oxford(
        spec.atlas_name,
        data_dir=str(data_dir),
        symmetric_split=symmetric_split,
        verbose=verbose,
    )

    maps = getattr(atlas, "maps", getattr(atlas, "filename", None))
    if maps is None:
        raise ValueError(f"Fetcher did not return atlas maps for {spec.atlas_name}")
    if isinstance(maps, nib.spatialimages.SpatialImage):
        atlas_img = maps
    else:
        atlas_img = nib.load(str(maps))

    if hasattr(atlas, "lut") and atlas.lut is not None:
        labels = [
            RegionLabel(index=int(row["index"]), name=str(row["name"]))
            for _, row in atlas.lut.iterrows()
        ]
    elif hasattr(atlas, "labels"):
        labels = [
            RegionLabel(index=i, name=str(name))
            for i, name in enumerate(atlas.labels)
        ]
    else:
        raise ValueError(f"Fetcher did not return labels for {spec.atlas_name}")

    return atlas_img, labels


def write_lut(labels: Sequence[RegionLabel], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "name"], delimiter="\t")
        writer.writeheader()
        for label in labels:
            writer.writerow({"index": label.index, "name": label.name})


def resample_labels_to_reference(
    atlas_img: nib.spatialimages.SpatialImage,
    ref_img: nib.spatialimages.SpatialImage,
) -> np.ndarray:
    atlas_data = np.rint(np.asanyarray(atlas_img.dataobj)).astype(np.int16)
    if atlas_data.ndim != 3:
        raise ValueError(f"Expected a 3D max-probability atlas, got shape {atlas_data.shape}")

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


def save_nifti(
    data: np.ndarray,
    ref_img: nib.spatialimages.SpatialImage,
    path: Path,
    dtype: np.dtype,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = ref_img.header.copy()
    hdr.set_data_shape(data.shape)
    hdr.set_data_dtype(dtype)
    img = nib.Nifti1Image(data.astype(dtype), ref_img.affine, hdr)
    nib.save(img, str(path))


def mask_for_indices(label_img: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    ids = np.array(list(indices), dtype=np.int16)
    if ids.size == 0:
        return np.zeros(label_img.shape, dtype=np.uint8)
    return np.isin(label_img, ids).astype(np.uint8)


def append_mask_row(
    rows: List[Dict[str, object]],
    mask_type: str,
    mask_name: str,
    labels: Sequence[RegionLabel],
    n_voxels: int,
    path: Path,
) -> None:
    rows.append(
        {
            "mask_type": mask_type,
            "mask_name": mask_name,
            "label_indices": ";".join(str(label.index) for label in labels),
            "label_names": ";".join(label.name for label in labels),
            "n_voxels": n_voxels,
            "path": str(path),
        }
    )


def save_mask(
    label_img: np.ndarray,
    ref_img: nib.spatialimages.SpatialImage,
    labels: Sequence[RegionLabel],
    path: Path,
) -> int:
    mask = mask_for_indices(label_img, [label.index for label in labels])
    save_nifti(mask, ref_img, path, np.uint8)
    return int(mask.sum())


def write_summary(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mask_type",
        "mask_name",
        "label_indices",
        "label_names",
        "n_voxels",
        "path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_single_region_masks(
    scope: str,
    label_img: np.ndarray,
    ref_img: nib.spatialimages.SpatialImage,
    labels: Sequence[RegionLabel],
    out: Path,
    rows: List[Dict[str, object]],
) -> None:
    for label in labels:
        if label.is_background:
            continue
        path = out / "masks" / scope / f"roi-{scope}-{label.index:03d}_{sanitize_name(label.name)}_mask.nii.gz"
        n_voxels = save_mask(label_img, ref_img, [label], path)
        append_mask_row(rows, scope, label.name, [label], n_voxels, path)


def save_bilateral_masks(
    scope: str,
    label_img: np.ndarray,
    ref_img: nib.spatialimages.SpatialImage,
    labels: Sequence[RegionLabel],
    out: Path,
    rows: List[Dict[str, object]],
) -> None:
    by_base_name: Dict[str, List[RegionLabel]] = {}
    for label in labels:
        if label.is_background:
            continue
        _, base_name = label.side_and_base_name
        by_base_name.setdefault(base_name, []).append(label)

    for base_name, grouped_labels in sorted(by_base_name.items()):
        if len(grouped_labels) < 2:
            continue
        path = out / "masks" / "bilateral" / scope / f"roi-{scope}-bilateral-{sanitize_name(base_name)}_mask.nii.gz"
        n_voxels = save_mask(label_img, ref_img, grouped_labels, path)
        append_mask_row(rows, f"bilateral_{scope}", f"Bilateral {base_name}", grouped_labels, n_voxels, path)


def save_all_scope_mask(
    scope: str,
    label_img: np.ndarray,
    ref_img: nib.spatialimages.SpatialImage,
    labels: Sequence[RegionLabel],
    out: Path,
    rows: List[Dict[str, object]],
) -> np.ndarray:
    foreground_labels = [label for label in labels if not label.is_background]
    path = out / "masks" / "atlas" / f"roi-HarvardOxford-{scope}-all_mask.nii.gz"
    mask = mask_for_indices(label_img, [label.index for label in foreground_labels])
    save_nifti(mask, ref_img, path, np.uint8)
    append_mask_row(rows, f"{scope}_all", f"HarvardOxford {scope} all", foreground_labels, int(mask.sum()), path)
    return mask


def labels_matching(labels: Sequence[RegionLabel], patterns: Sequence[str]) -> List[RegionLabel]:
    matched: List[RegionLabel] = []
    for label in labels:
        if label.is_background:
            continue
        if any(pattern.lower() in label.name.lower() for pattern in patterns):
            matched.append(label)
    return matched


def save_group_masks(
    label_images: Dict[str, np.ndarray],
    labels_by_scope: Dict[str, List[RegionLabel]],
    ref_img: nib.spatialimages.SpatialImage,
    out: Path,
    rows: List[Dict[str, object]],
) -> None:
    group_specs = [
        ("Hippocampus", "subcortical", ("Hippocampus",)),
        ("PFC", "cortical", PFC_LABEL_PATTERNS),
        ("Frontal", "cortical", ("Frontal",)),
        ("OFC", "cortical", ("Frontal Orbital Cortex",)),
    ]

    for group_name, scope, patterns in group_specs:
        labels = labels_matching(labels_by_scope.get(scope, []), patterns)
        if not labels:
            continue
        path = out / "masks" / "group" / f"roi-HarvardOxford-{sanitize_name(group_name)}_mask.nii.gz"
        n_voxels = save_mask(label_images[scope], ref_img, labels, path)
        append_mask_row(rows, "group", group_name, labels, n_voxels, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_ATLAS_DIR)
    parser.add_argument("--reference-fmri", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", choices=["0", "25", "50"], default="25")
    parser.add_argument("--resolution", choices=["1mm", "2mm"], default="1mm")
    parser.add_argument(
        "--no-symmetric-split",
        action="store_true",
        help="Do not ask nilearn to split symmetric labels into left/right masks.",
    )
    parser.add_argument(
        "--skip-group-masks",
        action="store_true",
        help="Only write atlas/ROI/bilateral masks, not convenience group masks such as Hippocampus and PFC.",
    )
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    if not args.reference_fmri.exists():
        raise FileNotFoundError(f"Reference fMRI/MNI image not found: {args.reference_fmri}")

    symmetric_split = not args.no_symmetric_split
    ref_img = nib.load(str(args.reference_fmri))
    out = args.output_dir

    summary_rows: List[Dict[str, object]] = []
    label_images: Dict[str, np.ndarray] = {}
    labels_by_scope: Dict[str, List[RegionLabel]] = {}
    all_masks: List[np.ndarray] = []

    for spec in atlas_specs(args.threshold, args.resolution):
        atlas_img, labels = fetch_harvard_oxford(
            spec,
            data_dir=args.data_dir,
            symmetric_split=symmetric_split,
            verbose=args.verbose,
        )
        label_img = resample_labels_to_reference(atlas_img, ref_img)
        label_images[spec.scope] = label_img
        labels_by_scope[spec.scope] = labels

        lut_path = out / "labels" / f"HarvardOxford-{spec.scope}-{spec.atlas_name}_lut.tsv"
        write_lut(labels, lut_path)

        dseg_path = out / "resampled" / (
            f"HarvardOxford-{spec.atlas_name}_"
            f"space-antsmni_res-2p5_desc-symmetric{int(symmetric_split)}_dseg.nii.gz"
        )
        save_nifti(label_img, ref_img, dseg_path, np.int16)
        foreground_labels = [label for label in labels if not label.is_background]
        append_mask_row(
            summary_rows,
            "dseg",
            f"HarvardOxford {spec.scope} {spec.atlas_name}",
            foreground_labels,
            int(np.sum(label_img > 0)),
            dseg_path,
        )

        scope_mask = save_all_scope_mask(spec.scope, label_img, ref_img, labels, out, summary_rows)
        all_masks.append(scope_mask)
        save_single_region_masks(spec.scope, label_img, ref_img, labels, out, summary_rows)
        save_bilateral_masks(spec.scope, label_img, ref_img, labels, out, summary_rows)

    if all_masks:
        all_mask = np.any(np.stack(all_masks, axis=0).astype(bool), axis=0).astype(np.uint8)
        all_path = out / "masks" / "atlas" / "roi-HarvardOxford-all_mask.nii.gz"
        save_nifti(all_mask, ref_img, all_path, np.uint8)
        all_labels = [
            label
            for labels in labels_by_scope.values()
            for label in labels
            if not label.is_background
        ]
        append_mask_row(summary_rows, "atlas_all", "HarvardOxford all", all_labels, int(all_mask.sum()), all_path)

    if not args.skip_group_masks:
        save_group_masks(label_images, labels_by_scope, ref_img, out, summary_rows)

    summary_path = out / "harvard_oxford_mask_summary.csv"
    write_summary(summary_rows, summary_path)

    print(f"[data_dir] {args.data_dir}")
    print(f"[reference] {args.reference_fmri}")
    print(f"[output] {out}")
    print(f"[summary] {summary_path}")
    print(f"[done] wrote {len(summary_rows)} mask entries")


if __name__ == "__main__":
    main()
