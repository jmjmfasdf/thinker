#!/usr/bin/env python3
"""Shared Harvard--Oxford ROI contracts and high-resolution MNI figures.

This module is deliberately display/post-processing only.  It does not fit a
neural model or alter the 110-parcel first-level estimands used by Analyses 06
and 07.  The eight bilateral regions below operationalize the anatomical ROI
set in Sung et al. with the Harvard--Oxford atlas available in this project.
The mapping is fixed independently of the current effect estimates.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from nibabel.processing import resample_from_to


SCRIPT_DIR = Path(__file__).resolve().parent
ATLAS_ROOT = (
    SCRIPT_DIR
    / "legacy"
    / "06_representational_mechanism"
    / "atlas"
    / "harvard_oxford"
    / "fsl"
    / "data"
    / "atlases"
)
MNI_TEMPLATE_CANDIDATES = (
    # Native 1-mm TemplateFlow MNI152NLin6Asym brain.  Its 182 x 218 x 182
    # RAS grid and affine exactly match the bundled 1-mm Harvard--Oxford atlas.
    Path(
        "/home/jeongmin/.cache/templateflow/tpl-MNI152NLin6Asym/"
        "tpl-MNI152NLin6Asym_res-01_desc-brain_T1w.nii.gz"
    ),
    # Backward-compatible display fallback.  This is resampled only when the
    # native 1-mm template is unavailable.
    Path(
        "/home/jeongmin/fmri/atari/derivatives/ants_mni/sub001-1/Session2/"
        "MNI152_T1_2.5x2.5x2.5mm_brain.nii"
    ),
)

# Matplotlib otherwise injects the current timestamp into every PDF.  Fixed
# metadata keeps shared anatomy/maps byte-stable across game-specific runs so
# artifact hashes in each manifest refer to the same file.
PDF_METADATA = {
    "Creator": "Thinker research_script",
    "CreationDate": None,
    "ModDate": None,
}

ROI_MAPPING_VERSION = "sung8-harvard-oxford-v1"
ROI_MAPPING_SOURCE = (
    "Sung et al. Figure 5 anatomical set; Harvard--Oxford bilateral "
    "operationalization fixed independently of current effects"
)

# Sung used AAL3 (and JuBrain for preSMA).  Harvard--Oxford cannot isolate
# preSMA from SMA, so this mapping is explicitly labelled SMA/preSMA proxy.
PREDEFINED_ROIS: tuple[dict[str, Any], ...] = (
    {
        "roi_key": "vlPFC",
        "roi_label": "vlPFC",
        "anatomical_label": "Inferior frontal gyrus, pars triangularis",
        "components": (
            "ho_cortical_left_inferior_frontal_gyrus_pars_triangularis",
            "ho_cortical_right_inferior_frontal_gyrus_pars_triangularis",
        ),
    },
    {
        "roi_key": "dlPFC",
        "roi_label": "dlPFC",
        "anatomical_label": "Middle frontal gyrus",
        "components": (
            "ho_cortical_left_middle_frontal_gyrus",
            "ho_cortical_right_middle_frontal_gyrus",
        ),
    },
    {
        "roi_key": "OFC",
        "roi_label": "OFC",
        "anatomical_label": "Frontal orbital cortex",
        "components": (
            "ho_cortical_left_frontal_orbital_cortex",
            "ho_cortical_right_frontal_orbital_cortex",
        ),
    },
    {
        "roi_key": "ACC",
        "roi_label": "ACC",
        "anatomical_label": "Anterior cingulate gyrus",
        "components": (
            "ho_cortical_left_cingulate_gyrus_anterior_division",
            "ho_cortical_right_cingulate_gyrus_anterior_division",
        ),
    },
    {
        "roi_key": "SMA_preSMA",
        "roi_label": "SMA/preSMA",
        "anatomical_label": "Juxtapositional lobule (SMA/preSMA proxy)",
        "components": (
            "ho_cortical_left_juxtapositional_lobule_cortex_formerly_supplementary_motor_cortex",
            "ho_cortical_right_juxtapositional_lobule_cortex_formerly_supplementary_motor_cortex",
        ),
    },
    {
        "roi_key": "V1",
        "roi_label": "V1",
        "anatomical_label": "Intracalcarine cortex (V1 proxy)",
        "components": (
            "ho_cortical_left_intracalcarine_cortex",
            "ho_cortical_right_intracalcarine_cortex",
        ),
    },
    {
        "roi_key": "HPC",
        "roi_label": "HPC",
        "anatomical_label": "Hippocampus",
        "components": (
            "ho_subcortical_left_hippocampus",
            "ho_subcortical_right_hippocampus",
        ),
    },
    {
        "roi_key": "vStr",
        "roi_label": "vStr",
        "anatomical_label": "Accumbens (ventral-striatum proxy)",
        "components": (
            "ho_subcortical_left_accumbens",
            "ho_subcortical_right_accumbens",
        ),
    },
)

ROI_ORDER = tuple(str(row["roi_key"]) for row in PREDEFINED_ROIS)
ROI_LABELS = {str(row["roi_key"]): str(row["roi_label"]) for row in PREDEFINED_ROIS}

# Display-only labels use the exact bilateral Harvard--Oxford parcel name with
# the hemisphere prefix removed.  Keeping these separate from
# ``anatomical_label`` avoids presenting the Sung-to-Harvard--Oxford proxy name
# as though it were an atlas label.
ROI_ATLAS_COMPONENT_LABELS = {
    "vlPFC": ("Inferior frontal gyrus, pars triangularis",),
    "dlPFC": ("Middle frontal gyrus",),
    "OFC": ("Frontal orbital cortex",),
    "ACC": ("Cingulate gyrus, anterior division",),
    "SMA_preSMA": (
        "Juxtapositional lobule cortex "
        "[formerly supplementary motor cortex]",
    ),
    "V1": ("Intracalcarine cortex",),
    "HPC": ("Hippocampus",),
    "vStr": ("Accumbens",),
}

# Eight fixed, colorblind-conscious hues make the anatomical definitions easy
# to track without implying an ordered effect scale.
ROI_ANATOMY_COLORS = {
    "vlPFC": "#4477AA",
    "dlPFC": "#EE6677",
    "OFC": "#228833",
    "ACC": "#CCBB44",
    "SMA_preSMA": "#AA3377",
    "V1": "#66CCEE",
    "HPC": "#EE7733",
    "vStr": "#332288",
}

CROSS_RED_YELLOW = LinearSegmentedColormap.from_list(
    "cross_red_yellow",
    ("#C8191E", "#ED3B24", "#F57C20", "#FDBE32", "#FFE45C"),
)
SIGNED_BLUE_RED_YELLOW = LinearSegmentedColormap.from_list(
    "signed_blue_red_yellow",
    ("#244A78", "#6FA6D8", "#F4F4F4", "#C8191E", "#FFE45C"),
)
INK = "#222426"
GRAY = "#6C737A"
ROI_ATLAS_ALIASES = {
    # planning_brain_common uses "opercular" while the FSL XML uses
    # "operculum" for these two Harvard--Oxford labels.
    "ho_cortical_left_frontal_opercular_cortex": (
        "ho_cortical_left_frontal_operculum_cortex"
    ),
    "ho_cortical_right_frontal_opercular_cortex": (
        "ho_cortical_right_frontal_operculum_cortex"
    ),
    "ho_cortical_left_parietal_opercular_cortex": (
        "ho_cortical_left_parietal_operculum_cortex"
    ),
    "ho_cortical_right_parietal_opercular_cortex": (
        "ho_cortical_right_parietal_operculum_cortex"
    ),
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def predefined_roi_mapping_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for definition in PREDEFINED_ROIS:
        for component in definition["components"]:
            rows.append(
                {
                    "mapping_version": ROI_MAPPING_VERSION,
                    "mapping_source": ROI_MAPPING_SOURCE,
                    "roi_key": str(definition["roi_key"]),
                    "roi_label": str(definition["roi_label"]),
                    "anatomical_label": str(definition["anatomical_label"]),
                    "hemisphere": (
                        "left" if "_left_" in component else "right"
                    ),
                    "component_roi": str(component),
                    "component_weight": 0.5,
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != 16 or frame["roi_key"].nunique() != 8:
        raise AssertionError("Predefined ROI mapping must be 8 bilateral pairs")
    return frame


def aggregate_predefined_rois(
    frame: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    grouping_columns: Sequence[str],
) -> pd.DataFrame:
    """Equal-average left/right parcel effects within each predefined ROI."""
    mapping = predefined_roi_mapping_frame()
    required = set(grouping_columns) | {"roi"} | set(value_columns)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Predefined ROI input lacks columns: {missing}")
    selected = frame.merge(
        mapping[["roi_key", "roi_label", "anatomical_label", "component_roi"]],
        left_on="roi",
        right_on="component_roi",
        how="inner",
        validate="many_to_one",
    )
    expected_groups = selected.groupby(
        list(grouping_columns) + ["roi_key"], dropna=False
    )["component_roi"].nunique()
    if expected_groups.empty or not expected_groups.eq(2).all():
        bad = expected_groups[~expected_groups.eq(2)]
        raise ValueError(
            "Every predefined ROI needs both hemispheres; bad groups="
            f"{bad.head().to_dict()}"
        )
    for column in value_columns:
        selected[column] = pd.to_numeric(selected[column], errors="raise")
        if not np.all(np.isfinite(selected[column].to_numpy(float))):
            raise ValueError(f"Non-finite predefined ROI values in {column}")
    aggregations: dict[str, tuple[str, str]] = {
        column: (column, "mean") for column in value_columns
    }
    aggregations["n_component_parcels"] = ("component_roi", "nunique")
    result = (
        selected.groupby(
            list(grouping_columns)
            + ["roi_key", "roi_label", "anatomical_label"],
            sort=False,
            dropna=False,
        )
        .agg(**aggregations)
        .reset_index()
    )
    result["mapping_version"] = ROI_MAPPING_VERSION
    result["hemisphere_weighting"] = "equal_left_right_mean"
    return result


def _atlas_label_key(prefix: str, label: str) -> str:
    return prefix + re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


@lru_cache(maxsize=1)
def load_mni_resources() -> dict[str, Any]:
    """Load the analysis atlas and a display-only 1-mm MNI T1 underlay."""
    contracts = (
        (
            "ho_cortical_",
            "HarvardOxford/HarvardOxford-cortl-maxprob-thr25-1mm.nii.gz",
            "HarvardOxford-Cortical-Lateralized.xml",
            False,
        ),
        (
            "ho_subcortical_",
            "HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz",
            "HarvardOxford-Subcortical.xml",
            True,
        ),
    )
    subcortical = (
        "Thalamus", "Caudate", "Putamen", "Pallidum", "Hippocampus",
        "Amygdala", "Accumbens",
    )
    resources: list[tuple[np.ndarray, dict[str, int]]] = []
    reference = None
    brain = None
    for prefix, nii_name, xml_name, is_subcortical in contracts:
        nii_path = ATLAS_ROOT / nii_name
        xml_path = ATLAS_ROOT / xml_name
        if not nii_path.is_file() or not xml_path.is_file():
            raise FileNotFoundError(f"Missing Harvard--Oxford resource: {nii_path}")
        image = nib.as_closest_canonical(nib.load(str(nii_path)))
        if reference is None:
            reference = image
        elif image.shape != reference.shape or not np.allclose(
            image.affine, reference.affine
        ):
            raise ValueError("Harvard--Oxford grids do not match")
        labels = {
            _atlas_label_key(prefix, element.text or ""): int(element.attrib["index"]) + 1
            for element in ET.parse(xml_path).getroot().findall(".//label")
        }
        if is_subcortical:
            labels = {
                key: value for key, value in labels.items()
                if any(token.lower() in key for token in subcortical)
            }
        atlas = np.asarray(image.dataobj, dtype=np.int16)
        resources.append((atlas, labels))
        brain = atlas > 0 if brain is None else brain | (atlas > 0)
    if reference is None or brain is None or sum(len(x[1]) for x in resources) != 110:
        raise ValueError("Expected exactly 110 analysis parcels")
    template_path = next(
        (
            path
            for path in MNI_TEMPLATE_CANDIDATES
            if path.is_file() and path.stat().st_size > 0
        ),
        None,
    )
    if template_path is None:
        raise FileNotFoundError(
            "Missing anatomical template; checked "
            f"{[str(path) for path in MNI_TEMPLATE_CANDIDATES]}"
        )
    template_image = nib.as_closest_canonical(nib.load(str(template_path)))
    if template_image.shape == reference.shape and np.allclose(
        template_image.affine, reference.affine
    ):
        anatomy_image = template_image
        anatomy_resampled = False
    else:
        anatomy_image = resample_from_to(
            template_image,
            (reference.shape, reference.affine),
            order=3,
        )
        anatomy_resampled = True
    anatomy = np.asarray(anatomy_image.dataobj, dtype=np.float32)
    anatomy[~np.isfinite(anatomy) | (anatomy < 0)] = 0
    anatomy_brain = (anatomy > 0) | np.asarray(brain, dtype=bool)
    return {
        "atlases": resources,
        "reference": reference,
        "brain": brain,
        "anatomy": anatomy,
        "anatomy_brain": anatomy_brain,
        "anatomy_source": str(template_path),
        "anatomy_resampled": bool(anatomy_resampled),
        "anatomy_source_resolution_mm": tuple(
            float(value) for value in template_image.header.get_zooms()[:3]
        ),
    }


def _volume_from_effects(
    effects: pd.DataFrame,
    *,
    value_column: str,
    resources: Mapping[str, Any],
) -> np.ndarray:
    shape = resources["reference"].shape
    volume = np.full(shape, np.nan, dtype=np.float32)
    matched: set[str] = set()
    midpoint = shape[0] // 2
    for atlas, labels in resources["atlases"]:
        for row in effects.itertuples(index=False):
            roi = str(row.roi)
            atlas_roi = ROI_ATLAS_ALIASES.get(roi, roi)
            if atlas_roi not in labels:
                continue
            parcel = atlas == int(labels[atlas_roi])
            if "_left_" in roi:
                parcel[midpoint:] = False
            elif "_right_" in roi:
                parcel[:midpoint] = False
            volume[parcel] = float(getattr(row, value_column))
            matched.add(roi)
    expected = set(effects["roi"].astype(str))
    if matched != expected:
        raise ValueError(f"Unmatched effect parcels: {sorted(expected - matched)}")
    return volume


def _anatomical_projection(
    anatomy: np.ndarray, brain: np.ndarray, axis: int
) -> np.ndarray:
    import warnings

    values = np.where(brain & (anatomy > 0), anatomy, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        projection = np.nanpercentile(values, 70, axis=axis)
    finite = projection[np.isfinite(projection)]
    if finite.size:
        lo, hi = np.percentile(finite, [3, 99])
        projection = np.clip((projection - lo) / max(hi - lo, 1e-8), 0, 1)
    return np.asarray(projection, dtype=np.float32)


def _effect_projection(volume: np.ndarray, axis: int) -> np.ndarray:
    valid = np.isfinite(volume)
    magnitude = np.where(valid, np.abs(volume), -np.inf)
    index = np.argmax(magnitude, axis=axis)
    expanded = np.expand_dims(index, axis=axis)
    projection = np.take_along_axis(volume, expanded, axis=axis).squeeze(axis)
    projection[~np.any(valid, axis=axis)] = np.nan
    return projection


def draw_effect_projection(
    ax: mpl.axes.Axes,
    effects: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    resources: Mapping[str, Any],
    norm: Normalize,
    cmap: mpl.colors.Colormap,
) -> mpl.image.AxesImage:
    """Bilateral sagittal maximum projection on the 1-mm MNI underlay."""
    volume = _volume_from_effects(effects, value_column=value_column, resources=resources)
    anatomy = resources["anatomy"]
    brain = resources["brain"]
    midpoint = volume.shape[0] // 2
    bg_l = _anatomical_projection(anatomy[:midpoint], brain[:midpoint], 0).T
    bg_r = _anatomical_projection(anatomy[midpoint:], brain[midpoint:], 0).T
    fx_l = _effect_projection(volume[:midpoint], 0).T
    fx_r = _effect_projection(volume[midpoint:], 0).T
    gap = np.full((bg_l.shape[0], 8), np.nan, dtype=np.float32)
    background = np.concatenate([np.fliplr(bg_l), gap, bg_r], axis=1)
    effect = np.concatenate([np.fliplr(fx_l), gap, fx_r], axis=1)
    support_l = np.any(brain[:midpoint], axis=0).T
    support_r = np.any(brain[midpoint:], axis=0).T
    support = np.concatenate(
        [np.fliplr(support_l), np.zeros((support_l.shape[0], 8), bool), support_r],
        axis=1,
    )
    ax.imshow(
        np.ma.masked_where(~support, background),
        origin="lower",
        cmap="Greys",
        vmin=0,
        vmax=1,
        interpolation="bicubic",
        alpha=0.72,
        rasterized=True,
    )
    image = ax.imshow(
        np.ma.masked_invalid(effect),
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        alpha=0.96,
        rasterized=True,
    )
    ax.contour(support.astype(float), levels=[0.5], colors="#62696D", linewidths=0.28)
    ax.set_title(title, fontsize=7.0, fontweight="normal", color="black", pad=2)
    ax.text(0.24, 0.02, "L", transform=ax.transAxes, ha="center", fontsize=5.5, color=GRAY)
    ax.text(0.76, 0.02, "R", transform=ax.transAxes, ha="center", fontsize=5.5, color=GRAY)
    ax.set_axis_off()
    return image


def save_effect_brain_maps(
    path: Path,
    *,
    panels: Sequence[tuple[str, pd.DataFrame]],
    value_column: str,
    title: str,
    subtitle: str,
    metric_label: str,
    signed: bool,
    n_columns: int = 4,
) -> None:
    resources = load_mni_resources()
    values = np.concatenate(
        [
            panel[value_column].to_numpy(float)
            for _, panel in panels
            if not panel.empty
        ]
    ) if any(not panel.empty for _, panel in panels) else np.asarray([0.0])
    maximum = max(float(np.max(np.abs(values))), 1e-6)
    if signed:
        norm: Normalize = TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)
        cmap = SIGNED_BLUE_RED_YELLOW
    else:
        norm = Normalize(vmin=0.0, vmax=maximum)
        cmap = CROSS_RED_YELLOW
    n_columns = max(1, min(int(n_columns), len(panels)))
    n_rows = int(np.ceil(len(panels) / n_columns))
    fig = plt.figure(figsize=(6.0, 1.15 + 1.48 * n_rows), layout="constrained")
    outer = fig.add_gridspec(2, 1, height_ratios=[0.22, 1], hspace=0.02)
    header = fig.add_subplot(outer[0]); header.axis("off")
    header.text(0.5, 0.66, title, ha="center", fontsize=8.0, color="black")
    header.text(0.5, 0.08, subtitle, ha="center", fontsize=6.2, color=GRAY)
    grid = outer[1].subgridspec(n_rows, n_columns + 1, width_ratios=[1] * n_columns + [0.06], wspace=0.03, hspace=0.08)
    images = []
    for index, (panel_title, effects) in enumerate(panels):
        row, column = divmod(index, n_columns)
        ax = fig.add_subplot(grid[row, column])
        images.append(
            draw_effect_projection(
                ax,
                effects,
                value_column=value_column,
                title=panel_title,
                resources=resources,
                norm=norm,
                cmap=cmap,
            )
        )
        if effects.empty:
            ax.text(0.5, 0.18, "No corrected parcel", transform=ax.transAxes, ha="center", fontsize=5.5, color=GRAY)
    for index in range(len(panels), n_rows * n_columns):
        row, column = divmod(index, n_columns)
        ax = fig.add_subplot(grid[row, column]); ax.axis("off")
    cax = fig.add_subplot(grid[:, -1])
    colorbar = fig.colorbar(images[-1] if images else mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    colorbar.set_label(metric_label, fontsize=5.8, labelpad=2)
    colorbar.ax.tick_params(labelsize=5.3, length=1.5, pad=1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=300,
        facecolor="white",
        metadata=PDF_METADATA,
    )
    plt.close(fig)


def _oriented_slice(values: np.ndarray, axis: int, index: int) -> np.ndarray:
    # ``load_mni_resources`` returns a canonical RAS volume.  With
    # ``imshow(origin='lower')``, transposition places superior/anterior at the
    # top and keeps left on the left.  np.rot90 would reverse the vertical
    # coordinate in all three planes.
    if axis == 0:
        return np.asarray(values[index, :, :]).T
    if axis == 1:
        return np.asarray(values[:, index, :]).T
    return np.asarray(values[:, :, index]).T


def _largest_roi_slice(
    mask: np.ndarray,
    *,
    axis: int,
    reference: nib.spatialimages.SpatialImage,
) -> tuple[int, float]:
    """Return the most informative slice and its MNI coordinate.

    For a bilateral sagittal display, the negative-x component is used so the
    selected section does not collapse to the interhemispheric midpoint.  The
    horizontal and coronal sections include both hemispheres.
    """
    support = np.asarray(mask, dtype=bool)
    coordinates = np.argwhere(support)
    if coordinates.size == 0:
        raise ValueError("Cannot select a slice from an empty ROI mask")
    candidates = coordinates
    if axis == 0:
        world = nib.affines.apply_affine(reference.affine, coordinates)
        left = world[:, 0] <= 0
        if np.any(left):
            candidates = coordinates[left]
    counts = np.bincount(candidates[:, axis], minlength=mask.shape[axis])
    maxima = np.flatnonzero(counts == counts.max())
    center = float(np.mean(candidates[:, axis]))
    index = int(maxima[np.argmin(np.abs(maxima.astype(float) - center))])
    voxel = (np.asarray(reference.shape, dtype=float) - 1.0) / 2.0
    voxel[axis] = float(index)
    coordinate = float(
        nib.affines.apply_affine(reference.affine, voxel)[axis]
    )
    return index, coordinate


def _crop_anatomical_slice(
    background: np.ndarray,
    support: np.ndarray,
    overlay: np.ndarray,
    *,
    padding: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove empty margins while retaining the full brain cross-section."""
    coordinates = np.argwhere(support)
    if coordinates.size == 0:
        return background, support, overlay
    lower = np.maximum(coordinates.min(axis=0) - int(padding), 0)
    upper = np.minimum(
        coordinates.max(axis=0) + int(padding) + 1,
        np.asarray(support.shape),
    )
    rows = slice(int(lower[0]), int(upper[0]))
    columns = slice(int(lower[1]), int(upper[1]))
    return background[rows, columns], support[rows, columns], overlay[rows, columns]


def _format_mni_coordinate(axis_name: str, coordinate: float) -> str:
    value = int(np.rint(coordinate))
    return f"{axis_name}={value:+d} mm"


def save_predefined_roi_anatomy(path: Path) -> None:
    """Sung-Figure-5-style orthogonal anatomy for the eight fixed ROI groups."""
    resources = load_mni_resources()
    mapping = predefined_roi_mapping_frame()
    anatomy = resources["anatomy"]
    brain = resources["anatomy_brain"]
    reference = resources["reference"]
    if resources["anatomy_resampled"] or not np.allclose(
        resources["anatomy_source_resolution_mm"], (1.0, 1.0, 1.0)
    ):
        raise ValueError(
            "Predefined ROI anatomy requires a native 1-mm MNI template; "
            f"got {resources['anatomy_source']} at "
            f"{resources['anatomy_source_resolution_mm']} mm"
        )
    if set(ROI_ATLAS_COMPONENT_LABELS) != set(ROI_ORDER):
        raise ValueError("Every predefined ROI needs an atlas component label")
    if set(ROI_ANATOMY_COLORS) != set(ROI_ORDER):
        raise ValueError("Every predefined ROI needs a unique display color")
    if len(set(ROI_ANATOMY_COLORS.values())) != len(ROI_ORDER):
        raise ValueError("Predefined ROI anatomy colors must be unique")

    # Four columns by two rows, while retaining large panels on a landscape
    # supplement page rather than squeezing them into a 6-inch portrait page.
    figure = plt.figure(figsize=(13.0, 5.9))
    outer = figure.add_gridspec(
        2,
        4,
        left=0.012,
        right=0.992,
        bottom=0.025,
        top=0.985,
        hspace=0.13,
        wspace=0.035,
    )
    for panel_index, roi_key in enumerate(ROI_ORDER):
        definition = mapping[mapping["roi_key"].eq(roi_key)]
        mask_frame = definition.rename(columns={"component_roi": "roi"}).copy()
        mask_frame["value"] = 1.0
        mask_volume = _volume_from_effects(
            mask_frame, value_column="value", resources=resources
        )
        mask = np.isfinite(mask_volume)
        if not np.any(mask):
            raise ValueError(f"Empty predefined ROI mask: {roi_key}")
        subgrid = outer[panel_index // 4, panel_index % 4].subgridspec(
            3,
            3,
            height_ratios=[0.38, 0.22, 1.0],
            hspace=0.0,
            wspace=0.025,
        )
        title_axis = figure.add_subplot(subgrid[0, :])
        title_axis.axis("off")
        title_axis.text(
            0.5,
            0.78,
            ROI_LABELS[roi_key],
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="black",
        )
        component_text = ", ".join(ROI_ATLAS_COMPONENT_LABELS[roi_key])
        if len(component_text) > 48:
            split_at = component_text.rfind(" ", 0, 48)
            if split_at > 0:
                component_text = (
                    component_text[:split_at]
                    + "\n"
                    + component_text[split_at + 1 :]
                )
        title_axis.text(
            0.5,
            0.22,
            f"({component_text})",
            ha="center",
            va="center",
            fontsize=7.4,
            color=GRAY,
            linespacing=0.92,
        )
        view_specs = (
            ("Sagittal", 0, "x"),
            ("Horizontal", 2, "z"),
            ("Coronal", 1, "y"),
        )
        color = ROI_ANATOMY_COLORS[roi_key]
        for view_index, (view_label, axis, axis_name) in enumerate(view_specs):
            index, coordinate = _largest_roi_slice(
                mask, axis=axis, reference=reference
            )
            view_title_axis = figure.add_subplot(subgrid[1, view_index])
            view_title_axis.axis("off")
            view_title_axis.text(
                0.5,
                0.48,
                f"{view_label}\n{_format_mni_coordinate(axis_name, coordinate)}",
                ha="center",
                va="center",
                fontsize=7.4,
                fontweight="normal",
                color="black",
                linespacing=0.95,
            )
            ax = figure.add_subplot(subgrid[2, view_index])
            background = _oriented_slice(anatomy, axis, index)
            support = _oriented_slice(brain.astype(float), axis, index) > 0
            overlay = _oriented_slice(mask.astype(float), axis, index)
            overlay[overlay < 0.5] = np.nan
            background, support, overlay = _crop_anatomical_slice(
                background, support, overlay
            )
            finite_bg = background[support & np.isfinite(background)]
            high = float(np.percentile(finite_bg, 99.5)) if finite_bg.size else 1.0
            ax.imshow(
                np.ma.masked_where(~support, background),
                cmap="Greys",
                vmin=0,
                vmax=max(high, 1e-6),
                interpolation="bilinear",
                origin="lower",
                rasterized=True,
            )
            ax.imshow(
                np.ma.masked_invalid(overlay),
                cmap=mpl.colors.ListedColormap([color]),
                vmin=0,
                vmax=1,
                interpolation="nearest",
                alpha=0.90,
                origin="lower",
                rasterized=True,
            )
            ax.contour(
                np.isfinite(overlay).astype(float),
                levels=[0.5],
                colors=[color],
                linewidths=0.45,
            )
            if axis in (1, 2):
                ax.text(
                    0.02,
                    0.02,
                    "L",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=6.2,
                    color=GRAY,
                )
                ax.text(
                    0.98,
                    0.02,
                    "R",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=6.2,
                    color=GRAY,
                )
            ax.set_axis_off()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=300,
        facecolor="white",
        metadata=PDF_METADATA,
    )
    plt.close(figure)


def configure_publication_fonts() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": "#55595D",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": GRAY,
            "ytick.color": GRAY,
        }
    )


configure_publication_fonts()
