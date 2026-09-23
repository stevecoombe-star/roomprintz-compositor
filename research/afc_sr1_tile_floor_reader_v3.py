"""Tracked AFC-SR1 V3 projective, single-image floor-line reader.

V3 deliberately reuses the frozen V2 raster admission policy while replacing
only its two-family selector with the certified finite/directional model.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import cv2
import numpy as np

from research.afc_sr1_tile_floor_reader import (
    HOMOGENEOUS_EPSILON,
    MAX_SOURCE_LONG_EDGE_V2,
    MAX_SOURCE_PIXELS_V2,
    POLICY,
    Segment,
    _analysis_dimensions,
    _analysis_identity,
    _line_through,
    _map_analysis_line_to_input,
    _normalize_line,
    _normalize_point,
    _pair_from_rank,
    _valid_roi_polygon,
)

READER_MODULE_VERSION = "afc-sr1-tile-floor-reader/v4"
POLICY_VERSION = "afc-sr1-ts2-extractor-policy/v4"
FAMILY_PAIR_INDEPENDENCE_DIAGNOSTICS_VERSION = "afc-sr1-family-pair-independence-diagnostics/v1"
RHO_INFINITY = 8.0
TOP_K = 8
CHORDAL_DELTA_MIN = 0.15
DIRECTIONAL_SEPARATION_DEGREES = 15.0
INDEPENDENT_DIRECTION_FIELD_MIN_MEDIAN_DEGREES = 10.0
STABILITY_MAX_PX = 18.0
Model = Literal["finite", "directional"]


@dataclass(frozen=True)
class Family:
    model: Model
    vp: np.ndarray
    supporters: tuple[Segment, ...]
    residuals: np.ndarray
    rho: float
    source_hypothesis: np.ndarray

    @property
    def total_length(self) -> float:
        return float(sum(item.length for item in self.supporters))

    @property
    def capped_length(self) -> float:
        return float(sum(min(item.length, 100.0) for item in self.supporters))


def _vector(value: np.ndarray) -> list[float]:
    return [float(item) for item in value]


def _rho(vp: np.ndarray, diagonal: float) -> float:
    point = _normalize_point(vp)
    if point is None:
        return float("inf")
    return float(np.linalg.norm(point[:2]) / (max(abs(float(point[2])), HOMOGENEOUS_EPSILON) * diagonal))


def classify_vp(vp: np.ndarray, diagonal: float, *, exact_directional: bool = False) -> tuple[Model | None, np.ndarray | None, float]:
    """Classifies after homogeneous normalization; tiny z is not finite authority."""
    point = _normalize_point(vp)
    if point is None:
        return None, None, float("inf")
    rho = _rho(point, diagonal)
    if exact_directional or point[2] == 0.0 or rho > RHO_INFINITY:
        direction = point[:2]
        magnitude = float(np.linalg.norm(direction))
        if magnitude <= HOMOGENEOUS_EPSILON:
            return None, None, rho
        return "directional", _normalize_point(np.asarray([direction[0] / magnitude, direction[1] / magnitude, 0.0])), rho
    return "finite", point, rho


def _direction_fit(segments: Sequence[Segment]) -> np.ndarray | None:
    """Length-weighted RP1 double-angle mean with exact zero projective z."""
    cosine = sine = 0.0
    for item in segments:
        angle = math.atan2(float(item.p2[1] - item.p1[1]), float(item.p2[0] - item.p1[0]))
        cosine += item.length * math.cos(2.0 * angle)
        sine += item.length * math.sin(2.0 * angle)
    if math.hypot(cosine, sine) <= HOMOGENEOUS_EPSILON:
        return None
    angle = 0.5 * math.atan2(sine, cosine)
    return _normalize_point(np.asarray([math.cos(angle), math.sin(angle), 0.0], dtype=np.float64))


def _finite_fit(segments: Sequence[Segment]) -> np.ndarray | None:
    if len(segments) < 2:
        return None
    matrix = np.asarray([item.line * math.sqrt(item.length) for item in segments], dtype=np.float64)
    _, _, vectors = np.linalg.svd(matrix, full_matrices=False)
    return _normalize_point(vectors[-1])


def _residuals(segments: Sequence[Segment], model: Model, vp: np.ndarray, diagonal: float) -> np.ndarray:
    if model == "finite":
        if abs(float(vp[2])) <= HOMOGENEOUS_EPSILON:
            return np.full(len(segments), np.inf)
        point = vp / vp[2]
        return np.abs(np.asarray([item.line @ point for item in segments], dtype=np.float64))
    direction = vp[:2] / np.linalg.norm(vp[:2])
    return np.abs(np.asarray([item.line[:2] @ direction for item in segments], dtype=np.float64)) * diagonal


def _family_diag(family: Family) -> dict[str, Any]:
    return {
        "vpClass": family.model,
        "rho": family.rho,
        "normalizedHomogeneousVp": _vector(family.vp),
        "direction": _vector(family.vp[:2]) if family.model == "directional" else None,
        "supportCount": len(family.supporters),
        "supportTotalLengthPx": family.total_length,
        "cappedSupportLengthPx": family.capped_length,
        "medianResidualPx": float(np.median(family.residuals)),
        "p90ResidualPx": float(np.percentile(family.residuals, 90)),
        "refinement": "two_round_reselect_refit",
    }


def _family_support_geometry(families: Sequence[Family]) -> dict[str, Any]:
    """Projects final-family supporter endpoints without changing V3 evidence."""
    segments: dict[int, Segment] = {}
    memberships: list[dict[str, Any]] = []
    for family_index, family in enumerate(families):
        supporter_indices = [item.detector_index for item in family.supporters]
        assert len(supporter_indices) == len(family.supporters)
        memberships.append({
            "familyIndex": family_index,
            "supporterDetectorIndices": supporter_indices,
        })
        for supporter in family.supporters:
            existing = segments.setdefault(supporter.detector_index, supporter)
            assert existing is supporter or (
                np.array_equal(existing.p1, supporter.p1) and
                np.array_equal(existing.p2, supporter.p2)
            )
    return {
        "coordinateSpace": "analysis-pixel/v1",
        "authority": "none",
        "role": "observation_only",
        "excludedFromCanonicalEvidence": True,
        "segments": [
            {
                "detectorIndex": detector_index,
                "x1": float(segment.p1[0]),
                "y1": float(segment.p1[1]),
                "x2": float(segment.p2[0]),
                "y2": float(segment.p2[1]),
            }
            for detector_index, segment in sorted(segments.items())
        ],
        "families": memberships,
    }


def _axial_angle(segment: Segment) -> float:
    """Returns the segment orientation in RP1, in the half-open range [0, pi)."""
    return math.atan2(
        float(segment.p2[1] - segment.p1[1]),
        float(segment.p2[0] - segment.p1[0]),
    ) % math.pi


def _axial_summary(segments: Sequence[Segment]) -> dict[str, Any]:
    """Unweighted double-angle descriptive statistics for observed supporters."""
    angles = np.asarray([_axial_angle(segment) for segment in segments], dtype=np.float64)
    if not len(angles):
        return {
            "supporterCount": 0,
            "axialMeanDegrees": None,
            "axialMedianDegrees": None,
            "axialCircularStdDevDegrees": None,
            "axialIqrDegrees": None,
        }
    cosine = float(np.cos(2.0 * angles).sum())
    sine = float(np.sin(2.0 * angles).sum())
    resultant = math.hypot(cosine, sine) / len(angles)
    if not math.isfinite(resultant) or resultant <= HOMOGENEOUS_EPSILON:
        return {
            "supporterCount": len(angles),
            "axialMeanDegrees": None,
            "axialMedianDegrees": None,
            "axialCircularStdDevDegrees": None,
            "axialIqrDegrees": None,
        }
    resultant = min(1.0, resultant)
    mean = (0.5 * math.atan2(sine, cosine)) % math.pi
    # Express each RP1 orientation around its double-angle mean before applying
    # ordinary robust summaries, so the 0/180 degree boundary is not a split.
    centered = ((angles - mean + math.pi / 2.0) % math.pi) - math.pi / 2.0
    median = (mean + float(np.median(centered))) % math.pi
    q25, q75 = np.percentile(centered, (25, 75))
    return {
        "supporterCount": len(angles),
        "axialMeanDegrees": math.degrees(mean),
        "axialMedianDegrees": math.degrees(median),
        "axialCircularStdDevDegrees": math.degrees(math.sqrt(-0.5 * math.log(resultant))),
        "axialIqrDegrees": math.degrees(float(q75 - q25)),
    }


def _residual_summary(segments: Sequence[Segment], model: Model, vp: np.ndarray, diagonal: float) -> dict[str, Any] | None:
    """Describes residuals using the exact V3 residual implementation."""
    if not np.all(np.isfinite(vp)):
        return None
    if (model == "finite" and abs(float(vp[2])) <= HOMOGENEOUS_EPSILON) or (
        model == "directional" and np.linalg.norm(vp[:2]) <= HOMOGENEOUS_EPSILON
    ):
        return None
    try:
        values = _residuals(segments, model, vp, diagonal)
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return None
    if not np.all(np.isfinite(values)):
        return None
    return {
        "supporterCount": len(values),
        "medianResidualPx": float(np.median(values)) if len(values) else None,
        "p90ResidualPx": float(np.percentile(values, 90)) if len(values) else None,
        "withinExistingInlierBandCount": int(
            np.count_nonzero(values <= POLICY["consensus"]["inlier_residual_px"])
        ),
    }


def _predicted_direction(segment: Segment, family: Family) -> float | None:
    """Predicts the local undirected line orientation from an existing V3 model."""
    try:
        if family.model == "finite":
            if abs(float(family.vp[2])) <= HOMOGENEOUS_EPSILON:
                return None
            point = family.vp / family.vp[2]
            dx = float(point[0] - segment.midpoint[0])
            dy = float(point[1] - segment.midpoint[1])
        else:
            dx, dy = float(family.vp[0]), float(family.vp[1])
        if not math.isfinite(dx) or not math.isfinite(dy) or math.hypot(dx, dy) <= HOMOGENEOUS_EPSILON:
            return None
        return math.atan2(dy, dx) % math.pi
    except (FloatingPointError, ValueError, ZeroDivisionError):
        return None


def _axial_disagreement_degrees(first: float, second: float) -> float:
    """Returns RP1 angular distance in the inclusive interval [0, 90] degrees."""
    delta = abs((first - second) % math.pi)
    return math.degrees(min(delta, math.pi - delta))


def _direction_field_disagreement(
    segments: Sequence[Segment], first: Family, second: Family
) -> dict[str, Any] | None:
    values: list[float] = []
    for segment in segments:
        first_direction = _predicted_direction(segment, first)
        second_direction = _predicted_direction(segment, second)
        if first_direction is None or second_direction is None:
            return None
        values.append(_axial_disagreement_degrees(first_direction, second_direction))
    if not all(math.isfinite(value) and 0.0 <= value <= 90.0 for value in values):
        return None
    return {
        "supporterCount": len(values),
        "medianDegrees": float(np.median(values)) if values else None,
        "p90Degrees": float(np.percentile(values, 90)) if values else None,
    }


def _cross_fit_inlier_band_count(
    supporters: Sequence[Segment], model: Model, vp: np.ndarray, diagonal: float
) -> int:
    """Counts cross-fit supporters in the existing V3 consensus inlier band."""
    values = _residuals(supporters, model, vp, diagonal)
    if not np.all(np.isfinite(values)):
        raise ValueError("cross-fit residuals must be finite for a valid pair")
    return int(np.count_nonzero(values <= POLICY["consensus"]["inlier_residual_px"]))


def _pair_independent_direction_measurements(
    first: Family, second: Family, diagonal: float
) -> dict[str, Any]:
    """Computes only the measurements consumed by the eligibility gate."""
    first_ids = {segment.detector_index for segment in first.supporters}
    second_ids = {segment.detector_index for segment in second.supporters}
    first_support_count = len(first.supporters)
    second_support_count = len(second.supporters)
    overlap_fraction_of_smaller = (
        len(first_ids & second_ids) / min(len(first_ids), len(second_ids))
        if first_ids and second_ids else None
    )
    first_inlier_band_count = _cross_fit_inlier_band_count(
        first.supporters, second.model, second.vp, diagonal
    )
    second_inlier_band_count = _cross_fit_inlier_band_count(
        second.supporters, first.model, first.vp, diagonal
    )
    first_field = _direction_field_disagreement(first.supporters, first, second)
    second_field = _direction_field_disagreement(second.supporters, first, second)
    first_region_median_degrees = (
        first_field["medianDegrees"] if first_field is not None else None
    )
    second_region_median_degrees = (
        second_field["medianDegrees"] if second_field is not None else None
    )
    strong_region_median_degrees = (
        max(first_region_median_degrees, second_region_median_degrees)
        if isinstance(first_region_median_degrees, float)
        and isinstance(second_region_median_degrees, float)
        else None
    )
    return {
        "overlapFractionOfSmaller": overlap_fraction_of_smaller,
        "firstSupportCount": first_support_count,
        "secondSupportCount": second_support_count,
        "firstInlierBandCount": first_inlier_band_count,
        "secondInlierBandCount": second_inlier_band_count,
        "firstInlierBandFraction": first_inlier_band_count / first_support_count,
        "secondInlierBandFraction": second_inlier_band_count / second_support_count,
        "firstRegionMedianDegrees": first_region_median_degrees,
        "secondRegionMedianDegrees": second_region_median_degrees,
        "strongRegionMedianDegrees": strong_region_median_degrees,
    }


def evaluate_independent_direction_eligibility(
    *,
    overlap_fraction_of_smaller: float | None,
    first_support_count: int,
    second_support_count: int,
    first_inlier_band_count: int,
    second_inlier_band_count: int,
    first_region_median_degrees: float | None,
    second_region_median_degrees: float | None,
) -> dict[str, Any]:
    """Applies the frozen Boolean independent-direction eligibility contract."""
    if first_support_count <= 0 or second_support_count <= 0:
        raise ValueError("eligibility requires non-empty family support")
    first_inlier_band_fraction = first_inlier_band_count / first_support_count
    second_inlier_band_fraction = second_inlier_band_count / second_support_count
    strong_region_median_degrees = (
        max(first_region_median_degrees, second_region_median_degrees)
        if isinstance(first_region_median_degrees, float)
        and isinstance(second_region_median_degrees, float)
        else None
    )
    stage_one_rejects = (
        overlap_fraction_of_smaller is not None
        and overlap_fraction_of_smaller >= 0.5
        and first_inlier_band_count * 2 >= first_support_count
        and second_inlier_band_count * 2 >= second_support_count
    )
    if stage_one_rejects:
        eligible = False
        failed_stage: int | None = 1
        rejection_reason: str | None = "duplicate_or_interchangeable_families"
    elif (
        isinstance(strong_region_median_degrees, float)
        and math.isfinite(strong_region_median_degrees)
        and strong_region_median_degrees >= INDEPENDENT_DIRECTION_FIELD_MIN_MEDIAN_DEGREES
    ):
        eligible = True
        failed_stage = None
        rejection_reason = None
    else:
        eligible = False
        failed_stage = 2
        rejection_reason = "insufficient_direction_field_separation"
    return {
        "eligible": eligible,
        "failedStage": failed_stage,
        "rejectionReason": rejection_reason,
        "overlapFractionOfSmaller": overlap_fraction_of_smaller,
        "firstSupportCount": first_support_count,
        "secondSupportCount": second_support_count,
        "firstInlierBandCount": first_inlier_band_count,
        "secondInlierBandCount": second_inlier_band_count,
        "firstInlierBandFraction": first_inlier_band_fraction,
        "secondInlierBandFraction": second_inlier_band_fraction,
        "firstRegionMedianDegrees": first_region_median_degrees,
        "secondRegionMedianDegrees": second_region_median_degrees,
        "strongRegionMedianDegrees": strong_region_median_degrees,
    }


def _evaluate_pair_independent_direction_eligibility(
    first: Family, second: Family, diagonal: float
) -> dict[str, Any]:
    """Projects science-local pair measurements into canonical eligibility evidence."""
    measurements = _pair_independent_direction_measurements(first, second, diagonal)
    return evaluate_independent_direction_eligibility(
        overlap_fraction_of_smaller=measurements["overlapFractionOfSmaller"],
        first_support_count=measurements["firstSupportCount"],
        second_support_count=measurements["secondSupportCount"],
        first_inlier_band_count=measurements["firstInlierBandCount"],
        second_inlier_band_count=measurements["secondInlierBandCount"],
        first_region_median_degrees=measurements["firstRegionMedianDegrees"],
        second_region_median_degrees=measurements["secondRegionMedianDegrees"],
    )


def _family_pair_independence_diagnostics(
    families: Sequence[Family], diagonal: float
) -> dict[str, Any]:
    """Projects final-family observations only; no selection path reads this."""
    orientations = [
        {"familyIndex": family_index, **_axial_summary(family.supporters)}
        for family_index, family in enumerate(families)
    ]
    pairs: list[dict[str, Any]] = []
    for first_index, first in enumerate(families):
        first_by_detector = {segment.detector_index: segment for segment in first.supporters}
        for second_index, second in enumerate(families[first_index + 1:], first_index + 1):
            second_by_detector = {segment.detector_index: segment for segment in second.supporters}
            first_ids = set(first_by_detector)
            second_ids = set(second_by_detector)
            shared_ids = first_ids & second_ids
            union_ids = first_ids | second_ids
            first_only_ids = first_ids - second_ids
            second_only_ids = second_ids - first_ids
            # Retain first-family order, then append B-only evidence, for a
            # deterministic union midpoint set without serializing endpoints.
            union_segments = [
                segment for segment in first.supporters
                if segment.detector_index in union_ids
            ] + [
                segment for segment in second.supporters
                if segment.detector_index in second_only_ids
            ]
            shared_segments = [
                segment for segment in first.supporters
                if segment.detector_index in shared_ids
            ]
            pairs.append({
                "familyIndices": [first_index, second_index],
                "overlap": {
                    "sharedSupporterCount": len(shared_ids),
                    "unionSupporterCount": len(union_ids),
                    "jaccard": len(shared_ids) / len(union_ids) if union_ids else None,
                    "overlapFractionOfSmaller": (
                        len(shared_ids) / min(len(first_ids), len(second_ids))
                        if first_ids and second_ids else None
                    ),
                    "familyASupporterCount": len(first.supporters),
                    "familyBSupporterCount": len(second.supporters),
                },
                "exclusiveSupport": {
                    "sharedSupportLengthPx": float(
                        sum(first_by_detector[index].length for index in shared_ids)
                    ),
                    "firstOnlySupporterCount": len(first_only_ids),
                    "secondOnlySupporterCount": len(second_only_ids),
                    "firstOnlySupportLengthPx": float(
                        sum(first_by_detector[index].length for index in first_only_ids)
                    ),
                    "secondOnlySupportLengthPx": float(
                        sum(second_by_detector[index].length for index in second_only_ids)
                    ),
                },
                "crossFit": {
                    "firstSupportersAgainstSecond": _residual_summary(
                        first.supporters, second.model, second.vp, diagonal
                    ),
                    "secondSupportersAgainstFirst": _residual_summary(
                        second.supporters, first.model, first.vp, diagonal
                    ),
                },
                "predictedDirectionFieldDisagreement": {
                    "onFirstSupporterMidpoints": _direction_field_disagreement(
                        first.supporters, first, second
                    ),
                    "onSecondSupporterMidpoints": _direction_field_disagreement(
                        second.supporters, first, second
                    ),
                    "onUnionSupporterMidpoints": _direction_field_disagreement(
                        union_segments, first, second
                    ),
                    "onSharedSupporterMidpoints": _direction_field_disagreement(
                        shared_segments, first, second
                    ),
                },
            })
    return {
        "contractVersion": FAMILY_PAIR_INDEPENDENCE_DIAGNOSTICS_VERSION,
        "coordinateSpace": "analysis-pixel/v1",
        "authority": "none",
        "role": "observation_only",
        "excludedFromCanonicalEvidence": True,
        "familyOrientationSummaries": orientations,
        "pairs": pairs,
    }


def _family_passes(family: Family) -> bool:
    return (
        len(family.supporters) >= POLICY["consensus"]["min_support_count"]
        and family.total_length >= POLICY["consensus"]["min_support_total_length_px"]
        and float(np.median(family.residuals)) <= POLICY["consensus"]["max_median_residual_px"]
        and float(np.percentile(family.residuals, 90)) <= POLICY["consensus"]["max_p90_residual_px"]
    )


def _refine(candidates: Sequence[Segment], seeds: Sequence[Segment], model: Model, diagonal: float, source: np.ndarray) -> Family | None:
    supporters = list(seeds)
    if len(supporters) < POLICY["consensus"]["min_support_count"]:
        return None
    for _ in range(2):
        vp = _finite_fit(supporters) if model == "finite" else _direction_fit(supporters)
        if vp is None:
            return None
        actual, normalized, rho = classify_vp(vp, diagonal, exact_directional=model == "directional")
        if actual != model or normalized is None:
            return None
        values = _residuals(candidates, model, normalized, diagonal)
        supporters = [item for index, item in enumerate(candidates) if values[index] <= POLICY["consensus"]["inlier_residual_px"]]
        if len(supporters) < POLICY["consensus"]["min_support_count"]:
            return None
    vp = _finite_fit(supporters) if model == "finite" else _direction_fit(supporters)
    if vp is None:
        return None
    actual, normalized, rho = classify_vp(vp, diagonal, exact_directional=model == "directional")
    if actual != model or normalized is None:
        return None
    return Family(model, normalized, tuple(supporters), _residuals(supporters, model, normalized, diagonal), rho, source)


def _chordal(left: np.ndarray, right: np.ndarray) -> float:
    return float(min(np.linalg.norm(left - right), np.linalg.norm(left + right)))


def _distinct(left: Family | np.ndarray, right: Family | np.ndarray, diagonal: float) -> tuple[bool, dict[str, float]]:
    left_vp, right_vp = (left.vp if isinstance(left, Family) else left), (right.vp if isinstance(right, Family) else right)
    lc, lp, _ = classify_vp(left_vp, diagonal, exact_directional=left_vp[2] == 0.0)
    rc, rp, _ = classify_vp(right_vp, diagonal, exact_directional=right_vp[2] == 0.0)
    if lp is None or rp is None:
        return False, {"chordal": 0.0, "directionAngleDegrees": 0.0}
    angle = 180.0
    if lc == rc == "directional":
        angle = math.degrees(math.acos(min(1.0, max(-1.0, abs(float(np.dot(lp[:2], rp[:2])))))))
    chordal = _chordal(lp, rp)
    return chordal >= CHORDAL_DELTA_MIN and angle >= DIRECTIONAL_SEPARATION_DEGREES, {"chordal": chordal, "directionAngleDegrees": angle}


def _hypothesis_beam(candidates: Sequence[Segment], diagonal: float) -> tuple[list[tuple[np.ndarray, tuple[Segment, ...], Model]], dict[str, Any]]:
    total = math.comb(len(candidates), 2)
    cap = POLICY["consensus"]["ransac_pair_hypotheses_max"]
    if total <= cap:
        pairs = [(first, second) for first in range(len(candidates) - 1) for second in range(first + 1, len(candidates))]
        strategy = "exhaustive"
    else:
        generator = np.random.default_rng(POLICY["consensus"]["pair_sampling_seed"])
        pairs = [_pair_from_rank(len(candidates), int(rank)) for rank in generator.choice(total, size=cap, replace=False)]
        strategy = "seeded_sample"
    proposals: list[tuple[tuple[float, int, float, int], np.ndarray, tuple[Segment, ...], Model]] = []
    for ordinal, (first, second) in enumerate(pairs):
        raw = np.cross(candidates[first].line, candidates[second].line)
        model, vp, _ = classify_vp(raw, diagonal, exact_directional=raw[2] == 0.0)
        if model is None or vp is None:
            continue
        values = _residuals(candidates, model, vp, diagonal)
        indices = np.flatnonzero(values <= POLICY["consensus"]["inlier_residual_px"])
        if len(indices) < POLICY["consensus"]["min_support_count"]:
            continue
        supporters = tuple(candidates[int(index)] for index in indices)
        proposals.append(((sum(min(item.length, 100.0) for item in supporters), len(supporters), -float(np.median(values[indices])), -ordinal), vp, supporters, model))
    beam: list[tuple[np.ndarray, tuple[Segment, ...], Model]] = []
    duplicates = 0
    for _, vp, supporters, model in sorted(proposals, key=lambda item: item[0], reverse=True):
        if all(_distinct(vp, old[0], diagonal)[0] for old in beam):
            beam.append((vp, supporters, model))
            if len(beam) == TOP_K:
                break
        else:
            duplicates += 1
    return beam, {"hypothesisStrategy": strategy, "hypotheses": len(pairs), "totalPairCount": total, "beamSize": len(beam), "beamDuplicateRejections": duplicates}


def _family_sort_key(family: Family) -> tuple[Any, ...]:
    return (
        family.model,
        tuple(float(item) for item in family.vp),
        float(np.median(family.residuals)),
        float(np.percentile(family.residuals, 90)),
        -family.capped_length,
        -len(family.supporters),
    )


def _discover_families(candidates: Sequence[Segment], width: int, height: int) -> tuple[list[Family], dict[str, Any]]:
    diagonal = math.hypot(width, height)
    beam, diagnostics = _hypothesis_beam(candidates, diagonal)
    valid: list[Family] = []
    attempts: list[dict[str, Any]] = []
    for rank, (hypothesis, supporters, primary) in enumerate(beam, 1):
        for model in (primary, "directional" if primary == "finite" else "finite"):
            initial = hypothesis if model == primary else (_direction_fit(supporters) if model == "directional" else _finite_fit(supporters))
            if initial is None:
                attempts.append({"beamRank": rank, "model": model, "valid": False, "reason": "initial_refit_failed"})
                continue
            actual, initial, _ = classify_vp(initial, diagonal, exact_directional=model == "directional")
            if actual != model or initial is None:
                attempts.append({"beamRank": rank, "model": model, "valid": False, "reason": "conditioning_class_mismatch"})
                continue
            seeds = tuple(item for index, item in enumerate(candidates) if _residuals(candidates, model, initial, diagonal)[index] <= POLICY["consensus"]["inlier_residual_px"])
            family = _refine(candidates, seeds, model, diagonal, hypothesis)
            if family is None or not _family_passes(family):
                attempts.append({"beamRank": rank, "model": model, "valid": False, "reason": "family_gates_failed"})
            else:
                valid.append(family)
                attempts.append({"beamRank": rank, "model": model, "valid": True, **_family_diag(family)})
    deduped: list[Family] = []
    for family in sorted(valid, key=_family_sort_key):
        if all(_distinct(family, old, diagonal)[0] for old in deduped):
            deduped.append(family)
    diagnostics.update({"attempts": attempts, "validBeforeDedupe": len(valid), "validFamilyCount": len(deduped), "finalFamilies": [_family_diag(item) for item in deduped]})
    return deduped, diagnostics


def line_distance(left: np.ndarray, right: np.ndarray, width: int, height: int) -> float:
    left, right = _normalize_line(left), _normalize_line(right)
    if left is None or right is None:
        return float("inf")
    distance = max(abs(float(left @ probe - right @ probe)) for probe in (np.asarray([0.0, 0.0, 1.0]), np.asarray([float(width), 0.0, 1.0]), np.asarray([0.0, float(height), 1.0]), np.asarray([float(width), float(height), 1.0])))
    return 0.0 if distance <= HOMOGENEOUS_EPSILON else distance


def _stability(first: Family, second: Family, line: np.ndarray, width: int, height: int) -> dict[str, Any]:
    diagonal = math.hypot(width, height)
    split_lines: list[np.ndarray] = []
    for parity in (0, 1):
        a = sorted(first.supporters, key=lambda item: (item.midpoint[0], item.midpoint[1], item.length, item.detector_index))[parity::2]
        b = sorted(second.supporters, key=lambda item: (item.midpoint[0], item.midpoint[1], item.length, item.detector_index))[parity::2]
        av = _finite_fit(a) if first.model == "finite" else _direction_fit(a)
        bv = _finite_fit(b) if second.model == "finite" else _direction_fit(b)
        if av is None or bv is None:
            return {"stable": False, "reason": "class_preserving_split_refit_failed"}
        ac, av, _ = classify_vp(av, diagonal, exact_directional=first.model == "directional")
        bc, bv, _ = classify_vp(bv, diagonal, exact_directional=second.model == "directional")
        if ac != first.model or bc != second.model or av is None or bv is None:
            return {"stable": False, "reason": "class_preserving_split_refit_failed"}
        split = _normalize_line(np.cross(av, bv))
        if split is None:
            return {"stable": False, "reason": "split_floor_line_degenerate"}
        split_lines.append(split)
    probes = (np.asarray([0.0, 0.0, 1.0]), np.asarray([float(width), 0.0, 1.0]), np.asarray([0.0, float(height), 1.0]), np.asarray([float(width), float(height), 1.0]))
    deltas = [[abs(float(split @ probe - line @ probe)) for probe in probes] for split in split_lines]
    maximum = max(max(row) for row in deltas)
    return {"stable": maximum <= STABILITY_MAX_PX, "classPreserving": True, "splitFloorLines": [_vector(item) for item in split_lines], "splitVsFullProbeDistancesPx": deltas, "maxSplitVsFullProbeDistancePx": maximum}


def _pair_key(pair: dict[str, Any]) -> tuple[Any, ...]:
    return (-pair["basinSupport"], pair["stability"]["maxSplitVsFullProbeDistancePx"], float(np.median(pair["first"].residuals)) + float(np.median(pair["second"].residuals)), float(np.percentile(pair["first"].residuals, 90)) + float(np.percentile(pair["second"].residuals, 90)), -(pair["first"].capped_length + pair["second"].capped_length), -(len(pair["first"].supporters) + len(pair["second"].supporters)), tuple(sorted(((pair["first"].model, tuple(float(x) for x in pair["first"].vp)), (pair["second"].model, tuple(float(x) for x in pair["second"].vp))))), tuple(float(x) for x in pair["line"]))


def _pair_evidence(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "familyIndices": [pair["i"], pair["j"]],
        "families": [_family_diag(pair["first"]), _family_diag(pair["second"])],
        "floorLineAnalysis": _vector(pair["line"]),
        "basinSupport": pair["basinSupport"],
        "stability": pair["stability"],
        "distinctness": pair["distinctness"],
        "independentDirectionEligibility": pair["independentDirectionEligibility"],
    }


def _select_pair(candidates: Sequence[Segment], width: int, height: int) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    families, discovery = _discover_families(candidates, width, height)
    diagnostics: dict[str, Any] = {
        "candidateDiscovery": discovery,
        "familySupportGeometry": _family_support_geometry(families),
        "familyPairIndependenceDiagnostics": _family_pair_independence_diagnostics(
            families, math.hypot(width, height)
        ),
        "validFamilyCount": len(families),
        "candidateUnorderedPairCount": len(families) * (len(families) - 1) // 2,
    }
    pairs: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    eligibility_rejected: list[dict[str, Any]] = []
    stable_projectively_valid_pair_count = 0
    diagonal = math.hypot(width, height)
    for i, first in enumerate(families):
        for j, second in enumerate(families[i + 1:], i + 1):
            distinct, detail = _distinct(first, second, diagonal)
            if not distinct:
                invalid.append({"familyIndices": [i, j], "reason": "not_projectively_distinct", "distinctness": detail})
                continue
            line = _normalize_line(np.cross(first.vp, second.vp))
            if line is None:
                invalid.append({"familyIndices": [i, j], "reason": "floor_line_degenerate"})
                continue
            stability = _stability(first, second, line, width, height)
            if not stability["stable"]:
                invalid.append({"familyIndices": [i, j], "reason": stability.get("reason", "unstable_vanishing_line"), "stability": stability})
                continue
            stable_projectively_valid_pair_count += 1
            eligibility = _evaluate_pair_independent_direction_eligibility(
                first, second, diagonal
            )
            pair = {
                "i": i,
                "j": j,
                "first": first,
                "second": second,
                "line": line,
                "stability": stability,
                "distinctness": detail,
                "independentDirectionEligibility": eligibility,
            }
            if eligibility["eligible"]:
                pairs.append(pair)
            else:
                eligibility_rejected.append({
                    "familyIndices": [i, j],
                    **eligibility,
                })
    for pair in pairs:
        pair["basinSupport"] = sum(line_distance(pair["line"], other["line"], width, height) <= STABILITY_MAX_PX for other in pairs)
    diagnostics.update({
        "stableProjectivelyValidPairCount": stable_projectively_valid_pair_count,
        "eligiblePairCount": len(pairs),
        "validPairCount": len(pairs),
        "invalidPairs": invalid,
        "independentDirectionEligibilityRejectedPairs": eligibility_rejected,
        "validPairUniverse": [_pair_evidence(pair) for pair in pairs],
    })
    if not pairs:
        return None, diagnostics
    pairs.sort(key=_pair_key)
    winner = pairs[0]
    diagnostics["winningPair"] = _pair_evidence(winner)
    return winner, diagnostics


def _admit_v2_segments(image_bytes: bytes, roi: Sequence[Sequence[float]]) -> tuple[dict[str, Any], list[Segment], np.ndarray | None]:
    identity = hashlib.sha256(image_bytes).hexdigest()
    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        return {"inputImageSha256": identity, "reason": "invalid_input_image"}, [], None
    height, width = decoded.shape[:2]
    base: dict[str, Any] = {"inputImageSha256": identity, "analysisImage": {"decodedWidth": int(width), "decodedHeight": int(height)}, "opencvVersion": cv2.__version__}
    if max(width, height) > MAX_SOURCE_LONG_EDGE_V2 or width * height > MAX_SOURCE_PIXELS_V2:
        base["analysisIdentity"] = _analysis_identity(decoded, "identity", width, height)
        return {**base, "reason": "source_raster_too_large"}, [], None
    if max(width, height) < 1264:
        base["analysisIdentity"] = _analysis_identity(decoded, "identity", width, height)
        return {**base, "reason": "below_reference_analysis_long_edge"}, [], None
    if not _valid_roi_polygon(roi):
        return {**base, "reason": "invalid_roi"}, [], None
    aw, ah = _analysis_dimensions(width, height)
    raster = decoded if (aw, ah) == (width, height) else cv2.resize(decoded, (aw, ah), interpolation=cv2.INTER_AREA)
    raster = np.ascontiguousarray(raster, dtype=np.uint8)
    base["analysisIdentity"] = _analysis_identity(raster, "identity" if raster.shape[:2] == decoded.shape[:2] else "downscale_long_edge", width, height)
    polygon = np.asarray([[round(float(x) * aw), round(float(y) * ah)] for x, y in roi], dtype=np.int32)
    raw = np.zeros((ah, aw), dtype=np.uint8)
    cv2.fillPoly(raw, [polygon], 255)
    mask = cv2.erode(raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29)))
    base["roi"] = {"rawMaskPixels": int(np.count_nonzero(raw)), "erodedMaskPixels": int(np.count_nonzero(mask)), "polygonPixelCoordinates": [[int(x), int(y)] for x, y in polygon]}
    if not np.any(mask):
        return {**base, "reason": "impossible_eroded_roi"}, [], raster
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.cvtColor(raster, cv2.COLOR_BGR2GRAY))
    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(enhanced)[0]
    candidates: list[Segment] = []
    overlap = 0
    if detected is not None:
        for index, value in enumerate(detected.reshape(-1, 4)):
            x1, y1, x2, y2 = (float(item) for item in value)
            samples = [(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t) for t in np.linspace(0.0, 1.0, 9)]
            inside = [0 <= round(x) < aw and 0 <= round(y) < ah and mask[round(y), round(x)] > 0 for x, y in samples]
            if sum(bool(value) for value in inside) >= 7:
                overlap += 1
            length = math.hypot(x2 - x1, y2 - y1)
            p1, p2 = np.asarray([x1, y1, 1.0]), np.asarray([x2, y2, 1.0])
            line = _line_through(p1, p2)
            if line is not None and length >= 24.0 and all(inside):
                candidates.append(Segment(index, p1, p2, line, length, ((x1 + x2) / 2.0, (y1 + y2) / 2.0)))
    candidates.sort(key=lambda item: (float(item.p1[0]), float(item.p1[1]), float(item.p2[0]), float(item.p2[1]), item.length, item.detector_index))
    base["segmentCounts"] = {"raw": 0 if detected is None else len(detected), "roiOverlapDiagnosticAtLeastSevenOfNine": overlap, "admittedAllNineInside": len(candidates)}
    return base, candidates, raster


def read_floor_vanishing_line(image_bytes: bytes, roi_polygon_source_normalized: Sequence[Sequence[float]], policy_version: str = POLICY_VERSION) -> dict[str, Any]:
    if policy_version != POLICY_VERSION:
        return {"status": "rejected", "policyVersion": policy_version, "reason": "unsupported_policy_version", "diagnostics": {}}
    if not isinstance(image_bytes, bytes) or not image_bytes:
        return {"status": "rejected", "policyVersion": policy_version, "reason": "invalid_input_image", "diagnostics": {}}
    diagnostics, candidates, raster = _admit_v2_segments(image_bytes, roi_polygon_source_normalized)
    reason = diagnostics.pop("reason", None)
    if reason:
        return {"status": "rejected", "policyVersion": policy_version, "reason": reason, "diagnostics": diagnostics}
    if raster is None or len(candidates) < 12:
        return {"status": "rejected", "policyVersion": policy_version, "reason": "insufficient_segments", "diagnostics": diagnostics}
    winner, selection = _select_pair(candidates, raster.shape[1], raster.shape[0])
    diagnostics.update(selection)
    if winner is None:
        reason = (
            "no_independent_direction_pair"
            if selection["stableProjectivelyValidPairCount"] > 0
            else "no_stable_valid_pair"
        )
        return {"status": "rejected", "policyVersion": policy_version, "reason": reason, "diagnostics": diagnostics}
    mapped = _map_analysis_line_to_input(winner["line"], diagnostics["analysisIdentity"]["scaleX"], diagnostics["analysisIdentity"]["scaleY"])
    if mapped is None:
        return {"status": "rejected", "policyVersion": policy_version, "reason": "degenerate_vanishing_line", "diagnostics": diagnostics}
    return {"status": "usable", "policyVersion": policy_version, "analysisImage": diagnostics["analysisImage"], "analysisIdentity": diagnostics["analysisIdentity"], "floorVanishingLinePixel": {"a": float(mapped[0]), "b": float(mapped[1]), "c": float(mapped[2])}, "diagnostics": diagnostics}
