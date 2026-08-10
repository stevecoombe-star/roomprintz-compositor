"""Deterministic AFC-SR1 image-only floor-vanishing-line reader.

This research module reads encoded pixels plus a source-normalized analysis ROI.
It deliberately has no room-corner, endpoint, seam, or compositor-route logic.

The v2 policy uses an aspect-preserving, downscale-only 1264px-long-edge
analysis raster.  Analysis dimensions use non-negative half-up rounding
(``floor(value + 0.5)``); the dimension corresponding to the input long edge
is then set exactly to 1264.  The returned line is mapped from that raster
back into the decoded-input pixel basis before canonical normalization.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import cv2
import numpy as np

READER_MODULE_VERSION = "afc-sr1-tile-floor-reader/v1"
V2_READER_MODULE_VERSION = "afc-sr1-tile-floor-reader/v2"
POLICY_VERSION = "afc-sr1-ts2-extractor-policy/v1"
V2_POLICY_VERSION = "afc-sr1-ts2-extractor-policy/v2"
REFERENCE_WIDTH = 1264
REFERENCE_HEIGHT = 848
REFERENCE_LONG_EDGE = 1264
MAX_SOURCE_LONG_EDGE_V2 = 8192
MAX_SOURCE_PIXELS_V2 = 36_000_000
HOMOGENEOUS_EPSILON = 1e-8
POLICY: dict[str, Any] = {
    "version": POLICY_VERSION,
    "reference_grid": [REFERENCE_WIDTH, REFERENCE_HEIGHT],
    "roi": {
        "erosion_kernel": [29, 29],
        "sample_count": 9,
        "candidate_requires_all_samples_inside": True,
    },
    "preprocess": {"grayscale": True, "clahe_clip_limit": 2.0, "clahe_grid": [8, 8]},
    "detector": {"name": "OpenCV LSD_REFINE_STD", "min_length_px": 24.0},
    "consensus": {
        "ransac_pair_hypotheses_max": 25_000,
        "pair_sampling_seed": 20_260_810,
        "inlier_residual_px": 4.0,
        "min_support_count": 6,
        "min_support_total_length_px": 300.0,
        "max_median_residual_px": 2.5,
        "max_p90_residual_px": 3.75,
    },
    "stability": {"max_floor_line_split_delta_px": 18.0},
}


def _analysis_dimensions(width: int, height: int) -> tuple[int, int]:
    """Returns v2 aspect-preserving analysis dimensions with a fixed long edge."""
    long_edge = max(width, height)
    if long_edge < REFERENCE_LONG_EDGE:
        raise ValueError("below_reference_analysis_long_edge")
    if long_edge == REFERENCE_LONG_EDGE:
        return width, height
    scale = REFERENCE_LONG_EDGE / long_edge
    analysis_width = int(math.floor(scale * width + 0.5))
    analysis_height = int(math.floor(scale * height + 0.5))
    if width >= height:
        analysis_width = REFERENCE_LONG_EDGE
    else:
        analysis_height = REFERENCE_LONG_EDGE
    return analysis_width, analysis_height


def _analysis_identity(
    image: np.ndarray, mode: Literal["identity", "downscale_long_edge"], input_width: int, input_height: int
) -> dict[str, Any]:
    """Binds the exact BGR uint8 C-order buffer passed to grayscale."""
    contiguous = np.ascontiguousarray(image, dtype=np.uint8)
    height, width = contiguous.shape[:2]
    return {
        "mode": mode,
        "analysisWidth": int(width),
        "analysisHeight": int(height),
        "scaleX": float(input_width / width),
        "scaleY": float(input_height / height),
        "referenceLongEdge": REFERENCE_LONG_EDGE,
        "resampler": "identity" if mode == "identity" else "opencv-inter-area/v1",
        "pixelFormat": "bgr8",
        "pixelBufferSha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _map_analysis_line_to_input(
    line: np.ndarray, scale_x: float, scale_y: float
) -> np.ndarray | None:
    """Applies H^-T for H=diag(scale_x, scale_y, 1), then canonicalizes."""
    return _normalize_line(
        np.asarray([line[0] / scale_x, line[1] / scale_y, line[2]], dtype=np.float64)
    )


@dataclass(frozen=True)
class Segment:
    detector_index: int
    p1: np.ndarray
    p2: np.ndarray
    line: np.ndarray
    length: float
    midpoint: tuple[float, float]


def _as_vector(value: np.ndarray | None) -> list[float] | None:
    return None if value is None else [float(component) for component in value]


def _canonical_zero(value: float) -> float:
    return 0.0 if value == 0.0 else float(value)


def _normalize_line(line: np.ndarray) -> np.ndarray | None:
    normal_length = float(np.hypot(line[0], line[1]))
    if normal_length <= HOMOGENEOUS_EPSILON or not np.all(np.isfinite(line)):
        return None
    normalized = line / normal_length
    if normalized[1] < -HOMOGENEOUS_EPSILON or (
        abs(float(normalized[1])) <= HOMOGENEOUS_EPSILON and normalized[0] < 0
    ):
        normalized = -normalized
    return np.asarray([_canonical_zero(value) for value in normalized], dtype=np.float64)


def _normalize_point(point: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(point))
    if length <= HOMOGENEOUS_EPSILON or not np.all(np.isfinite(point)):
        return None
    normalized = point / length
    first_significant = next(
        (component for component in normalized if abs(float(component)) > HOMOGENEOUS_EPSILON),
        0.0,
    )
    if first_significant < 0:
        normalized = -normalized
    return np.asarray([_canonical_zero(value) for value in normalized], dtype=np.float64)


def _finite_point(point: np.ndarray) -> np.ndarray | None:
    if abs(float(point[2])) <= HOMOGENEOUS_EPSILON:
        return None
    result = point / point[2]
    return result if np.all(np.isfinite(result)) else None


def _line_through(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    return _normalize_line(np.cross(first, second))


def _residuals_to_vp(segments: list[Segment], vp: np.ndarray, diagonal: float) -> np.ndarray:
    euclidean = _finite_point(vp)
    if euclidean is not None:
        return np.abs(np.asarray([segment.line @ euclidean for segment in segments], dtype=np.float64))
    direction = vp[:2]
    direction_length = float(np.linalg.norm(direction))
    if direction_length <= HOMOGENEOUS_EPSILON:
        return np.full(len(segments), np.inf)
    direction = direction / direction_length
    return np.abs(np.asarray([segment.line[:2] @ direction for segment in segments], dtype=np.float64)) * diagonal


def _fit_vp(segments: list[Segment]) -> np.ndarray | None:
    if len(segments) < 2:
        return None
    matrix = np.asarray(
        [segment.line * math.sqrt(segment.length) for segment in segments],
        dtype=np.float64,
    )
    _, _, right_vectors = np.linalg.svd(matrix, full_matrices=False)
    return _normalize_point(right_vectors[-1])


def _pair_from_rank(segment_count: int, rank: int) -> tuple[int, int]:
    """Maps itertools.combinations order back from a pair rank without storing all pairs."""
    remaining = rank
    for first in range(segment_count - 1):
        row_length = segment_count - first - 1
        if remaining < row_length:
            return first, first + 1 + remaining
        remaining -= row_length
    raise ValueError("pair rank outside candidate range")


def _hypothesis_pairs(
    candidate_count: int, seed_offset: int
) -> tuple[list[tuple[int, int]], Literal["exhaustive", "seeded_sample"], int]:
    total_pair_count = math.comb(candidate_count, 2)
    max_hypotheses = POLICY["consensus"]["ransac_pair_hypotheses_max"]
    if total_pair_count <= max_hypotheses:
        return list(itertools.combinations(range(candidate_count), 2)), "exhaustive", total_pair_count
    generator = np.random.default_rng(POLICY["consensus"]["pair_sampling_seed"] + seed_offset)
    ranks = generator.choice(total_pair_count, size=max_hypotheses, replace=False)
    return [
        _pair_from_rank(candidate_count, int(rank))
        for rank in ranks
    ], "seeded_sample", total_pair_count


def _discover_family(
    candidates: list[Segment], width: int, height: int, seed_offset: int
) -> tuple[list[Segment], np.ndarray | None, dict[str, Any]]:
    if len(candidates) < POLICY["consensus"]["min_support_count"]:
        return [], None, {"reason": "insufficient_candidates", "hypothesis_strategy": "exhaustive", "hypotheses": 0}

    pairs, strategy, total_pair_count = _hypothesis_pairs(len(candidates), seed_offset)
    diagonal = math.hypot(width, height)
    threshold = POLICY["consensus"]["inlier_residual_px"]
    best_indices: np.ndarray | None = None
    best_key = (-1.0, -1, float("-inf"))
    for first, second in pairs:
        hypothesis = _normalize_point(np.cross(candidates[first].line, candidates[second].line))
        if hypothesis is None:
            continue
        residual = _residuals_to_vp(candidates, hypothesis, diagonal)
        indices = np.flatnonzero(residual <= threshold)
        if len(indices) < POLICY["consensus"]["min_support_count"]:
            continue
        score = float(sum(min(candidates[int(index)].length, 100.0) for index in indices))
        key = (score, int(len(indices)), -float(np.median(residual[indices])))
        if key > best_key:
            best_key = key
            best_indices = indices

    base = {
        "hypothesis_strategy": strategy,
        "hypotheses": len(pairs),
        "total_pair_count": total_pair_count,
    }
    if best_indices is None:
        return [], None, {**base, "reason": "no_ransac_consensus"}

    supporters = [candidates[int(index)] for index in best_indices]
    vp = _fit_vp(supporters)
    for _ in range(2):
        if vp is None:
            return [], None, {**base, "reason": "refinement_failed"}
        residual = _residuals_to_vp(candidates, vp, diagonal)
        supporters = [segment for index, segment in enumerate(candidates) if residual[index] <= threshold]
        vp = _fit_vp(supporters)

    if vp is None:
        return [], None, {**base, "reason": "refinement_failed"}
    residual = _residuals_to_vp(supporters, vp, diagonal)
    return supporters, vp, {
        **base,
        "support_count": len(supporters),
        "support_total_length_px": float(sum(segment.length for segment in supporters)),
        "median_residual_px": float(np.median(residual)) if len(residual) else None,
        "p90_residual_px": float(np.percentile(residual, 90)) if len(residual) else None,
        "max_residual_px": float(np.max(residual)) if len(residual) else None,
        "vp_homogeneous": _as_vector(vp),
    }


def _passes_family_gates(family: list[Segment], diagnostics: dict[str, Any]) -> bool:
    return (
        len(family) >= POLICY["consensus"]["min_support_count"]
        and diagnostics.get("support_total_length_px", 0.0) >= POLICY["consensus"]["min_support_total_length_px"]
        and diagnostics.get("median_residual_px") is not None
        and diagnostics["median_residual_px"] <= POLICY["consensus"]["max_median_residual_px"]
        and diagnostics.get("p90_residual_px") is not None
        and diagnostics["p90_residual_px"] <= POLICY["consensus"]["max_p90_residual_px"]
    )


def _family_rejection_reason(family: list[Segment], diagnostics: dict[str, Any]) -> str:
    if (
        len(family) < POLICY["consensus"]["min_support_count"]
        or diagnostics.get("support_total_length_px", 0.0) < POLICY["consensus"]["min_support_total_length_px"]
    ):
        return "weak_family_support"
    return "high_vp_residual"


def _floor_line_stability(
    family_a: list[Segment], family_b: list[Segment], full_line: np.ndarray, width: int, height: int
) -> dict[str, Any]:
    ordered_a = sorted(
        family_a,
        key=lambda segment: (segment.midpoint[0], segment.midpoint[1], segment.length, segment.detector_index),
    )
    ordered_b = sorted(
        family_b,
        key=lambda segment: (segment.midpoint[0], segment.midpoint[1], segment.length, segment.detector_index),
    )
    split_lines: list[np.ndarray] = []
    for parity in (0, 1):
        first_vp = _fit_vp(ordered_a[parity::2])
        second_vp = _fit_vp(ordered_b[parity::2])
        if first_vp is None or second_vp is None:
            return {"stable": False, "reason": "split_refit_failed"}
        line = _normalize_line(np.cross(first_vp, second_vp))
        if line is None:
            return {"stable": False, "reason": "split_floor_line_degenerate"}
        split_lines.append(line)

    probes = (
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([float(width), 0.0, 1.0]),
        np.asarray([0.0, float(height), 1.0]),
        np.asarray([float(width), float(height), 1.0]),
    )
    deltas = [
        [abs(float(candidate @ probe - full_line @ probe)) for probe in probes]
        for candidate in split_lines
    ]
    max_delta = max(max(row) for row in deltas)
    return {
        "stable": max_delta <= POLICY["stability"]["max_floor_line_split_delta_px"],
        "split_floor_lines": [_as_vector(line) for line in split_lines],
        "split_vs_full_probe_distances_px": deltas,
        "max_split_vs_full_probe_distance_px": max_delta,
    }


def _valid_roi_polygon(value: Sequence[Sequence[float]] | object) -> bool:
    if not isinstance(value, Sequence) or len(value) < 3:
        return False
    points: list[tuple[float, float]] = []
    for point in value:
        if not isinstance(point, Sequence) or len(point) != 2:
            return False
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return False
        if not math.isfinite(x) or not math.isfinite(y) or not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            return False
        points.append((x, y))
    twice_area = sum(
        first[0] * second[1] - second[0] * first[1]
        for first, second in zip(points, points[1:] + points[:1])
    )
    return abs(twice_area) > HOMOGENEOUS_EPSILON


def _rejected(
    reason: str, policy_version: str, diagnostics: dict[str, Any]
) -> dict[str, Any]:
    return {
        "status": "rejected",
        "policyVersion": policy_version,
        "reason": reason,
        "diagnostics": diagnostics,
    }


def read_floor_vanishing_line(
    image_bytes: bytes,
    roi_polygon_source_normalized: Sequence[Sequence[float]],
    policy_version: str = POLICY_VERSION,
) -> dict[str, Any]:
    """Returns canonical pixel-basis floor-line evidence or a fail-closed rejection."""
    if policy_version not in {POLICY_VERSION, V2_POLICY_VERSION}:
        return _rejected("unsupported_policy_version", policy_version, {})
    if not isinstance(image_bytes, bytes) or not image_bytes:
        return _rejected("invalid_input_image", policy_version, {})
    image_identity = hashlib.sha256(image_bytes).hexdigest()
    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        return _rejected("invalid_input_image", policy_version, {"inputImageSha256": image_identity})
    height, width = decoded.shape[:2]
    analysis_image = {"decodedWidth": int(width), "decodedHeight": int(height)}
    base_diagnostics: dict[str, Any] = {
        "inputImageSha256": image_identity,
        "analysisImage": analysis_image,
        "opencvVersion": cv2.__version__,
    }
    analysis_identity: dict[str, Any] | None = None
    extraction_image = decoded
    if policy_version == V2_POLICY_VERSION:
        if max(width, height) > MAX_SOURCE_LONG_EDGE_V2 or width * height > MAX_SOURCE_PIXELS_V2:
            # This decoded-raster resource gate precedes both downscale and
            # every CV stage.  Its identity binds the decoded BGR input.
            analysis_identity = _analysis_identity(decoded, "identity", width, height)
            base_diagnostics["analysisIdentity"] = analysis_identity
            return _rejected("source_raster_too_large", policy_version, base_diagnostics)
        if max(width, height) < REFERENCE_LONG_EDGE:
            # Domain rejection has no detector pass.  Its identity still binds
            # the decoded BGR raster whose size caused that fail-closed result.
            analysis_identity = _analysis_identity(decoded, "identity", width, height)
            base_diagnostics["analysisIdentity"] = analysis_identity
            return _rejected("below_reference_analysis_long_edge", policy_version, base_diagnostics)
        analysis_width, analysis_height = _analysis_dimensions(width, height)
        if (analysis_width, analysis_height) == (width, height):
            analysis_mode: Literal["identity", "downscale_long_edge"] = "identity"
        else:
            analysis_mode = "downscale_long_edge"
            extraction_image = cv2.resize(
                decoded, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA
            )
        extraction_image = np.ascontiguousarray(extraction_image, dtype=np.uint8)
        analysis_identity = _analysis_identity(extraction_image, analysis_mode, width, height)
        base_diagnostics["analysisIdentity"] = analysis_identity
    elif (width, height) != (REFERENCE_WIDTH, REFERENCE_HEIGHT):
        return _rejected("unsupported_analysis_grid", policy_version, base_diagnostics)
    if not _valid_roi_polygon(roi_polygon_source_normalized):
        return _rejected("invalid_roi", policy_version, base_diagnostics)

    analysis_height, analysis_width = extraction_image.shape[:2]
    polygon = np.asarray(
        [
            [round(float(x) * analysis_width), round(float(y) * analysis_height)]
            for x, y in roi_polygon_source_normalized
        ],
        dtype=np.int32,
    )
    raw_roi = np.zeros((analysis_height, analysis_width), dtype=np.uint8)
    cv2.fillPoly(raw_roi, [polygon], 255)
    roi = cv2.erode(raw_roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29)))
    raw_mask_pixels = int(np.count_nonzero(raw_roi))
    eroded_mask_pixels = int(np.count_nonzero(roi))
    base_diagnostics["roi"] = {
        "rawMaskPixels": raw_mask_pixels,
        "erodedMaskPixels": eroded_mask_pixels,
        "polygonPixelCoordinates": [[int(x), int(y)] for x, y in polygon],
    }
    if eroded_mask_pixels == 0:
        return _rejected("impossible_eroded_roi", policy_version, base_diagnostics)

    grayscale = cv2.cvtColor(extraction_image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grayscale)
    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(enhanced)[0]
    raw_count = 0 if detected is None else len(detected)
    roi_overlap_count = 0
    candidates: list[Segment] = []
    if detected is not None:
        for detector_index, candidate in enumerate(detected.reshape(-1, 4)):
            x1, y1, x2, y2 = (float(component) for component in candidate)
            samples = [(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t) for t in np.linspace(0.0, 1.0, 9)]
            inside = [
                0 <= round(x) < analysis_width and 0 <= round(y) < analysis_height and roi[round(y), round(x)] > 0
                for x, y in samples
            ]
            if sum(inside) >= 7:
                roi_overlap_count += 1
            length = math.hypot(x2 - x1, y2 - y1)
            p1, p2 = np.asarray([x1, y1, 1.0]), np.asarray([x2, y2, 1.0])
            line = _line_through(p1, p2)
            if line is not None and length >= POLICY["detector"]["min_length_px"] and all(inside):
                candidates.append(
                    Segment(
                        detector_index,
                        p1,
                        p2,
                        line,
                        length,
                        ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    )
                )

    # OpenCV does not document an output ordering contract. The explicit key is
    # also the pair-enumeration tie-break; detector index only resolves bytewise
    # equal segment geometry.
    candidates.sort(
        key=lambda segment: (
            float(segment.p1[0]),
            float(segment.p1[1]),
            float(segment.p2[0]),
            float(segment.p2[1]),
            segment.length,
            segment.detector_index,
        )
    )
    base_diagnostics["segmentCounts"] = {
        "raw": raw_count,
        "roiOverlapDiagnosticAtLeastSevenOfNine": roi_overlap_count,
        "admittedAllNineInside": len(candidates),
    }
    if len(candidates) < POLICY["consensus"]["min_support_count"] * 2:
        return _rejected("insufficient_segments", policy_version, base_diagnostics)

    first, first_vp, first_diagnostics = _discover_family(
        candidates, analysis_width, analysis_height, seed_offset=0
    )
    base_diagnostics["firstFamily"] = first_diagnostics
    if first_vp is None:
        return _rejected("first_family_not_found", policy_version, base_diagnostics)
    if not _passes_family_gates(first, first_diagnostics):
        return _rejected(_family_rejection_reason(first, first_diagnostics), policy_version, base_diagnostics)

    first_ids = {id(segment) for segment in first}
    remaining = [segment for segment in candidates if id(segment) not in first_ids]
    second, second_vp, second_diagnostics = _discover_family(
        remaining, analysis_width, analysis_height, seed_offset=1
    )
    base_diagnostics["secondFamily"] = second_diagnostics
    if second_vp is None:
        return _rejected("second_family_not_found", policy_version, base_diagnostics)
    if not _passes_family_gates(second, second_diagnostics):
        return _rejected(_family_rejection_reason(second, second_diagnostics), policy_version, base_diagnostics)

    finite_first, finite_second = _finite_point(first_vp), _finite_point(second_vp)
    if (
        finite_first is not None
        and finite_second is not None
        and np.linalg.norm(finite_first[:2] - finite_second[:2]) < 40.0
    ):
        return _rejected("near_identical_vanishing_points", policy_version, base_diagnostics)
    floor_line = _normalize_line(np.cross(first_vp, second_vp))
    if floor_line is None:
        return _rejected("degenerate_vanishing_line", policy_version, base_diagnostics)

    stability = _floor_line_stability(
        first, second, floor_line, analysis_width, analysis_height
    )
    base_diagnostics["stability"] = stability
    if not stability["stable"]:
        return _rejected("unstable_vanishing_line", policy_version, base_diagnostics)
    output_floor_line = floor_line
    if analysis_identity is not None:
        output_floor_line = _map_analysis_line_to_input(
            floor_line, analysis_identity["scaleX"], analysis_identity["scaleY"]
        )
        if output_floor_line is None:
            return _rejected("degenerate_vanishing_line", policy_version, base_diagnostics)
    return {
        "status": "usable",
        "policyVersion": policy_version,
        "analysisImage": analysis_image,
        **({"analysisIdentity": analysis_identity} if analysis_identity is not None else {}),
        "floorVanishingLinePixel": {
            "a": float(output_floor_line[0]),
            "b": float(output_floor_line[1]),
            "c": float(output_floor_line[2]),
        },
        "diagnostics": base_diagnostics,
    }
