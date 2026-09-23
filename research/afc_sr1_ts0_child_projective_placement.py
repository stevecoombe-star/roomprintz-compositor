"""Frozen AFC-SR1 TS0 child placement policy.

This research module estimates one parent-to-child translation.  It has no
provider, floor-line, TR0, Track 1a, or semantic-placement authority.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from decimal import Decimal
from typing import Any, Sequence

import cv2
import numpy as np

POLICY_VERSION = "afc-sr1-ts0-child-projective-placement-policy/v1"
SCHEMA_VERSION = "afc-sr1-ts0-child-projective-placement/v1"
MODULE_VERSION = SCHEMA_VERSION
TRANSFORM_TYPE = "translation"
TRANSFORM_DIRECTION = "parent_to_child"
ORIENTATION = 1
COORDINATE_SPACE = "source-normalized/v1"
MASK_ROLE = "registration_exclusion_support_only_not_placement_authority"
SUPPORTED_MASK_LABELS = frozenset(
    {
        "STRICT_EMPTY_POLYGON_USED_AS_REGISTRATION_EXCLUSION_MASK_ONLY",
        "NON_AUTHORITATIVE_RESEARCH_MASK_ONLY",
    }
)
EXPECTED_OPENCV_VERSION = "4.11.0"
EXPECTED_NUMPY_VERSION = "2.4.6"

SIFT_PARAMETERS: dict[str, Any] = {
    "nfeatures": 4000,
    "nOctaveLayers": 3,
    "contrastThreshold": 0.04,
    "edgeThreshold": 10,
    "sigma": 1.6,
}
AKAZE_PARAMETERS: dict[str, Any] = {
    "descriptor_type": cv2.AKAZE_DESCRIPTOR_MLDB,
    "descriptor_size": 0,
    "descriptor_channels": 3,
    "threshold": 0.001,
    "nOctaves": 4,
    "nOctaveLayers": 4,
    "diffusivity": cv2.KAZE_DIFF_PM_G2,
}
REJECTION_REASONS = frozenset(
    {
        "invalid_source_image",
        "invalid_target_image",
        "registration_mask_missing",
        "lineage_mismatch",
        "insufficient_correspondence",
        "degenerate_correspondence_geometry",
        "insufficient_spatial_coverage",
        "fit_residual_exceeds_limit",
        "validation_residual_exceeds_limit",
        "nonprojective_drift_detected",
        "translation_not_finite",
        "deterministic_replay_failed",
    }
)


def _jcs_number(value: float | int) -> str:
    """Serialize finite IEEE-754 values using RFC 8785/ECMAScript thresholds."""
    if isinstance(value, bool):
        return "true" if value else "false"
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("canonical JSON does not permit non-finite numbers")
    if number == 0:
        return "0"
    negative = number < 0
    absolute = abs(number)
    representation = repr(absolute)
    if 1e-6 <= absolute < 1e21:
        if "e" in representation:
            representation = format(Decimal(representation), "f")
        if "." in representation:
            representation = representation.rstrip("0").rstrip(".")
    else:
        if "e" not in representation:
            representation = format(absolute, ".15e")
        mantissa, exponent = representation.lower().split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        exponent_value = int(exponent)
        representation = f"{mantissa}e{'+' if exponent_value >= 0 else ''}{exponent_value}"
    return ("-" if negative else "") + representation


def canonical_json(value: Any) -> str:
    """Return the receipt's RFC 8785-compatible canonical JSON representation."""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _jcs_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON object keys must be strings")
        # Receipt keys are ASCII; UTF-16 and code-point sorting are identical.
        return "{" + ",".join(
            f"{canonical_json(key)}:{canonical_json(value[key])}" for key in sorted(value)
        ) + "}"
    raise TypeError(f"unsupported canonical JSON value: {type(value)!r}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _runtime_identity() -> dict[str, Any]:
    return {
        "placementModuleVersion": MODULE_VERSION,
        "opencvVersion": cv2.__version__,
        "numpyVersion": np.__version__,
        "cvRngSeed": 0,
        "cvNumThreads": 1,
    }


def runtime_is_supported() -> bool:
    return cv2.__version__ == EXPECTED_OPENCV_VERSION and np.__version__ == EXPECTED_NUMPY_VERSION


def _decode_bgr(image_bytes: bytes) -> np.ndarray | None:
    if not image_bytes:
        return None
    try:
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except cv2.error:
        return None
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        return None
    return image


def image_basis(image_bytes: bytes, image: np.ndarray | None) -> dict[str, Any]:
    height, width = image.shape[:2] if image is not None else (None, None)
    return {
        "sha256": sha256_bytes(image_bytes),
        "byteCount": len(image_bytes),
        "decodedWidth": int(width) if width is not None else None,
        "decodedHeight": int(height) if height is not None else None,
        "orientation": ORIENTATION,
    }


def _valid_lineage_structure(ts0_lineage: Any) -> bool:
    basis_keys = {
        "sha256",
        "byteCount",
        "decodedWidth",
        "decodedHeight",
        "orientation",
    }
    return (
        isinstance(ts0_lineage, dict)
        and set(ts0_lineage) == {"parent", "child"}
        and isinstance(ts0_lineage["parent"], dict)
        and isinstance(ts0_lineage["child"], dict)
        and set(ts0_lineage["parent"]) == basis_keys
        and set(ts0_lineage["child"]) == basis_keys
    )


def usable_structure_mask(
    height: int, width: int, polygon_norm: Sequence[Sequence[float]]
) -> np.ndarray:
    """Rasterize exactly: round(norm*dim), fill, 9x9 dilation twice, invert."""
    points = np.asarray(
        [[int(round(float(x) * width)), int(round(float(y) * height))] for x, y in polygon_norm],
        dtype=np.int32,
    )
    excluded = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(excluded, [points], 255)
    excluded = cv2.dilate(excluded, np.ones((9, 9), np.uint8), iterations=2)
    return np.where(excluded > 0, 0, 255).astype(np.uint8)


def percentile90(values: np.ndarray) -> float:
    if values.size == 0:
        return math.inf
    return float(np.percentile(values, 90, method="linear"))


def translation_from(src: np.ndarray, dst: np.ndarray) -> tuple[float, float]:
    translation = np.median(dst - src, axis=0)
    return float(translation[0]), float(translation[1])


def translation_residuals(
    src: np.ndarray, dst: np.ndarray, tx: float, ty: float
) -> np.ndarray:
    predicted = src + np.asarray([tx, ty], dtype=np.float64)
    return np.linalg.norm(predicted - dst, axis=1)


def h_norm_of(
    tx: float, ty: float, parent_width: int, parent_height: int, child_width: int, child_height: int
) -> list[list[float]]:
    return [
        [float(parent_width / child_width), 0.0, float(tx / child_width)],
        [0.0, float(parent_height / child_height), float(ty / child_height)],
        [0.0, 0.0, 1.0],
    ]


def holdout_split(
    src: np.ndarray, parent_width: int, parent_height: int
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    cells = np.asarray(
        [
            min(3, max(0, int(x / parent_width * 4)))
            + 4 * min(3, max(0, int(y / parent_height * 4)))
            for x, y in src
        ]
    )
    fit = np.where(cells % 2 == 0)[0]
    validation = np.where(cells % 2 == 1)[0]
    method = "4x4_even_odd"
    if len(fit) < 8 or len(validation) < 4:
        median_x = np.median(src[:, 0])
        fit = np.where(src[:, 0] <= median_x)[0]
        validation = np.where(src[:, 0] > median_x)[0]
        method = "median_x_fallback"
    if len(fit) < 8 or len(validation) < 4:
        return None, None, method
    return fit, validation, method


def spatial_coverage(points: np.ndarray, width: int, height: int) -> dict[str, Any]:
    if len(points) == 0:
        return {
            "occupiedCells": 0,
            "quadrants": 0,
            "xExtentFraction": 0.0,
            "yExtentFraction": 0.0,
            "collinearityScore": 1.0,
        }
    cells: set[tuple[int, int]] = set()
    quadrants: set[tuple[int, int]] = set()
    for x, y in points:
        cx = min(3, max(0, int(x / width * 4)))
        cy = min(3, max(0, int(y / height * 4)))
        cells.add((cx, cy))
        quadrants.add((0 if x < width / 2 else 1, 0 if y < height / 2 else 1))
    centered = points - points.mean(axis=0)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    collinearity = float(
        1.0
        - (
            singular_values[1] / singular_values[0]
            if singular_values[0] > 1e-9
            else 0.0
        )
    )
    return {
        "occupiedCells": len(cells),
        "quadrants": len(quadrants),
        "xExtentFraction": float((points[:, 0].max() - points[:, 0].min()) / width),
        "yExtentFraction": float((points[:, 1].max() - points[:, 1].min()) / height),
        "collinearityScore": collinearity,
    }


def maximum_cell_p90(
    src: np.ndarray, residuals: np.ndarray, width: int, height: int
) -> float:
    buckets: dict[tuple[int, int], list[float]] = {}
    for (x, y), residual in zip(src, residuals):
        cell = (
            min(3, max(0, int(x / width * 4))),
            min(3, max(0, int(y / height * 4))),
        )
        buckets.setdefault(cell, []).append(float(residual))
    if not buckets:
        return math.inf
    return max(percentile90(np.asarray(bucket, dtype=np.float64)) for bucket in buckets.values())


def _ratio_matches(descriptors_a: np.ndarray, descriptors_b: np.ndarray, norm: int) -> list[Any]:
    matcher = cv2.BFMatcher(norm, crossCheck=False)
    retained = []
    for pair in matcher.knnMatch(descriptors_a, descriptors_b, k=2):
        if len(pair) < 2:
            continue
        match, next_match = pair
        if match.distance < 0.75 * next_match.distance:
            retained.append(match)
    retained.sort(key=lambda item: (item.queryIdx, item.trainIdx, item.distance))
    return retained


def sift_correspondences(
    parent_gray: np.ndarray,
    child_gray: np.ndarray,
    parent_mask: np.ndarray,
    child_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    cv2.setRNGSeed(0)
    sift = cv2.SIFT_create(**SIFT_PARAMETERS)
    parent_keypoints, parent_descriptors = sift.detectAndCompute(parent_gray, parent_mask)
    child_keypoints, child_descriptors = sift.detectAndCompute(child_gray, child_mask)
    parent_count, child_count = len(parent_keypoints), len(child_keypoints)
    if (
        parent_descriptors is None
        or child_descriptors is None
        or parent_count < 8
        or child_count < 8
    ):
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty.copy(), parent_count, child_count
    matches = _ratio_matches(parent_descriptors, child_descriptors, cv2.NORM_L2)
    if len(matches) < 12:
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty.copy(), parent_count, child_count
    src = np.float64([parent_keypoints[item.queryIdx].pt for item in matches])
    dst = np.float64([child_keypoints[item.trainIdx].pt for item in matches])
    return src, dst, parent_count, child_count


def akaze_correspondences(
    parent_gray: np.ndarray,
    child_gray: np.ndarray,
    parent_mask: np.ndarray,
    child_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    cv2.setRNGSeed(0)
    akaze = cv2.AKAZE_create(**AKAZE_PARAMETERS)
    parent_keypoints, parent_descriptors = akaze.detectAndCompute(parent_gray, parent_mask)
    child_keypoints, child_descriptors = akaze.detectAndCompute(child_gray, child_mask)
    parent_count, child_count = len(parent_keypoints), len(child_keypoints)
    if (
        parent_descriptors is None
        or child_descriptors is None
        or parent_count < 8
        or child_count < 8
    ):
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty.copy(), parent_count, child_count
    matches = _ratio_matches(parent_descriptors, child_descriptors, cv2.NORM_HAMMING)
    if len(matches) < 20:
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty.copy(), parent_count, child_count
    src = np.float64([parent_keypoints[item.queryIdx].pt for item in matches])
    dst = np.float64([child_keypoints[item.trainIdx].pt for item in matches])
    return src, dst, parent_count, child_count


def edge_validation(
    parent_gray: np.ndarray,
    child_gray: np.ndarray,
    tx: float,
    ty: float,
    child_mask: np.ndarray,
) -> dict[str, Any]:
    parent_edges = cv2.Canny(parent_gray, 80, 160)
    child_edges = cv2.Canny(child_gray, 80, 160)
    child_height, child_width = child_gray.shape
    h_px = np.asarray([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])
    warped = cv2.warpPerspective(
        parent_edges, h_px, (child_width, child_height), flags=cv2.INTER_NEAREST
    )
    child_dilated = cv2.dilate(child_edges, np.ones((3, 3), np.uint8), iterations=1)
    support = (child_mask > 0) & (warped > 0)
    support_count = int(np.count_nonzero(support))
    hit_count = int(np.count_nonzero(support & (child_dilated > 0)))
    return {
        "supportCount": support_count,
        "hitCount": hit_count,
        "hitRate": float(hit_count / support_count) if support_count else 0.0,
        "refitApplied": False,
    }


def _blank_diagnostics() -> dict[str, Any]:
    return {
        "sift": {
            "parentKeypoints": None,
            "childKeypoints": None,
            "goodMatches": None,
            "finalInliers": None,
            "inlierRule": "residual_px < 3.0",
            "fitP90Px": None,
        },
        "holdout": {
            "partition": None,
            "fitCount": None,
            "validationCount": None,
            "fitTranslationPx": None,
            "validationP90Px": None,
        },
        "coverage": {
            "occupiedCells": None,
            "quadrants": None,
            "xExtentFraction": None,
            "yExtentFraction": None,
            "collinearityScore": None,
            "maxCellP90Px": None,
        },
        "akaze": {
            "parentKeypoints": None,
            "childKeypoints": None,
            "goodMatches": None,
            "transferP90Px": None,
            "refitApplied": False,
        },
        "canny": {
            "supportCount": None,
            "hitCount": None,
            "hitRate": None,
            "refitApplied": False,
        },
        "thresholds": {
            "minimumSiftMatches": 12,
            "minimumFinalInliers": 40,
            "maximumFitP90Px": 2.0,
            "maximumValidationP90Px": 3.5,
            "maximumCellP90Px": 5.0,
            "minimumOccupiedCells": 4,
            "minimumQuadrants": 2,
            "minimumXExtentFraction": 0.25,
            "minimumYExtentFraction": 0.2,
            "maximumCollinearityScore": 0.85,
            "minimumAkazeMatches": 20,
            "maximumAkazeTransferP90Px": 3.5,
            "minimumEdgeSupport": 2000,
            "minimumEdgeHitRate": 0.55,
            "percentileMethod": "linear",
        },
    }


def _valid_polygon(polygon: Sequence[Sequence[float]] | None) -> bool:
    if polygon is None or len(polygon) < 3:
        return False
    try:
        return all(
            len(point) == 2 and all(math.isfinite(float(coordinate)) for coordinate in point)
            for point in polygon
        )
    except (TypeError, ValueError):
        return False


def _mask_identity(
    polygon: Sequence[Sequence[float]] | None,
    evidence_label: str | None,
    parent_mask: np.ndarray | None,
    child_mask: np.ndarray | None,
) -> dict[str, Any] | None:
    if not _valid_polygon(polygon) or evidence_label not in SUPPORTED_MASK_LABELS:
        return None
    identity: dict[str, Any] = {
        "coordinateSpace": COORDINATE_SPACE,
        "role": MASK_ROLE,
        "evidenceLabel": evidence_label,
        "polygon": [[float(x), float(y)] for x, y in polygon],
        "rasterization": {
            "pixelConversion": "int(round(norm * dimension))",
            "dilationKernel": [9, 9],
            "dilationIterations": 2,
            "usableMaskConvention": "inverse_uint8_255",
        },
        "parentUsableMaskSha256": sha256_bytes(parent_mask.tobytes()) if parent_mask is not None else None,
        "childUsableMaskSha256": sha256_bytes(child_mask.tobytes()) if child_mask is not None else None,
    }
    identity["maskDigest"] = sha256_bytes(canonical_json(identity).encode("utf-8"))
    return identity


def _finish_receipt(
    *,
    started: float,
    source_basis: dict[str, Any],
    target_basis: dict[str, Any],
    lineage: dict[str, Any],
    mask_identity: dict[str, Any] | None,
    diagnostics: dict[str, Any],
    status: str,
    reason: str | None,
    translation_px: dict[str, float] | None = None,
    h_norm: list[list[float]] | None = None,
) -> dict[str, Any]:
    if status not in {"usable", "rejected"}:
        raise ValueError("invalid placement status")
    if status == "rejected" and reason not in REJECTION_REASONS:
        raise ValueError("invalid placement rejection reason")
    preimage = {
        "schemaVersion": SCHEMA_VERSION,
        "policyVersion": POLICY_VERSION,
        "sourceImageBasis": source_basis,
        "targetImageBasis": target_basis,
        "ts0Lineage": lineage,
        "registrationMaskIdentity": mask_identity,
        "transformType": TRANSFORM_TYPE,
        "transformDirection": TRANSFORM_DIRECTION,
        "translationPx": translation_px,
        "H_norm": h_norm,
        "diagnostics": diagnostics,
        "runtimeIdentity": _runtime_identity(),
        "status": status,
        "reason": reason,
    }
    canonical = canonical_json(preimage)
    return {
        **preimage,
        "evidenceCanonicalJson": canonical,
        "evidenceDigest": {
            "algorithm": "sha256",
            "encoding": "hex",
            "value": sha256_bytes(canonical.encode("utf-8")),
        },
        "elapsedMs": float((time.perf_counter() - started) * 1000.0),
    }


def place_ts0_child(
    parent_bytes: bytes,
    child_bytes: bytes,
    polygon_norm: Sequence[Sequence[float]] | None,
    evidence_label: str | None,
    ts0_lineage: dict[str, Any],
    policy_version: str = POLICY_VERSION,
) -> dict[str, Any]:
    """Execute the frozen translation-only policy and return a bound receipt."""
    started = time.perf_counter()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    diagnostics = _blank_diagnostics()
    parent = _decode_bgr(parent_bytes)
    source_basis = image_basis(parent_bytes, parent)
    unknown_target_basis = image_basis(child_bytes, None)
    if parent is None:
        return _finish_receipt(
            started=started,
            source_basis=source_basis,
            target_basis=unknown_target_basis,
            lineage=ts0_lineage,
            mask_identity=None,
            diagnostics=diagnostics,
            status="rejected",
            reason="invalid_source_image",
        )
    child = _decode_bgr(child_bytes)
    target_basis = image_basis(child_bytes, child)
    if child is None:
        return _finish_receipt(
            started=started,
            source_basis=source_basis,
            target_basis=target_basis,
            lineage=ts0_lineage,
            mask_identity=None,
            diagnostics=diagnostics,
            status="rejected",
            reason="invalid_target_image",
        )
    if not _valid_lineage_structure(ts0_lineage):
        return _finish_receipt(
            started=started,
            source_basis=source_basis,
            target_basis=target_basis,
            lineage=ts0_lineage,
            mask_identity=None,
            diagnostics=diagnostics,
            status="rejected",
            reason="lineage_mismatch",
        )
    if ts0_lineage["parent"] != source_basis:
        return _finish_receipt(
            started=started,
            source_basis=source_basis,
            target_basis=target_basis,
            lineage=ts0_lineage,
            mask_identity=None,
            diagnostics=diagnostics,
            status="rejected",
            reason="invalid_source_image",
        )
    if ts0_lineage["child"] != target_basis:
        return _finish_receipt(
            started=started,
            source_basis=source_basis,
            target_basis=target_basis,
            lineage=ts0_lineage,
            mask_identity=None,
            diagnostics=diagnostics,
            status="rejected",
            reason="invalid_target_image",
        )
    parent_height, parent_width = parent.shape[:2]
    child_height, child_width = child.shape[:2]
    parent_mask = child_mask = None
    mask_identity = _mask_identity(polygon_norm, evidence_label, None, None)
    if mask_identity is None:
        return _finish_receipt(
            started=started,
            source_basis=source_basis,
            target_basis=target_basis,
            lineage=ts0_lineage,
            mask_identity=None,
            diagnostics=diagnostics,
            status="rejected",
            reason="registration_mask_missing",
        )
    try:
        parent_mask = usable_structure_mask(
            parent_height, parent_width, polygon_norm or []
        )
        child_mask = usable_structure_mask(
            child_height, child_width, polygon_norm or []
        )
    except (OverflowError, ValueError, cv2.error):
        return _finish_receipt(
            started=started,
            source_basis=source_basis,
            target_basis=target_basis,
            lineage=ts0_lineage,
            mask_identity=None,
            diagnostics=diagnostics,
            status="rejected",
            reason="registration_mask_missing",
        )
    mask_identity = _mask_identity(
        polygon_norm, evidence_label, parent_mask, child_mask
    )
    # Unsupported requested policy/runtime rejects under the sole supported
    # receipt policy identity; the request value is never claimed in evidence.
    if policy_version != POLICY_VERSION or not runtime_is_supported():
        return _finish_receipt(
            started=started,
            source_basis=source_basis,
            target_basis=target_basis,
            lineage=ts0_lineage,
            mask_identity=mask_identity,
            diagnostics=diagnostics,
            status="rejected",
            reason="deterministic_replay_failed",
        )

    parent_gray = cv2.cvtColor(parent, cv2.COLOR_BGR2GRAY)
    child_gray = cv2.cvtColor(child, cv2.COLOR_BGR2GRAY)
    src, dst, sift_parent_count, sift_child_count = sift_correspondences(
        parent_gray, child_gray, parent_mask, child_mask
    )
    diagnostics["sift"].update(
        {
            "parentKeypoints": sift_parent_count,
            "childKeypoints": sift_child_count,
            "goodMatches": int(len(src)),
        }
    )
    if len(src) < 12:
        return _finish_receipt(
            started=started, source_basis=source_basis, target_basis=target_basis,
            lineage=ts0_lineage, mask_identity=mask_identity, diagnostics=diagnostics,
            status="rejected", reason="insufficient_correspondence",
        )

    fit_indices, validation_indices, partition = holdout_split(src, parent_width, parent_height)
    diagnostics["holdout"]["partition"] = partition
    if fit_indices is None or validation_indices is None:
        return _finish_receipt(
            started=started, source_basis=source_basis, target_basis=target_basis,
            lineage=ts0_lineage, mask_identity=mask_identity, diagnostics=diagnostics,
            status="rejected", reason="degenerate_correspondence_geometry",
        )
    diagnostics["holdout"].update(
        {"fitCount": int(len(fit_indices)), "validationCount": int(len(validation_indices))}
    )
    holdout_tx, holdout_ty = translation_from(src[fit_indices], dst[fit_indices])
    if not (math.isfinite(holdout_tx) and math.isfinite(holdout_ty)):
        return _finish_receipt(
            started=started, source_basis=source_basis, target_basis=target_basis,
            lineage=ts0_lineage, mask_identity=mask_identity, diagnostics=diagnostics,
            status="rejected", reason="translation_not_finite",
        )
    diagnostics["holdout"]["fitTranslationPx"] = {"tx": holdout_tx, "ty": holdout_ty}
    validation_p90 = percentile90(
        translation_residuals(
            src[validation_indices], dst[validation_indices], holdout_tx, holdout_ty
        )
    )
    diagnostics["holdout"]["validationP90Px"] = validation_p90
    if validation_p90 > 3.5:
        return _finish_receipt(
            started=started, source_basis=source_basis, target_basis=target_basis,
            lineage=ts0_lineage, mask_identity=mask_identity, diagnostics=diagnostics,
            status="rejected", reason="validation_residual_exceeds_limit",
        )

    tx, ty = translation_from(src, dst)
    if not (math.isfinite(tx) and math.isfinite(ty)):
        return _finish_receipt(
            started=started, source_basis=source_basis, target_basis=target_basis,
            lineage=ts0_lineage, mask_identity=mask_identity, diagnostics=diagnostics,
            status="rejected", reason="translation_not_finite",
        )
    all_residuals = translation_residuals(src, dst, tx, ty)
    inliers = all_residuals < 3.0
    inlier_count = int(np.count_nonzero(inliers))
    diagnostics["sift"]["finalInliers"] = inlier_count
    if inlier_count < 40:
        return _finish_receipt(
            started=started, source_basis=source_basis, target_basis=target_basis,
            lineage=ts0_lineage, mask_identity=mask_identity, diagnostics=diagnostics,
            status="rejected", reason="insufficient_correspondence",
        )
    fit_p90 = percentile90(all_residuals[inliers])
    diagnostics["sift"]["fitP90Px"] = fit_p90
    if fit_p90 > 2.0:
        return _finish_receipt(
            started=started, source_basis=source_basis, target_basis=target_basis,
            lineage=ts0_lineage, mask_identity=mask_identity, diagnostics=diagnostics,
            status="rejected", reason="fit_residual_exceeds_limit",
        )

    coverage = spatial_coverage(src[inliers], parent_width, parent_height)
    diagnostics["coverage"].update(coverage)
    if (
        coverage["occupiedCells"] < 4
        or coverage["quadrants"] < 2
        or coverage["xExtentFraction"] < 0.25
        or coverage["yExtentFraction"] < 0.20
        or coverage["collinearityScore"] > 0.85
    ):
        return _finish_receipt(
            started=started, source_basis=source_basis, target_basis=target_basis,
            lineage=ts0_lineage, mask_identity=mask_identity, diagnostics=diagnostics,
            status="rejected", reason="insufficient_spatial_coverage",
        )
    max_cell_p90 = maximum_cell_p90(
        src[inliers], all_residuals[inliers], parent_width, parent_height
    )
    diagnostics["coverage"]["maxCellP90Px"] = max_cell_p90
    if max_cell_p90 > 5.0:
        return _finish_receipt(
            started=started, source_basis=source_basis, target_basis=target_basis,
            lineage=ts0_lineage, mask_identity=mask_identity, diagnostics=diagnostics,
            status="rejected", reason="nonprojective_drift_detected",
        )

    akaze_src, akaze_dst, akaze_parent_count, akaze_child_count = akaze_correspondences(
        parent_gray, child_gray, parent_mask, child_mask
    )
    diagnostics["akaze"].update(
        {
            "parentKeypoints": akaze_parent_count,
            "childKeypoints": akaze_child_count,
            "goodMatches": int(len(akaze_src)),
        }
    )
    if len(akaze_src) < 20:
        return _finish_receipt(
            started=started, source_basis=source_basis, target_basis=target_basis,
            lineage=ts0_lineage, mask_identity=mask_identity, diagnostics=diagnostics,
            status="rejected", reason="insufficient_correspondence",
        )
    akaze_p90 = percentile90(translation_residuals(akaze_src, akaze_dst, tx, ty))
    diagnostics["akaze"]["transferP90Px"] = akaze_p90
    if akaze_p90 > 3.5:
        return _finish_receipt(
            started=started, source_basis=source_basis, target_basis=target_basis,
            lineage=ts0_lineage, mask_identity=mask_identity, diagnostics=diagnostics,
            status="rejected", reason="validation_residual_exceeds_limit",
        )

    edge = edge_validation(parent_gray, child_gray, tx, ty, child_mask)
    diagnostics["canny"].update(edge)
    if edge["supportCount"] < 2000 or edge["hitRate"] < 0.55:
        return _finish_receipt(
            started=started, source_basis=source_basis, target_basis=target_basis,
            lineage=ts0_lineage, mask_identity=mask_identity, diagnostics=diagnostics,
            status="rejected", reason="validation_residual_exceeds_limit",
        )

    translation_px = {"tx": tx, "ty": ty}
    return _finish_receipt(
        started=started,
        source_basis=source_basis,
        target_basis=target_basis,
        lineage=ts0_lineage,
        mask_identity=mask_identity,
        diagnostics=diagnostics,
        status="usable",
        reason=None,
        translation_px=translation_px,
        h_norm=h_norm_of(tx, ty, parent_width, parent_height, child_width, child_height),
    )


def receipt_digest_is_valid(receipt: dict[str, Any]) -> bool:
    """Verify the canonical evidence and digest without trusting envelope fields."""
    canonical = receipt.get("evidenceCanonicalJson")
    digest = receipt.get("evidenceDigest")
    if not isinstance(canonical, str) or not isinstance(digest, dict):
        return False
    try:
        preimage = json.loads(canonical)
        if canonical_json(preimage) != canonical:
            return False
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    envelope = {
        key: value
        for key, value in receipt.items()
        if key not in {"evidenceCanonicalJson", "evidenceDigest", "elapsedMs"}
    }
    return (
        preimage == envelope
        and digest
        == {
            "algorithm": "sha256",
            "encoding": "hex",
            "value": sha256_bytes(canonical.encode("utf-8")),
        }
    )
