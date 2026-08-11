"""Explicit external-corpus certifier for frozen TS0 child placement."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.afc_sr1_ts0_child_projective_placement import (
    ORIENTATION,
    POLICY_VERSION,
    place_ts0_child,
    receipt_digest_is_valid,
)

FLOAT_TOLERANCE = 1e-12
RUNTIME_LIMIT_MS = 15_000.0


def _read_verified(path: Path, expected_sha256: str) -> bytes:
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise AssertionError(f"{path}: expected SHA-256 {expected_sha256}, got {actual}")
    return data


def _basis(data: bytes) -> dict[str, Any]:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise AssertionError("certification image failed to decode")
    height, width = image.shape[:2]
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "byteCount": len(data),
        "decodedWidth": int(width),
        "decodedHeight": int(height),
        "orientation": ORIENTATION,
    }


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def _require_close(label: str, actual: float, expected: float) -> float:
    delta = abs(float(actual) - float(expected))
    if not math.isfinite(delta) or delta > FLOAT_TOLERANCE:
        raise AssertionError(
            f"{label}: expected {expected!r}, got {actual!r}, delta {delta!r}"
        )
    return delta


def _compare_expected(
    label: str, receipt: dict[str, Any], expected: dict[str, Any]
) -> float:
    _require_equal(f"{label} status", receipt["status"], "usable")
    _require_equal(f"{label} expected status", expected["status"], "PASS")
    diagnostics = receipt["diagnostics"]
    comparisons = [
        ("tx", receipt["translationPx"]["tx"], expected["translationPx"]["tx"]),
        ("ty", receipt["translationPx"]["ty"], expected["translationPx"]["ty"]),
        (
            "holdout tx",
            diagnostics["holdout"]["fitTranslationPx"]["tx"],
            expected["holdout_fit_translationPx"]["tx"],
        ),
        (
            "holdout ty",
            diagnostics["holdout"]["fitTranslationPx"]["ty"],
            expected["holdout_fit_translationPx"]["ty"],
        ),
        (
            "holdout p90",
            diagnostics["holdout"]["validationP90Px"],
            expected["holdout_validation_p90"],
        ),
        ("fit p90", diagnostics["sift"]["fitP90Px"], expected["fit_p90"]),
        (
            "maximum cell p90",
            diagnostics["coverage"]["maxCellP90Px"],
            expected["max_cell_p90"],
        ),
        (
            "coverage x extent",
            diagnostics["coverage"]["xExtentFraction"],
            expected["spatial_coverage"]["x_extent_frac"],
        ),
        (
            "coverage y extent",
            diagnostics["coverage"]["yExtentFraction"],
            expected["spatial_coverage"]["y_extent_frac"],
        ),
        (
            "coverage collinearity",
            diagnostics["coverage"]["collinearityScore"],
            expected["spatial_coverage"]["collinearity_score"],
        ),
        (
            "AKAZE transfer p90",
            diagnostics["akaze"]["transferP90Px"],
            expected["akaze_transfer_p90"],
        ),
        (
            "edge hit rate",
            diagnostics["canny"]["hitRate"],
            expected["edge"]["edge_hit_rate"],
        ),
    ]
    deltas = [_require_close(f"{label} {name}", actual, wanted) for name, actual, wanted in comparisons]
    integer_comparisons = [
        ("SIFT matches", diagnostics["sift"]["goodMatches"], expected["n_sift"]),
        ("SIFT inliers", diagnostics["sift"]["finalInliers"], expected["n_inliers"]),
        (
            "occupied cells",
            diagnostics["coverage"]["occupiedCells"],
            expected["spatial_coverage"]["occupied_cells"],
        ),
        (
            "quadrants",
            diagnostics["coverage"]["quadrants"],
            expected["spatial_coverage"]["quadrants"],
        ),
        ("AKAZE matches", diagnostics["akaze"]["goodMatches"], expected["akaze_n"]),
        ("edge support", diagnostics["canny"]["supportCount"], expected["edge"]["edge_support"]),
    ]
    for name, actual, wanted in integer_comparisons:
        _require_equal(f"{label} {name}", actual, wanted)
    return max(deltas)


def _certify_pair(case: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    parent = _read_verified(Path(case["parentPath"]), case["parentSha256"])
    child = _read_verified(Path(case["childPath"]), case["childSha256"])
    lineage = {"parent": _basis(parent), "child": _basis(child)}
    first = place_ts0_child(
        parent,
        child,
        case["polygon"],
        case["maskEvidenceLabel"],
        lineage,
        POLICY_VERSION,
    )
    replay = place_ts0_child(
        parent,
        child,
        case["polygon"],
        case["maskEvidenceLabel"],
        lineage,
        POLICY_VERSION,
    )
    if not receipt_digest_is_valid(first) or not receipt_digest_is_valid(replay):
        raise AssertionError(f"{case['label']}: receipt digest validation failed")
    deterministic = (
        first["evidenceCanonicalJson"] == replay["evidenceCanonicalJson"]
        and first["evidenceDigest"] == replay["evidenceDigest"]
    )
    if not deterministic:
        raise AssertionError(f"{case['label']}: deterministic replay differs")
    if first["elapsedMs"] >= RUNTIME_LIMIT_MS:
        raise AssertionError(
            f"{case['label']}: runtime {first['elapsedMs']:.3f} ms exceeds {RUNTIME_LIMIT_MS:.0f}"
        )
    max_delta = _compare_expected(case["label"], first, expected)
    return {
        "room": case["room"],
        "label": case["label"],
        "parentSha256": case["parentSha256"],
        "childSha256": case["childSha256"],
        "status": first["status"],
        "translationPx": first["translationPx"],
        "diagnostics": first["diagnostics"],
        "receiptDigest": first["evidenceDigest"],
        "runtimeIdentity": first["runtimeIdentity"],
        "runtimeMs": first["elapsedMs"],
        "replayRuntimeMs": replay["elapsedMs"],
        "deterministic": deterministic,
        "maximumFrozenMetricDelta": max_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).with_name(
            "afc_sr1_ts0_child_placement_certification_fixture.json"
        ),
    )
    parser.add_argument(
        "--expected-artifact",
        type=Path,
        help=(
            "Optional provenance cross-check against frozen_v1_twelve_pair_rerun.json; "
            "tracked fixture metrics remain certification authority."
        ),
    )
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    if fixture.get("imageOrientation") != ORIENTATION:
        raise AssertionError("certification fixture orientation convention differs")
    expected_by_label = fixture["expectedMetrics"]
    cases = fixture["pairs"]
    if len(cases) != 12 or set(expected_by_label) != {case["label"] for case in cases}:
        raise AssertionError("certification fixture must bind exactly 12 pairs and expectations")
    if args.expected_artifact is not None:
        frozen = json.loads(args.expected_artifact.read_text(encoding="utf-8"))
        frozen_by_label = {row["pair"]: row for row in frozen["pairs"]}
        if set(frozen_by_label) != set(expected_by_label):
            raise AssertionError("optional provenance artifact pair identities differ")
        for label, tracked in expected_by_label.items():
            frozen_row = frozen_by_label[label]
            for key, expected_value in tracked.items():
                if frozen_row.get(key) != expected_value:
                    raise AssertionError(
                        f"{label}: tracked {key} differs from optional provenance artifact"
                    )
    results = [
        _certify_pair(case, expected_by_label[case["label"]])
        for case in cases
    ]
    report = {
        "schemaVersion": fixture["schemaVersion"],
        "policyVersion": POLICY_VERSION,
        "status": "certified",
        "passCount": sum(row["status"] == "usable" for row in results),
        "rejectCount": sum(row["status"] != "usable" for row in results),
        "allSha256Verified": True,
        "allDeterministic": all(row["deterministic"] for row in results),
        "provenanceCrossChecked": args.expected_artifact is not None,
        "runtimeLimitMs": RUNTIME_LIMIT_MS,
        "maximumRuntimeMs": max(row["runtimeMs"] for row in results),
        "maximumFrozenMetricDelta": max(row["maximumFrozenMetricDelta"] for row in results),
        "pairs": results,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, RuntimeError, ValueError) as error:
        print(f"TS0 child placement certification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
