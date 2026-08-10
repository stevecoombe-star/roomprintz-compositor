"""Explicit external-corpus parity harness for the frozen AFC-SR1 TS2 cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.afc_sr1_tile_floor_reader import read_floor_vanishing_line

LINE_TOLERANCE = 1e-9
DIAGNOSTIC_TOLERANCE = 1e-9


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paths_by_sha(corpus_root: Path, wanted: set[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted(corpus_root.rglob("*.png")):
        digest = _sha256(path)
        if digest in wanted:
            if digest in paths:
                raise RuntimeError(f"multiple corpus PNGs share fixture SHA-256 {digest}")
            paths[digest] = path
    missing = wanted - paths.keys()
    if missing:
        raise RuntimeError(f"missing expected corpus PNG SHA-256 values: {', '.join(sorted(missing))}")
    return paths


def _require_close(label: str, actual: float, expected: float, tolerance: float) -> float:
    delta = abs(actual - expected)
    if delta > tolerance:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}, delta {delta!r}")
    return delta


def _family_diagnostic(result: dict[str, Any], index: int) -> dict[str, Any]:
    key = "firstFamily" if index == 0 else "secondFamily"
    return result["diagnostics"][key]


def _certify_case(case: dict[str, Any], image_path: Path) -> dict[str, Any]:
    image_bytes = image_path.read_bytes()
    started = time.perf_counter()
    first = read_floor_vanishing_line(image_bytes, case["roiPolygonSourceNormalized"])
    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    replay = read_floor_vanishing_line(image_bytes, case["roiPolygonSourceNormalized"])
    if first != replay:
        raise AssertionError(f"{case['room']}-{case['generation']}: deterministic replay differs")
    if first["status"] != case["expectedStatus"]:
        raise AssertionError(f"{case['room']}-{case['generation']}: unexpected status {first['status']!r}")
    if first["status"] != "usable":
        raise AssertionError(f"{case['room']}-{case['generation']}: unusable result {first['reason']!r}")

    diagnostics = first["diagnostics"]
    counts = diagnostics["segmentCounts"]
    if counts["raw"] != case["segmentCounts"]["raw"]:
        raise AssertionError(f"{case['room']}-{case['generation']}: raw segment count differs")
    if counts["admittedAllNineInside"] != case["segmentCounts"]["admitted"]:
        raise AssertionError(f"{case['room']}-{case['generation']}: admitted segment count differs")
    if first["analysisImage"] != {"decodedWidth": 1264, "decodedHeight": 848}:
        raise AssertionError(f"{case['room']}-{case['generation']}: decoded dimensions differ")

    line = first["floorVanishingLinePixel"]
    expected_line = case["floorVanishingLinePixel"]
    coefficient_delta = max(
        _require_close(
            f"{case['room']}-{case['generation']} floor line coefficient {component}",
            line[component],
            expected,
            LINE_TOLERANCE,
        )
        for component, expected in zip(("a", "b", "c"), expected_line)
    )
    for index, expected_family in enumerate(case["families"]):
        actual_family = _family_diagnostic(first, index)
        if actual_family["support_count"] != expected_family["supportCount"]:
            raise AssertionError(f"{case['room']}-{case['generation']}: family {index} support differs")
        _require_close(
            f"{case['room']}-{case['generation']} family {index} median residual",
            actual_family["median_residual_px"],
            expected_family["medianResidualPx"],
            DIAGNOSTIC_TOLERANCE,
        )
        _require_close(
            f"{case['room']}-{case['generation']} family {index} p90 residual",
            actual_family["p90_residual_px"],
            expected_family["p90ResidualPx"],
            DIAGNOSTIC_TOLERANCE,
        )
        if actual_family["hypothesis_strategy"] != "exhaustive":
            raise AssertionError(f"{case['room']}-{case['generation']}: expected exhaustive hypotheses")
    _require_close(
        f"{case['room']}-{case['generation']} stability",
        diagnostics["stability"]["max_split_vs_full_probe_distance_px"],
        case["stabilityMaxProbeDistancePx"],
        DIAGNOSTIC_TOLERANCE,
    )
    return {
        "room": case["room"],
        "generation": case["generation"],
        "sha256Verified": _sha256(image_path) == case["imageSha256"],
        "status": first["status"],
        "rawSegments": counts["raw"],
        "admittedSegments": counts["admittedAllNineInside"],
        "familySupportCounts": [
            diagnostics["firstFamily"]["support_count"],
            diagnostics["secondFamily"]["support_count"],
        ],
        "familyMedianResidualPx": [
            diagnostics["firstFamily"]["median_residual_px"],
            diagnostics["secondFamily"]["median_residual_px"],
        ],
        "familyP90ResidualPx": [
            diagnostics["firstFamily"]["p90_residual_px"],
            diagnostics["secondFamily"]["p90_residual_px"],
        ],
        "floorVanishingLinePixel": line,
        "referenceFloorVanishingLinePixel": dict(zip(("a", "b", "c"), expected_line)),
        "maxCoefficientDelta": coefficient_delta,
        "stable": diagnostics["stability"]["stable"],
        "runtimeMs": elapsed_ms,
        "hypotheses": [
            diagnostics["firstFamily"]["hypotheses"],
            diagnostics["secondFamily"]["hypotheses"],
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path(__file__).with_name("afc_sr1_ts2_development_fixture.json"),
    )
    args = parser.parse_args()
    if not args.corpus_root.is_dir():
        raise SystemExit(f"corpus root does not exist or is not a directory: {args.corpus_root}")
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    wanted = {
        row["imageSha256"]
        for row in [*fixture["cases"], *fixture.get("negativeControls", [])]
    }
    paths = _paths_by_sha(args.corpus_root, wanted)
    rows = [_certify_case(case, paths[case["imageSha256"]]) for case in fixture["cases"]]
    negatives = []
    for control in fixture.get("negativeControls", []):
        result = read_floor_vanishing_line(
            paths[control["imageSha256"]].read_bytes(),
            control["roiPolygonSourceNormalized"],
        )
        if result["status"] != control["expectedStatus"]:
            raise AssertionError(
                f"{control['room']} {control['label']}: expected status "
                f"{control['expectedStatus']!r}, got {result['status']!r}"
            )
        if result.get("reason") != control["expectedReason"]:
            raise AssertionError(
                f"{control['room']} {control['label']}: expected reason "
                f"{control['expectedReason']!r}, got {result.get('reason')!r}"
            )
        negatives.append(
            {
                "room": control["room"],
                "label": control["label"],
                "sha256Verified": _sha256(paths[control["imageSha256"]]) == control["imageSha256"],
                "status": result["status"],
                "reason": result.get("reason"),
                "segmentCounts": result.get("diagnostics", {}).get("segmentCounts"),
            }
        )
    runtimes = [row["runtimeMs"] for row in rows]
    print(
        json.dumps(
            {
                "status": "certified",
                "cases": rows,
                "negativeControls": negatives,
                "runtimeMs": {
                    "min": min(runtimes),
                    "median": statistics.median(runtimes),
                    "max": max(runtimes),
                },
                "maxFloorLineCoefficientDelta": max(row["maxCoefficientDelta"] for row in rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, RuntimeError) as error:
        print(f"TR1 certification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
