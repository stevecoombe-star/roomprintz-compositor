"""C/D-only in-memory covariance certifier for AFC-SR1 tile reader v2.

The caller must explicitly enumerate the six frozen development images with
``--image SHA256=PATH``.  This avoids corpus scanning: the harness never
discovers, hashes, or reads any image that was not explicitly named by the
operator.  Synthetic inputs are OpenCV INTER_LINEAR upscales encoded in memory
only; no derived image is written to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.afc_sr1_tile_floor_reader import V2_POLICY_VERSION, read_floor_vanishing_line

UPSCALE_FACTORS = (1.0, 1.5, 2.0, 4.0)
BELOW_REFERENCE_FACTORS = (0.5, 0.75)


def _half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _parse_image(value: str) -> tuple[str, Path]:
    digest, separator, path = value.partition("=")
    if separator != "=" or not re_fullmatch_sha256(digest) or not path:
        raise argparse.ArgumentTypeError("--image must be SHA256=PATH")
    return digest, Path(path)


def re_fullmatch_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_line(line: list[float] | np.ndarray) -> np.ndarray:
    result = np.asarray(line, dtype=np.float64)
    result /= math.hypot(float(result[0]), float(result[1]))
    if result[1] < 0 or (result[1] == 0 and result[0] < 0):
        result = -result
    result[result == 0] = 0
    return result


def _transformed_expected_line(line: list[float], scale_x: float, scale_y: float) -> np.ndarray:
    return _canonical_line([line[0] / scale_x, line[1] / scale_y, line[2]])


def _resampled_png(image: np.ndarray, factor: float, interpolation: int) -> tuple[bytes, tuple[int, int]]:
    height, width = image.shape[:2]
    resized_width, resized_height = _half_up(width * factor), _half_up(height * factor)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    ok, encoded = cv2.imencode(".png", resized)
    if not ok:
        raise RuntimeError("failed to encode deterministic synthetic image")
    return bytes(encoded), (resized_width, resized_height)


def _load_explicit_images(specifications: list[tuple[str, Path]]) -> dict[str, tuple[np.ndarray, bytes]]:
    images: dict[str, tuple[np.ndarray, bytes]] = {}
    for expected_sha, path in specifications:
        data = path.read_bytes()
        actual_sha = hashlib.sha256(data).hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(f"explicit image digest mismatch: {path}")
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"explicit image is not OpenCV-decodable: {path}")
        if expected_sha in images:
            raise RuntimeError(f"duplicate explicit image digest: {expected_sha}")
        images[expected_sha] = (image, data)
    return images


def _run_case(case: dict[str, Any], source: tuple[np.ndarray, bytes], factor: float) -> dict[str, Any]:
    image, original_bytes = source
    if factor == 1.0:
        input_bytes, (input_width, input_height) = original_bytes, (image.shape[1], image.shape[0])
    else:
        input_bytes, (input_width, input_height) = _resampled_png(image, factor, cv2.INTER_LINEAR)
    started = time.perf_counter()
    result = read_floor_vanishing_line(input_bytes, case["roiPolygonSourceNormalized"], V2_POLICY_VERSION)
    runtime_ms = (time.perf_counter() - started) * 1_000.0
    replay = read_floor_vanishing_line(input_bytes, case["roiPolygonSourceNormalized"], V2_POLICY_VERSION)
    if result != replay:
        raise AssertionError(f"{case['room']}-{case['generation']} scale {factor}: replay differs")
    if result["status"] != "usable":
        raise AssertionError(
            f"{case['room']}-{case['generation']} scale {factor}: {result.get('reason', 'not usable')}"
        )
    diagnostics = result["diagnostics"]
    first, second = diagnostics["firstFamily"], diagnostics["secondFamily"]
    if first["hypothesis_strategy"] != "exhaustive" or second["hypothesis_strategy"] != "exhaustive":
        raise AssertionError(f"{case['room']}-{case['generation']} scale {factor}: non-exhaustive")
    line = result["floorVanishingLinePixel"]
    expected = _transformed_expected_line(
        case["floorVanishingLinePixel"], input_width / image.shape[1], input_height / image.shape[0]
    )
    actual = np.asarray([line["a"], line["b"], line["c"]], dtype=np.float64)
    return {
        "case": f"{case['room']}-{case['generation']}",
        "scale": factor,
        "inputDimensions": [input_width, input_height],
        "analysisDimensions": [
            result["analysisIdentity"]["analysisWidth"],
            result["analysisIdentity"]["analysisHeight"],
        ],
        "analysisMode": result["analysisIdentity"]["mode"],
        "scaleX": result["analysisIdentity"]["scaleX"],
        "scaleY": result["analysisIdentity"]["scaleY"],
        "status": result["status"],
        "rawSegments": diagnostics["segmentCounts"]["raw"],
        "admittedSegments": diagnostics["segmentCounts"]["admittedAllNineInside"],
        "familySupportCounts": [first["support_count"], second["support_count"]],
        "hypothesisStrategy": "exhaustive",
        "lineErrorMaxCoefficient": float(np.max(np.abs(actual - expected))),
        "runtimeMs": runtime_ms,
    }


def _run_below_reference(case: dict[str, Any], source: tuple[np.ndarray, bytes], factor: float) -> dict[str, Any]:
    image, _ = source
    input_bytes, dimensions = _resampled_png(image, factor, cv2.INTER_LINEAR)
    result = read_floor_vanishing_line(input_bytes, case["roiPolygonSourceNormalized"], V2_POLICY_VERSION)
    if (result["status"], result.get("reason")) != ("rejected", "below_reference_analysis_long_edge"):
        raise AssertionError(f"{case['room']}-{case['generation']} scale {factor}: expected domain rejection")
    return {"case": f"{case['room']}-{case['generation']}", "scale": factor, "inputDimensions": dimensions,
            "status": result["status"], "reason": result["reason"]}


def _run_raw_control(control: dict[str, Any], source: tuple[np.ndarray, bytes], factor: float) -> dict[str, Any]:
    image, original_bytes = source
    if factor == 1.0:
        input_bytes = original_bytes
    else:
        input_bytes, _ = _resampled_png(image, factor, cv2.INTER_LINEAR)
    result = read_floor_vanishing_line(input_bytes, control["roiPolygonSourceNormalized"], V2_POLICY_VERSION)
    if result["status"] != "rejected":
        raise AssertionError(f"{control['room']} RAW scale {factor}: unexpectedly usable")
    return {"case": f"{control['room']}-RAW", "scale": factor, "status": result["status"],
            "reason": result.get("reason")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=Path(__file__).with_name("afc_sr1_ts2_development_fixture.json"))
    parser.add_argument("--image", type=_parse_image, action="append", required=True,
                        help="Explicit frozen C/D image mapping SHA256=PATH; repeated once per image.")
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    images = _load_explicit_images(args.image)
    cases = fixture["cases"]
    controls = fixture.get("negativeControls", [])
    needed = {row["imageSha256"] for row in [*cases, *controls]}
    if set(images) != needed:
        raise SystemExit("explicit --image SHA set must equal the frozen C/D fixture SHA set exactly")
    rows = [_run_case(case, images[case["imageSha256"]], factor)
            for case in cases for factor in UPSCALE_FACTORS]
    below = [_run_below_reference(case, images[case["imageSha256"]], factor)
             for case in cases for factor in BELOW_REFERENCE_FACTORS]
    raw = [_run_raw_control(control, images[control["imageSha256"]], factor)
           for control in controls for factor in (1.0, 2.0)]
    runtimes = [row["runtimeMs"] for row in rows]
    print(json.dumps({
        "status": "certified",
        "upscaleResampler": "opencv-inter-linear/development-only",
        "cases": rows,
        "belowReference": below,
        "rawControls": raw,
        "runtimeMs": {"min": min(runtimes), "median": statistics.median(runtimes), "max": max(runtimes)},
        "maxLineError": max(row["lineErrorMaxCoefficient"] for row in rows),
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, RuntimeError) as error:
        print(f"TR2 v2 certification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
