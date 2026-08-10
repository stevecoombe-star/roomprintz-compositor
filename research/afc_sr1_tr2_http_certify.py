"""External C/D-only HTTP certification for the AFC-SR1 TR2 reader seam."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import requests

ROUTE = "/api/research/afc-sr1/tile-floor-vanishing-line"
LINE_TOLERANCE = 1e-9
RESEARCH_PROFILE = "afc-sr1-tr2-tile-floor-reader/v1"
POLICY_VERSION = "afc-sr1-ts2-extractor-policy/v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_cd_development_candidate(path: Path, root: Path) -> bool:
    """Never hash A/B/mirrored paths while locating the explicitly permitted C/D corpus."""
    names = [part.lower().replace("_", "-") for part in path.relative_to(root).parts]
    root_name = root.name.lower().replace("_", "-")
    if root_name in {"c", "d", "room-c", "room-d", "roomc", "roomd"}:
        return True
    return any(
        name in {"c", "d", "room-c", "room-d", "roomc", "roomd"} or
        name.startswith("room-c-") or name.startswith("room-d-")
        for name in names
    )


def _paths_by_sha_cd_only(corpus_root: Path, wanted: set[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted(corpus_root.rglob("*.png")):
        if not _is_cd_development_candidate(path, corpus_root):
            continue
        digest = _sha256(path)
        if digest in wanted:
            if digest in paths:
                raise RuntimeError(f"multiple C/D PNGs share fixture SHA-256 {digest}")
            paths[digest] = path
    missing = wanted - paths.keys()
    if missing:
        raise RuntimeError(
            "missing expected C/D development PNG SHA-256 values; "
            "point --corpus-root at the frozen Room C/D corpus: " + ", ".join(sorted(missing))
        )
    return paths


def _post(base_url: str, image_bytes: bytes, polygon: list[list[float]]) -> dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}{ROUTE}",
        json={
            "researchProfile": RESEARCH_PROFILE,
            "policyVersion": POLICY_VERSION,
            "imageBase64": base64.b64encode(image_bytes).decode("ascii"),
            "roi": {"coordinateSpace": "source-normalized/v1", "polygon": polygon},
        },
        timeout=20,
    )
    if response.status_code != 200:
        raise AssertionError(f"TR2 route returned HTTP {response.status_code}: {response.text}")
    body = response.json()
    if not isinstance(body, dict):
        raise AssertionError("TR2 route did not return a JSON object receipt")
    return body


def _assert_replay(first: dict[str, Any], replay: dict[str, Any], label: str) -> None:
    for field in (
        "status", "reason", "floorVanishingLinePixel", "evidenceCanonicalJson", "evidenceDigest",
        "runtimeIdentity",
    ):
        if first.get(field) != replay.get(field):
            raise AssertionError(f"{label}: replay differs in {field}")


def _certify_usable(case: dict[str, Any], image_path: Path, base_url: str) -> dict[str, Any]:
    image_bytes = image_path.read_bytes()
    first = _post(base_url, image_bytes, case["roiPolygonSourceNormalized"])
    replay = _post(base_url, image_bytes, case["roiPolygonSourceNormalized"])
    _assert_replay(first, replay, f"{case['room']}-{case['generation']}")
    if first.get("status") != "usable":
        raise AssertionError(f"{case['room']}-{case['generation']}: expected usable, got {first!r}")
    line = first.get("floorVanishingLinePixel")
    if not isinstance(line, dict):
        raise AssertionError(f"{case['room']}-{case['generation']}: usable response omitted pixel line")
    expected = case["floorVanishingLinePixel"]
    delta = max(abs(float(line[key]) - float(value)) for key, value in zip(("a", "b", "c"), expected))
    if delta > LINE_TOLERANCE:
        raise AssertionError(f"{case['room']}-{case['generation']}: floor-line delta {delta} exceeds {LINE_TOLERANCE}")
    return {
        "room": case["room"],
        "generation": case["generation"],
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "status": first["status"],
        "floorVanishingLinePixel": line,
        "maxCoefficientDelta": delta,
        "receiptDigest": first["evidenceDigest"],
        "runtimeIdentity": first["runtimeIdentity"],
    }


def _certify_raw(control: dict[str, Any], image_path: Path, base_url: str) -> dict[str, Any]:
    first = _post(base_url, image_path.read_bytes(), control["roiPolygonSourceNormalized"])
    replay = _post(base_url, image_path.read_bytes(), control["roiPolygonSourceNormalized"])
    _assert_replay(first, replay, f"{control['room']} {control['label']}")
    if (first.get("status"), first.get("reason")) != (
        control["expectedStatus"], control["expectedReason"],
    ):
        raise AssertionError(
            f"{control['room']} {control['label']}: expected "
            f"{control['expectedStatus']!r}/{control['expectedReason']!r}, got "
            f"{first.get('status')!r}/{first.get('reason')!r}"
        )
    return {
        "room": control["room"],
        "label": control["label"],
        "sha256": _sha256(image_path),
        "status": first["status"],
        "reason": first["reason"],
        "receiptDigest": first["evidenceDigest"],
        "runtimeIdentity": first["runtimeIdentity"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
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
    allowed = [*fixture["cases"], *fixture["negativeControls"]]
    paths = _paths_by_sha_cd_only(args.corpus_root, {row["imageSha256"] for row in allowed})
    usable = [_certify_usable(case, paths[case["imageSha256"]], args.base_url) for case in fixture["cases"]]
    raw = [_certify_raw(control, paths[control["imageSha256"]], args.base_url) for control in fixture["negativeControls"]]
    print(json.dumps({
        "status": "certified",
        "usableCases": usable,
        "rawControls": raw,
        "maxFloorLineCoefficientDelta": max(row["maxCoefficientDelta"] for row in usable),
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, RuntimeError, requests.RequestException) as error:
        print(f"TR2 HTTP certification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
