"""Bounded research HTTP adapter for the frozen AFC-SR1 TR1 reader.

This module owns transport validation and deterministic evidence identity only.
It deliberately does not import the FastAPI application or any product/provider code.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import math
import os
import re
import time
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from research.afc_sr1_tile_floor_reader import (
    POLICY_VERSION,
    READER_MODULE_VERSION,
    V2_POLICY_VERSION,
    V2_READER_MODULE_VERSION,
    read_floor_vanishing_line,
)
from research.afc_sr1_tile_floor_reader_v3 import (
    POLICY_VERSION as V4_POLICY_VERSION,
    READER_MODULE_VERSION as V4_READER_MODULE_VERSION,
    read_floor_vanishing_line as read_floor_vanishing_line_v3,
)

RESEARCH_PROFILE = "afc-sr1-tr2-tile-floor-reader/v1"
RESULT_SCHEMA_VERSION = "afc-sr1-tr2-tile-floor-reader-result/v1"
V2_RESEARCH_PROFILE = "afc-sr1-tr2-tile-floor-reader/v2"
V2_RESULT_SCHEMA_VERSION = "afc-sr1-tr2-tile-floor-reader-result/v2"
V4_RESEARCH_PROFILE = "afc-sr1-tr2-tile-floor-reader/v4"
V4_RESULT_SCHEMA_VERSION = "afc-sr1-tr2-tile-floor-reader-result/v4"
ROI_COORDINATE_SPACE = "source-normalized/v1"
MAX_DECODED_IMAGE_BYTES_V1 = 8 * 1024 * 1024
MAX_DECODED_IMAGE_BYTES_V2 = 32 * 1024 * 1024
# Backward-compatible name for existing v1 tests and callers.
MAX_DECODED_IMAGE_BYTES = MAX_DECODED_IMAGE_BYTES_V1
MAX_BASE64_PAYLOAD_CHARS = 4 * ((MAX_DECODED_IMAGE_BYTES_V1 + 2) // 3)
_DATA_IMAGE_PREFIX = re.compile(r"^data:image/[A-Za-z0-9.+-]+;base64,")
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
logger = logging.getLogger(__name__)


class ReaderRoiRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coordinateSpace: Literal["source-normalized/v1"]
    polygon: list[tuple[float, float]] = Field(min_length=3, max_length=32)

    @field_validator("polygon")
    @classmethod
    def polygon_is_finite_non_degenerate(cls, polygon: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if any(
            not math.isfinite(coordinate) or coordinate < 0.0 or coordinate > 1.0
            for point in polygon
            for coordinate in point
        ):
            raise ValueError("ROI polygon coordinates must be finite values in [0, 1].")
        twice_area = sum(
            first[0] * second[1] - second[0] * first[1]
            for first, second in zip(polygon, [*polygon[1:], polygon[0]])
        )
        if abs(twice_area) <= 1e-8:
            raise ValueError("ROI polygon must be non-degenerate.")
        return polygon


class TileFloorReaderRequest(BaseModel):
    """Strict image-only input contract; semantic AFC fields are forbidden."""

    model_config = ConfigDict(extra="forbid")

    researchProfile: Literal[
        "afc-sr1-tr2-tile-floor-reader/v1", "afc-sr1-tr2-tile-floor-reader/v2",
        "afc-sr1-tr2-tile-floor-reader/v4"
    ]
    policyVersion: Literal[
        "afc-sr1-ts2-extractor-policy/v1", "afc-sr1-ts2-extractor-policy/v2",
        "afc-sr1-ts2-extractor-policy/v4"
    ]
    imageBase64: str
    roi: ReaderRoiRequest

    @model_validator(mode="after")
    def reject_empty_image(self) -> "TileFloorReaderRequest":
        if not self.imageBase64.strip():
            raise ValueError("imageBase64 must not be empty.")
        expected_pairs = {
            (RESEARCH_PROFILE, POLICY_VERSION),
            (V2_RESEARCH_PROFILE, V2_POLICY_VERSION),
            (V4_RESEARCH_PROFILE, V4_POLICY_VERSION),
        }
        if (self.researchProfile, self.policyVersion) not in expected_pairs:
            raise ValueError("researchProfile and policyVersion must be an exact supported pair.")
        return self


def reader_route_enabled() -> bool:
    """Uses an explicit, opt-in gate and evaluates it per request for deployment clarity."""
    return os.getenv("AFC_SR1_TR2_READER_ENABLED", "").strip().lower() in _TRUTHY_ENV_VALUES


def _canonical_json(payload: dict[str, Any]) -> str:
    """Frozen server-owned receipt serialization: UTF-8 JSON, sorted compact keys, no NaN."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _decode_transport_image(image_base64: str, max_decoded_image_bytes: int) -> bytes:
    """Accepts raw base64 or data:image/...;base64 without altering encoded bytes."""
    value = image_base64.strip()
    if value.startswith("data:"):
        prefix = _DATA_IMAGE_PREFIX.match(value)
        if prefix is None:
            raise HTTPException(status_code=400, detail="imageBase64 must use data:image/...;base64 when data-URL encoded.")
        payload = value[prefix.end():]
    else:
        payload = value
    max_base64_payload_chars = 4 * ((max_decoded_image_bytes + 2) // 3)
    limit_mib = max_decoded_image_bytes // (1024 * 1024)
    if len(payload) > max_base64_payload_chars:
        raise HTTPException(
            status_code=413, detail=f"imageBase64 exceeds the {limit_mib} MiB decoded image limit."
        )
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=400, detail="imageBase64 is not valid base64.") from error
    if len(decoded) > max_decoded_image_bytes:
        raise HTTPException(
            status_code=413, detail=f"imageBase64 exceeds the {limit_mib} MiB decoded image limit."
        )
    return decoded


def _runtime_identity(policy_version: str) -> dict[str, str]:
    return {
        "readerModuleVersion": (
            READER_MODULE_VERSION if policy_version == POLICY_VERSION else
            V2_READER_MODULE_VERSION if policy_version == V2_POLICY_VERSION else
            V4_READER_MODULE_VERSION
        ),
        "opencvVersion": cv2.__version__,
        "numpyVersion": np.__version__,
    }


def _image_identity(image_bytes: bytes, reader_result: dict[str, Any]) -> dict[str, Any]:
    analysis_image = reader_result.get("analysisImage")
    if not isinstance(analysis_image, dict):
        analysis_image = reader_result.get("diagnostics", {}).get("analysisImage")
    width = analysis_image.get("decodedWidth") if isinstance(analysis_image, dict) else None
    height = analysis_image.get("decodedHeight") if isinstance(analysis_image, dict) else None
    return {
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "byteCount": len(image_bytes),
        "decodedWidth": width if isinstance(width, int) else None,
        "decodedHeight": height if isinstance(height, int) else None,
    }


def _evidence_diagnostics(diagnostics: Any) -> dict[str, Any]:
    """The small, stable TR1 audit subset that is bound into receipt replay identity."""
    value = diagnostics if isinstance(diagnostics, dict) else {}
    counts = value.get("segmentCounts") if isinstance(value.get("segmentCounts"), dict) else {}
    first = value.get("firstFamily") if isinstance(value.get("firstFamily"), dict) else {}
    second = value.get("secondFamily") if isinstance(value.get("secondFamily"), dict) else {}
    stability = value.get("stability") if isinstance(value.get("stability"), dict) else {}
    return {
        "rawSegmentCount": counts.get("raw"),
        "admittedAllNineInsideCount": counts.get("admittedAllNineInside"),
        "familySupportCounts": [first.get("support_count"), second.get("support_count")],
        "familyMedianResidualsPx": [first.get("median_residual_px"), second.get("median_residual_px")],
        "familyP90ResidualsPx": [first.get("p90_residual_px"), second.get("p90_residual_px")],
        "stabilityMaxProbeDistancePx": stability.get("max_split_vs_full_probe_distance_px"),
        "hypothesisStrategies": [first.get("hypothesis_strategy"), second.get("hypothesis_strategy")],
    }


def _v3_evidence_diagnostics(diagnostics: Any) -> dict[str, Any]:
    """Binds the complete deterministic V3 selection authority into a receipt."""
    value = diagnostics if isinstance(diagnostics, dict) else {}
    return {
        "segmentCounts": value.get("segmentCounts"),
        "candidateDiscovery": value.get("candidateDiscovery"),
        "validFamilyCount": value.get("validFamilyCount"),
        "candidateUnorderedPairCount": value.get("candidateUnorderedPairCount"),
        "stableProjectivelyValidPairCount": value.get("stableProjectivelyValidPairCount"),
        "eligiblePairCount": value.get("eligiblePairCount"),
        "validPairCount": value.get("validPairCount"),
        "invalidPairs": value.get("invalidPairs"),
        "independentDirectionEligibilityRejectedPairs": (
            value.get("independentDirectionEligibilityRejectedPairs")
        ),
        "validPairUniverse": value.get("validPairUniverse"),
        "finalFamilies": value.get("candidateDiscovery", {}).get("finalFamilies")
        if isinstance(value.get("candidateDiscovery"), dict) else None,
        "winningPair": value.get("winningPair"),
    }


def _roi_identity(roi: ReaderRoiRequest) -> tuple[list[list[float]], dict[str, Any]]:
    polygon = [[float(x), float(y)] for x, y in roi.polygon]
    identity = {"coordinateSpace": ROI_COORDINATE_SPACE, "polygon": polygon}
    return polygon, {**identity, "roiDigest": _sha256_text(_canonical_json(identity))}


def execute_tile_floor_reader(request: TileFloorReaderRequest) -> dict[str, Any]:
    """Runs TR1 once and envelopes its output without changing reader policy or pixels."""
    started = time.perf_counter()
    max_decoded_image_bytes = MAX_DECODED_IMAGE_BYTES_V2 if request.policyVersion in {V2_POLICY_VERSION, V4_POLICY_VERSION} else MAX_DECODED_IMAGE_BYTES_V1
    image_bytes = _decode_transport_image(request.imageBase64, max_decoded_image_bytes)
    polygon, roi_identity = _roi_identity(request.roi)
    result = (read_floor_vanishing_line_v3(image_bytes, polygon, request.policyVersion)
              if request.policyVersion == V4_POLICY_VERSION
              else read_floor_vanishing_line(image_bytes, polygon, request.policyVersion))
    image_identity = _image_identity(image_bytes, result)
    runtime_identity = _runtime_identity(request.policyVersion)
    diagnostics = result.get("diagnostics", {})
    is_v2 = request.policyVersion == V2_POLICY_VERSION
    is_v4 = request.policyVersion == V4_POLICY_VERSION
    research_profile = V4_RESEARCH_PROFILE if is_v4 else V2_RESEARCH_PROFILE if is_v2 else RESEARCH_PROFILE
    result_schema_version = V4_RESULT_SCHEMA_VERSION if is_v4 else V2_RESULT_SCHEMA_VERSION if is_v2 else RESULT_SCHEMA_VERSION
    analysis_identity = result.get("analysisIdentity")
    if analysis_identity is None and isinstance(diagnostics, dict):
        analysis_identity = diagnostics.get("analysisIdentity")
    status = result.get("status")
    if status not in {"usable", "rejected"}:
        raise RuntimeError("TR1 returned an invalid status.")

    preimage: dict[str, Any] = {
        "schemaVersion": result_schema_version,
        "researchProfile": research_profile,
        "policyVersion": request.policyVersion,
        "image": image_identity,
        "roi": roi_identity,
        "runtime": runtime_identity,
        "status": status,
        "diagnostics": _v3_evidence_diagnostics(diagnostics) if is_v4 else _evidence_diagnostics(diagnostics),
    }
    response: dict[str, Any] = {
        "schemaVersion": result_schema_version,
        "researchProfile": research_profile,
        "policyVersion": request.policyVersion,
        "status": status,
        "imageIdentity": image_identity,
        "roiIdentity": roi_identity,
        "runtimeIdentity": runtime_identity,
        "diagnostics": diagnostics,
    }
    if is_v2 or (is_v4 and isinstance(analysis_identity, dict)):
        if not isinstance(analysis_identity, dict):
            raise RuntimeError("TR1 v2 result was missing analysis identity.")
        response["analysisIdentity"] = analysis_identity
        preimage["analysisIdentity"] = analysis_identity
    if status == "usable":
        line = result.get("floorVanishingLinePixel")
        if not isinstance(line, dict) or not all(
            isinstance(line.get(component), float) and math.isfinite(line[component])
            for component in ("a", "b", "c")
        ):
            raise RuntimeError("TR1 usable result was missing a finite pixel line.")
        response["floorVanishingLinePixel"] = line
        preimage["floorVanishingLinePixel"] = line
    else:
        reason = result.get("reason")
        if not isinstance(reason, str):
            raise RuntimeError("TR1 rejection result was missing its reason.")
        response["reason"] = reason
        preimage["reason"] = reason

    evidence_canonical_json = _canonical_json(preimage)
    response["evidenceCanonicalJson"] = evidence_canonical_json
    response["evidenceDigest"] = {
        "algorithm": "sha256",
        "encoding": "hex",
        "value": _sha256_text(evidence_canonical_json),
    }
    response["elapsedMs"] = (time.perf_counter() - started) * 1_000.0
    logger.info(
        "afc_sr1_tr2_tile_reader profile=%s policy=%s image_sha256=%s dimensions=%sx%s status=%s reason=%s elapsed_ms=%.3f",
        research_profile,
        request.policyVersion,
        image_identity["sha256"],
        image_identity["decodedWidth"],
        image_identity["decodedHeight"],
        status,
        response.get("reason"),
        response["elapsedMs"],
    )
    return response
