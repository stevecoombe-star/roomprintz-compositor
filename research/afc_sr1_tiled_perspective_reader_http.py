"""Strict HTTP transport for the certified AFC-SR1 TILED perspective reader.

This adapter authenticates immutable TILED bytes before invoking the reader.
It has no provider, V3/V4/TR2, placement, ROI, or room-geometry authority.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import math
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from research.afc_sr1_tiled_perspective_reader import (
    TiledPerspectiveLatticeFailure,
    read_tiled_perspective_lattice,
)

RESEARCH_PROFILE = "afc-sr1-tiled-perspective-reader/s1"
MAX_DECODED_IMAGE_BYTES = 32 * 1024 * 1024


class ClaimedImageIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    sha256: str
    byteCount: int = Field(gt=0)
    decodedWidth: int = Field(gt=0)
    decodedHeight: int = Field(gt=0)
    mimeType: Literal["image/jpeg", "image/png", "image/webp"]
    orientation: Literal[1]

    @field_validator("sha256")
    @classmethod
    def sha256_is_lowercase_hex(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("sha256 must be a lowercase SHA-256 hex digest.")
        return value


class TiledPerspectiveReaderRequest(BaseModel):
    """Image-only, exact-bytes contract for the S1 reader."""

    model_config = ConfigDict(extra="forbid", strict=True)

    researchProfile: Literal["afc-sr1-tiled-perspective-reader/s1"]
    imageBase64: str = Field(min_length=4)
    claimedIdentity: ClaimedImageIdentity


def _transport_fail(reason: str) -> ValueError:
    return ValueError(f"tiled_perspective_transport_identity_failure:{reason}")


def _decode_canonical_base64(value: str) -> bytes:
    if len(value) > 4 * ((MAX_DECODED_IMAGE_BYTES + 2) // 3):
        raise _transport_fail("payload_too_large")
    try:
        encoded = value.encode("ascii")
        decoded = base64.b64decode(encoded, validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise _transport_fail("base64_invalid") from error
    if not decoded or len(decoded) > MAX_DECODED_IMAGE_BYTES:
        raise _transport_fail("decoded_bytes_invalid")
    return decoded


def _magic_mime(encoded: bytes) -> Literal["image/jpeg", "image/png", "image/webp"] | None:
    if encoded.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if encoded.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(encoded) >= 12 and encoded[:4] == b"RIFF" and encoded[8:12] == b"WEBP":
        return "image/webp"
    return None


def _decode_identity(encoded: bytes, claimed: ClaimedImageIdentity) -> dict[str, Any]:
    if hashlib.sha256(encoded).hexdigest() != claimed.sha256:
        raise _transport_fail("sha256_mismatch")
    if len(encoded) != claimed.byteCount:
        raise _transport_fail("byte_count_mismatch")
    if _magic_mime(encoded) != claimed.mimeType:
        raise _transport_fail("mime_magic_mismatch")

    decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
        raise _transport_fail("opencv_decode_failed")
    height, width = decoded.shape[:2]
    if width != claimed.decodedWidth or height != claimed.decodedHeight:
        raise _transport_fail("decoded_dimensions_mismatch")
    try:
        with Image.open(io.BytesIO(encoded)) as image:
            if int(image.getexif().get(274, 1)) != 1:
                raise _transport_fail("orientation_not_identity")
    except ValueError:
        raise
    except Exception as error:
        raise _transport_fail("orientation_inspection_failed") from error
    return {
        "sha256": claimed.sha256,
        "byteCount": claimed.byteCount,
        "decodedWidth": width,
        "decodedHeight": height,
        "mimeType": claimed.mimeType,
        "orientation": 1,
    }


def _finite_tuple(values: tuple[tuple[float, float], ...]) -> list[list[float]]:
    result = [[float(x), float(y)] for x, y in values]
    if not all(math.isfinite(value) for point in result for value in point):
        raise RuntimeError("TILED reader returned a non-finite quadrilateral.")
    return result


def execute_tiled_perspective_reader(request: TiledPerspectiveReaderRequest) -> dict[str, Any]:
    """Verify exact TILED bytes and execute the certified S1 reader once."""
    encoded = _decode_canonical_base64(request.imageBase64)
    decoded_identity = _decode_identity(encoded, request.claimedIdentity)
    try:
        result = read_tiled_perspective_lattice(encoded)
    except TiledPerspectiveLatticeFailure as error:
        return {
            "status": "failed",
            "reason": error.reason,
            "decodedIdentity": decoded_identity,
            "readerVersion": RESEARCH_PROFILE,
        }
    return {
        "status": "ok",
        "decodedIdentity": decoded_identity,
        "readerVersion": result.version,
        "authoritativeQuadSourceNormalized": _finite_tuple(
            result.authoritative_quad_source_normalized
        ),
        "authoritativeQuadPixel": _finite_tuple(result.authoritative_quad_pixel),
        "authoritativeCore": {
            "rows": result.authoritative_core.rows,
            "columns": result.authoritative_core.columns,
            "j0": result.authoritative_core.j0,
            "i0": result.authoritative_core.i0,
            "cellIds": list(result.authoritative_core.cell_ids),
        },
        "selectedComponentTileCount": result.selected_component.tile_count,
        "rawQuadrilateralCount": result.raw_quadrilateral_count,
        "deduplicatedCellCount": result.deduplicated_cell_count,
        "reprojectionMeanPx": result.reprojection_mean_px,
        "reprojectionMaxPx": result.reprojection_max_px,
    }
