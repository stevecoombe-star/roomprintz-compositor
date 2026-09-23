"""Strict, env-gated HTTP adapter for frozen TS0 child placement."""
from __future__ import annotations

import base64
import binascii
import math
import os
import re
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from research.afc_sr1_ts0_child_projective_placement import (
    COORDINATE_SPACE,
    MASK_ROLE,
    place_ts0_child,
)

MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_BASE64_PAYLOAD_CHARS = 4 * ((MAX_IMAGE_BYTES + 2) // 3)
_DATA_IMAGE_PREFIX = re.compile(r"^data:image/[A-Za-z0-9.+-]+;base64,")
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


class PlacementImageIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    byteCount: int = Field(gt=0)
    decodedWidth: int = Field(gt=0)
    decodedHeight: int = Field(gt=0)
    orientation: Literal[1]


class Ts0LineageBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parent: PlacementImageIdentity
    child: PlacementImageIdentity


class RegistrationExclusion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coordinateSpace: Literal["source-normalized/v1"]
    role: Literal["registration_exclusion_support_only_not_placement_authority"]
    evidenceLabel: Literal[
        "STRICT_EMPTY_POLYGON_USED_AS_REGISTRATION_EXCLUSION_MASK_ONLY",
        "NON_AUTHORITATIVE_RESEARCH_MASK_ONLY",
    ]
    polygon: list[tuple[float, float]] = Field(min_length=3, max_length=32)

    @field_validator("polygon")
    @classmethod
    def polygon_is_finite(
        cls, polygon: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        # Off-frame normalized coordinates are intentionally permitted.  The
        # frozen rasterizer rounds them and lets fillPoly clip naturally.
        if any(not math.isfinite(value) for point in polygon for value in point):
            raise ValueError("registration exclusion polygon must be finite")
        return polygon


class Ts0ChildPlacementRequest(BaseModel):
    """The only accepted transport fields; semantic/downstream fields are forbidden."""

    model_config = ConfigDict(extra="forbid")

    policyVersion: Literal["afc-sr1-ts0-child-projective-placement-policy/v1"]
    parentImageBase64: str
    childImageBase64: str
    registrationExclusion: RegistrationExclusion
    ts0Lineage: Ts0LineageBinding

    @field_validator("parentImageBase64", "childImageBase64")
    @classmethod
    def image_is_present(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("image base64 must not be empty")
        return value


def placement_route_enabled() -> bool:
    return (
        os.getenv("AFC_SR1_TS0_CHILD_PLACEMENT_ENABLED", "").strip().lower()
        in _TRUTHY_ENV_VALUES
    )


def _decode_transport_image(value: str) -> bytes:
    encoded = value.strip()
    if encoded.startswith("data:"):
        match = _DATA_IMAGE_PREFIX.match(encoded)
        if match is None:
            raise HTTPException(
                status_code=400,
                detail="image base64 must use data:image/...;base64 when data-URL encoded.",
            )
        encoded = encoded[match.end():]
    if len(encoded) > MAX_BASE64_PAYLOAD_CHARS:
        raise HTTPException(status_code=413, detail="image exceeds the 32 MiB decoded limit.")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=400, detail="image is not valid base64.") from error
    if len(decoded) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image exceeds the 32 MiB decoded limit.")
    return decoded


def execute_ts0_child_placement(request: Ts0ChildPlacementRequest) -> dict:
    parent_bytes = _decode_transport_image(request.parentImageBase64)
    child_bytes = _decode_transport_image(request.childImageBase64)
    exclusion = request.registrationExclusion
    if exclusion.coordinateSpace != COORDINATE_SPACE or exclusion.role != MASK_ROLE:
        raise RuntimeError("validated registration exclusion constants changed")
    return place_ts0_child(
        parent_bytes=parent_bytes,
        child_bytes=child_bytes,
        polygon_norm=exclusion.polygon,
        evidence_label=exclusion.evidenceLabel,
        ts0_lineage=request.ts0Lineage.model_dump(),
        policy_version=request.policyVersion,
    )
