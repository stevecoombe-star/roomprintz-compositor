"""Non-scientific process-gate readiness for AFC-SR1 research routes."""
from __future__ import annotations
import os

from research.afc_sr1_tr2_tile_floor_reader_http import reader_route_enabled
from research.afc_sr1_ts0_child_projective_placement_http import (
    placement_route_enabled,
)

AFC_SR1_READINESS_SCHEMA_VERSION = "afc-sr1-readiness/v2"
AFC_SR1_TS0_GENERATOR_PROFILE = "afc-sr1-tile-grid-scaffold/v1"
AFC_SR1_TS0_REQUESTED_MODEL_ID = "NBP"


def afc_sr1_readiness() -> dict[str, bool | str]:
    """Report process gate state without executing any scientific operation."""
    generator_environment_present = bool(os.getenv("GEMINI_API_KEY", "").strip())
    return {
        "schemaVersion": AFC_SR1_READINESS_SCHEMA_VERSION,
        "readerEnabled": reader_route_enabled(),
        "placementEnabled": placement_route_enabled(),
        "ts0GeneratorReady": generator_environment_present,
        "ts0GeneratorProfile": AFC_SR1_TS0_GENERATOR_PROFILE,
        "ts0RequestedModelId": AFC_SR1_TS0_REQUESTED_MODEL_ID,
    }
