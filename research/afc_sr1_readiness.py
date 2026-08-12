"""Non-scientific process-gate readiness for AFC-SR1 research routes."""
from __future__ import annotations

from research.afc_sr1_tr2_tile_floor_reader_http import reader_route_enabled
from research.afc_sr1_ts0_child_projective_placement_http import (
    placement_route_enabled,
)

AFC_SR1_READINESS_SCHEMA_VERSION = "afc-sr1-readiness/v1"


def afc_sr1_readiness() -> dict[str, bool | str]:
    """Report process gate state without executing any scientific operation."""
    return {
        "schemaVersion": AFC_SR1_READINESS_SCHEMA_VERSION,
        "readerEnabled": reader_route_enabled(),
        "placementEnabled": placement_route_enabled(),
    }
