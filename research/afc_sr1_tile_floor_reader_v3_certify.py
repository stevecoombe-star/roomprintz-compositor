"""Explicit A/C/D-only tracked V3 certification; no corpus discovery."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.afc_sr1_tr2_tile_floor_reader_http import (
    TileFloorReaderRequest,
    V3_POLICY_VERSION,
    V3_RESEARCH_PROFILE,
    execute_tile_floor_reader,
)

C_GT0 = 0.7063703325987577
PARITY = {
    "C-RAW": 0.7037731582393056,
    "A-T1": 0.04116740051886033, "A-T2": 0.03623876295467647, "A-T3": 0.04548174342150267,
    "C-T1": 0.6829908429659618, "C-T2": 0.694415664007315, "C-T3": 0.6903928230051386,
    "D-T1": 0.8405945296341184, "D-T2": 0.8502407650518156, "D-T3": 0.8510208987624672,
}
V2 = {
    "C-T1": 0.6767253259089697, "C-T2": 0.6967860417339513, "C-T3": 0.6904606574937419,
    "D-T1": 0.8391650201059466, "D-T2": 0.8368235659537211, "D-T3": 0.8199513029993906,
}
ROWS = {
    "A-RAW": ("d5f9b40ffbe2789d4756d72162b1d4a38d505da67b36e5deb9b2a9e0ec5bb686", [[.10837393593189963,.9249876495993743],[1,.93],[.688,.648],[.356,.648]], "NL"),
    "A-T1": ("d01c445fc6f1ffea785f74c6ca15ff564cb4b7ed20b1f7c7c2554d9de2420a95", [[.10837393593189963,.9249876495993743],[1,.93],[.688,.648],[.356,.648]], "NL"),
    "A-T2": ("e72b690631ec8c51039babb459b899f6b49106539b77ed23e2029bc9da790e7d", [[.10837393593189963,.9249876495993743],[1,.93],[.688,.648],[.356,.648]], "NL"),
    "A-T3": ("db357c9765e81411e936d3425f5ffdefd7e195d588c3343f4b6389c80aae427c", [[.10837393593189963,.9249876495993743],[1,.93],[.688,.648],[.356,.648]], "NL"),
    "C-RAW": ("b7283bb606d09bc7803543bfcbca14aa5f2041cfb91c5eb590d69d167355243f", [[.045,1],[1,.82],[.415,.616],[.077,.694]], "NL"),
    "C-T1": ("93c4c40764c863246cd58976c5bc242689f6dfb0e71cb952248872daac76da92", [[.045,1],[1,.82],[.415,.616],[.077,.694]], "NL"),
    "C-T2": ("66aa1803fa3cb45516ae639d0bce22eec0afa4fd89ca92e64d412901ce930056", [[.045,1],[1,.82],[.415,.616],[.077,.694]], "NL"),
    "C-T3": ("c81e678c1e908d901fea256f56983fcd253e6bb3dbc421c5a492aad5d056d053", [[.045,1],[1,.82],[.415,.616],[.077,.694]], "NL"),
    "D-RAW": ("82d72e591c535f8ce5174b9771b47cd007b004817bca6ccd3170a2e4e6387e96", [[0,.844],[1,.876],[.453,.626],[.041,.716]], "NL"),
    "D-T1": ("38bbcb5c948a8de27b47a4b832cf6f672416403dc5449946eaaeb13df39bd174", [[0,.844],[1,.876],[.453,.626],[.041,.716]], "NL"),
    "D-T2": ("664a3fea09dac5baf123edba871291b1abcb67a9bac5893380c62cb7d77b66c0", [[0,.844],[1,.876],[.453,.626],[.041,.716]], "NL"),
    "D-T3": ("802b0126d0714252acc2fe6f15a7b646cd90e7121900d484bf511b989c0e8558", [[0,.844],[1,.876],[.453,.626],[.041,.716]], "NL"),
}


def image_spec(value: str) -> tuple[str, Path]:
    digest, separator, name = value.partition("=")
    if separator != "=" or digest not in {row[0] for row in ROWS.values()} or not name:
        raise argparse.ArgumentTypeError("--image must be one required SHA256=PATH mapping")
    return digest, Path(name)


def run_bridge(command: str, cwd: Path | None, payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = subprocess.run(
        shlex.split(command),
        cwd=cwd,
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"tracked UI bridge exited {completed.returncode}: {completed.stderr.strip()}"
        )
    result = json.loads(completed.stdout)
    if not isinstance(result, list) or len(result) != len(payload):
        raise RuntimeError("tracked UI bridge returned an invalid response")
    return result


def strip_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: strip_timing(item) for key, item in value.items() if key not in {"elapsedMs", "runtimeMs"}}
    if isinstance(value, list):
        return [strip_timing(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=image_spec, action="append", required=True)
    parser.add_argument("--bridge-command", required=True, help="Explicit command for tracked UI certification bridge.")
    parser.add_argument("--bridge-cwd", type=Path, help="Working directory for the explicit UI bridge command.")
    parser.add_argument("--output", type=Path, help="Optional JSON report output path.")
    args = parser.parse_args()
    supplied = dict(args.image)
    required = {row[0] for row in ROWS.values()}
    if set(supplied) != required or len(args.image) != len(required):
        raise SystemExit("exactly the twelve declared A/C/D SHA256=PATH mappings are required")
    data: dict[str, bytes] = {}
    for digest, path in supplied.items():
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError(f"explicit image SHA mismatch: {path}")
        data[digest] = content

    rows: dict[str, dict[str, Any]] = {}
    for label, (digest, roi, anchor) in ROWS.items():
        image_base64 = base64.b64encode(data[digest]).decode("ascii")
        request = TileFloorReaderRequest.model_validate({
            "researchProfile": V3_RESEARCH_PROFILE,
            "policyVersion": V3_POLICY_VERSION,
            "imageBase64": image_base64,
            "roi": {"coordinateSpace": "source-normalized/v1", "polygon": roi},
        })
        first = execute_tile_floor_reader(request)
        replay = execute_tile_floor_reader(request)
        payload = [{
            "receipt": first,
            "imageBase64": image_base64,
            "roi": {"coordinateSpace": "source-normalized/v1", "polygon": roi},
            "sourcePolygon": [{"x": x, "y": y} for x, y in roi],
            "truncatedAnchor": anchor,
        }]
        bridge_error = None
        bridge_first: dict[str, Any] | None = None
        bridge_replay: dict[str, Any] | None = None
        bridge_ms = 0.0
        try:
            started = time.perf_counter()
            bridge_first = run_bridge(args.bridge_command, args.bridge_cwd, payload)[0]
            bridge_ms = (time.perf_counter() - started) * 1000.0
            bridge_replay = run_bridge(args.bridge_command, args.bridge_cwd, payload)[0]
        except Exception as error:
            bridge_error = f"{type(error).__name__}: {error}"
        handoff = (
            bridge_first.get("projectiveHandoff")
            if isinstance(bridge_first, dict) and bridge_first.get("validationStatus") == "accepted"
            else None
        )
        seam = (
            handoff.get("prior", {}).get("seamT")
            if isinstance(handoff, dict) and handoff.get("status") == "usable"
            else None
        )
        row: dict[str, Any] = {
            "sha256": digest,
            "receipt": first,
            "readerStatus": first["status"],
            "readerReason": first.get("reason"),
            "analysisIdentity": first.get("analysisIdentity"),
            "readerReplayEqual": strip_timing(first) == strip_timing(replay),
            "bridge": bridge_first,
            "bridgeError": bridge_error,
            "bridgeReplayEqual": bridge_first is not None and bridge_first == bridge_replay,
            "projectiveHandoff": handoff,
            "seamT": seam,
            "readerMs": first["elapsedMs"],
            "bridgeMs": bridge_ms,
            "directPathMs": first["elapsedMs"] + bridge_ms,
        }
        row["portParity"] = None if label not in PARITY or seam is None else abs(seam - PARITY[label])
        if label in V2:
            row["v2Comparison"] = {
                "v2SeamT": V2[label],
                "v3SeamT": seam,
                "v3MinusV2": None if seam is None else seam - V2[label],
                "absoluteDelta": None if seam is None else abs(seam - V2[label]),
            }
        rows[label] = row

    tiled = [rows[f"{room}-T{generation}"] for room in "ACD" for generation in range(1, 4)]

    def values(room: str) -> list[float | None]:
        return [rows[f"{room}-T{generation}"]["seamT"] for generation in range(1, 4)]

    ranges = {
        room: max(values(room)) - min(values(room)) if all(value is not None for value in values(room)) else None  # type: ignore[arg-type]
        for room in "ACD"
    }
    c_errors = {
        label: None if rows[label]["seamT"] is None else abs(rows[label]["seamT"] - C_GT0)
        for label in ("C-RAW", "C-T1", "C-T2", "C-T3")
    }
    all_replay = all(row["readerReplayEqual"] and row["bridgeReplayEqual"] for row in rows.values())
    all_port_parity = all(
        rows[label]["portParity"] is not None and rows[label]["portParity"] <= 1e-12
        for label in PARITY
    )
    c_tiled_errors = [c_errors[label] for label in ("C-T1", "C-T2", "C-T3")]
    bars = {
        "cRaw": rows["C-RAW"]["readerStatus"] == "usable" and
            rows["C-RAW"]["seamT"] is not None and c_errors["C-RAW"] <= .025,
        "aRawNoAdvisory": rows["A-RAW"]["seamT"] is None,
        "dRawNoAdvisory": rows["D-RAW"]["seamT"] is None,
        "nineTiledUsable": all(row["seamT"] is not None for row in tiled),
        "cPhysical": all(error is not None and error <= .025 for error in c_tiled_errors) and
            ranges["C"] is not None and ranges["C"] <= .030,
        "aRange": ranges["A"] is not None and ranges["A"] <= .030, "dRange": ranges["D"] is not None and ranges["D"] <= .030,
        "replay": all_replay,
        "portParity": all_port_parity,
    }
    runtime_paths = {f"{label}-direct": row["directPathMs"] for label, row in rows.items() if label.endswith("RAW")}
    for label in (f"{room}-T{generation}" for room in "ACD" for generation in range(1, 4)):
        raw = rows[f"{label[0]}-RAW"]
        tiled_row = rows[label]
        tiled_row["fallbackPathMs"] = raw["directPathMs"] + tiled_row["directPathMs"]
        runtime_paths[f"{label}-fallback"] = tiled_row["fallbackPathMs"]
    bars["runtime"] = all(value < 15_000 for value in runtime_paths.values())
    runtime = {
        "pathsMs": runtime_paths,
        "min": min(runtime_paths.values()),
        "median": statistics.median(runtime_paths.values()),
        "max": max(runtime_paths.values()),
        "worstPath": max(runtime_paths, key=runtime_paths.get),
        "included": ["RAW V3 adapter/reader", "UI receipt validation", "TR0", "Track 1a",
                     "fallback TILED V3 adapter/reader when applicable"],
        "excluded": ["fixed-byte SHA verification", "replay executions", "NBP/TS0 generation"],
    }
    report = {
        "status": "certified" if all(bars.values()) else "failed",
        "corpus": [{"label": label, "sha256": row[0]} for label, row in ROWS.items()],
        "rows": rows,
        "bars": bars,
        "cGt0": C_GT0,
        "cErrors": c_errors,
        "ranges": ranges,
        "runtimeMs": runtime,
        "historicalV2Comparisons": {
            label: rows[label]["v2Comparison"] for label in V2
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(json.dumps({
            "status": report["status"],
            "bars": bars,
            "seamTs": {label: row["seamT"] for label, row in rows.items()},
            "output": str(args.output),
        }, indent=2, sort_keys=True))
    else:
        print(encoded)
    if report["status"] != "certified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
