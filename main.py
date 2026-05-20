import os
import io
import base64
import hashlib
import json
import math
import inspect
import time
from contextvars import ContextVar
from datetime import datetime
from uuid import UUID, uuid4
from urllib.parse import quote, urlparse
from typing import Any, Literal, Optional, Tuple, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator
from PIL import Image, ImageChops, ImageFilter
import requests
try:
    from PIL import ImageDraw
    _HAS_IMAGE_DRAW = True
except Exception:
    ImageDraw = None
    _HAS_IMAGE_DRAW = False
try:
    from PIL import ImageFont
except Exception:
    ImageFont = None

from google import genai
from google.genai import types

# ---------- GEMINI / NANO BANANA CONFIG ----------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set. "
        "Run `export GEMINI_API_KEY=...` before starting the server."
    )

client = genai.Client(api_key=GEMINI_API_KEY)

# Default model: Nano Banana Pro
DEFAULT_MODEL_NAME = "gemini-3-pro-image-preview"
MODEL_VERSION_ALIASES: Dict[str, str] = {
    # Preferred Vibode shorthand contract.
    "nbp": "gemini-3-pro-image-preview",
    "nb2": "gemini-3.1-flash-image-preview",
    # Backward-compatible raw model IDs.
    "gemini-3-pro-image-preview": "gemini-3-pro-image-preview",
    "gemini-3.1-flash-image-preview": "gemini-3.1-flash-image-preview",
    # Legacy aliases kept for compatibility.
    "gemini-3": "gemini-3-pro-image-preview",
    "gemini-3-pro": "gemini-3-pro-image-preview",
    "gemini-2.5": "gemini-2.5-flash-image",
    "gemini-2.5-flash-image": "gemini-2.5-flash-image",
}

# Toggle prompt logging with env var: DEBUG_ROOMPRINTZ_PROMPT=1
DEBUG_ROOMPRINTZ_PROMPT = os.getenv("DEBUG_ROOMPRINTZ_PROMPT", "0") == "1"

# Strict Stage 3 prompt dump flag (route-level, exact prompt text).
DEBUG_ROOMPRINTZ_STAGE3_PROMPT = os.getenv("DEBUG_ROOMPRINTZ_STAGE3_PROMPT", "0") == "1"

# Gate Gemini prompt logging behind explicit env opt-in.
VIBODE_LOG_GEMINI_PROMPTS = (
    os.getenv("VIBODE_LOG_GEMINI_PROMPTS", "false").strip().lower() in ("1", "true", "yes", "on")
)

# Toggle ratio debug return
DEBUG_ROOMPRINTZ_RATIO = os.getenv("DEBUG_ROOMPRINTZ_RATIO", "0") == "1"
# Toggle ingest checkpoint image metrics logging (off by default)
DEBUG_USER_SKU_INGEST_METRICS = os.getenv("DEBUG_USER_SKU_INGEST_METRICS", "0") == "1"

# Optional: cap input size to keep cost + latency down (resize down only; never upscale)
# Set to "" to disable.
MAX_INPUT_LONG_EDGE = os.getenv("ROOMPRINTZ_MAX_INPUT_LONG_EDGE", "2048").strip()
MAX_INPUT_LONG_EDGE_INT = (
    int(MAX_INPUT_LONG_EDGE) if MAX_INPUT_LONG_EDGE.isdigit() else None
)
STAGE_ROOM_MAX_REFERENCE_IMAGES = 8

# ✅ CHANGE: beta testing wants ALL ratios for Gemini 2.5 Flash.
# We keep the env var, but default it to "1" so Flash is NOT forced to 1:1.
ALLOW_FLASH_NON_SQUARE = os.getenv("ROOMPRINTZ_ALLOW_FLASH_NON_SQUARE", "1") == "1"

# User SKU ingest config
USER_SKU_MAX_INPUT_BYTES = 12 * 1024 * 1024  # 12 MB
USER_SKU_NORMALIZED_MAX_DIM = 1536
USER_SKU_NORMALIZED_PADDING_RATIO = 0.03
USER_SKU_FORCED_PREVIEW_BG_RGB = (237, 237, 237)
USER_SKU_INGEST_TIMEOUT_SECONDS = 10.0
SUPABASE_SIGNED_URL_TTL_SECONDS = 7 * 24 * 60 * 60
SUPABASE_STORAGE_UPLOAD_TIMEOUT_SECONDS = 20.0
SUPABASE_USAGE_WRITE_TIMEOUT_SECONDS = 2.5
SUPABASE_URL = (os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or "").strip()
SUPABASE_SERVICE_KEY = (os.getenv("SUPABASE_SERVICE_KEY") or "").strip()
SUPABASE_SERVICE_ROLE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
SUPABASE_STORAGE_BUCKET = (
    os.getenv("SUPABASE_STORAGE_BUCKET")
    or os.getenv("NEXT_PUBLIC_SUPABASE_STORAGE_BUCKET")
    or ""
).strip()
_SUPABASE_STORAGE_BUCKET_CACHE: Optional[str] = None

USER_SKU_BG_REMOVAL_PROMPT = (
    "Isolate the product and replace the entire background with a flat, uniform, solid light grey color (#F2F2F2).\n"
    "Treat screenshot/clipboard/card captures as source only and isolate only the real product object.\n"
    "Remove and ignore any border, frame, card, clipboard shape, source UI, or screenshot container shadow.\n"
    "The background must be a single solid color.\n"
    "No gradients.\n"
    "No shadows.\n"
    "No floor.\n"
    "No texture.\n"
    "No checkerboard pattern.\n"
    "Do not simulate transparency.\n"
    "Preserve the product's exact shape, scale, and materials.\n"
    "Do not brighten, whiten, normalize, or globally increase contrast/exposure on the product.\n"
    "Preserve highlight detail and subtle near-white tonal gradients; avoid flattening light areas into pure white.\n"
    "Retain material texture and fine detail (fabric texture, stitching, grain, and surface detail), especially on pale/neutral products.\n"
    "Keep clean but natural edges and clear separation from the background; do not let light object edges melt or fade.\n"
    "Avoid aggressive cleanup, beautification, or stylization; prefer faithful reproduction of the original product appearance.\n"
    "Include a small margin around the product."
)
USER_SKU_CLIPBOARD_ISOLATION_PROMPT = (
    "Extract only the real furniture/product object from this image.\n"
    "The source may be a screenshot, card, clipboard, or UI capture.\n"
    "Do not keep or recreate any outer rectangle, border, frame, card, clipboard, UI, source shadow, or filler background.\n"
    "Return only the product and preserve its true silhouette.\n"
    "Prefer transparent background; if not possible, use flat #F2F2F2 background."
)
USER_SKU_FOREGROUND_COLOR_DISTANCE_THRESHOLD = 24
USER_SKU_MIN_FOREGROUND_AREA_RATIO = 0.008
USER_SKU_MAX_FOREGROUND_AREA_RATIO = 0.96
USER_SKU_RECT_FRAME_BBOX_COVER_RATIO = 0.88
USER_SKU_RECT_FRAME_FILL_RATIO = 0.84
USER_SKU_BORDER_HEAVY_EDGE_RATIO = 0.20
USER_SKU_UNIFORM_BG_DOMINANCE_RATIO = 0.48


def resolve_model_name(model_version: Optional[str]) -> str:
    """
    Map modelVersion aliases from the frontend into a concrete Gemini model ID.

    Default/fallback remains Nano Banana Pro.
    """
    if not model_version or model_version.strip() == "":
        return DEFAULT_MODEL_NAME

    v = model_version.strip().lower()
    return MODEL_VERSION_ALIASES.get(v, model_version)


# ---------- ASPECT RATIO NORMALIZATION ----------

AspectRatio = Literal[
    "auto",
    "1:1",
    "3:2",
    "2:3",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
]
Stage4StyleMode = Literal[
    "style_room",
    "accessories",
    "wall_art",
    "shelves",
    "curtains",
    "ceiling_light",
]

RATIO_MAP: Dict[str, float] = {
    "1:1": 1.0,
    "3:2": 3 / 2,
    "2:3": 2 / 3,
    "3:4": 3 / 4,
    "4:3": 4 / 3,
    "4:5": 4 / 5,
    "5:4": 5 / 4,
    "9:16": 9 / 16,
    "16:9": 16 / 9,
    "21:9": 21 / 9,
}

SUPPORTED_RATIOS_ORDERED = ["1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]

_REQUEST_ID_CTX: ContextVar[str] = ContextVar("roomprintz_request_id", default="-")
_REQUEST_ID_HEADER_MAX_LEN = 100
_REQUEST_ID_ALLOWED_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-:"
)
_VIBODE_CORRELATION_ID_MAX_LEN = 140
_VIBODE_CORRELATION_ALLOWED_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-:"
)
_OPERATION_ID_CTX: ContextVar[Optional[str]] = ContextVar("roomprintz_operation_id", default=None)
_ATTEMPT_ID_CTX: ContextVar[Optional[str]] = ContextVar("roomprintz_attempt_id", default=None)
_ROUTE_PATH_CTX: ContextVar[Optional[str]] = ContextVar("roomprintz_route_path", default=None)
_PROVIDER_ATTEMPT_SEQ_CTX: ContextVar[int] = ContextVar("roomprintz_provider_attempt_seq", default=0)
_USER_ID_CTX: ContextVar[Optional[str]] = ContextVar("roomprintz_user_id", default=None)
_USER_EMAIL_CTX: ContextVar[Optional[str]] = ContextVar("roomprintz_user_email", default=None)
_ROOM_ID_CTX: ContextVar[Optional[str]] = ContextVar("roomprintz_room_id", default=None)
_VERSION_ID_CTX: ContextVar[Optional[str]] = ContextVar("roomprintz_version_id", default=None)
_ASSET_ID_CTX: ContextVar[Optional[str]] = ContextVar("roomprintz_asset_id", default=None)
_WORKFLOW_TYPE_CTX: ContextVar[Optional[str]] = ContextVar("roomprintz_workflow_type", default=None)
_ACTION_TYPE_CTX: ContextVar[Optional[str]] = ContextVar("roomprintz_action_type", default=None)
_SOURCE_TRIGGER_CTX: ContextVar[Optional[str]] = ContextVar("roomprintz_source_trigger", default=None)
VIBODE_REQUEST_ID_HEADER = "x-vibode-request-id"
VIBODE_OPERATION_ID_HEADER = "x-vibode-operation-id"
VIBODE_ATTEMPT_ID_HEADER = "x-vibode-attempt-id"
VIBODE_USER_ID_HEADER = "x-vibode-user-id"
VIBODE_USER_EMAIL_HEADER = "x-vibode-user-email"
VIBODE_ROOM_ID_HEADER = "x-vibode-room-id"
VIBODE_VERSION_ID_HEADER = "x-vibode-version-id"
VIBODE_ASSET_ID_HEADER = "x-vibode-asset-id"
VIBODE_WORKFLOW_TYPE_HEADER = "x-vibode-workflow-type"
VIBODE_ACTION_TYPE_HEADER = "x-vibode-action-type"
VIBODE_SOURCE_TRIGGER_HEADER = "x-vibode-source-trigger"


def get_request_id() -> str:
    return _REQUEST_ID_CTX.get()


def get_operation_id() -> Optional[str]:
    return _OPERATION_ID_CTX.get()


def get_attempt_id() -> Optional[str]:
    return _ATTEMPT_ID_CTX.get()


def get_route_path() -> Optional[str]:
    return _ROUTE_PATH_CTX.get()


def get_user_id() -> Optional[str]:
    return _USER_ID_CTX.get()


def get_user_email() -> Optional[str]:
    return _USER_EMAIL_CTX.get()


def get_room_id() -> Optional[str]:
    return _ROOM_ID_CTX.get()


def get_version_id() -> Optional[str]:
    return _VERSION_ID_CTX.get()


def get_asset_id() -> Optional[str]:
    return _ASSET_ID_CTX.get()


def get_workflow_type() -> Optional[str]:
    return _WORKFLOW_TYPE_CTX.get()


def get_action_type() -> Optional[str]:
    return _ACTION_TYPE_CTX.get()


def get_source_trigger() -> Optional[str]:
    return _SOURCE_TRIGGER_CTX.get()


def _next_provider_attempt_id() -> str:
    inbound_attempt_id = get_attempt_id()
    if inbound_attempt_id:
        next_seq = _PROVIDER_ATTEMPT_SEQ_CTX.get() + 1
        _PROVIDER_ATTEMPT_SEQ_CTX.set(next_seq)
        if next_seq == 1:
            return inbound_attempt_id
        return f"{inbound_attempt_id}:{next_seq}"
    return uuid4().hex


def _sanitize_inbound_request_id(header_value: Optional[str]) -> Optional[str]:
    if not header_value:
        return None

    candidate = header_value.strip()
    if not candidate:
        return None
    if len(candidate) > _REQUEST_ID_HEADER_MAX_LEN:
        return None
    if any(ch not in _REQUEST_ID_ALLOWED_CHARS for ch in candidate):
        return None

    return candidate


def _sanitize_vibode_correlation_id(header_value: Optional[str]) -> Optional[str]:
    if not header_value:
        return None
    candidate = str(header_value).strip()
    if not candidate:
        return None
    if len(candidate) > _VIBODE_CORRELATION_ID_MAX_LEN:
        return None
    if any(ch not in _VIBODE_CORRELATION_ALLOWED_CHARS for ch in candidate):
        return None
    return candidate


def _sanitize_optional_header_value(header_value: Optional[str]) -> Optional[str]:
    if header_value is None:
        return None
    candidate = str(header_value).strip()
    if not candidate:
        return None
    return candidate


def _sanitize_uuid_header_value(header_value: Optional[str]) -> Optional[str]:
    candidate = _sanitize_optional_header_value(header_value)
    if not candidate:
        return None
    try:
        return str(UUID(candidate))
    except Exception:
        return None


def _log_value(value: Any) -> str:
    if isinstance(value, str):
        return value.replace("\n", "\\n")
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=True, default=str)
    return str(value)


def log_event(event: str, **fields: Any) -> None:
    parts = [f"event={event}", f"request_id={get_request_id()}"]
    for key in sorted(fields.keys()):
        parts.append(f"{key}={_log_value(fields[key])}")
    print("[roomprintz]", " ".join(parts))


def _log_gemini_prompt_debug(*, function_name: str, model_name: str, prompt: str) -> None:
    if not VIBODE_LOG_GEMINI_PROMPTS:
        return
    route = get_route_path() or "(unknown)"
    request_id = get_request_id()
    print(
        "[roomprintz] "
        f"event=gemini_prompt_debug route={route} request_id={request_id} "
        f"model_name={model_name} function={function_name} prompt_chars={len(prompt)}"
    )
    print("----- BEGIN GEMINI PROMPT -----")
    print(prompt)
    print("----- END GEMINI PROMPT -----")


PASTE_TO_PLACE_JOB_ID_HEADER = "x-vibode-paste-to-place-job-id"
PASTE_TO_PLACE_SCOPE_ID_HEADER = "x-vibode-paste-to-place-scope-id"
PASTE_TO_PLACE_CANCELLED_CODE = "PASTE_TO_PLACE_CANCELLED"
PASTE_TO_PLACE_JOB_TTL_SECONDS = 10 * 60
PASTE_TO_PLACE_MAX_CANCELLED_IDS_PER_SCOPE = 128
# Best-effort in-memory cancellation; it may not stop an already-started
# provider generation request. Full provider request abort is a future pass.
_PASTE_TO_PLACE_JOBS_BY_SCOPE: Dict[str, Dict[str, Any]] = {}


def _cleanup_expired_paste_to_place_scopes(now: Optional[float] = None) -> None:
    now_ts = time.time() if now is None else now
    expired_scope_ids = [
        scope_id
        for scope_id, state in _PASTE_TO_PLACE_JOBS_BY_SCOPE.items()
        if now_ts - float(state.get("updatedAt") or 0) > PASTE_TO_PLACE_JOB_TTL_SECONDS
    ]
    for scope_id in expired_scope_ids:
        _PASTE_TO_PLACE_JOBS_BY_SCOPE.pop(scope_id, None)


def _extract_operation_id_from_job_id(scope_id: str, job_id: str) -> Optional[int]:
    prefix = f"{scope_id}:"
    if not job_id.startswith(prefix):
        return None
    operation_id = job_id[len(prefix) :].strip()
    if not operation_id:
        return None
    try:
        return int(operation_id)
    except (TypeError, ValueError):
        return None


def _snapshot_paste_to_place_scope_state(scope_id: str, job_id: Optional[str] = None) -> Dict[str, Any]:
    _cleanup_expired_paste_to_place_scopes()
    scope_state = _PASTE_TO_PLACE_JOBS_BY_SCOPE.get(scope_id)
    if not scope_state:
        return {
            "exists": False,
            "latestJobId": None,
            "latestOperationId": None,
            "cancelledCount": 0,
            "jobIsCancelled": False if job_id else None,
        }
    cancelled_job_ids = scope_state.get("cancelledJobIds", set())
    if not isinstance(cancelled_job_ids, set):
        cancelled_job_ids = set(cancelled_job_ids or [])
    return {
        "exists": True,
        "latestJobId": scope_state.get("latestJobId"),
        "latestOperationId": scope_state.get("latestOperationId"),
        "cancelledCount": len(cancelled_job_ids),
        "jobIsCancelled": (job_id in cancelled_job_ids) if job_id else None,
    }


def mark_latest(scope_id: str, job_id: str) -> None:
    _cleanup_expired_paste_to_place_scopes()
    scope_state = _PASTE_TO_PLACE_JOBS_BY_SCOPE.setdefault(
        scope_id,
        {"latestJobId": None, "latestOperationId": None, "cancelledJobIds": set(), "updatedAt": time.time()},
    )
    operation_id = _extract_operation_id_from_job_id(scope_id, job_id)
    latest_operation_id = scope_state.get("latestOperationId")
    # Keep latest monotonic because older queued jobs may arrive after newer ones.
    if operation_id is not None and (latest_operation_id is None or operation_id >= latest_operation_id):
        scope_state["latestJobId"] = job_id
        scope_state["latestOperationId"] = operation_id
    elif operation_id is None:
        scope_state["latestJobId"] = job_id
    scope_state["updatedAt"] = time.time()


def mark_cancelled(scope_id: str, job_id: str) -> None:
    _cleanup_expired_paste_to_place_scopes()
    scope_state = _PASTE_TO_PLACE_JOBS_BY_SCOPE.setdefault(
        scope_id,
        {"latestJobId": None, "latestOperationId": None, "cancelledJobIds": set(), "updatedAt": time.time()},
    )
    cancelled_job_ids = scope_state.setdefault("cancelledJobIds", set())
    cancelled_job_ids.add(job_id)
    while len(cancelled_job_ids) > PASTE_TO_PLACE_MAX_CANCELLED_IDS_PER_SCOPE:
        removable_job_ids = [cancelled_id for cancelled_id in cancelled_job_ids if cancelled_id != job_id]
        cancelled_job_ids.remove(removable_job_ids[0] if removable_job_ids else next(iter(cancelled_job_ids)))
    scope_state["updatedAt"] = time.time()


def get_state(scope_id: str, job_id: str) -> Literal["active", "stale", "cancelled", "unknown"]:
    _cleanup_expired_paste_to_place_scopes()
    scope_state = _PASTE_TO_PLACE_JOBS_BY_SCOPE.get(scope_id)
    if not scope_state:
        return "unknown"
    scope_state["updatedAt"] = time.time()
    if job_id in scope_state.get("cancelledJobIds", set()):
        return "cancelled"
    latest_job_id = scope_state.get("latestJobId")
    if not latest_job_id:
        return "unknown"
    if latest_job_id != job_id:
        return "stale"
    return "active"


def _extract_paste_to_place_control(
    request: Request,
    route: str,
    *,
    log_missing_headers: bool = False,
) -> Optional[Dict[str, str]]:
    job_id = (request.headers.get(PASTE_TO_PLACE_JOB_ID_HEADER) or "").strip()
    scope_id = (request.headers.get(PASTE_TO_PLACE_SCOPE_ID_HEADER) or "").strip()
    if not job_id or not scope_id:
        if log_missing_headers:
            log_event(
                "paste_to_place_headers_missing",
                route=route,
                has_scope_id_header=bool(scope_id),
                has_job_id_header=bool(job_id),
            )
        return None
    operation_id = _extract_operation_id_from_job_id(scope_id, job_id)
    before_mark_latest = _snapshot_paste_to_place_scope_state(scope_id, job_id)
    mark_latest(scope_id, job_id)
    after_mark_latest = _snapshot_paste_to_place_scope_state(scope_id, job_id)
    state_after_mark_latest = get_state(scope_id, job_id)
    log_event(
        "paste_to_place_request_arrived",
        route=route,
        scope_id=scope_id,
        job_id=job_id,
        parsed_operation_id=operation_id,
        latest_job_id_before=before_mark_latest["latestJobId"],
        latest_operation_id_before=before_mark_latest["latestOperationId"],
        latest_job_id_after=after_mark_latest["latestJobId"],
        latest_operation_id_after=after_mark_latest["latestOperationId"],
        state_after_mark_latest=state_after_mark_latest,
    )
    return {"scopeId": scope_id, "jobId": job_id}


def _ensure_paste_to_place_job_active(
    control: Optional[Dict[str, str]],
    route: str,
    checkpoint: str,
) -> Optional[JSONResponse]:
    if not control:
        return None
    scope_id = control["scopeId"]
    job_id = control["jobId"]
    state = get_state(scope_id, job_id)
    log_event(
        "paste_to_place_checkpoint",
        route=route,
        checkpoint=checkpoint,
        scope_id=scope_id,
        job_id=job_id,
        state=state,
    )
    if state not in ("stale", "cancelled"):
        return None
    log_event(
        "paste_to_place_job_early_exit",
        route=route,
        checkpoint=checkpoint,
        scope_id=scope_id,
        job_id=job_id,
        state=state,
    )
    return JSONResponse(
        status_code=409,
        content={
            "code": PASTE_TO_PLACE_CANCELLED_CODE,
            "cancelled": True,
            "reason": state,
        },
    )


def resolve_model_name_for_route(route: str, model_version: Optional[str]) -> str:
    model_name = resolve_model_name(model_version)
    log_event(
        "model_version_resolved",
        route=route,
        model_version=model_version,
        model_name=model_name,
    )
    return model_name


def summarize_prompt(prompt: str) -> Dict[str, Any]:
    prompt_text = prompt or ""
    first_line = next((line.strip() for line in prompt_text.splitlines() if line.strip()), "")
    if len(first_line) > 120:
        first_line = f"{first_line[:117]}..."
    return {
        "prompt_len": len(prompt_text),
        "prompt_hash": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12],
        "prompt_first_line": first_line,
    }


def _safe_open_image(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        print("[_safe_open_image] Failed to open input image:", e)
        raise


def choose_closest_aspect_ratio(width: int, height: int) -> str:
    """Pick the closest preset ratio to the uploaded image's native ratio."""
    if width <= 0 or height <= 0:
        return "4:3"
    native = width / height

    best = "4:3"
    best_dist = float("inf")
    for k in SUPPORTED_RATIOS_ORDERED:
        dist = abs(native - RATIO_MAP[k])
        if dist < best_dist:
            best_dist = dist
            best = k
    return best


def resize_down_if_needed(img: Image.Image, max_long_edge: Optional[int]) -> Image.Image:
    """Resize down so the longer edge <= max_long_edge, preserving aspect ratio. Never upscale."""
    if not max_long_edge:
        return img

    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return img

    scale = max_long_edge / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), resample=Image.LANCZOS)


def crop_to_aspect_ratio_fill(img: Image.Image, target_ratio: float) -> Image.Image:
    """
    Fill/Crop normalization with an upward bias:
    - If we need to crop vertically (image too tall), we bias crop lower (keep more top).
    - If we need to crop horizontally (image too wide), we center crop.
    """
    w, h = img.size
    if w <= 0 or h <= 0:
        return img

    current_ratio = w / h

    if abs(current_ratio - target_ratio) < 1e-6:
        return img

    if current_ratio > target_ratio:
        # Too wide -> crop width (center)
        new_w = int(round(h * target_ratio))
        new_w = min(new_w, w)
        left = (w - new_w) // 2
        right = left + new_w
        return img.crop((left, 0, right, h))

    # Too tall -> crop height (upward bias)
    new_h = int(round(w / target_ratio))
    new_h = min(new_h, h)

    bias = 0.65
    max_top = h - new_h
    top = int(round(max_top * (1.0 - bias)))
    top = max(0, min(top, max_top))
    bottom = top + new_h
    return img.crop((0, top, w, bottom))


def image_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def normalize_image_bytes_for_ratio(
    image_bytes: bytes,
    requested_ratio: Optional[str],
    model_name: str,
) -> Tuple[bytes, str]:
    """
    Returns (normalized_png_bytes, applied_ratio_str).

    Fresh upload behavior:
    - requested_ratio None/"auto" => choose closest preset from uploaded image ratio.
    - Fill/Crop to that ratio with upward bias.
    - Optionally resize down.
    - For gemini-2.5-flash-image, we no longer force 1:1 (beta wants all ratios),
      unless ROOMPRINTZ_ALLOW_FLASH_NON_SQUARE is explicitly set to 0.
    """
    img = _safe_open_image(image_bytes)
    w, h = img.size

    ratio_choice = (requested_ratio or "auto").strip().lower()

    if ratio_choice == "auto":
        chosen = choose_closest_aspect_ratio(w, h)
    else:
        normalized = ratio_choice.replace("x", ":")
        if normalized not in RATIO_MAP:
            print(
                f"[normalize_image_bytes_for_ratio] Unknown aspectRatio '{requested_ratio}', defaulting to auto."
            )
            chosen = choose_closest_aspect_ratio(w, h)
        else:
            chosen = normalized

    # Guardrail (opt-out only now):
    if (
        model_name.strip().lower() == "gemini-2.5-flash-image"
        and chosen != "1:1"
        and not ALLOW_FLASH_NON_SQUARE
    ):
        print(
            "[normalize_image_bytes_for_ratio] Forcing aspect ratio to 1:1 for gemini-2.5-flash-image (ROOMPRINTZ_ALLOW_FLASH_NON_SQUARE=0)."
        )
        chosen = "1:1"

    log_event(
        "aspect_ratio_policy",
        continuation=False,
        source_canvas_width=w,
        source_canvas_height=h,
        mapped_aspect_ratio=chosen,
        aspect_ratio_applied=True,
    )

    target_ratio = RATIO_MAP[chosen]
    cropped = crop_to_aspect_ratio_fill(img, target_ratio)
    cropped = resize_down_if_needed(cropped, MAX_INPUT_LONG_EDGE_INT)

    return image_to_png_bytes(cropped), chosen


def prepare_passthrough_png_bytes(image_bytes: bytes) -> bytes:
    """
    Continuation mode:
    - Do NOT normalize aspect ratio
    - Do NOT crop
    - Do optionally resize down (uniform) to cap cost/latency
    - Always send PNG
    """
    img = _safe_open_image(image_bytes)
    img = resize_down_if_needed(img, MAX_INPUT_LONG_EDGE_INT)
    return image_to_png_bytes(img)


def log_continuation_aspect_ratio_omitted(route: str, **fields: Any) -> None:
    log_event(
        "aspect_ratio_policy",
        route=route,
        continuation=True,
        aspect_ratio_applied=False,
        **fields,
    )


# ---------- PROMPT BUILDING (ROOMPRINTZ ENGINE) ----------

BASE_ROOMPRINTZ_INSTRUCTIONS = """
You are a professional real-estate photo editor for MLS listings.

General rules:
- Preserve the room's geometry, perspective, and camera angle.
- Keep windows, doors, walls, floors, ceilings, and built-in elements consistent.
- Do not add new furniture or decor unless explicitly asked.
- Do not remove structural elements (walls, windows, doors).
- Keep edits subtle, photorealistic, and suitable for real-estate marketing.
- Do not add any text, logos, or watermarks.
"""

ENHANCE_FRAGMENT = """
Step 1 — Enhance photo quality:
- Correct white balance so the scene looks neutral and natural.
- Optimize exposure: recover highlights, open up shadows, and maintain good contrast.
- Improve dynamic range for a bright, inviting interior without looking HDR or fake.
- Increase sharpness and clarity slightly so details are crisp but not oversharpened.
- Reduce noise or grain, especially in darker areas.
- Keep the overall style realistic and suitable for real-estate MLS listings.
"""

DECLUTTER_FRAGMENT = """
Step 2 — Declutter and clean the room:
- Remove small personal items, clutter, and mess from surfaces and floors.
- Examples: toys, clothes, laundry baskets, trash, cables, countertop clutter, small decor that feels busy.
- Keep key furniture pieces that define the room (sofa, dining table, bed, side tables, TV console).
- Keep built-in fixtures and major appliances.
- Do NOT remove walls, windows, doors, radiators, or major built-in cabinetry.
- After cleanup, the room should feel tidy, neutral, and ready to photograph for a listing.
"""

REPAIR_FRAGMENT = """
Step 3 — Repair visible damage:
- Fix holes, cracks, dents, stains, scuffs, and peeling paint on walls, ceilings, and floors.
- Match the original material and texture (e.g., drywall, hardwood, tile, carpet).
- Keep existing patterns and seams consistent.
- Do not change the overall color or style of surfaces; just repair them so they look well-maintained.
"""

EMPTY_ROOM_FRAGMENT = """
Step 4 — Empty the room:
- Remove all movable furniture and decor items from the room.
- Remove sofas, chairs, tables, lamps, rugs, wall art, small decor, and personal items.
- Keep only the fixed architectural shell: walls, ceilings, floors, windows, doors, built-in cabinetry, and radiators.
- The result should be a completely empty but clean room shell, ready for virtual staging or inspection.
"""

RENOVATE_ROOM_FRAGMENT = """
Step 5 — Renovate the room finishes:
- Update worn or outdated finishes so the room looks freshly renovated.
- You may:
  - Repaint walls and ceilings in a clean, modern, neutral color.
  - Upgrade flooring to a high-quality, contemporary material appropriate to the room (e.g., hardwood, tile, or modern carpet).
  - Refresh built-in cabinetry, trim, and doors so they look new and well-maintained.
- Preserve the room's layout, window positions, and overall function (e.g., keep it clearly a bedroom, living room, or kitchen).
- Keep the style broadly appealing for real-estate buyers, not overly themed.
"""

REPAINT_WALLS_FRAGMENT = """
Repaint walls and ceilings:
- Repaint all walls and ceilings in a clean, modern, neutral off-white suitable for real-estate listings.
- Keep trim, doors, and windows crisp and clean; update their paint as needed but keep them neutral.
- Do not change the room layout or remove architectural details; only update the paint.
"""

FLOORING_CARPET_FRAGMENT = """
Change flooring to carpet:
- Replace existing flooring with a soft, neutral, medium-light carpet that works well for real-estate photos.
- Keep floor level, perspective, and room dimensions the same; only change the material and texture.
"""

FLOORING_HARDWOOD_FRAGMENT = """
Change flooring to hardwood:
- Replace existing flooring with light, natural hardwood planks.
- Planks should be realistic, with subtle grain and consistent direction across the room.
- Keep floor level, perspective, and transitions consistent with the original layout.
"""

FLOORING_TILE_FRAGMENT = """
Change flooring to tile:
- Replace existing flooring with large-format, modern, neutral floor tiles.
- Keep grout lines subtle and evenly spaced, aligned with the room's perspective.
- Preserve thresholds and transitions to other rooms.
"""

ROOM_TYPE_HINTS = {
    "living-room": "This is a living room / lounge. It must clearly remain a living room with seating and social area, not a bedroom.",
    "family-room": "This is a family room / den. It should remain a casual, comfortable gathering space with seating, not a bedroom.",
    "bedroom": "This is a bedroom. It must clearly remain a bedroom with a bed as the primary focal point, not a living or dining room.",
    "kitchen": "This is a kitchen. Keep it clearly a kitchen with cabinetry, countertops, appliances, and do not convert it into another room type.",
    "bathroom": "This is a bathroom. It must remain a bathroom with fixtures like sink, toilet, and/or shower or tub.",
    "dining-room": "This is a dining room. It should remain a dining room with a dining table as a main element, not a bedroom or living room.",
    "office": "This is a home office / study. It must remain an office space, not a bedroom or living room.",
    "office-den": "This is a home office / den. It must remain an office / den space, not a bedroom or living room.",
    "other": "This room has a specific existing function. Preserve that function and do not transform it into a different type of room.",
}

STYLE_PROMPTS = {
    "modern-luxury": (
        " Modern luxury style, high-end finishes, neutral palette with warm whites, "
        "soft grays, brushed brass / gold accents, marble textures, curated statement lighting, "
        "large comfortable sectional or sofa, designer coffee table, subtle art, and layered textures."
    ),
    "japandi": (
        " Japandi style, calm, minimal, and warm, light woods, low-profile furniture, "
        "soft textiles, neutral tones, no clutter, an emphasis on negative space and tranquility."
    ),
    "scandinavian": (
        " Scandinavian minimalism, bright and airy, white walls, light oak floors, "
        "functional furniture, cozy textiles, simple art, plants for a subtle pop of green."
    ),
    "coastal": (
        " Coastal bright style, beach-adjacent, soft blues and greens, white and sand tones, "
        "light woods, woven textures, relaxed but elegant decor, lots of light and openness."
    ),
    "urban-loft": (
        " Urban loft style, industrial elements like exposed brick or concrete, "
        "black metal accents, modern furniture with clean lines, dramatic lighting, and bold art."
    ),
    "farmhouse": (
        " Modern farmhouse chic, warm and rustic, wood textures, whites and creams, "
        "large comfortable sofa, vintage-inspired decor, but clean and not cluttered."
    ),
}


def build_roomprintz_prompt(
    enhance_photo: bool,
    cleanup_room: bool,
    repair_damage: bool,
    empty_room: bool,
    renovate_room: bool,
    repaint_walls: bool,
    flooring_preset: Optional[str],
    style_id: Optional[str] = None,
    room_type: Optional[str] = None,
) -> str:
    fragments = [BASE_ROOMPRINTZ_INSTRUCTIONS.strip()]

    if room_type:
        key = room_type.strip().lower()
        hint = ROOM_TYPE_HINTS.get(key) or (
            "This room has a specific existing function. Preserve that function and "
            "do not convert it into a different type of room."
        )
        fragments.append(
            f"Room type context:\n- {hint}\n- All edits must keep the room clearly consistent with this function."
        )

    fragments.append(
        "You are given a single interior room photo. Edit this photo in-place according to the steps below."
    )

    if enhance_photo:
        fragments.append(ENHANCE_FRAGMENT.strip())
    if cleanup_room:
        fragments.append(DECLUTTER_FRAGMENT.strip())
    if repair_damage:
        fragments.append(REPAIR_FRAGMENT.strip())
    if empty_room:
        fragments.append(EMPTY_ROOM_FRAGMENT.strip())
    if renovate_room:
        fragments.append(RENOVATE_ROOM_FRAGMENT.strip())
    if repaint_walls:
        fragments.append(REPAINT_WALLS_FRAGMENT.strip())

    if flooring_preset:
        preset = flooring_preset.lower()
        if preset == "carpet":
            fragments.append(FLOORING_CARPET_FRAGMENT.strip())
        elif preset == "hardwood":
            fragments.append(FLOORING_HARDWOOD_FRAGMENT.strip())
        elif preset == "tile":
            fragments.append(FLOORING_TILE_FRAGMENT.strip())

    if style_id:
        style_detail = STYLE_PROMPTS.get(style_id, "")
        staging_block = f"""
Step 4 — Virtual staging in '{style_id}' style:
- Virtually stage the room using photorealistic furniture and decor that fits this style.
- Keep the room's architecture, windows, and layout the same.
- Replace or enhance existing furniture and decor so the overall scene matches this style description:
  {style_detail}
- Do not add any people, text, or logos.
"""
        fragments.append(staging_block.strip())

    fragments.append(
        """
Output requirements:
- Return a single, high-quality edited image.
- The edit must look like a real photograph, not an illustration or painting.
- Do not alter the room's basic layout, window views, or camera angle.
""".strip()
    )

    final_prompt = "\n\n".join(fragments)

    return final_prompt


def build_stage_room_model_decided_staging_prompt(
    reference_image_count: int,
    reference_item_labels: Optional[List[str]] = None,
    room_type: Optional[str] = None,
) -> str:
    item_count = max(1, int(reference_image_count))
    fragments: List[str] = [
        "You are an expert interior staging and photorealistic image-editing model.",
        (
            "Task:\n"
            "Stage the provided room photo by adding the provided furniture/product reference images into the room."
        ),
        (
            "Inputs:\n"
            "- The first image is the room/base image to edit.\n"
            "- The additional reference images are the furniture/product items to place into the room.\n"
            f"- You are given {item_count} furniture reference images. Place all {item_count} items into the room exactly once unless an item is physically impossible to fit."
        ),
    ]
    if room_type:
        key = room_type.strip().lower()
        hint = ROOM_TYPE_HINTS.get(key) or (
            "This room has a specific existing function. Preserve that function and "
            "do not convert it into a different type of room."
        )
        fragments.append(f"Room type context:\n- {hint}")

    normalized_labels = [label.strip() for label in (reference_item_labels or []) if isinstance(label, str) and label.strip()]
    if normalized_labels:
        fragments.append(
            "Furniture reference items:\n"
            + "\n".join(f"- {label}" for label in normalized_labels)
        )

    fragments.append(
        (
            "Placement behavior:\n"
            "- Decide the best plausible location for each item based on room layout, floor plane, walls, camera perspective, and interior design logic.\n"
            "- Arrange all provided items naturally as one coherent staged room.\n"
            "- Prioritize larger anchor furniture first (beds, sofas, dining tables), then place secondary/accent items naturally around them.\n"
            "- Respect existing architecture and do not change room structure.\n"
            "- Match each item's scale, perspective, contact with floor/walls, shadows, lighting direction, and occlusion realistically.\n"
            "- Keep items recognizable and faithful to their reference images."
        )
    )
    fragments.append(
        (
            "Important constraints:\n"
            "- Do not ignore any provided furniture reference image.\n"
            "- Do not only enhance the room photo; the main task is furniture placement.\n"
            "- Only add the provided furniture reference items; do not add unrelated new furniture or decor.\n"
            "- Do not remove or alter structural elements.\n"
            "- Preserve architecture, camera angle, perspective, windows, doors, walls, floors, ceilings, built-ins, and overall lighting consistency.\n"
            "- Return one single photorealistic final staged room image."
        )
    )
    return "\n\n".join(fragments)


# ---------- FASTAPI APP ----------

app = FastAPI()


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    vibode_request_id = _sanitize_vibode_correlation_id(request.headers.get(VIBODE_REQUEST_ID_HEADER))
    request_id = (
        vibode_request_id
        or _sanitize_inbound_request_id(request.headers.get("X-Request-Id"))
        or uuid4().hex[:12]
    )
    operation_id = _sanitize_vibode_correlation_id(request.headers.get(VIBODE_OPERATION_ID_HEADER))
    attempt_id = _sanitize_vibode_correlation_id(request.headers.get(VIBODE_ATTEMPT_ID_HEADER))
    user_id = _sanitize_uuid_header_value(request.headers.get(VIBODE_USER_ID_HEADER))
    user_email = _sanitize_optional_header_value(request.headers.get(VIBODE_USER_EMAIL_HEADER))
    room_id = _sanitize_uuid_header_value(request.headers.get(VIBODE_ROOM_ID_HEADER))
    version_id = _sanitize_uuid_header_value(request.headers.get(VIBODE_VERSION_ID_HEADER))
    asset_id = _sanitize_uuid_header_value(request.headers.get(VIBODE_ASSET_ID_HEADER))
    workflow_type = _sanitize_optional_header_value(request.headers.get(VIBODE_WORKFLOW_TYPE_HEADER))
    action_type = _sanitize_optional_header_value(request.headers.get(VIBODE_ACTION_TYPE_HEADER))
    source_trigger = _sanitize_optional_header_value(request.headers.get(VIBODE_SOURCE_TRIGGER_HEADER))
    token = _REQUEST_ID_CTX.set(request_id)
    operation_token = _OPERATION_ID_CTX.set(operation_id)
    attempt_token = _ATTEMPT_ID_CTX.set(attempt_id)
    user_id_token = _USER_ID_CTX.set(user_id)
    user_email_token = _USER_EMAIL_CTX.set(user_email)
    room_id_token = _ROOM_ID_CTX.set(room_id)
    version_id_token = _VERSION_ID_CTX.set(version_id)
    asset_id_token = _ASSET_ID_CTX.set(asset_id)
    workflow_type_token = _WORKFLOW_TYPE_CTX.set(workflow_type)
    action_type_token = _ACTION_TYPE_CTX.set(action_type)
    source_trigger_token = _SOURCE_TRIGGER_CTX.set(source_trigger)
    route_token = _ROUTE_PATH_CTX.set(request.url.path if request.url and request.url.path else None)
    seq_token = _PROVIDER_ATTEMPT_SEQ_CTX.set(0)
    try:
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response
    finally:
        _PROVIDER_ATTEMPT_SEQ_CTX.reset(seq_token)
        _ROUTE_PATH_CTX.reset(route_token)
        _SOURCE_TRIGGER_CTX.reset(source_trigger_token)
        _ACTION_TYPE_CTX.reset(action_type_token)
        _WORKFLOW_TYPE_CTX.reset(workflow_type_token)
        _ASSET_ID_CTX.reset(asset_id_token)
        _VERSION_ID_CTX.reset(version_id_token)
        _ROOM_ID_CTX.reset(room_id_token)
        _USER_EMAIL_CTX.reset(user_email_token)
        _USER_ID_CTX.reset(user_id_token)
        _ATTEMPT_ID_CTX.reset(attempt_token)
        _OPERATION_ID_CTX.reset(operation_token)
        _REQUEST_ID_CTX.reset(token)


# ---------- MODELS ----------

class HealthResponse(BaseModel):
    status: str


class PasteToPlaceCancelRequest(BaseModel):
    scopeId: str
    jobId: str


class StageRoomReferenceImage(BaseModel):
    imageBase64: Optional[str] = None
    mimeType: Optional[str] = None
    sourceUrl: Optional[str] = None
    label: Optional[str] = None
    skuId: Optional[str] = None


class StageRoomRequest(BaseModel):
    imageBase64: str
    styleId: Optional[str] = None

    enhancePhoto: bool = False
    cleanupRoom: bool = False
    repairDamage: bool = False
    emptyRoom: bool = False
    renovateRoom: bool = False

    repaintWalls: bool = False
    flooringPreset: Optional[str] = None

    roomType: Optional[str] = None
    modelVersion: Optional[str] = None

    # Used only on fresh uploads
    aspectRatio: Optional[AspectRatio] = "auto"

    # ✅ Continuation mode (Continue from this image)
    isContinuation: bool = False
    placementIntent: Optional[str] = None
    referenceImageUrls: Optional[List[str]] = None
    referenceImageBase64s: Optional[List[str]] = None
    referenceImages: Optional[List[StageRoomReferenceImage]] = None


class StageRoomResponse(BaseModel):
    imageUrl: str
    appliedAspectRatio: Optional[str] = None  # debug only


class VibodePlacement(BaseModel):
    nodeId: str
    zIndex: Optional[int] = None
    skuId: Optional[str] = None
    skuImageBase64: str
    cxPx: float
    cyPx: float
    rPx: Optional[float] = None


class VibodeComposeRequest(BaseModel):
    roomImageBase64: str
    placements: List[VibodePlacement]
    enhancePhoto: bool = True
    modelVersion: Optional[str] = None
    aspectRatio: Optional[AspectRatio] = "auto"


class FreezeBaseImageInput(BaseModel):
    signedUrl: Optional[str] = None
    base64: Optional[str] = None
    widthPx: Optional[int] = None
    heightPx: Optional[int] = None
    # Backward compatibility for older clients.
    width: Optional[int] = None
    height: Optional[int] = None


class FreezePayloadV2Input(BaseModel):
    baseImage: FreezeBaseImageInput
    vibodeIntent: Optional[Dict[str, Any]] = None


class VibodeFreezeRequest(BaseModel):
    freeze: FreezePayloadV2Input
    collectionId: Optional[str] = None
    bundleId: Optional[str] = None
    enhancePhoto: bool = True
    heavyDeclutter: bool = False
    modelVersion: Optional[str] = None
    aspectRatio: Optional[AspectRatio] = "auto"


class VibodeEligibleSku(BaseModel):
    skuId: str
    label: Optional[str] = None
    defaultPxWidth: Optional[int] = None
    defaultPxHeight: Optional[int] = None
    realWidthFt: Optional[float] = None
    realDepthFt: Optional[float] = None
    variants: Optional[List[Dict[str, Any]]] = None


class VibodeVibeRequest(BaseModel):
    roomImageBase64: str
    collectionId: Optional[str] = None
    bundleId: Optional[str] = None
    eligibleSkus: List[VibodeEligibleSku]
    targetCount: Optional[int] = None
    enhancePhoto: bool = True
    modelVersion: Optional[str] = None
    aspectRatio: Optional[AspectRatio] = "auto"


class VibodeStageRunRequest(BaseModel):
    stage: Literal[1, 2, 3, 4, 5]
    isContinuation: Optional[bool] = None
    baseImageId: Optional[str] = None
    roomImageBase64: Optional[str] = None
    baseImageUrl: Optional[str] = None
    baseImageBase64: Optional[str] = None
    collectionId: Optional[str] = None
    bundleId: Optional[str] = None
    eligibleSkus: Optional[List[VibodeEligibleSku]] = None
    targetCount: Optional[int] = None
    enhancePhoto: bool = True
    cleanupRoom: bool = False
    repairDamage: bool = False
    emptyRoom: bool = False
    stage1Mode: Optional[Literal["enhance_declutter", "empty_room"]] = None
    heavyDeclutter: bool = False
    renovateRoom: bool = False
    repaintWalls: bool = False
    flooringPreset: Optional[str] = None
    roomType: Optional[str] = None
    stage4Mode: Optional[Stage4StyleMode] = None
    modelVersion: Optional[str] = None
    aspectRatio: Optional[AspectRatio] = "auto"


class ScenePlacementBbox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class ScenePlacement(BaseModel):
    placementId: str
    skuId: str
    label: Optional[str] = None
    source: Optional[str] = None
    bbox: Optional[ScenePlacementBbox] = None
    rotationDeg: Optional[float] = None
    stageAdded: Optional[int] = None
    locked: Optional[bool] = None


class EligibleSkuVariant(BaseModel):
    imageUrl: str


class EligibleSku(BaseModel):
    skuId: str
    label: str
    source: Optional[str] = None
    variants: List[EligibleSkuVariant]


class VibodeEditRunTarget(BaseModel):
    placementId: Optional[str] = None
    skuId: Optional[str] = None
    xNorm: Optional[float] = None
    yNorm: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None


class VibodeEditRunRequest(BaseModel):
    baseImageUrl: Optional[str] = None
    action: Literal["add", "remove", "swap", "rotate"]
    placements: List[ScenePlacement] = Field(default_factory=list)
    target: Optional[VibodeEditRunTarget] = None
    params: Optional[Dict[str, Any]] = None
    placementId: Optional[str] = None
    xNorm: Optional[float] = None
    yNorm: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    rotationDegrees: Optional[float] = None
    eligibleSkus: Optional[List[EligibleSku]] = None
    sceneId: Optional[str] = None
    modelVersion: Optional[str] = None
    aspectRatio: Optional[AspectRatio] = "auto"


class VibodeEditRunResponse(BaseModel):
    imageUrl: str
    placements: List[ScenePlacement]


class VibodeComposeResponse(BaseModel):
    imageUrl: str
    appliedAspectRatio: Optional[str] = None


class VibodeRemoveMark(BaseModel):
    id: str
    x: float
    y: float
    r: float
    labelIndex: Optional[int] = None


class VibodeRemoveRequest(BaseModel):
    cleanBase64: str
    marks: List[VibodeRemoveMark]
    modelVersion: Optional[str] = None
    aspectRatio: Optional[AspectRatio] = "auto"


class VibodeSwapReplacementAsset(BaseModel):
    kind: str = "sku"
    skuId: Optional[str] = None
    imageUrl: str


class VibodeSwapMark(BaseModel):
    id: str
    x: float
    y: float
    replacement: VibodeSwapReplacementAsset


class VibodeSwapRequest(BaseModel):
    cleanBase64: str
    marks: List[VibodeSwapMark]
    replacementAssets: List[VibodeSwapReplacementAsset]
    modelVersion: Optional[str] = None


class VibodeSwapResponse(BaseModel):
    imageUrl: str


class VibodeRotateMark(BaseModel):
    id: str
    x: float
    y: float
    angleDeg: float


class VibodeRotateRequest(BaseModel):
    freezePayload: Dict[str, Any]
    baseImageUrl: Optional[str] = None
    cleanBase64: Optional[str] = None
    marks: Optional[List[VibodeRotateMark]] = None
    modelVersion: Optional[str] = None
    aspectRatio: Optional[AspectRatio] = "auto"


class VibodeUserSkuIngestRequest(BaseModel):
    imageUrl: Optional[str] = None
    imageBase64: Optional[str] = None
    label: Optional[str] = None
    model: Optional[str] = None
    normalization: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def validate_image_input(self) -> "VibodeUserSkuIngestRequest":
        has_image_url = bool(self.imageUrl and self.imageUrl.strip())
        has_image_base64 = bool(self.imageBase64 and self.imageBase64.strip())
        if has_image_url == has_image_base64:
            raise ValueError("Exactly one of imageUrl or imageBase64 must be provided.")
        return self


class VibodeUserSku(BaseModel):
    skuId: str
    label: str
    variants: List[str]
    sourceUrl: Optional[str] = None
    status: Literal["ready", "failed"]
    reason: Optional[str] = None


class VibodeUserSkuIngestResponse(BaseModel):
    userSku: VibodeUserSku


def call_gemini_with_prompt(
    image_png_bytes: bytes,
    prompt: str,
    model_name: str,
    aspect_ratio: Optional[str] = None,
    additional_image_png_bytes_list: Optional[List[bytes]] = None,
) -> bytes:
    """
    If aspect_ratio is None, we OMIT image_config.aspect_ratio entirely
    (premium continuation behavior).
    """
    started_at = time.perf_counter()
    provider_attempt_id = _next_provider_attempt_id()
    accounting_status = "failed"
    accounting_error_code: Optional[str] = None
    accounting_error_message: Optional[str] = None
    accounting_usage_metrics: Dict[str, Any] = {}
    response: Optional[Any] = None
    logged_terminal_failure = False
    additional_images = additional_image_png_bytes_list or []
    log_event(
        "model_call_start",
        function="call_gemini_with_prompt",
        model_name=model_name,
        modality="image+text",
        aspect_ratio=aspect_ratio if aspect_ratio else "(omitted)",
        image_count=1 + len(additional_images),
        additional_image_count=len(additional_images),
        input_png_bytes=len(image_png_bytes),
    )
    try:
        config_kwargs = {"response_modalities": ["IMAGE"]}

        if aspect_ratio:
            config_kwargs["image_config"] = types.ImageConfig(aspect_ratio=aspect_ratio)

        contents = [
            prompt,
            types.Part(
                inline_data=types.Blob(
                    data=image_png_bytes,
                    mime_type="image/png",
                )
            ),
        ]
        for extra_png_bytes in additional_images:
            contents.append(
                types.Part(
                    inline_data=types.Blob(
                        data=extra_png_bytes,
                        mime_type="image/png",
                    )
                )
            )

        _log_gemini_prompt_debug(
            function_name="call_gemini_with_prompt",
            model_name=model_name,
            prompt=prompt,
        )
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        accounting_usage_metrics = _extract_provider_usage_metrics(response)

        try:
            candidate = response.candidates[0]
            part = candidate.content.parts[0]
            out_bytes = part.inline_data.data
        except Exception as e:
            log_event(
                "model_call_extract_failed",
                function="call_gemini_with_prompt",
                model_name=model_name,
                modality="image+text",
                aspect_ratio=aspect_ratio if aspect_ratio else "(omitted)",
                error=str(e),
            )
            logged_terminal_failure = True
            raise RuntimeError("Could not extract generated image from Gemini response")

        if not out_bytes:
            logged_terminal_failure = True
            log_event(
                "model_call_failed",
                function="call_gemini_with_prompt",
                model_name=model_name,
                modality="image+text",
                aspect_ratio=aspect_ratio if aspect_ratio else "(omitted)",
                error="Gemini returned empty image bytes",
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )
            raise RuntimeError("Gemini returned empty image bytes")

        log_event(
            "model_call_success",
            function="call_gemini_with_prompt",
            model_name=model_name,
            modality="image+text",
            aspect_ratio=aspect_ratio if aspect_ratio else "(omitted)",
            output_png_bytes=len(out_bytes),
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )
        accounting_status = "success"
        return out_bytes

    except Exception as e:
        accounting_error_code = type(e).__name__
        accounting_error_message = str(e)
        if response is not None and not accounting_usage_metrics:
            accounting_usage_metrics = _extract_provider_usage_metrics(response)
        if not logged_terminal_failure:
            log_event(
                "model_call_failed",
                function="call_gemini_with_prompt",
                model_name=model_name,
                modality="image+text",
                aspect_ratio=aspect_ratio if aspect_ratio else "(omitted)",
                error=str(e),
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )
        raise
    finally:
        _write_gemini_usage_event_best_effort(
            attempt_id=provider_attempt_id,
            model_name=model_name,
            status=accounting_status,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
            error_code=accounting_error_code,
            error_message=accounting_error_message,
            usage_metrics=accounting_usage_metrics,
        )


def run_fusion(
    image_png_bytes: bytes,
    style_id: Optional[str],
    enhance_photo: bool,
    cleanup_room: bool,
    repair_damage: bool,
    empty_room: bool,
    renovate_room: bool,
    repaint_walls: bool,
    flooring_preset: Optional[str],
    room_type: Optional[str],
    model_name: str,
    aspect_ratio: Optional[str],
    reference_image_png_bytes_list: Optional[List[bytes]] = None,
    prompt_override: Optional[str] = None,
    prompt_intent: str = "default_roomprintz",
    prompt_version: str = "roomprintz_base_v1",
) -> bytes:
    prompt = (
        prompt_override
        if isinstance(prompt_override, str) and prompt_override.strip()
        else build_roomprintz_prompt(
            enhance_photo=enhance_photo,
            cleanup_room=cleanup_room,
            repair_damage=repair_damage,
            empty_room=empty_room,
            renovate_room=renovate_room,
            repaint_walls=repaint_walls,
            flooring_preset=flooring_preset,
            style_id=style_id,
            room_type=room_type,
        )
    )
    prompt_summary = summarize_prompt(prompt)
    log_event(
        "stage_room_prompt_dispatch",
        route="/stage-room",
        prompt_intent=prompt_intent,
        prompt_version=prompt_version,
        **prompt_summary,
    )

    return call_gemini_with_prompt(
        image_png_bytes=image_png_bytes,
        prompt=prompt,
        model_name=model_name,
        aspect_ratio=aspect_ratio,
        additional_image_png_bytes_list=reference_image_png_bytes_list,
    )


def run_photo_tools(
    image_png_bytes: bytes,
    enhance_photo: bool,
    cleanup_room: bool,
    repair_damage: bool,
    empty_room: bool,
    renovate_room: bool,
    repaint_walls: bool,
    flooring_preset: Optional[str],
    room_type: Optional[str],
    model_name: str,
    aspect_ratio: Optional[str],
    reference_image_png_bytes_list: Optional[List[bytes]] = None,
    prompt_override: Optional[str] = None,
    prompt_intent: str = "default_roomprintz",
    prompt_version: str = "roomprintz_base_v1",
) -> bytes:
    return run_fusion(
        image_png_bytes=image_png_bytes,
        style_id=None,
        enhance_photo=enhance_photo,
        cleanup_room=cleanup_room,
        repair_damage=repair_damage,
        empty_room=empty_room,
        renovate_room=renovate_room,
        repaint_walls=repaint_walls,
        flooring_preset=flooring_preset,
        room_type=room_type,
        model_name=model_name,
        aspect_ratio=aspect_ratio,
        reference_image_png_bytes_list=reference_image_png_bytes_list,
        prompt_override=prompt_override,
        prompt_intent=prompt_intent,
        prompt_version=prompt_version,
    )


def _collect_stage_room_reference_png_bytes(
    req: StageRoomRequest,
    max_additional_refs: int = STAGE_ROOM_MAX_REFERENCE_IMAGES,
) -> List[bytes]:
    # Preferred source: structured candidate refs from referenceImages[].imageBase64.
    # Legacy fallback: referenceImageBase64s only when structured refs have no embedded images.
    structured_payloads: List[str] = []
    for ref in req.referenceImages or []:
        payload = (ref.imageBase64 or "").strip()
        if payload:
            structured_payloads.append(payload)

    legacy_payloads: List[str] = []
    if not structured_payloads:
        for payload in req.referenceImageBase64s or []:
            trimmed = (payload or "").strip()
            if trimmed:
                legacy_payloads.append(trimmed)

    raw_payloads = structured_payloads if structured_payloads else legacy_payloads
    reference_payloads: List[str] = []
    seen_payloads: set[str] = set()
    for payload in raw_payloads:
        if payload in seen_payloads:
            continue
        seen_payloads.add(payload)
        reference_payloads.append(payload)

    additional_png_bytes: List[bytes] = []
    for payload in reference_payloads[:max_additional_refs]:
        try:
            raw_bytes = _decode_base64_image(payload)
            additional_png_bytes.append(prepare_sku_png_bytes(raw_bytes))
        except Exception as e:
            print("[/stage-room] Skipping invalid embedded reference image:", str(e))

    return additional_png_bytes


def make_data_url(image_bytes: bytes, mime_type: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _decode_base64_image(data: str) -> bytes:
    try:
        trimmed = data.strip()
        if trimmed.startswith("data:") and "," in trimmed:
            trimmed = trimmed.split(",", 1)[1]
        return base64.b64decode(trimmed)
    except Exception as e:
        print("[_decode_base64_image] Failed to decode base64:", e)
        raise


def _fetch_image_bytes_from_url(image_url: str, timeout_seconds: float = 15.0) -> bytes:
    try:
        response = requests.get(image_url, timeout=timeout_seconds)
        response.raise_for_status()
        if not response.content:
            raise RuntimeError("Empty image payload")
        return response.content
    except Exception as e:
        print("[_fetch_image_bytes_from_url] Failed to fetch image URL:", image_url, "| Error:", e)
        raise


def _fetch_image_bytes_from_url_limited(
    image_url: str,
    timeout_seconds: float = USER_SKU_INGEST_TIMEOUT_SECONDS,
    max_bytes: int = USER_SKU_MAX_INPUT_BYTES,
) -> Tuple[bytes, Optional[str]]:
    try:
        with requests.get(image_url, timeout=timeout_seconds, stream=True) as response:
            response.raise_for_status()
            content_type_header = (response.headers.get("Content-Type") or "").strip()
            content_type = (
                content_type_header.split(";", 1)[0].strip().lower()
                if content_type_header
                else None
            )

            content_length_header = (response.headers.get("Content-Length") or "").strip()
            if content_length_header.isdigit() and int(content_length_header) > max_bytes:
                raise RuntimeError(
                    f"Remote image exceeds max size ({content_length_header} > {max_bytes} bytes)."
                )

            payload = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                payload.extend(chunk)
                if len(payload) > max_bytes:
                    raise RuntimeError(
                        f"Remote image exceeds max size while downloading (> {max_bytes} bytes)."
                    )

            if not payload:
                raise RuntimeError("Empty image payload.")

            return bytes(payload), content_type
    except Exception as e:
        print(
            "[_fetch_image_bytes_from_url_limited] Failed to fetch image URL:",
            image_url,
            "| Error:",
            e,
        )
        raise


def _decode_base64_image_with_mime(data: str) -> Tuple[bytes, Optional[str]]:
    try:
        trimmed = (data or "").strip()
        inferred_mime: Optional[str] = None
        payload = trimmed

        if trimmed.startswith("data:") and "," in trimmed:
            metadata, payload = trimmed.split(",", 1)
            media = metadata[5:]
            if ";" in media:
                media = media.split(";", 1)[0]
            inferred_mime = media.strip().lower() or None

        try:
            decoded = base64.b64decode(payload, validate=True)
        except Exception:
            decoded = base64.b64decode(payload)

        if not decoded:
            raise RuntimeError("Empty base64 payload.")
        if len(decoded) > USER_SKU_MAX_INPUT_BYTES:
            raise RuntimeError(
                f"Base64 image exceeds max size ({len(decoded)} > {USER_SKU_MAX_INPUT_BYTES} bytes)."
            )

        return decoded, inferred_mime
    except Exception as e:
        print("[_decode_base64_image_with_mime] Failed to decode base64 payload:", e)
        raise


def _infer_image_mime_type(image_bytes: bytes, fallback_mime: Optional[str] = None) -> str:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            image_format = (img.format or "").upper()
        if image_format in Image.MIME:
            return Image.MIME[image_format]
    except Exception:
        pass

    if fallback_mime and fallback_mime.strip():
        return fallback_mime.strip().lower()
    return "application/octet-stream"


def _convert_image_bytes_to_png(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        rgba = img.convert("RGBA")
        return image_to_png_bytes(rgba)


def _has_transparency(png_bytes: bytes) -> bool:
    with Image.open(io.BytesIO(png_bytes)) as img:
        rgba = img.convert("RGBA")
        alpha = rgba.getchannel("A")
        lo, hi = alpha.getextrema()
        return not (lo == 255 and hi == 255)


def _assert_has_transparency(png_bytes: bytes) -> None:
    if not _has_transparency(png_bytes):
        raise RuntimeError(
            "Background removal output has no transparency (alpha fully opaque)."
        )


def _estimate_corner_background_rgb(img_rgb: Image.Image) -> Tuple[int, int, int]:
    width, height = img_rgb.size
    patch = max(2, int(round(min(width, height) * 0.04)))
    corners = [
        (0, 0, patch, patch),
        (max(0, width - patch), 0, width, patch),
        (0, max(0, height - patch), patch, height),
        (max(0, width - patch), max(0, height - patch), width, height),
    ]
    samples: List[Tuple[int, int, int]] = []
    for box in corners:
        samples.extend(list(img_rgb.crop(box).getdata()))
    if not samples:
        return (242, 242, 242)
    total = len(samples)
    return (
        int(round(sum(px[0] for px in samples) / total)),
        int(round(sum(px[1] for px in samples) / total)),
        int(round(sum(px[2] for px in samples) / total)),
    )


def _histogram_percentile(hist: List[int], percentile: float) -> int:
    total = float(sum(hist))
    if total <= 0:
        return 0
    threshold = max(1.0, total * min(1.0, max(0.0, percentile)))
    running = 0.0
    for idx, count in enumerate(hist):
        running += count
        if running >= threshold:
            return idx
    return max(0, len(hist) - 1)


def _estimate_local_background_rgb(
    img_rgb: Image.Image,
    bbox: Tuple[int, int, int, int],
) -> Tuple[int, int, int]:
    width, height = img_rgb.size
    left = max(0, min(width, bbox[0]))
    top = max(0, min(height, bbox[1]))
    right = max(left + 1, min(width, bbox[2]))
    bottom = max(top + 1, min(height, bbox[3]))
    region = img_rgb.crop((left, top, right, bottom))
    region_w, region_h = region.size
    if region_w < 2 or region_h < 2:
        return _estimate_corner_background_rgb(img_rgb)

    band = max(1, int(round(min(region_w, region_h) * 0.08)))
    samples: List[Tuple[int, int, int]] = []
    pixels = region.load()

    for x in range(region_w):
        for y in range(band):
            samples.append(pixels[x, y])
            samples.append(pixels[x, max(0, region_h - 1 - y)])
    for y in range(band, max(band, region_h - band)):
        for x in range(band):
            samples.append(pixels[x, y])
            samples.append(pixels[max(0, region_w - 1 - x), y])

    if not samples:
        return _estimate_corner_background_rgb(img_rgb)

    rs = sorted(px[0] for px in samples)
    gs = sorted(px[1] for px in samples)
    bs = sorted(px[2] for px in samples)
    mid_idx = len(samples) // 2
    return (rs[mid_idx], gs[mid_idx], bs[mid_idx])


def _find_user_sku_outer_bbox_from_binary_mask(
    binary_mask: Image.Image,
) -> Optional[Tuple[int, int, int, int]]:
    mask = binary_mask.convert("L")
    width, height = mask.size
    if width <= 0 or height <= 0:
        return None

    pixels = mask.load()
    row_counts = [0] * height
    col_counts = [0] * width
    for y in range(height):
        row_total = 0
        for x in range(width):
            if pixels[x, y] >= 128:
                row_total += 1
                col_counts[x] += 1
        row_counts[y] = row_total

    min_row_pixels = max(1, int(round(width * 0.002)))
    min_col_pixels = max(1, int(round(height * 0.002)))

    top = next((y for y, count in enumerate(row_counts) if count >= min_row_pixels), None)
    bottom = next(
        (y for y in range(height - 1, -1, -1) if row_counts[y] >= min_row_pixels),
        None,
    )
    left = next((x for x, count in enumerate(col_counts) if count >= min_col_pixels), None)
    right = next(
        (x for x in range(width - 1, -1, -1) if col_counts[x] >= min_col_pixels),
        None,
    )

    if top is None or bottom is None or left is None or right is None:
        return mask.getbbox()
    if left > right or top > bottom:
        return mask.getbbox()

    return (left, top, right + 1, bottom + 1)


def _detect_user_sku_outer_bbox_from_flat_background(
    candidate_rgba: Image.Image,
) -> Tuple[Optional[Tuple[int, int, int, int]], Image.Image]:
    candidate_rgb = candidate_rgba.convert("RGB")
    bg_r, bg_g, bg_b = _estimate_corner_background_rgb(candidate_rgb)
    bg = Image.new("RGB", candidate_rgb.size, (bg_r, bg_g, bg_b))
    diff_rgb = ImageChops.difference(candidate_rgb, bg)

    # Use the strongest per-channel difference so pale products are less likely to be clipped.
    diff_r, diff_g, diff_b = diff_rgb.split()
    diff_max = ImageChops.lighter(ImageChops.lighter(diff_r, diff_g), diff_b)
    diff_hist = diff_max.histogram()
    diff_p90 = _histogram_percentile(diff_hist, 0.90)
    diff_p97 = _histogram_percentile(diff_hist, 0.97)
    threshold = max(
        8,
        min(
            40,
            int(
                round(
                    max(
                        USER_SKU_FOREGROUND_COLOR_DISTANCE_THRESHOLD * 0.45,
                        (diff_p90 * 0.55) + (diff_p97 * 0.45),
                    )
                )
            ),
        ),
    )
    outer_mask = diff_max.point(lambda v: 255 if v >= threshold else 0, mode="L")
    bbox = _find_user_sku_outer_bbox_from_binary_mask(outer_mask)
    return bbox, outer_mask


def _count_mask_pixels(mask: Image.Image) -> int:
    return sum(1 for px in mask.getdata() if px >= 128)


def _compute_user_sku_image_debug_metrics(image_png_bytes: bytes) -> Dict[str, Any]:
    with Image.open(io.BytesIO(image_png_bytes)) as img:
        source_mode = img.mode or ""
        rgba = img.convert("RGBA")
        width, height = rgba.size
        total_pixels = max(1, width * height)

        alpha = rgba.getchannel("A")
        alpha_hist = alpha.histogram()
        alpha_nonzero = sum(alpha_hist[1:])
        alpha_ge_250 = sum(alpha_hist[250:])
        alpha_lo, alpha_hi = alpha.getextrema()
        has_alpha = not (alpha_lo == 255 and alpha_hi == 255)

        gray = rgba.convert("L")
        hist = gray.histogram()
        lum_total = float(sum(hist))
        weighted_lum = sum(idx * count for idx, count in enumerate(hist))
        mean_luminance = (weighted_lum / lum_total) if lum_total > 0 else 0.0

        def _percentile_from_hist(percentile: float) -> int:
            threshold = max(1.0, lum_total * percentile)
            running = 0.0
            for idx, count in enumerate(hist):
                running += count
                if running >= threshold:
                    return idx
            return 255

        p95_luminance = _percentile_from_hist(0.95)
        p99_luminance = _percentile_from_hist(0.99)

        lum_ge_245 = sum(hist[245:])
        lum_ge_250 = sum(hist[250:])
        lum_ge_254 = sum(hist[254:])

        content_mask: Optional[Image.Image] = None
        if has_alpha:
            content_mask = alpha.point(lambda a: 255 if a > 0 else 0, mode="L")
        else:
            # Approximate non-background content when alpha isn't available.
            content_mask = gray.point(lambda v: 255 if v < 250 else 0, mode="L")
        bbox = content_mask.getbbox() if content_mask else None
        content_bbox = (
            {"left": bbox[0], "top": bbox[1], "right": bbox[2], "bottom": bbox[3]}
            if bbox
            else None
        )

        metrics: Dict[str, Any] = {
            "width": width,
            "height": height,
            "mode": source_mode or rgba.mode,
            "has_alpha": has_alpha,
            "mean_luminance": round(mean_luminance, 4),
            "p95_luminance": p95_luminance,
            "p99_luminance": p99_luminance,
            "lum_ge_245_ratio": round(lum_ge_245 / float(total_pixels), 6),
            "lum_ge_250_ratio": round(lum_ge_250 / float(total_pixels), 6),
            "lum_ge_254_ratio": round(lum_ge_254 / float(total_pixels), 6),
            "content_bbox": content_bbox,
        }
        if has_alpha:
            metrics["alpha_nonzero_ratio"] = round(alpha_nonzero / float(total_pixels), 6)
            metrics["alpha_ge_250_ratio"] = round(alpha_ge_250 / float(total_pixels), 6)
        return metrics


def _build_user_sku_foreground_mask(img_rgba: Image.Image) -> Image.Image:
    rgba = img_rgba.convert("RGBA")
    alpha = rgba.getchannel("A")
    alpha_lo, _ = alpha.getextrema()
    if alpha_lo < 255:
        base_mask = alpha.point(lambda a: 255 if a >= 12 else 0, mode="L")
    else:
        rgb = rgba.convert("RGB")
        bg_r, bg_g, bg_b = _estimate_corner_background_rgb(rgb)
        bg = Image.new("RGB", rgb.size, (bg_r, bg_g, bg_b))
        diff = ImageChops.difference(rgb, bg).convert("L")
        base_mask = diff.point(
            lambda v: 255 if v >= USER_SKU_FOREGROUND_COLOR_DISTANCE_THRESHOLD else 0,
            mode="L",
        )

    # Keep shape tight while removing tiny noise + holes.
    base_mask = base_mask.filter(ImageFilter.MaxFilter(size=3))
    base_mask = base_mask.filter(ImageFilter.MinFilter(size=3))
    return base_mask


def _fill_user_sku_mask_internal_holes(binary_mask: Image.Image) -> Image.Image:
    if not _HAS_IMAGE_DRAW or ImageDraw is None or not hasattr(ImageDraw, "floodfill"):
        # Fallback when flood fill is unavailable: mild closing to reduce visible gaps.
        closed = binary_mask.filter(ImageFilter.MaxFilter(size=5))
        return closed.filter(ImageFilter.MinFilter(size=3))

    width, height = binary_mask.size
    traced = binary_mask.convert("L")
    seeds = [
        (0, 0),
        (max(0, width - 1), 0),
        (0, max(0, height - 1)),
        (max(0, width - 1), max(0, height - 1)),
    ]
    for x in range(width):
        seeds.append((x, 0))
        seeds.append((x, max(0, height - 1)))
    for y in range(height):
        seeds.append((0, y))
        seeds.append((max(0, width - 1), y))

    # Mark border-connected background as 64, leaving enclosed 0-valued holes untouched.
    for seed in seeds:
        if traced.getpixel(seed) == 0:
            ImageDraw.floodfill(traced, seed, 64)

    # Convert enclosed holes (0) to foreground while keeping traced background (64) transparent.
    return traced.point(lambda v: 0 if v == 64 else 255, mode="L")


def _refine_user_sku_foreground_mask(
    source_rgba: Image.Image,
    candidate_mask: Image.Image,
) -> Image.Image:
    candidate_binary = candidate_mask.convert("L").point(lambda v: 255 if v >= 128 else 0, mode="L")
    bbox = candidate_binary.getbbox()
    if not bbox:
        return candidate_binary

    width, height = candidate_binary.size
    pad = max(4, int(round(min(width, height) * 0.015)))
    expanded_bbox = (
        max(0, bbox[0] - pad),
        max(0, bbox[1] - pad),
        min(width, bbox[2] + pad),
        min(height, bbox[3] + pad),
    )
    bbox_gate = Image.new("L", candidate_binary.size, 0)
    bbox_gate.paste(255, expanded_bbox)

    source_rgb = source_rgba.convert("RGB")
    bg_r, bg_g, bg_b = _estimate_local_background_rgb(source_rgb, expanded_bbox)
    bg = Image.new("RGB", source_rgb.size, (bg_r, bg_g, bg_b))
    source_diff = ImageChops.difference(source_rgb, bg).convert("L")
    diff_hist = source_diff.crop(expanded_bbox).histogram()
    diff_p55 = _histogram_percentile(diff_hist, 0.55)
    diff_p75 = _histogram_percentile(diff_hist, 0.75)
    diff_p90 = _histogram_percentile(diff_hist, 0.90)
    adaptive_threshold = max(
        8,
        min(
            40,
            int(round((diff_p55 * 0.30) + (diff_p75 * 0.50) + (diff_p90 * 0.20))),
        ),
    )
    strong_threshold = min(56, adaptive_threshold + 10)

    source_fg_soft = source_diff.point(lambda v: 255 if v >= adaptive_threshold else 0, mode="L")
    source_fg_strong = source_diff.point(lambda v: 255 if v >= strong_threshold else 0, mode="L")
    source_fg_soft = ImageChops.multiply(source_fg_soft, bbox_gate)
    source_fg_strong = ImageChops.multiply(source_fg_strong, bbox_gate)

    candidate_seed = candidate_binary.filter(ImageFilter.MinFilter(size=3))
    candidate_guardrail = candidate_binary.filter(ImageFilter.MaxFilter(size=5))
    candidate_guardrail = candidate_guardrail.filter(ImageFilter.MaxFilter(size=5))
    source_fg_guarded = ImageChops.multiply(source_fg_soft, candidate_guardrail)

    # Preserve low-contrast boundaries near the candidate contour.
    edge_map = source_rgb.convert("L").filter(ImageFilter.FIND_EDGES)
    edge_hist = edge_map.crop(expanded_bbox).histogram()
    edge_threshold = max(
        10,
        min(30, int(round(_histogram_percentile(edge_hist, 0.75) * 0.80))),
    )
    edge_support = edge_map.point(lambda v: 255 if v >= edge_threshold else 0, mode="L")
    edge_support = ImageChops.multiply(edge_support, bbox_gate)
    edge_support = ImageChops.multiply(edge_support, candidate_guardrail)

    combined = ImageChops.lighter(source_fg_guarded, source_fg_strong)
    combined = ImageChops.lighter(combined, edge_support)
    combined = ImageChops.lighter(combined, candidate_seed)

    source_alpha = source_rgba.getchannel("A")
    source_alpha_lo, _ = source_alpha.getextrema()
    if source_alpha_lo < 255:
        source_alpha_mask = source_alpha.point(lambda a: 255 if a >= 8 else 0, mode="L")
        source_alpha_mask = ImageChops.multiply(source_alpha_mask, bbox_gate)
        combined = ImageChops.lighter(combined, source_alpha_mask)

    combined = combined.filter(ImageFilter.MaxFilter(size=3))
    combined = combined.filter(ImageFilter.MinFilter(size=3))
    return _fill_user_sku_mask_internal_holes(combined)


def _compute_user_sku_mask_stats(mask: Image.Image) -> Dict[str, Any]:
    binary_mask = mask.convert("L").point(lambda v: 255 if v >= 128 else 0, mode="L")
    width, height = binary_mask.size
    total_pixels = max(1, width * height)
    fg_pixels = _count_mask_pixels(binary_mask)
    bbox = binary_mask.getbbox()
    if not bbox or fg_pixels <= 0:
        return {
            "has_foreground": False,
            "fg_pixels": 0,
            "area_ratio": 0.0,
            "bbox_cover_w": 0.0,
            "bbox_cover_h": 0.0,
            "fill_ratio": 0.0,
            "edge_touch_ratio": 0.0,
            "edge_touch_sides": 0,
        }

    bbox_w = max(1, bbox[2] - bbox[0])
    bbox_h = max(1, bbox[3] - bbox[1])
    bbox_pixels = max(1, bbox_w * bbox_h)
    edge_band = max(1, int(round(min(width, height) * 0.02)))

    top = binary_mask.crop((0, 0, width, edge_band))
    bottom = binary_mask.crop((0, max(0, height - edge_band), width, height))
    left = binary_mask.crop((0, 0, edge_band, height))
    right = binary_mask.crop((max(0, width - edge_band), 0, width, height))
    edge_pixels = _count_mask_pixels(top) + _count_mask_pixels(bottom) + _count_mask_pixels(left) + _count_mask_pixels(right)
    touched_sides = sum(1 for region in (top, bottom, left, right) if region.getbbox() is not None)

    return {
        "has_foreground": True,
        "fg_pixels": fg_pixels,
        "area_ratio": fg_pixels / float(total_pixels),
        "bbox_cover_w": bbox_w / float(max(1, width)),
        "bbox_cover_h": bbox_h / float(max(1, height)),
        "fill_ratio": fg_pixels / float(bbox_pixels),
        "edge_touch_ratio": min(1.0, edge_pixels / float(max(1, fg_pixels))),
        "edge_touch_sides": touched_sides,
    }


def _dominant_quantized_color_ratio(img_rgb: Image.Image) -> float:
    width, height = img_rgb.size
    long_edge = max(width, height)
    sample = img_rgb
    if long_edge > 512:
        scale = 512.0 / float(long_edge)
        sample = img_rgb.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            resample=Image.BILINEAR,
        )

    counts: Dict[Tuple[int, int, int], int] = {}
    total = 0
    for r, g, b in sample.getdata():
        key = (r // 32, g // 32, b // 32)
        counts[key] = counts.get(key, 0) + 1
        total += 1
    if total <= 0 or not counts:
        return 0.0
    return max(counts.values()) / float(total)


def _analyze_user_sku_candidate(
    original_png_bytes: bytes,
    candidate_png_bytes: bytes,
) -> Dict[str, Any]:
    with Image.open(io.BytesIO(original_png_bytes)) as src_img:
        src_rgb = src_img.convert("RGB")
    with Image.open(io.BytesIO(candidate_png_bytes)) as candidate_img:
        candidate_rgba = candidate_img.convert("RGBA")

    mask = _build_user_sku_foreground_mask(candidate_rgba)
    mask_stats = _compute_user_sku_mask_stats(mask)
    uniform_bg_dominance_ratio = _dominant_quantized_color_ratio(src_rgb)

    rectangular_outer_frame_likelihood = (
        mask_stats["has_foreground"]
        and mask_stats["bbox_cover_w"] >= USER_SKU_RECT_FRAME_BBOX_COVER_RATIO
        and mask_stats["bbox_cover_h"] >= USER_SKU_RECT_FRAME_BBOX_COVER_RATIO
        and mask_stats["fill_ratio"] >= USER_SKU_RECT_FRAME_FILL_RATIO
        and mask_stats["edge_touch_sides"] >= 3
    )
    edge_touch_border_heavy_content = (
        mask_stats["edge_touch_ratio"] >= USER_SKU_BORDER_HEAVY_EDGE_RATIO
        or mask_stats["edge_touch_sides"] >= 3
    )
    large_uniform_background_dominance = (
        uniform_bg_dominance_ratio >= USER_SKU_UNIFORM_BG_DOMINANCE_RATIO
    )
    likely_screenshot_card_composition = rectangular_outer_frame_likelihood and (
        edge_touch_border_heavy_content or large_uniform_background_dominance
    )
    low_confidence_segmentation_fallback = (
        not mask_stats["has_foreground"]
        or mask_stats["area_ratio"] < USER_SKU_MIN_FOREGROUND_AREA_RATIO
        or mask_stats["area_ratio"] > USER_SKU_MAX_FOREGROUND_AREA_RATIO
        or likely_screenshot_card_composition
    )

    return {
        "rectangular_outer_frame_likelihood": rectangular_outer_frame_likelihood,
        "edge_touch_border_heavy_content": edge_touch_border_heavy_content,
        "edge_touch_ratio": mask_stats["edge_touch_ratio"],
        "edge_touch_sides": mask_stats["edge_touch_sides"],
        "large_uniform_background_dominance": large_uniform_background_dominance,
        "uniform_bg_dominance_ratio": uniform_bg_dominance_ratio,
        "likely_screenshot_card_composition": likely_screenshot_card_composition,
        "low_confidence_segmentation_fallback": low_confidence_segmentation_fallback,
        "foreground_area_ratio": mask_stats["area_ratio"],
        "bbox_cover_w": mask_stats["bbox_cover_w"],
        "bbox_cover_h": mask_stats["bbox_cover_h"],
        "fill_ratio": mask_stats["fill_ratio"],
    }


def _extract_user_sku_product_crops(
    candidate_png_bytes: bytes,
    source_png_bytes: Optional[bytes] = None,
) -> Tuple[bytes, bytes, Dict[str, Any]]:
    with Image.open(io.BytesIO(candidate_png_bytes)) as img:
        candidate_rgba = img.convert("RGBA")

    # Primary fast path for Gemini RGB isolates:
    # detect only an outer bbox from flat-background color difference and crop directly.
    outer_bbox, outer_mask = _detect_user_sku_outer_bbox_from_flat_background(candidate_rgba)
    if outer_bbox:
        outer_stats = _compute_user_sku_mask_stats(outer_mask)
        if outer_stats["has_foreground"]:
            tight_cutout_rgba = candidate_rgba.crop(outer_bbox)
            tight_rgb = candidate_rgba.convert("RGB").crop(outer_bbox)
            return image_to_png_bytes(tight_cutout_rgba), image_to_png_bytes(tight_rgb), outer_stats

    source_rgba = candidate_rgba
    if source_png_bytes:
        with Image.open(io.BytesIO(source_png_bytes)) as source_img:
            decoded_source_rgba = source_img.convert("RGBA")
        if decoded_source_rgba.size == candidate_rgba.size:
            source_rgba = decoded_source_rgba

    candidate_mask = _build_user_sku_foreground_mask(candidate_rgba)
    mask = _refine_user_sku_foreground_mask(source_rgba, candidate_mask)
    stats = _compute_user_sku_mask_stats(mask)
    if not stats["has_foreground"]:
        raise RuntimeError("No product foreground detected after isolation.")

    bbox = candidate_mask.getbbox() or mask.getbbox()
    if not bbox:
        raise RuntimeError("Unable to compute product crop bounds.")

    original_alpha = source_rgba.getchannel("A")
    alpha_lo, _ = original_alpha.getextrema()
    if alpha_lo < 255:
        combined_alpha = ImageChops.multiply(original_alpha, mask)
    else:
        combined_alpha = mask

    cutout_rgba = source_rgba.copy()
    cutout_rgba.putalpha(combined_alpha)
    tight_cutout_rgba = cutout_rgba.crop(bbox)
    tight_rgb = source_rgba.convert("RGB").crop(bbox)
    return image_to_png_bytes(tight_cutout_rgba), image_to_png_bytes(tight_rgb), stats


def _resolve_supabase_storage_bucket(base_url: str, service_key: str) -> str:
    global _SUPABASE_STORAGE_BUCKET_CACHE

    if _SUPABASE_STORAGE_BUCKET_CACHE:
        return _SUPABASE_STORAGE_BUCKET_CACHE

    configured_bucket = (SUPABASE_STORAGE_BUCKET or "").strip()
    discovered_bucket_ids: List[str] = []

    try:
        response = requests.get(
            f"{base_url}/storage/v1/bucket",
            headers={
                "Authorization": f"Bearer {service_key}",
                "apikey": service_key,
            },
            timeout=SUPABASE_STORAGE_UPLOAD_TIMEOUT_SECONDS,
        )
        if response.status_code < 400:
            raw_buckets = response.json() if response.content else []
            if isinstance(raw_buckets, list):
                for item in raw_buckets:
                    if not isinstance(item, dict):
                        continue
                    bucket_id = str(item.get("id") or item.get("name") or "").strip()
                    if bucket_id:
                        discovered_bucket_ids.append(bucket_id)
        else:
            print(
                "[_resolve_supabase_storage_bucket] bucket list failed:",
                response.status_code,
                response.text[:200],
            )
    except Exception as e:
        print("[_resolve_supabase_storage_bucket] bucket discovery error:", e)

    chosen_bucket = configured_bucket
    if discovered_bucket_ids:
        if not chosen_bucket or chosen_bucket not in discovered_bucket_ids:
            preferred = (
                "vibode-user-skus",
                "user-skus",
                "assets",
                "images",
                "public",
            )
            chosen_bucket = next(
                (bucket for bucket in preferred if bucket in discovered_bucket_ids),
                discovered_bucket_ids[0],
            )
            if configured_bucket and configured_bucket != chosen_bucket:
                print(
                    "[_resolve_supabase_storage_bucket] configured bucket not found, using discovered bucket:",
                    {"configured": configured_bucket, "chosen": chosen_bucket},
                )

    if not chosen_bucket:
        raise RuntimeError(
            "No Supabase storage bucket resolved. Set SUPABASE_STORAGE_BUCKET."
        )

    _SUPABASE_STORAGE_BUCKET_CACHE = chosen_bucket
    return chosen_bucket


def _require_supabase_storage_config() -> Tuple[str, str, str]:
    if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL or NEXT_PUBLIC_SUPABASE_URL is required.")
    if not SUPABASE_SERVICE_KEY:
        raise RuntimeError("SUPABASE_SERVICE_KEY is required.")
    base_url = SUPABASE_URL.rstrip("/")
    bucket = _resolve_supabase_storage_bucket(base_url, SUPABASE_SERVICE_KEY)
    return base_url, SUPABASE_SERVICE_KEY, bucket


def _supabase_storage_upload_bytes(object_path: str, payload: bytes, mime_type: str) -> None:
    base_url, service_key, bucket = _require_supabase_storage_config()
    normalized_path = object_path.strip().lstrip("/")
    if not normalized_path:
        raise RuntimeError("object_path is required for upload.")

    upload_url = f"{base_url}/storage/v1/object/{bucket}/{quote(normalized_path, safe='/')}"
    headers = {
        "Authorization": f"Bearer {service_key}",
        "apikey": service_key,
        "Content-Type": mime_type,
        "x-upsert": "true",
    }

    response = requests.post(
        upload_url,
        headers=headers,
        data=payload,
        timeout=SUPABASE_STORAGE_UPLOAD_TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Supabase upload failed ({response.status_code}): {response.text[:300]}"
        )


def _supabase_storage_create_signed_url(
    object_path: str,
    expires_in_seconds: int = SUPABASE_SIGNED_URL_TTL_SECONDS,
) -> str:
    base_url, service_key, bucket = _require_supabase_storage_config()
    normalized_path = object_path.strip().lstrip("/")
    if not normalized_path:
        raise RuntimeError("object_path is required for signed URL.")

    sign_url = f"{base_url}/storage/v1/object/sign/{bucket}/{quote(normalized_path, safe='/')}"
    headers = {
        "Authorization": f"Bearer {service_key}",
        "apikey": service_key,
        "Content-Type": "application/json",
    }
    response = requests.post(
        sign_url,
        headers=headers,
        json={"expiresIn": int(max(1, expires_in_seconds))},
        timeout=SUPABASE_STORAGE_UPLOAD_TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Supabase signed URL failed ({response.status_code}): {response.text[:300]}"
        )

    payload = response.json() if response.content else {}
    signed_url_value = (
        payload.get("signedURL")
        or payload.get("signedUrl")
        or payload.get("signed_url")
        or ""
    )
    signed_url_value = str(signed_url_value).strip()
    if not signed_url_value:
        raise RuntimeError("Supabase sign response missing signed URL.")

    # Supabase can return signedURL values missing the "/storage/v1" prefix.
    if signed_url_value.startswith("http://") or signed_url_value.startswith("https://"):
        parsed = urlparse(signed_url_value)
        if parsed.path.startswith("/storage/v1/"):
            normalized_signed_url = signed_url_value
        elif parsed.path.startswith("/object/"):
            normalized_signed_url = parsed._replace(path=f"/storage/v1{parsed.path}").geturl()
        else:
            log_event(
                "supabase_signed_url_unexpected_path_prefix",
                bucket=bucket,
                path=normalized_path,
                pathname=parsed.path,
            )
            normalized_signed_url = signed_url_value
    elif signed_url_value.startswith("/"):
        normalized_signed_path = signed_url_value
        if normalized_signed_path.startswith("/object/"):
            normalized_signed_path = f"/storage/v1{normalized_signed_path}"
        normalized_signed_url = f"{base_url}{normalized_signed_path}"
    else:
        # Supabase can return relative signed paths with different prefixes; normalize safely.
        if signed_url_value.startswith("storage/v1/"):
            normalized_signed_url = f"{base_url}/{signed_url_value}"
        elif signed_url_value.startswith("object/"):
            normalized_signed_url = f"{base_url}/storage/v1/{signed_url_value}"
        else:
            normalized_signed_url = f"{base_url}/storage/v1/{signed_url_value}"

    if DEBUG_ROOMPRINTZ_PROMPT or DEBUG_ROOMPRINTZ_RATIO:
        parsed_normalized_signed_url = urlparse(normalized_signed_url)
        log_event(
            "supabase_signed_url_normalization",
            bucket=bucket,
            path=normalized_path,
            normalization_occurred=(signed_url_value != normalized_signed_url),
            host=parsed_normalized_signed_url.netloc,
            pathname=parsed_normalized_signed_url.path,
            had_query_string=bool(parsed_normalized_signed_url.query),
        )
    return normalized_signed_url


def _require_supabase_usage_config() -> Tuple[str, str]:
    if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL or NEXT_PUBLIC_SUPABASE_URL is required.")
    service_key = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_SERVICE_KEY
    if not service_key:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY or SUPABASE_SERVICE_KEY is required.")
    return SUPABASE_URL.rstrip("/"), service_key


def _usage_value_to_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _usage_value_to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_usage_value_to_json(v) for v in value]
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return _usage_value_to_json(value.model_dump())
        except Exception:
            return str(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _usage_value_to_json(value.to_dict())
        except Exception:
            return str(value)
    return str(value)


def _extract_provider_usage_metrics(response: Any) -> Dict[str, Any]:
    usage_obj = None
    if response is not None:
        usage_obj = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
    usage_json = _usage_value_to_json(usage_obj)
    return usage_json if isinstance(usage_json, dict) else {}


def _extract_int_usage_value(usage_metrics: Dict[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        value = usage_metrics.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.strip():
            try:
                return int(float(value))
            except Exception:
                continue
    return None


def _write_gemini_usage_event_best_effort(
    *,
    attempt_id: str,
    model_name: str,
    status: str,
    latency_ms: int,
    error_code: Optional[str],
    error_message: Optional[str],
    usage_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        base_url, service_key = _require_supabase_usage_config()
        usage_data = usage_metrics or {}
        route_path = get_route_path() or "unknown"
        metadata_payload: Dict[str, Any] = {}
        if error_message:
            metadata_payload["error_message"] = (error_message or "")[:4000]
        if usage_data:
            metadata_payload["provider_usage"] = usage_data
        user_email = get_user_email()
        if user_email:
            metadata_payload["user_email"] = user_email
        full_payload = {
            "attempt_id": attempt_id,
            "retry_of_attempt_id": None,
            "is_retry": False,
            "request_id": get_request_id() if get_request_id() != "-" else None,
            "operation_id": get_operation_id(),
            "provider_request_id": None,
            "service": "roomprintz-compositor",
            "provider": "google_gemini",
            "model": model_name,
            "status": status,
            "latency_ms": max(0, int(latency_ms)),
            "error_code": error_code,
            "metadata": metadata_payload,
            "route": route_path,
            "user_id": get_user_id(),
            "room_id": get_room_id(),
            "version_id": get_version_id(),
            "asset_id": get_asset_id(),
            "workflow_type": get_workflow_type() or "unknown",
            "action_type": get_action_type() or "unknown",
            "source_trigger": get_source_trigger(),
            "input_tokens": _extract_int_usage_value(
                usage_data,
                "prompt_token_count",
                "prompt_tokens",
                "input_token_count",
                "input_tokens",
            ),
            "output_tokens": _extract_int_usage_value(
                usage_data,
                "candidates_token_count",
                "completion_token_count",
                "completion_tokens",
                "output_token_count",
                "output_tokens",
            ),
            "image_count": _extract_int_usage_value(
                usage_data,
                "image_count",
                "generated_image_count",
            ),
            "reference_image_count": None,
            "estimated_cost_usd": None,
        }
        minimal_payload = {
            "attempt_id": attempt_id,
            "retry_of_attempt_id": None,
            "is_retry": False,
            "request_id": get_request_id() if get_request_id() != "-" else None,
            "operation_id": get_operation_id(),
            "provider_request_id": None,
            "service": "roomprintz-compositor",
            "provider": "google_gemini",
            "model": model_name,
            "status": status,
            "latency_ms": max(0, int(latency_ms)),
            "error_code": error_code,
            "metadata": metadata_payload,
            "route": route_path,
            "source_trigger": get_source_trigger(),
            "user_id": get_user_id(),
            "room_id": get_room_id(),
            "version_id": get_version_id(),
            "asset_id": get_asset_id(),
            "workflow_type": get_workflow_type() or "unknown",
            "action_type": get_action_type() or "unknown",
            "input_tokens": None,
            "output_tokens": None,
            "image_count": None,
            "reference_image_count": None,
            "estimated_cost_usd": None,
        }
        headers = {
            "Authorization": f"Bearer {service_key}",
            "apikey": service_key,
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        }
        write_url = f"{base_url}/rest/v1/vibode_gemini_usage_events?on_conflict=attempt_id"
        response = requests.post(
            write_url,
            headers=headers,
            json=[full_payload],
            timeout=SUPABASE_USAGE_WRITE_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            fallback_response = requests.post(
                write_url,
                headers=headers,
                json=[minimal_payload],
                timeout=SUPABASE_USAGE_WRITE_TIMEOUT_SECONDS,
            )
            if fallback_response.status_code < 400:
                log_event(
                    "gemini_usage_accounting_write_fallback_succeeded",
                    attempt_id=attempt_id,
                    model_name=model_name,
                    usage_table="public.vibode_gemini_usage_events",
                )
                return
            log_event(
                "gemini_usage_accounting_write_failed",
                status_code=response.status_code,
                body=response.text[:280],
                fallback_status_code=fallback_response.status_code,
                fallback_body=fallback_response.text[:280],
                attempt_id=attempt_id,
                model_name=model_name,
                usage_table="public.vibode_gemini_usage_events",
            )
    except Exception as e:
        log_event(
            "gemini_usage_accounting_write_failed",
            error=str(e),
            attempt_id=attempt_id,
            model_name=model_name,
            usage_table="public.vibode_gemini_usage_events",
        )


def _run_user_sku_background_removal(image_png_bytes: bytes, model_name: str) -> bytes:
    return call_gemini_with_prompt(
        image_png_bytes=image_png_bytes,
        prompt=USER_SKU_BG_REMOVAL_PROMPT,
        model_name=model_name,
        aspect_ratio=None,
    )


def _run_user_sku_clipboard_product_isolation(image_png_bytes: bytes, model_name: str) -> bytes:
    return call_gemini_with_prompt(
        image_png_bytes=image_png_bytes,
        prompt=USER_SKU_CLIPBOARD_ISOLATION_PROMPT,
        model_name=model_name,
        aspect_ratio=None,
    )


def _normalize_user_sku_transparent_png(
    image_bytes: bytes,
    max_dimension: int = USER_SKU_NORMALIZED_MAX_DIM,
    padding_ratio: float = USER_SKU_NORMALIZED_PADDING_RATIO,
    background_rgb: Optional[Tuple[int, int, int]] = None,
) -> Tuple[bytes, Tuple[int, int]]:
    with Image.open(io.BytesIO(image_bytes)) as img:
        rgba = img.convert("RGBA")

    alpha = rgba.getchannel("A")
    non_transparent_box = alpha.getbbox()
    if non_transparent_box:
        trimmed = rgba.crop(non_transparent_box)
    else:
        trimmed = rgba

    width, height = trimmed.size
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid dimensions after background removal.")

    long_edge = max(width, height)
    if long_edge > max_dimension:
        scale = max_dimension / float(long_edge)
        resized = trimmed.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            resample=Image.LANCZOS,
        )
    else:
        resized = trimmed

    pad = max(2, int(round(max(resized.size) * max(0.0, padding_ratio))))
    padded = Image.new("RGBA", (resized.width + pad * 2, resized.height + pad * 2), (0, 0, 0, 0))
    padded.paste(resized, (pad, pad), resized)

    padded_long_edge = max(padded.size)
    if padded_long_edge > max_dimension:
        scale = max_dimension / float(padded_long_edge)
        padded = padded.resize(
            (
                max(1, int(round(padded.width * scale))),
                max(1, int(round(padded.height * scale))),
            ),
            resample=Image.LANCZOS,
        )

    if background_rgb is not None:
        flattened = Image.new("RGB", padded.size, background_rgb)
        flattened.paste(padded, (0, 0), padded)
        return image_to_png_bytes(flattened), flattened.size

    return image_to_png_bytes(padded), padded.size


def _normalize_user_sku_solid_bg_png(
    image_bytes: bytes,
    max_dimension: int,
    padding_ratio: float,
    bg_rgb: Tuple[int, int, int] = (242, 242, 242),
) -> Tuple[bytes, Tuple[int, int]]:
    with Image.open(io.BytesIO(image_bytes)) as img:
        rgb = img.convert("RGB")

    width, height = rgb.size
    if width <= 0 or height <= 0:
        raise RuntimeError("Invalid dimensions after background removal.")

    long_edge = max(width, height)
    if long_edge > max_dimension:
        scale = max_dimension / float(long_edge)
        resized = rgb.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            resample=Image.LANCZOS,
        )
    else:
        resized = rgb

    pad = max(2, int(round(max(resized.size) * max(0.0, padding_ratio))))
    padded = Image.new("RGB", (resized.width + pad * 2, resized.height + pad * 2), bg_rgb)
    padded.paste(resized, (pad, pad))

    padded_long_edge = max(padded.size)
    if padded_long_edge > max_dimension:
        scale = max_dimension / float(padded_long_edge)
        padded = padded.resize(
            (
                max(1, int(round(padded.width * scale))),
                max(1, int(round(padded.height * scale))),
            ),
            resample=Image.LANCZOS,
        )

    return image_to_png_bytes(padded), padded.size


def _draw_contrasted_line(
    draw: "ImageDraw.ImageDraw",
    points: Tuple[int, int, int, int],
    line_width: int,
    inner_color: Tuple[int, int, int] = (255, 255, 255),
    outer_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    outer_width = line_width + max(2, int(round(line_width * 0.65)))
    draw.line(points, fill=outer_color, width=outer_width)
    draw.line(points, fill=inner_color, width=line_width)


def _draw_swap_glyph(
    draw: "ImageDraw.ImageDraw",
    cx: int,
    cy: int,
    glyph_half_len: int,
) -> None:
    track_gap = max(8, int(round(glyph_half_len * 0.6)))
    top_y = cy - (track_gap // 2)
    bottom_y = cy + (track_gap // 2)
    left_x = cx - glyph_half_len
    right_x = cx + glyph_half_len
    line_width = max(3, int(round(glyph_half_len * 0.2)))
    head_len = max(7, int(round(glyph_half_len * 0.35)))
    head_height = max(6, int(round(glyph_half_len * 0.32)))

    # Top lane points right.
    _draw_contrasted_line(draw, (left_x, top_y, right_x, top_y), line_width=line_width)
    _draw_contrasted_line(draw, (right_x, top_y, right_x - head_len, top_y - head_height), line_width=line_width)
    _draw_contrasted_line(draw, (right_x, top_y, right_x - head_len, top_y + head_height), line_width=line_width)

    # Bottom lane points left.
    _draw_contrasted_line(draw, (right_x, bottom_y, left_x, bottom_y), line_width=line_width)
    _draw_contrasted_line(draw, (left_x, bottom_y, left_x + head_len, bottom_y - head_height), line_width=line_width)
    _draw_contrasted_line(draw, (left_x, bottom_y, left_x + head_len, bottom_y + head_height), line_width=line_width)


def _draw_swap_number_badge(
    draw: "ImageDraw.ImageDraw",
    cx: int,
    cy: int,
    marker_label: str,
    glyph_half_len: int,
    width: int,
    height: int,
) -> None:
    badge_radius = max(10, int(round(glyph_half_len * 0.48)))
    badge_cx = cx + glyph_half_len
    badge_cy = cy - glyph_half_len
    badge_cx = max(badge_radius, min(badge_cx, width - 1 - badge_radius))
    badge_cy = max(badge_radius, min(badge_cy, height - 1 - badge_radius))
    badge_bbox = (
        badge_cx - badge_radius,
        badge_cy - badge_radius,
        badge_cx + badge_radius,
        badge_cy + badge_radius,
    )
    outline_width = max(2, int(round(badge_radius * 0.18)))
    draw.ellipse(badge_bbox, fill=(0, 0, 0), outline=(255, 255, 255), width=outline_width)
    _draw_marker_label(
        draw=draw,
        cx=badge_cx,
        cy=badge_cy,
        marker_label=marker_label,
        radius=badge_radius,
    )


def _draw_red_marker_pixels(img: Image.Image, cx: int, cy: int, r: int) -> None:
    pixels = img.load()
    if pixels is None:
        return
    width, height = img.size
    radius = max(2, int(r))
    min_x = max(0, cx - radius)
    max_x = min(width - 1, cx + radius)
    min_y = max(0, cy - radius)
    max_y = min(height - 1, cy + radius)
    radius_sq = radius * radius
    for y in range(min_y, max_y + 1):
        dy = y - cy
        for x in range(min_x, max_x + 1):
            dx = x - cx
            if (dx * dx + dy * dy) <= radius_sq:
                pixels[x, y] = (220, 36, 36)


def order_vibode_placements(placements: List[VibodePlacement]) -> List[VibodePlacement]:
    has_z_index = any(placement.zIndex is not None for placement in placements)
    if has_z_index:
        return sorted(
            placements,
            key=lambda placement: (
                placement.zIndex if placement.zIndex is not None else float("inf"),
                placement.nodeId,
            ),
        )
    return sorted(placements, key=lambda placement: placement.nodeId)


def _load_marker_label_font(font_size: int):
    if ImageFont is None:
        return None
    for font_name in ("DejaVuSans-Bold.ttf", "Arial.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(font_name, size=font_size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _draw_marker_label(
    draw: "ImageDraw.ImageDraw",
    cx: int,
    cy: int,
    marker_label: str,
    radius: int,
    font_scale: float = 0.9,
    fill: Tuple[int, ...] = (255, 255, 255),
    stroke_fill: Tuple[int, ...] = (0, 0, 0),
    stroke_scale: float = 0.12,
) -> None:
    font_size = max(10, int(radius * font_scale))
    font = _load_marker_label_font(font_size)
    stroke_width = max(1, int(round(radius * stroke_scale)))
    try:
        draw.text(
            (cx, cy),
            marker_label,
            fill=fill,
            font=font,
            anchor="mm",
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
        return
    except Exception:
        pass

    try:
        bbox = draw.textbbox((0, 0), marker_label, font=font, stroke_width=stroke_width)
        left, top, right, bottom = bbox
        x = cx - ((right - left) / 2.0)
        y = cy - ((bottom - top) / 2.0)
    except Exception:
        x = cx
        y = cy
    draw.text(
        (x, y),
        marker_label,
        fill=fill,
        font=font,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
    )


VIBODE_COMPOSE_MARKER_STYLE = "remove_style_numbered_red_badges_v1"
VIBODE_COMPOSE_MARKER_RADIUS_PX = 24
VIBODE_COMPOSE_MARKER_COLOR_FAMILY = "red_white"
VIBODE_COMPOSE_MARKER_OPACITY = 1.0  # Keep markers fully opaque for deterministic index readability.
VIBODE_COMPOSE_MARKER_FILL_RGB = (220, 36, 36)
VIBODE_COMPOSE_MARKER_OUTLINE_RGB = (255, 255, 255)
VIBODE_COMPOSE_MARKER_LABEL_FILL_RGB = (255, 255, 255)
VIBODE_COMPOSE_MARKER_LABEL_STROKE_RGB = (0, 0, 0)
VIBODE_COMPOSE_MARKER_OUTLINE_WIDTH_SCALE = 0.14
VIBODE_COMPOSE_MARKER_OUTLINE_WIDTH_MIN_PX = 2


def draw_red_markers_overlay(image_png_bytes: bytes, placements: List[VibodePlacement]) -> bytes:
    # Image 2 is a semantic placement instruction map: robust numbered badges are preferred
    # here to keep marker-index mapping stable across multi-item compose runs.
    img = _safe_open_image(image_png_bytes)
    if img.mode != "RGB":
        img = img.convert("RGB")
    width, height = img.size
    max_radius = max(1, min(width, height) // 4)
    marker_fill = VIBODE_COMPOSE_MARKER_FILL_RGB
    marker_outline = VIBODE_COMPOSE_MARKER_OUTLINE_RGB
    if _HAS_IMAGE_DRAW and ImageDraw is not None:
        draw = ImageDraw.Draw(img)
        for idx, placement in enumerate(placements):
            radius = int(round(placement.rPx)) if placement.rPx else VIBODE_COMPOSE_MARKER_RADIUS_PX
            radius = max(16, radius)
            radius = min(radius, max_radius)
            cx = int(round(placement.cxPx))
            cy = int(round(placement.cyPx))
            cx = max(0, min(cx, width - 1))
            cy = max(0, min(cy, height - 1))
            badge_bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
            outline_width = max(
                VIBODE_COMPOSE_MARKER_OUTLINE_WIDTH_MIN_PX,
                int(round(radius * VIBODE_COMPOSE_MARKER_OUTLINE_WIDTH_SCALE)),
            )
            draw.ellipse(badge_bbox, fill=marker_fill, outline=marker_outline, width=outline_width)
            marker_label = str(idx + 1)
            _draw_marker_label(
                draw=draw,
                cx=cx,
                cy=cy,
                marker_label=marker_label,
                radius=radius,
                font_scale=0.92,
                fill=VIBODE_COMPOSE_MARKER_LABEL_FILL_RGB,
                stroke_fill=VIBODE_COMPOSE_MARKER_LABEL_STROKE_RGB,
                stroke_scale=0.14,
            )
    else:
        for placement in placements:
            radius = int(round(placement.rPx)) if placement.rPx else VIBODE_COMPOSE_MARKER_RADIUS_PX
            radius = max(16, radius)
            radius = min(radius, max_radius)
            cx = int(round(placement.cxPx))
            cy = int(round(placement.cyPx))
            cx = max(0, min(cx, width - 1))
            cy = max(0, min(cy, height - 1))
            _draw_red_marker_pixels(img, cx, cy, radius)
    return image_to_png_bytes(img)


def _draw_red_x_pixels(img: Image.Image, cx: int, cy: int, radius: int) -> None:
    pixels = img.load()
    if pixels is None:
        return
    width, height = img.size
    thickness = 3
    for offset in range(-radius, radius + 1):
        x1 = cx + offset
        y1 = cy + offset
        x2 = cx + offset
        y2 = cy - offset
        for spread in range(-thickness, thickness + 1):
            px1 = x1 + spread
            py1 = y1
            px2 = x2 + spread
            py2 = y2
            if 0 <= px1 < width and 0 <= py1 < height:
                pixels[px1, py1] = (255, 0, 0)
            if 0 <= px2 < width and 0 <= py2 < height:
                pixels[px2, py2] = (255, 0, 0)


def draw_red_x_overlay(image_png_bytes: bytes, marks: List[VibodeRemoveMark]) -> bytes:
    img = _safe_open_image(image_png_bytes)
    if img.mode != "RGB":
        img = img.convert("RGB")
    width, height = img.size
    max_radius = max(1, min(width, height) // 4)
    if _HAS_IMAGE_DRAW and ImageDraw is not None:
        draw = ImageDraw.Draw(img)
        for mark in marks:
            radius = int(round(mark.r)) if mark.r else 60
            radius = max(20, radius)
            radius = min(radius, max_radius)
            cx = int(round(mark.x))
            cy = int(round(mark.y))
            cx = max(0, min(cx, width - 1))
            cy = max(0, min(cy, height - 1))
            draw.line((cx - radius, cy - radius, cx + radius, cy + radius), fill=(255, 0, 0), width=6)
            draw.line((cx - radius, cy + radius, cx + radius, cy - radius), fill=(255, 0, 0), width=6)
            if mark.labelIndex is not None:
                _draw_marker_label(draw, cx, cy, str(mark.labelIndex), radius)
    else:
        for mark in marks:
            radius = int(round(mark.r)) if mark.r else 60
            radius = max(20, radius)
            radius = min(radius, max_radius)
            cx = int(round(mark.x))
            cy = int(round(mark.y))
            cx = max(0, min(cx, width - 1))
            cy = max(0, min(cy, height - 1))
            _draw_red_x_pixels(img, cx, cy, radius)
    return image_to_png_bytes(img)


def render_vibode_swap_overlay(image_png_bytes: bytes, marks: List[VibodeSwapMark]) -> bytes:
    img = _safe_open_image(image_png_bytes)
    if img.mode != "RGB":
        img = img.convert("RGB")
    width, height = img.size
    max_glyph_half_len = max(1, min(width, height) // 4)
    min_dim = min(width, height)
    base_half_len = int(round(min_dim * 0.045))
    base_half_len = max(20, min(base_half_len, 72, max_glyph_half_len))

    if _HAS_IMAGE_DRAW and ImageDraw is not None:
        draw = ImageDraw.Draw(img)
        for idx, mark in enumerate(marks):
            cx = int(round(mark.x))
            cy = int(round(mark.y))
            cx = max(0, min(cx, width - 1))
            cy = max(0, min(cy, height - 1))
            _draw_swap_glyph(draw=draw, cx=cx, cy=cy, glyph_half_len=base_half_len)
            _draw_swap_number_badge(
                draw=draw,
                cx=cx,
                cy=cy,
                marker_label=str(idx + 1),
                glyph_half_len=base_half_len,
                width=width,
                height=height,
            )
    else:
        # Keep a deterministic fallback if ImageDraw is unavailable.
        for mark in marks:
            radius = max(20, base_half_len)
            cx = int(round(mark.x))
            cy = int(round(mark.y))
            cx = max(0, min(cx, width - 1))
            cy = max(0, min(cy, height - 1))
            _draw_red_x_pixels(img, cx, cy, radius)

    return image_to_png_bytes(img)


def _clamp_rotation_degrees(angle_deg: float) -> float:
    if not math.isfinite(angle_deg):
        return 0.0
    return max(-180.0, min(180.0, float(angle_deg)))


def _normalized_to_pixel(normalized_value: float, size: int) -> int:
    if size <= 1:
        return 0
    clamped = max(0.0, min(1.0, float(normalized_value)))
    return max(0, min(int(round(clamped * (size - 1))), size - 1))


def _draw_colored_disc_pixels(
    img: Image.Image,
    cx: int,
    cy: int,
    radius: int,
    color: Tuple[int, int, int],
) -> None:
    pixels = img.load()
    if pixels is None:
        return
    width, height = img.size
    r = max(1, int(radius))
    min_x = max(0, cx - r)
    max_x = min(width - 1, cx + r)
    min_y = max(0, cy - r)
    max_y = min(height - 1, cy + r)
    radius_sq = r * r
    for y in range(min_y, max_y + 1):
        dy = y - cy
        dy_sq = dy * dy
        for x in range(min_x, max_x + 1):
            dx = x - cx
            if (dx * dx + dy_sq) <= radius_sq:
                pixels[x, y] = color


def _draw_colored_line_pixels(
    img: Image.Image,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    steps = max(abs(x2 - x1), abs(y2 - y1), 1)
    half = max(0, int(round(thickness / 2)))
    pixels = img.load()
    if pixels is None:
        return
    width, height = img.size

    for step in range(steps + 1):
        t = step / float(steps)
        x = int(round(x1 + ((x2 - x1) * t)))
        y = int(round(y1 + ((y2 - y1) * t)))
        for oy in range(-half, half + 1):
            py = y + oy
            if py < 0 or py >= height:
                continue
            for ox in range(-half, half + 1):
                px = x + ox
                if 0 <= px < width:
                    pixels[px, py] = color


def _draw_purple_marker_pixels(img: Image.Image, cx: int, cy: int, radius: int) -> None:
    pixels = img.load()
    if pixels is None:
        return
    width, height = img.size
    thickness = 3
    min_x = max(0, cx - radius - thickness)
    max_x = min(width - 1, cx + radius + thickness)
    min_y = max(0, cy - radius - thickness)
    max_y = min(height - 1, cy + radius + thickness)
    inner_radius = max(0, radius - thickness)
    outer_radius = radius + thickness
    inner_radius_sq = inner_radius * inner_radius
    outer_radius_sq = outer_radius * outer_radius
    purple = (160, 90, 255)
    for y in range(min_y, max_y + 1):
        dy = y - cy
        dy_sq = dy * dy
        for x in range(min_x, max_x + 1):
            dx = x - cx
            dist_sq = dx * dx + dy_sq
            if inner_radius_sq <= dist_sq <= outer_radius_sq:
                pixels[x, y] = purple


def render_vibode_rotate_overlay(image_png_bytes: bytes, marks: List[VibodeRotateMark]) -> bytes:
    img = _safe_open_image(image_png_bytes)
    if img.mode != "RGB":
        img = img.convert("RGB")
    width, height = img.size
    min_dim = min(width, height)
    marker_radius = max(18, int(round(min_dim * 0.038)))
    marker_radius = min(marker_radius, 64)
    line_width = max(3, int(round(marker_radius * 0.22)))
    min_arrow_len = max(marker_radius + 8, int(round(marker_radius * 1.1)))
    max_arrow_len = max(min_arrow_len + 10, int(round(marker_radius * 2.4)))
    purple = (160, 90, 255)

    if _HAS_IMAGE_DRAW and ImageDraw is not None:
        draw = ImageDraw.Draw(img)
        for idx, mark in enumerate(marks):
            clamped_angle_deg = _clamp_rotation_degrees(mark.angleDeg)
            cx = _normalized_to_pixel(mark.x, width)
            cy = _normalized_to_pixel(mark.y, height)

            marker_bbox = (
                cx - marker_radius,
                cy - marker_radius,
                cx + marker_radius,
                cy + marker_radius,
            )
            draw.ellipse(marker_bbox, outline=purple, width=max(3, int(round(marker_radius * 0.16))))
            _draw_marker_label(draw=draw, cx=cx, cy=cy, marker_label=str(idx + 1), radius=marker_radius)

            angle_fraction = abs(clamped_angle_deg) / 180.0
            arrow_len = int(round(min_arrow_len + ((max_arrow_len - min_arrow_len) * angle_fraction)))
            theta = math.radians(-clamped_angle_deg)
            end_x = int(round(cx + (math.sin(theta) * arrow_len)))
            end_y = int(round(cy - (math.cos(theta) * arrow_len)))
            end_x = max(0, min(end_x, width - 1))
            end_y = max(0, min(end_y, height - 1))

            draw.line((cx, cy, end_x, end_y), fill=purple, width=line_width)

            head_len = max(8, int(round(marker_radius * 0.45)))
            direction_theta = math.atan2((end_y - cy), (end_x - cx))
            left_theta = direction_theta + math.radians(150)
            right_theta = direction_theta - math.radians(150)
            left_x = int(round(end_x + (math.cos(left_theta) * head_len)))
            left_y = int(round(end_y + (math.sin(left_theta) * head_len)))
            right_x = int(round(end_x + (math.cos(right_theta) * head_len)))
            right_y = int(round(end_y + (math.sin(right_theta) * head_len)))
            draw.line((end_x, end_y, left_x, left_y), fill=purple, width=line_width)
            draw.line((end_x, end_y, right_x, right_y), fill=purple, width=line_width)
    else:
        for mark in marks:
            cx = _normalized_to_pixel(mark.x, width)
            cy = _normalized_to_pixel(mark.y, height)
            _draw_purple_marker_pixels(img, cx, cy, marker_radius)

    return image_to_png_bytes(img)


def scale_placements_for_resized_room_image(
    placements: List[VibodePlacement],
    original_size: Tuple[int, int],
    resized_size: Tuple[int, int],
) -> List[VibodePlacement]:
    orig_w, orig_h = original_size
    new_w, new_h = resized_size

    if orig_w <= 0 or orig_h <= 0 or new_w <= 0 or new_h <= 0:
        return placements

    if (orig_w, orig_h) == (new_w, new_h):
        return placements

    scale_x = new_w / float(orig_w)
    scale_y = new_h / float(orig_h)
    radius_scale = min(scale_x, scale_y)

    scaled_placements: List[VibodePlacement] = []
    for placement in placements:
        scaled_placements.append(
            placement.model_copy(
                update={
                    "cxPx": placement.cxPx * scale_x,
                    "cyPx": placement.cyPx * scale_y,
                    "rPx": (placement.rPx * radius_scale) if placement.rPx is not None else None,
                }
            )
        )

    if placements and scaled_placements:
        first_original = placements[0]
        first_scaled = scaled_placements[0]
        print(
            "[/vibode/compose] Scaling marker coordinates for resized room image:",
            {
                "orig_dims": (orig_w, orig_h),
                "new_dims": (new_w, new_h),
                "scale_x": scale_x,
                "scale_y": scale_y,
                "first_original": {
                    "nodeId": first_original.nodeId,
                    "cxPx": first_original.cxPx,
                    "cyPx": first_original.cyPx,
                    "rPx": first_original.rPx,
                },
                "first_scaled": {
                    "nodeId": first_scaled.nodeId,
                    "cxPx": first_scaled.cxPx,
                    "cyPx": first_scaled.cyPx,
                    "rPx": first_scaled.rPx,
                },
            },
        )

    return scaled_placements


def scale_marks_for_resized_image(
    marks: List[VibodeRemoveMark],
    original_size: Tuple[int, int],
    resized_size: Tuple[int, int],
) -> List[VibodeRemoveMark]:
    orig_w, orig_h = original_size
    new_w, new_h = resized_size

    if orig_w <= 0 or orig_h <= 0 or new_w <= 0 or new_h <= 0:
        return marks

    if (orig_w, orig_h) == (new_w, new_h):
        return marks

    scale_x = new_w / float(orig_w)
    scale_y = new_h / float(orig_h)
    radius_scale = min(scale_x, scale_y)

    scaled_marks: List[VibodeRemoveMark] = []
    for mark in marks:
        scaled_marks.append(
            mark.model_copy(
                update={
                    "x": mark.x * scale_x,
                    "y": mark.y * scale_y,
                    "r": mark.r * radius_scale,
                }
            )
        )
    return scaled_marks


def scale_swap_marks_for_resized_image(
    marks: List[VibodeSwapMark],
    original_size: Tuple[int, int],
    resized_size: Tuple[int, int],
) -> List[VibodeSwapMark]:
    orig_w, orig_h = original_size
    new_w, new_h = resized_size

    if orig_w <= 0 or orig_h <= 0 or new_w <= 0 or new_h <= 0:
        return marks

    if (orig_w, orig_h) == (new_w, new_h):
        return marks

    scale_x = new_w / float(orig_w)
    scale_y = new_h / float(orig_h)

    scaled_marks: List[VibodeSwapMark] = []
    for mark in marks:
        scaled_marks.append(
            mark.model_copy(
                update={
                    "x": mark.x * scale_x,
                    "y": mark.y * scale_y,
                }
            )
        )
    return scaled_marks


def _env_truthy(var_name: str) -> bool:
    value = os.getenv(var_name)
    if not value:
        return False
    return value.strip().lower() in ("1", "true", "yes")


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
    return None


def _extract_user_sku_preview_bg_override_flags(
    ingest_req: VibodeUserSkuIngestRequest,
    http_request: Request,
) -> Dict[str, Any]:
    normalization_block = ingest_req.normalization if isinstance(ingest_req.normalization, dict) else {}
    header_bg_rgb = (http_request.headers.get("x-roomprintz-normalized-preview-bg-rgb") or "").strip()
    header_bg_mode = (http_request.headers.get("x-roomprintz-normalized-preview-bg-mode") or "").strip()
    body_bg_mode_raw = normalization_block.get("previewBackgroundMode")
    body_bg_rgb_raw = normalization_block.get("previewBackgroundRgb")
    body_bg_mode = body_bg_mode_raw.strip() if isinstance(body_bg_mode_raw, str) else ""
    body_bg_rgb = body_bg_rgb_raw.strip() if isinstance(body_bg_rgb_raw, str) else ""
    disable_sampled = _coerce_optional_bool(normalization_block.get("disableSampledBackground"))
    disable_dominant = _coerce_optional_bool(normalization_block.get("disableDominantBackground"))

    override_requested = any(
        (
            bool(header_bg_rgb),
            bool(header_bg_mode),
            bool(body_bg_mode),
            bool(body_bg_rgb),
            disable_sampled is True,
            disable_dominant is True,
        )
    )

    return {
        "override_requested": override_requested,
        "header_bg_rgb_present": bool(header_bg_rgb),
        "header_bg_mode_present": bool(header_bg_mode),
        "body_bg_mode_present": bool(body_bg_mode),
        "body_bg_rgb_present": bool(body_bg_rgb),
        "disable_sampled_background": disable_sampled,
        "disable_dominant_background": disable_dominant,
    }


# Toggle full prompt logging for Vibode routes (off by default to avoid log spam)
VIBODE_LOG_PROMPTS = _env_truthy("VIBODE_LOG_PROMPTS")
# Strict validation guardrails for Vibode routes (off by default).
VIBODE_STRICT = _env_truthy("VIBODE_STRICT")


def _append_missing_nonempty_str(
    missing_fields: List[str],
    field_path: str,
    value: Optional[str],
) -> None:
    if not value or not value.strip():
        missing_fields.append(field_path)


def _reject_if_vibode_strict_missing(route_tag: str, missing_fields: List[str]) -> None:
    if not VIBODE_STRICT or not missing_fields:
        return
    print(
        f"[{route_tag}] VIBODE_STRICT reject missing required fields:",
        ", ".join(missing_fields),
    )
    raise HTTPException(
        status_code=400,
        detail=f"Missing required fields: {', '.join(missing_fields)}",
    )


def _collect_vibode_compose_missing_fields(req: VibodeComposeRequest) -> List[str]:
    missing_fields: List[str] = []
    _append_missing_nonempty_str(missing_fields, "roomImageBase64", req.roomImageBase64)
    if not req.placements:
        missing_fields.append("placements")
        return missing_fields

    for idx, placement in enumerate(req.placements):
        _append_missing_nonempty_str(missing_fields, f"placements[{idx}].nodeId", placement.nodeId)
        _append_missing_nonempty_str(
            missing_fields,
            f"placements[{idx}].skuImageBase64",
            placement.skuImageBase64,
        )
    return missing_fields


def _collect_vibode_vibe_missing_fields(req: VibodeVibeRequest) -> List[str]:
    missing_fields: List[str] = []
    _append_missing_nonempty_str(missing_fields, "roomImageBase64", req.roomImageBase64)
    if not req.eligibleSkus:
        missing_fields.append("eligibleSkus")
        return missing_fields

    for idx, sku in enumerate(req.eligibleSkus):
        _append_missing_nonempty_str(missing_fields, f"eligibleSkus[{idx}].skuId", sku.skuId)
    return missing_fields


def _collect_vibode_full_vibe_missing_fields(req: VibodeFreezeRequest) -> List[str]:
    missing_fields: List[str] = []
    base_image = req.freeze.baseImage if req.freeze else None
    if not base_image:
        missing_fields.append("freeze.baseImage")
        return missing_fields

    has_signed_url = bool(base_image.signedUrl and base_image.signedUrl.strip())
    has_base64 = bool(base_image.base64 and base_image.base64.strip())
    if not has_signed_url and not has_base64:
        missing_fields.append("freeze.baseImage.signedUrl")
    return missing_fields


def _collect_vibode_stage_run_missing_fields(req: VibodeStageRunRequest) -> List[str]:
    missing_fields: List[str] = []
    has_room_b64 = bool(req.roomImageBase64 and req.roomImageBase64.strip())
    has_base_b64 = bool(req.baseImageBase64 and req.baseImageBase64.strip())
    has_base_url = bool(req.baseImageUrl and req.baseImageUrl.strip())
    has_base_id = bool(req.baseImageId and req.baseImageId.strip())
    if not has_room_b64 and not has_base_b64 and not has_base_url and not has_base_id:
        missing_fields.append("roomImageBase64|baseImageBase64|baseImageUrl|baseImageId")

    if req.stage == 3 and not req.eligibleSkus:
        missing_fields.append("eligibleSkus")

    return missing_fields


def _collect_vibode_remove_missing_fields(req: VibodeRemoveRequest) -> List[str]:
    missing_fields: List[str] = []
    _append_missing_nonempty_str(missing_fields, "cleanBase64", req.cleanBase64)
    if not req.marks:
        missing_fields.append("marks")
        return missing_fields

    for idx, mark in enumerate(req.marks):
        _append_missing_nonempty_str(missing_fields, f"marks[{idx}].id", mark.id)
    return missing_fields


def _collect_vibode_swap_missing_fields(req: VibodeSwapRequest) -> List[str]:
    missing_fields: List[str] = []
    _append_missing_nonempty_str(missing_fields, "cleanBase64", req.cleanBase64)

    if not req.marks:
        missing_fields.append("marks")
    else:
        for idx, mark in enumerate(req.marks):
            _append_missing_nonempty_str(missing_fields, f"marks[{idx}].id", mark.id)
            if not mark.replacement:
                missing_fields.append(f"marks[{idx}].replacement")
            else:
                _append_missing_nonempty_str(
                    missing_fields,
                    f"marks[{idx}].replacement.imageUrl",
                    mark.replacement.imageUrl,
                )

    if not req.replacementAssets:
        missing_fields.append("replacementAssets")
    else:
        for idx, asset in enumerate(req.replacementAssets):
            _append_missing_nonempty_str(
                missing_fields,
                f"replacementAssets[{idx}].imageUrl",
                asset.imageUrl,
            )

    return missing_fields


def _short_prompt_hash(prompt: str, length: int = 12) -> str:
    normalized = (prompt or "").strip()
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:length]


def _prompt_first_line(prompt: str, max_chars: int = 80) -> str:
    if not prompt:
        return ""
    first_line = next((line.strip() for line in prompt.splitlines() if line.strip()), "").strip()
    if len(first_line) <= max_chars:
        return first_line
    return first_line[:max_chars].rstrip() + "..."


def _summarize_remove_marks(
    marks: List[VibodeRemoveMark],
    limit: int = 5,
) -> Dict[str, object]:
    preview: List[Tuple[Optional[int], float, float, float]] = []
    for mark in marks[:limit]:
        preview.append(
            (
                mark.labelIndex,
                round(mark.x, 1),
                round(mark.y, 1),
                round(mark.r, 1),
            )
        )
    return {
        "count": len(marks),
        "preview": preview,
    }


def _summarize_swap_marks(
    marks: List[VibodeSwapMark],
    replacement_assets: List[VibodeSwapReplacementAsset],
    limit: int = 5,
) -> Dict[str, object]:
    preview: List[Dict[str, object]] = []
    replacement_count = len(replacement_assets)
    for idx, mark in enumerate(marks[:limit]):
        mapped_asset = replacement_assets[idx] if idx < replacement_count else None
        preview.append(
            {
                "markerIndex": idx + 1,
                "markId": mark.id,
                "x": round(mark.x, 1),
                "y": round(mark.y, 1),
                "markReplacementSkuId": mark.replacement.skuId,
                "mappedAssetSkuId": mapped_asset.skuId if mapped_asset else None,
            }
        )
    return {
        "count": len(marks),
        "replacementAssets": replacement_count,
        "preview": preview,
    }


def _summarize_rotate_marks(
    marks: List[VibodeRotateMark],
    limit: int = 5,
) -> Dict[str, object]:
    preview: List[Dict[str, object]] = []
    for idx, mark in enumerate(marks[:limit]):
        preview.append(
            {
                "markerIndex": idx + 1,
                "markId": mark.id,
                "xNorm": round(mark.x, 4),
                "yNorm": round(mark.y, 4),
                "angleDeg": round(mark.angleDeg, 2),
            }
        )
    return {
        "count": len(marks),
        "preview": preview,
    }


def maybe_dump_prepared_room_images(
    room_clean_png_bytes: bytes,
    room_marked_png_bytes: bytes,
    output_dir: Optional[str] = None,
) -> None:
    if not _env_truthy("VIBODE_DUMP_ANNOTATED_IMAGE"):
        return
    try:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        resolved_dir = output_dir or os.getenv("VIBODE_DEBUG_DIR") or "tmp/vibode_debug"
        abs_dir = os.path.abspath(resolved_dir)
        os.makedirs(abs_dir, exist_ok=True)

        artifacts = (
            ("prepared_room_clean", room_clean_png_bytes),
            ("prepared_room_marked", room_marked_png_bytes),
        )
        for prefix, image_png_bytes in artifacts:
            abs_path = os.path.join(abs_dir, f"{prefix}_{timestamp}.png")
            print("[maybe_dump_prepared_room_images] target path:", abs_path)
            try:
                with open(abs_path, "wb") as handle:
                    handle.write(image_png_bytes)
                print("[maybe_dump_prepared_room_images] wrote:", abs_path)
            except Exception as e:
                print("[maybe_dump_prepared_room_images] failed:", e)
    except Exception as e:
        print("[maybe_dump_prepared_room_images] failed:", e)


def maybe_dump_vibode_swap_images(
    room_clean_png_bytes: bytes,
    room_swap_overlay_png_bytes: bytes,
    output_dir: Optional[str] = None,
) -> None:
    if not _env_truthy("VIBODE_DUMP_ANNOTATED_IMAGE"):
        return

    try:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        resolved_dir = output_dir or os.getenv("VIBODE_DEBUG_DIR") or "tmp/vibode_debug"
        abs_dir = os.path.abspath(resolved_dir)
        run_dir = os.path.join(abs_dir, f"swap_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        clean_path = os.path.join(run_dir, "clean_normalized.png")
        overlay_path = os.path.join(run_dir, "swap_overlay.png")
        preview_path = os.path.join(run_dir, "swap_preview_combined.png")

        with open(clean_path, "wb") as handle:
            handle.write(room_clean_png_bytes)
        with open(overlay_path, "wb") as handle:
            handle.write(room_swap_overlay_png_bytes)

        try:
            clean_img = _safe_open_image(room_clean_png_bytes)
            overlay_img = _safe_open_image(room_swap_overlay_png_bytes)
            preview_w = clean_img.width + overlay_img.width
            preview_h = max(clean_img.height, overlay_img.height)
            combined = Image.new("RGB", (preview_w, preview_h), color=(30, 30, 30))
            combined.paste(clean_img, (0, 0))
            combined.paste(overlay_img, (clean_img.width, 0))
            with open(preview_path, "wb") as handle:
                handle.write(image_to_png_bytes(combined))
        except Exception as e:
            print("[maybe_dump_vibode_swap_images] failed to write preview:", e)

        print(
            "[maybe_dump_vibode_swap_images] wrote:",
            {
                "clean_normalized": clean_path,
                "swap_overlay": overlay_path,
                "swap_preview_combined": preview_path,
            },
        )
    except Exception as e:
        print("[maybe_dump_vibode_swap_images] failed:", e)


def maybe_dump_vibode_rotate_images(
    room_clean_png_bytes: bytes,
    room_rotate_overlay_png_bytes: bytes,
    prompt_hash: str,
    output_dir: Optional[str] = None,
) -> None:
    if not _env_truthy("VIBODE_DUMP_ANNOTATED_IMAGE"):
        return

    try:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        resolved_dir = output_dir or os.getenv("VIBODE_DEBUG_DIR") or "tmp/vibode_debug"
        abs_dir = os.path.abspath(resolved_dir)
        prompt_tag = prompt_hash or "nohash"
        run_dir = os.path.join(abs_dir, f"rotate_{timestamp}_{prompt_tag}")
        os.makedirs(run_dir, exist_ok=True)

        clean_path = os.path.join(run_dir, "clean_normalized.png")
        overlay_path = os.path.join(run_dir, "rotate_overlay.png")
        preview_path = os.path.join(run_dir, "rotate_preview_combined.png")

        with open(clean_path, "wb") as handle:
            handle.write(room_clean_png_bytes)
        with open(overlay_path, "wb") as handle:
            handle.write(room_rotate_overlay_png_bytes)

        try:
            clean_img = _safe_open_image(room_clean_png_bytes)
            overlay_img = _safe_open_image(room_rotate_overlay_png_bytes)
            preview_w = clean_img.width + overlay_img.width
            preview_h = max(clean_img.height, overlay_img.height)
            combined = Image.new("RGB", (preview_w, preview_h), color=(30, 30, 30))
            combined.paste(clean_img, (0, 0))
            combined.paste(overlay_img, (clean_img.width, 0))
            with open(preview_path, "wb") as handle:
                handle.write(image_to_png_bytes(combined))
        except Exception as e:
            print("[maybe_dump_vibode_rotate_images] failed to write preview:", e)

        print(
            "[maybe_dump_vibode_rotate_images] wrote:",
            {
                "clean_normalized": clean_path,
                "rotate_overlay": overlay_path,
                "rotate_preview_combined": preview_path,
                "promptHash": prompt_hash,
            },
        )
    except Exception as e:
        print("[maybe_dump_vibode_rotate_images] failed:", e)


def prepare_sku_png_bytes(image_bytes: bytes) -> bytes:
    img = _safe_open_image(image_bytes)
    img = resize_down_if_needed(img, MAX_INPUT_LONG_EDGE_INT)
    return image_to_png_bytes(img)


def _extract_vibode_vibe_sku_image_ref(value: Any) -> Optional[str]:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.startswith("http://") or candidate.startswith("https://"):
            return candidate
        if candidate.startswith("data:image/"):
            return candidate
        return None

    if isinstance(value, list):
        for item in value:
            resolved = _extract_vibode_vibe_sku_image_ref(item)
            if resolved:
                return resolved
        return None

    if isinstance(value, dict):
        preferred_keys = [
            "imageUrl",
            "imageURL",
            "url",
            "pngUrl",
            "assetUrl",
            "src",
            "image",
            "asset",
        ]
        for key in preferred_keys:
            if key in value:
                resolved = _extract_vibode_vibe_sku_image_ref(value.get(key))
                if resolved:
                    return resolved
        for nested_value in value.values():
            resolved = _extract_vibode_vibe_sku_image_ref(nested_value)
            if resolved:
                return resolved
    return None


def _resolve_vibode_stage_run_room_raw_bytes(req: VibodeStageRunRequest) -> bytes:
    if req.roomImageBase64 and req.roomImageBase64.strip():
        return _decode_base64_image(req.roomImageBase64)

    if req.baseImageBase64 and req.baseImageBase64.strip():
        return _decode_base64_image(req.baseImageBase64)

    if req.baseImageUrl and req.baseImageUrl.strip():
        return _fetch_image_bytes_from_url(req.baseImageUrl.strip())

    if req.baseImageId and req.baseImageId.strip():
        base_image_id = req.baseImageId.strip()
        if base_image_id.startswith("http://") or base_image_id.startswith("https://"):
            return _fetch_image_bytes_from_url(base_image_id)
        if base_image_id.startswith("data:image/"):
            payload_b64 = base_image_id.split(",", 1)[1] if "," in base_image_id else base_image_id
            return _decode_base64_image(payload_b64)

    raise HTTPException(
        status_code=400,
        detail=(
            "Provide roomImageBase64, baseImageBase64, baseImageUrl, "
            "or a URL/data-URL baseImageId."
        ),
    )


def build_vibode_compose_prompt(
    placements: List[VibodePlacement],
    enhance_photo: bool,
) -> str:
    placements_count = len(placements)

    # Keep Image 1 (pixels) and Image 2 (placement metadata) explicitly separated so the
    # model preserves room fidelity while still using unambiguous marker indices.
    lines = [
        "You are a professional real-estate photo editor.",
        "",
        "Input images are provided in this exact order:",
        "1) Image 1 is the clean prepared room photo.",
        "   Use this as the only visual source for final room pixels.",
        f"2) Image 2 is the placement instruction map with numbered red circle markers (1..{placements_count}).",
        "   The placement instruction map is guidance metadata only.",
        "3) Each subsequent image is a single SKU item to insert.",
        "",
        "Sequential mapping:",
        "- marker #1 -> SKU image #3",
        "- marker #2 -> SKU image #4",
        "- continue sequentially for all markers and SKU images",
        "- Do not swap item indices.",
        "",
        "Placement rules:",
        "- Place each SKU item at its corresponding numbered marker location.",
        "- The center of each red marker is the intended placement location.",
        "- Markers are identifiers only and are not part of the final scene.",
        "",
        "Room preservation rules:",
        "- Use Image 1 as the only visual source for the room.",
        "- Preserve the existing room layout, walls, windows, doors, floors, lighting, and camera perspective.",
        "- Do not move, resize, rotate, or reframe the room camera.",
        "- Do not invent extra furniture or decor.",
        "- Only insert the provided SKU items.",
        "- Add realistic contact shadows so items sit naturally in the room.",
        "",
        "Placement-instruction-map rules:",
        "- The placement instruction map is not a visual source for final pixels.",
        "- Treat all guide graphics as temporary invisible metadata after placement is understood.",
        "- Do not render red circles, white numbers, outlines, labels, colors, halos, overlays, or annotation residue.",
        "- The final image must look like a normal real-estate photograph with no graphic overlays.",
        "",
        "Final quality check:",
        "- Before returning the image, inspect the final result carefully.",
        "- Ensure every SKU item is placed at its correct corresponding marker location.",
        "- Remove any remaining guide marks or overlay artifacts completely.",
        "- Do not add text, logos, or watermarks.",
    ]

    if enhance_photo:
        lines += [
            "",
            "Enhance photo quality subtly:",
            "- Correct white balance and exposure without changing room appearance.",
            "- Improve clarity and reduce noise while staying photorealistic.",
        ]

    return "\n".join(lines)

def build_vibode_vibe_prompt(
    eligible_skus: List[VibodeEligibleSku],
    collection_id: Optional[str],
    bundle_id: Optional[str],
    target_count: int,
    enhance_photo: bool,
) -> str:
    lines = [
        "Stage the room photo with furniture from the provided SKU assets.",
        "",
        "Constraints:",
        f"- Place approximately {target_count} furniture items using only provided SKU assets.",
        "- Maintain room geometry and lighting consistency.",
        "- Do not add text, logos, or watermarks.",
        "- Do not alter architecture (windows, doors, walls) beyond necessary furniture occlusion.",
        "",
    ]

    if collection_id and collection_id.strip():
        lines.append(f"Collection ID: {collection_id.strip()}")
    if bundle_id and bundle_id.strip():
        lines.append(f"Bundle ID: {bundle_id.strip()}")
    if collection_id or bundle_id:
        lines.append("")

    lines.append("SKU reference list (in order):")
    for idx, sku in enumerate(eligible_skus, start=1):
        label = (sku.label or "").strip()
        if label:
            lines.append(f"{idx}. {sku.skuId} - {label}")
        else:
            lines.append(f"{idx}. {sku.skuId}")

    if enhance_photo:
        lines += [
            "",
            "Subtly enhance photo quality while preserving the original room appearance.",
        ]

    if DEBUG_ROOMPRINTZ_PROMPT:
        print("\n===== VIBODE VIBE PROMPT SENT TO GEMINI =====\n")
        print("\n".join(lines))
        print("\n==============================================\n")
    return "\n".join(lines)


def build_stage1_prestage_prompt_v1(
    enhance_photo: bool,
    cleanup_room: bool,
    empty_room_mode: bool,
    room_type: Optional[str] = None,
) -> str:
    if empty_room_mode:
        return build_roomprintz_prompt(
            enhance_photo=enhance_photo,
            cleanup_room=False,
            repair_damage=False,
            empty_room=True,
            renovate_room=False,
            repaint_walls=False,
            flooring_preset=None,
            style_id=None,
            room_type=room_type,
        )

    return build_roomprintz_prompt(
        enhance_photo=enhance_photo,
        cleanup_room=cleanup_room,
        repair_damage=False,
        empty_room=False,
        renovate_room=False,
        repaint_walls=False,
        flooring_preset=None,
        style_id=None,
        room_type=room_type,
    )


def build_stage2_surfaces_prompt_v1(
    repair_damage: bool,
    heavy_declutter: bool,
    renovate_room: bool,
    repaint_walls: bool,
    flooring_preset: Optional[str],
    room_type: Optional[str] = None,
) -> str:
    fragments = [BASE_ROOMPRINTZ_INSTRUCTIONS.strip()]

    if room_type:
        key = room_type.strip().lower()
        hint = ROOM_TYPE_HINTS.get(key) or (
            "This room has a specific existing function. Preserve that function and "
            "do not convert it into a different type of room."
        )
        fragments.append(
            f"Room type context:\n- {hint}\n- All edits must keep the room clearly consistent with this function."
        )

    fragments.append(
        "You are given a single interior room photo. Edit this photo in-place for a surfaces/finishes pass."
    )

    if heavy_declutter:
        fragments.append(HEAVY_DECLUTTER_FRAGMENT.strip())
    if repair_damage:
        fragments.append(REPAIR_FRAGMENT.strip())
    if renovate_room:
        fragments.append(RENOVATE_ROOM_FRAGMENT.strip())
    if repaint_walls:
        fragments.append(REPAINT_WALLS_FRAGMENT.strip())

    if flooring_preset:
        preset = flooring_preset.lower()
        if preset == "carpet":
            fragments.append(FLOORING_CARPET_FRAGMENT.strip())
        elif preset == "hardwood":
            fragments.append(FLOORING_HARDWOOD_FRAGMENT.strip())
        elif preset == "tile":
            fragments.append(FLOORING_TILE_FRAGMENT.strip())

    fragments.append(
        """
Output requirements:
- Return a single, high-quality edited image.
- The edit must look like a real photograph, not an illustration or painting.
- Do not alter the room's basic layout, window views, or camera angle.
""".strip()
    )

    return "\n\n".join(fragments)


def build_stage3_furniture_prompt_v1(
    eligible_skus: List[VibodeEligibleSku],
    collection_id: Optional[str],
    bundle_id: Optional[str],
    target_count: int,
    enhance_photo: bool,
) -> str:
    base_prompt = build_vibode_vibe_prompt(
        eligible_skus=eligible_skus,
        collection_id=collection_id,
        bundle_id=bundle_id,
        target_count=target_count,
        enhance_photo=enhance_photo,
    )
    return (
        f"{base_prompt}\n\n"
        "Stage focus:\n"
        "- This is Stage 3 (furniture pass).\n"
        "- Prioritize primary furniture placement and realism.\n"
        "- Keep accessory styling minimal for now."
    )


def build_stage3_furniture_prompt_v2(
    eligible_skus: List[VibodeEligibleSku],
    collection_id: Optional[str],
    bundle_id: Optional[str],
    target_count: int,
    enhance_photo: bool,
) -> str:
    lines = [
        "Place the provided furniture SKUs into the room photo.",
        "",
        "Stage focus:",
        "- This is Stage 3 (furniture pass) only.",
        "- Place up to {target_count} furniture items, chosen ONLY from the SKU list.".format(
            target_count=target_count
        ),
        "",
        "Hard constraints:",
        "- Use ONLY provided SKU assets; no substitutes.",
        "- No blending/averaging/hybrids between SKUs.",
        "- Primary furniture only (sofa/bed/table/storage/seating).",
        "- Do not rearrange existing furniture unless absolutely necessary to place the selected SKUs.",
        "- If a SKU cannot be placed plausibly (scale/clearance), omit it rather than inventing anything else.",
        "- Preserve camera angle, perspective, vanishing lines, and room geometry.",
        "- Preserve architecture and surfaces; Stage 2 handled surfaces.",
        "- Do not apply cinematic/editorial grading or mood styling (Stage 5 handles final vibe).",
        "- Do not add text, logos, or watermarks.",
        "",
        "Strict exclusions (ban all accessories/decor):",
        "- Do not add rugs, plants, art, pillows, throws, lamps, vases, books, baskets, candles, mirrors, curtains/drapes, ceiling fixtures, or wall decor.",
        "",
        "Placement heuristics:",
        "- Place anchor pieces first (for example sofa, bed, dining table).",
        "- Keep clear walk paths and practical circulation.",
        "- Avoid blocking doors, door swings, and major passage openings.",
        "",
        "Layout stability:",
        "- If furniture already appears to be placed in a plausible layout, preserve its position.",
        "- Avoid shifting existing items unless absolutely necessary to place a new SKU.",
        "- Do not slightly move, rotate, or resize furniture that already fits naturally in the scene.",
        "- Treat the current layout as intentional staging, not a layout to be redesigned.",
        "",
    ]

    if collection_id and collection_id.strip():
        lines.append(f"Collection ID: {collection_id.strip()}")
    if bundle_id and bundle_id.strip():
        lines.append(f"Bundle ID: {bundle_id.strip()}")
    if collection_id or bundle_id:
        lines.append("")

    lines.append("SKU reference list (only these may be placed):")
    for idx, sku in enumerate(eligible_skus, start=1):
        label = (sku.label or "").strip()
        if label:
            lines.append(f"{idx}. {sku.skuId} - {label}")
        else:
            lines.append(f"{idx}. {sku.skuId}")

    if enhance_photo:
        lines += [
            "",
            "If enhance_photo is true: apply minor technical cleanup only (white balance, exposure, subtle clarity, light noise reduction).",
            "Do not make mood, style, cinematic, editorial, or creative lighting/color changes.",
        ]

    return "\n".join(lines)


STAGE4_PREAMBLE = """
You are a professional interior photo editor performing Stage 4 styling.
This is a non-destructive styling pass on an already-staged room image.
""".strip()


STAGE4_GLOBAL_RULES = """
Global hard constraints:
- Preserve room geometry and camera.
- Do NOT move, resize, replace, or remove existing furniture.
- Keep edits subtle, photorealistic, and editorial.
- Ensure all added items appear physically supported (no floating).
- Do not add text, logos, or watermarks.
""".strip()


STAGE4_STYLE_ROOM_FRAGMENT = """
Stage 4 — Editorial Accessories

Add a small number of tasteful editorial accessories to subtly style the room.

Allowed items:
books, small plant, decorative tray, vase, cushion, throw blanket, or small tabletop decor.

Styling guidance:
- Place accessories naturally on appropriate surfaces such as coffee tables, side tables, consoles, shelves, or sofas.
- Create one primary styling vignette (for example a coffee table or sofa composition) and a few smaller supporting accents.
- Favor restraint and leave some surfaces intentionally uncluttered so the room feels open and balanced.
- Choose accessories whose materials or tones harmonize with materials already present in the room.
- Arrange objects so they are visually readable from the camera perspective and subtly suggest a lived-in atmosphere.

Constraints:
- Preserve room geometry and camera.
- Do NOT move, resize, replace, or remove existing furniture.
- Add only small accessories.
- Maximum 5-8 items.
- Avoid blocking walkways or functional surfaces like dining tables or desks.
- Accessories must appear supported by surfaces and not float.
""".strip()


STAGE4_ACCESSORIES_FRAGMENT = """
Stage 4 — Accessories (Advanced)

Apply a concise editorial accessories pass only.
- Add only small accessories (for example: books, small plant, tray, vase, cushion, throw, small tabletop decor).
- Keep the styling restrained and natural, with one main vignette and a few subtle supporting accents.
- Do not add wall-mounted or architectural elements.
- Do NOT move, resize, replace, or remove existing furniture.
""".strip()


STAGE4_WALL_ART_FRAGMENT = """
Stage 4 — Wall Art (Advanced)

Add tasteful minimal framed wall art only.
- Keep pieces proportionate to wall size.
- Do not add oversized gallery walls.
- Do NOT move, resize, replace, or remove existing furniture.
""".strip()


STAGE4_SHELVES_FRAGMENT = """
Stage 4 — Shelves (Advanced)

Add minimal small floating wall shelves only.
- Maximum 1-2 shelves.
- Apply light styling only.
- Do NOT move, resize, replace, or remove existing furniture.
""".strip()


STAGE4_CURTAINS_FRAGMENT = """
Stage 4 — Curtains (Advanced)

Add tasteful neutral window dressing only.
- Use a soft editorial drape.
- Do not block light excessively.
- Preserve window geometry.
- Do NOT move, resize, replace, or remove existing furniture.
""".strip()


STAGE4_CEILING_LIGHT_FRAGMENT = """
Stage 4 — Ceiling Light (Advanced)

Add a tasteful ceiling light only if the room naturally supports one.
- Keep it proportionate.
- Do not introduce dramatic oversized fixtures.
- Preserve room geometry.
- Do NOT move, resize, replace, or remove existing furniture.
""".strip()


def build_stage4_styling_prompt_v1(stage4_mode: Stage4StyleMode) -> str:
    mode_fragments: Dict[str, str] = {
        "style_room": STAGE4_STYLE_ROOM_FRAGMENT,
        "accessories": STAGE4_ACCESSORIES_FRAGMENT,
        "wall_art": STAGE4_WALL_ART_FRAGMENT,
        "shelves": STAGE4_SHELVES_FRAGMENT,
        "curtains": STAGE4_CURTAINS_FRAGMENT,
        "ceiling_light": STAGE4_CEILING_LIGHT_FRAGMENT,
    }
    mode_key = (stage4_mode or "style_room").strip().lower()
    mode_fragment = mode_fragments.get(mode_key, STAGE4_STYLE_ROOM_FRAGMENT)
    final_prompt = "\n\n".join([STAGE4_PREAMBLE, mode_fragment, STAGE4_GLOBAL_RULES])

    return final_prompt


def build_stage5_final_vibe_prompt_v1(
    collection_id: Optional[str],
    bundle_id: Optional[str],
    enhance_photo: bool,
    heavy_declutter: bool,
) -> str:
    # Keep Full Vibe v1 behavior; this stage is editorial polish only.
    return build_full_vibe_prompt_sunlit_editorial_v1(
        collection_id=collection_id,
        bundle_id=bundle_id,
        enhance_photo=enhance_photo,
        heavy_declutter=heavy_declutter,
    )


STAGE5_PREAMBLE = """
You are a professional interior photo editor performing a non-destructive polish pass.

Goal:
- Apply a subtle full-vibe enhancement pass, not a redesign or restage.
- Preserve the exact room layout, architecture, and primary furniture positions.
""".strip()


STAGE5_SWAGGER_GRADE = """
Sunlit editorial lighting direction:
- Keep lighting natural and believable.
- Use soft daylight with slight warmth, gentle shadows, and balanced highlights.
- Improve texture clarity and add only mild natural vibrancy.
""".strip()


STAGE5_HARD_RULES = """
Hard rules (non-negotiable):
- Do not change layout.
- Do not shift perspective or camera angle.
- Do not add, remove, move, rotate, replace, or resize any furniture or objects.
- Do not add accessories or decor (no pillows, plants, rugs, wall art, drapes, lamps).
- Do not add text, logos, or watermarks.
""".strip()


def build_stage5_final_vibe_prompt_v3() -> str:
    final_prompt = "\n\n".join([STAGE5_PREAMBLE, STAGE5_SWAGGER_GRADE, STAGE5_HARD_RULES])
    return final_prompt


# ---- Full Vibe Prompt Anchor ----

FULL_VIBE_PREAMBLE = """
You are a professional interior photo editor performing a subtle, non-destructive polish pass.
Your job is to enhance the existing room realistically while preserving its identity, layout, and architecture.
Do not redesign, restage, or creatively reinterpret the space.
""".strip()


# ---- Optional Enhancement Fragment ----

ENHANCE_FRAGMENT = """
Step 1 — Enhance photo quality (technical pass only):
- Correct white balance so the scene looks natural and realistic; do NOT over-neutralize or remove a subtle warm tone.
- Optimize exposure: recover highlights and gently open shadows.
- Maintain realistic contrast (do NOT create HDR effects).
- Improve dynamic range for a bright, inviting interior.
- Increase sharpness and clarity slightly (avoid oversharpening).
- Reduce visible noise or grain.
- Enhancements must support the Step 3 Sunlit Editorial look and must not override its warmth.
- Preserve original room layout, geometry, and object placement.
""".strip()


# ---- Light Declutter Fragment ----

LIGHT_DECLUTTER_FRAGMENT = """
Step 2 — Light tidy pass:
- Remove existing minor non-essential clutter if present (for example: small stray items or visible loose cords).
- Never introduce new clutter, cables, wires, or additional objects.
- If unsure, leave surfaces clean and minimal.
- Keep all primary furniture and meaningful decor.
""".strip()


# ---- Heavy Declutter Fragment ----

HEAVY_DECLUTTER_FRAGMENT = """
Step 2 — Deep declutter and cleanup:
- Remove visible clutter and personal items from surfaces and floors.
- Examples: toys, clothes, laundry baskets, trash, cables, countertop clutter, small busy decor.
- Keep key furniture pieces that define the room (sofa, bed, table, chairs, TV console).
- Keep built-in fixtures and major appliances.
- Do NOT remove walls, windows, doors, radiators, or built-in cabinetry.
- Never introduce new clutter, cables, or replacement objects.
- After cleanup, the room should feel tidy, neutral, and ready for real-estate photography.
""".strip()


# ---- Full Vibe Editorial Core ----

FULL_VIBE_CORE = """
Step 3 — Apply subtle full-vibe editorial polish:
- Preserve the exact room layout, architecture, and primary furniture positions.
- Use soft natural daylight with slight warmth.
- Add gentle, believable shadows and balanced highlights.
- Improve texture clarity and mild natural vibrancy.
- Add minimal accent styling only if it enhances realism.
- At most 1–2 pillows total.
- Optional small coffee table styling.
- Optional small vase or plant.
- Do NOT add new primary furniture.
- Do NOT add rugs, wall art, drapes, or major decor.
""".strip()


# ---- Hard Prohibitions ----

HARD_RULES = """
Hard rules:
- Do not change layout.
- Do not shift camera angle or perspective.
- Do not alter architecture.
- Do not add large decor elements.
- Do not introduce new clutter.
- Do not add text, logos, or watermarks.
""".strip()


def build_full_vibe_prompt_sunlit_editorial_v1(
    collection_id: Optional[str],
    bundle_id: Optional[str],
    enhance_photo: bool,
    heavy_declutter: bool,
) -> str:
    sections: List[str] = [FULL_VIBE_PREAMBLE]

    context_lines: List[str] = []
    if collection_id and collection_id.strip():
        context_lines.append(f"Collection ID (context only): {collection_id.strip()}")
    if bundle_id and bundle_id.strip():
        context_lines.append(f"Bundle ID (context only): {bundle_id.strip()}")
    if context_lines:
        sections.append("\n".join(context_lines))

    if enhance_photo:
        sections.append(ENHANCE_FRAGMENT)
    if heavy_declutter:
        sections.append(HEAVY_DECLUTTER_FRAGMENT)
    else:
        sections.append(LIGHT_DECLUTTER_FRAGMENT)
    sections.append(FULL_VIBE_CORE)
    sections.append(HARD_RULES)

    final_prompt = "\n\n".join(sections)

    if DEBUG_ROOMPRINTZ_PROMPT:
        print("\n===== VIBODE FULL VIBE PROMPT SENT TO GEMINI =====\n")
        print(final_prompt)
        print("\n===================================================\n")
    return final_prompt


VIBODE_REMOVE_PROMPT = (
    "Remove objects under red X markers only. "
    "Do not restyle or add new objects. "
    "Preserve lighting/perspective."
)

VIBODE_SWAP_PROMPT = (
    "You will edit Image 1 using guidance from Image 2.\n"
    "For each numbered swap marker, remove the object at that marker and replace it with the corresponding replacement image.\n"
    "Marker #1 uses replacement Image #3, marker #2 uses Image #4, etc.\n"
    "Match perspective, scale to the removed object footprint, lighting, shadows, and occlusion.\n"
    "Do not change anything outside the swapped objects. Do not restage the room."
)


def build_vibode_rotate_prompt(marks: List[VibodeRotateMark]) -> str:
    marker_count = len(marks)
    lines = [
        "You are a professional real-estate photo editor.",
        "",
        "You are given exactly two images in this order:",
        "1) Image 1 is the clean/base room image and is the source of truth.",
        f"2) Image 2 is the same room with numbered purple rotate markers (1..{marker_count}) used only for guidance.",
        "",
        "Edit strategy:",
        "- Treat each marker as identifying the object nearest to that marker.",
        "- Keep every edited object in the same location and at the same scale.",
        "- Preserve lighting, shadows, and perspective.",
        "- Use marker order deterministically from #1 to #N.",
        "- Do not alter any other objects. Do not restage the room.",
        "- Do not leave any markers, arrows, numbers, text, logos, or watermarks in the final image.",
        "",
        "Apply these rotations in order:",
    ]

    for idx, mark in enumerate(marks):
        marker_index = idx + 1
        clamped_angle = _clamp_rotation_degrees(mark.angleDeg)
        abs_angle = abs(clamped_angle)
        if clamped_angle > 0:
            direction = "clockwise"
        elif clamped_angle < 0:
            direction = "counter-clockwise"
        else:
            direction = "clockwise (no-op)"

        lines.extend(
            [
                f"- Marker #{marker_index}: Rotate the object closest to marker #{marker_index} by {abs_angle:g} degrees {direction}.",
                "  Keep the object in the same location and at the same scale (no shifting or resizing).",
                "  Preserve lighting, shadows, and perspective.",
                "  Do not alter any other objects. Do not restage the room.",
            ]
        )

    return "\n".join(lines)


def _extract_rotate_marks(
    freeze_payload: Dict[str, Any],
    request_marks: Optional[List[VibodeRotateMark]],
) -> List[VibodeRotateMark]:
    if request_marks:
        raw_marks: object = request_marks
    else:
        candidate_marks: object = None
        vibode_intent_block = freeze_payload.get("vibodeIntent")
        if isinstance(vibode_intent_block, dict):
            rotate_block = vibode_intent_block.get("rotate")
            if isinstance(rotate_block, dict):
                nested_marks = rotate_block.get("marks")
                if isinstance(nested_marks, list):
                    candidate_marks = nested_marks
        if not isinstance(candidate_marks, list):
            candidate_marks = freeze_payload.get("rotateMarks")
        if not isinstance(candidate_marks, list):
            candidate_marks = freeze_payload.get("marks")
        if not isinstance(candidate_marks, list):
            rotate_block = freeze_payload.get("rotate")
            if isinstance(rotate_block, dict):
                nested_marks = rotate_block.get("marks")
                if isinstance(nested_marks, list):
                    candidate_marks = nested_marks
        raw_marks = candidate_marks

    if not isinstance(raw_marks, list) or not raw_marks:
        raise HTTPException(
            status_code=400,
            detail=(
                "No rotate marks found. Provide marks[] or freezePayload.rotateMarks[] "
                "(or freezePayload.marks[])."
            ),
        )

    parsed_marks: List[VibodeRotateMark] = []
    for idx, raw_mark in enumerate(raw_marks):
        try:
            if isinstance(raw_mark, VibodeRotateMark):
                mark = raw_mark
            elif isinstance(raw_mark, dict):
                mark = VibodeRotateMark(**raw_mark)
            else:
                raise ValueError("Each mark must be an object")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid rotate mark at index {idx}: {e}",
            )

        if not math.isfinite(mark.x) or not math.isfinite(mark.y):
            raise HTTPException(
                status_code=400,
                detail=f"marks[{idx}] has invalid coordinates; x and y must be finite numbers.",
            )

        clamped_x = max(0.0, min(1.0, float(mark.x)))
        clamped_y = max(0.0, min(1.0, float(mark.y)))
        clamped_angle = _clamp_rotation_degrees(mark.angleDeg)
        parsed_marks.append(
            mark.model_copy(
                update={
                    "x": clamped_x,
                    "y": clamped_y,
                    "angleDeg": clamped_angle,
                }
            )
        )

    return parsed_marks


def _vibode_edit_run_error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _coerce_finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _resolve_normalized_point(
    candidates: List[Tuple[str, Any, Any]],
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    for source, raw_x, raw_y in candidates:
        parsed_x = _coerce_finite_float(raw_x)
        parsed_y = _coerce_finite_float(raw_y)
        if parsed_x is None or parsed_y is None:
            continue
        return _clamp01(parsed_x), _clamp01(parsed_y), source
    return None, None, None


def _target_bbox_from_point(x: float, y: float, box_size: float = 0.18) -> ScenePlacementBbox:
    return _normalized_bbox_for_storage(
        ScenePlacementBbox(
            x=x - (box_size / 2.0),
            y=y - (box_size / 2.0),
            w=box_size,
            h=box_size,
        )
    )


def _find_scene_placement_index_for_point(
    placements: List[ScenePlacement], x: float, y: float
) -> int:
    hit_idx = -1
    hit_area = float("inf")
    for idx, placement in enumerate(placements):
        placement_bbox = placement.bbox
        if not placement_bbox:
            continue
        within_x = placement_bbox.x <= x <= (placement_bbox.x + placement_bbox.w)
        within_y = placement_bbox.y <= y <= (placement_bbox.y + placement_bbox.h)
        if not (within_x and within_y):
            continue
        area = placement_bbox.w * placement_bbox.h
        if area < hit_area:
            hit_area = area
            hit_idx = idx
    return hit_idx


def _sanitize_storage_segment(raw_value: Optional[str], fallback: str) -> str:
    source = (raw_value or "").strip()
    if not source:
        return fallback
    allowed = []
    for ch in source:
        if ch.isalnum() or ch in ("-", "_"):
            allowed.append(ch)
        elif ch in ("/", "\\", " "):
            allowed.append("_")
    normalized = "".join(allowed).strip("_")
    return normalized or fallback


def _find_scene_placement_index(placements: List[ScenePlacement], placement_id: str) -> int:
    for idx, placement in enumerate(placements):
        if placement.placementId == placement_id:
            return idx
    return -1


def _find_eligible_sku(eligible_skus: Optional[List[EligibleSku]], sku_id: str) -> Optional[EligibleSku]:
    if not eligible_skus:
        return None
    for sku in eligible_skus:
        if sku.skuId == sku_id:
            return sku
    return None


def _looks_like_png_image_ref(image_ref: str) -> bool:
    value = image_ref.strip().lower()
    if value.startswith("data:image/png"):
        return True
    parsed = urlparse(value)
    return parsed.path.endswith(".png")


def _select_eligible_sku_image_ref(sku: EligibleSku) -> Optional[str]:
    variants = sku.variants or []
    if not variants:
        return None
    refs: List[str] = []
    for variant in variants:
        candidate = (variant.imageUrl or "").strip()
        if candidate:
            refs.append(candidate)
    if not refs:
        return None
    png_ref = next((ref for ref in refs if _looks_like_png_image_ref(ref)), None)
    return png_ref or refs[0]


def _load_image_ref_bytes(image_ref: str) -> bytes:
    if image_ref.startswith("data:image/"):
        payload_b64 = (image_ref.split(",", 1)[1] if "," in image_ref else image_ref).strip()
        return _decode_base64_image(payload_b64)
    return _fetch_image_bytes_from_url(image_ref)


def _normalized_bbox_for_storage(bbox: ScenePlacementBbox) -> ScenePlacementBbox:
    w = min(1.0, max(0.01, float(bbox.w)))
    h = min(1.0, max(0.01, float(bbox.h)))
    x = min(1.0 - w, max(0.0, float(bbox.x)))
    y = min(1.0 - h, max(0.0, float(bbox.y)))
    return ScenePlacementBbox(x=x, y=y, w=w, h=h)


def _bbox_prompt_snippet(bbox: Optional[ScenePlacementBbox]) -> str:
    if not bbox:
        return "Target location is approximate; use context from existing object geometry."
    return (
        "The object to edit is located in bounding box normalized "
        f"({bbox.x:.4f}, {bbox.y:.4f}, {bbox.w:.4f}, {bbox.h:.4f})."
    )


def _target_area_prompt_snippet(bbox: Optional[ScenePlacementBbox]) -> str:
    if not bbox:
        return "The target edit should occur roughly in the intended target area."
    center_x = bbox.x + (bbox.w / 2.0)
    center_y = bbox.y + (bbox.h / 2.0)
    return f"The target edit should occur roughly near normalized location ({center_x:.4f}, {center_y:.4f})."


def _target_area_point_prompt_snippet(x_norm: float, y_norm: float) -> str:
    return (
        "The target edit should occur at normalized location "
        f"({x_norm:.4f}, {y_norm:.4f})."
    )


def _target_area_list_prompt_snippet(targets: List[ScenePlacementBbox]) -> str:
    if not targets:
        return "The target edits should occur roughly in the intended target areas."
    centers = ", ".join(
        f"({target.x + (target.w / 2.0):.4f}, {target.y + (target.h / 2.0):.4f})" for target in targets
    )
    return f"The target edits should occur roughly near normalized locations: {centers}."


def _remove_target_area_prompt_snippet(bbox: Optional[ScenePlacementBbox]) -> str:
    if not bbox:
        return (
            "Treat the provided normalized coordinate as the exact user-selected point for removal."
        )
    center_x = bbox.x + (bbox.w / 2.0)
    center_y = bbox.y + (bbox.h / 2.0)
    return (
        "Treat normalized location "
        f"({center_x:.4f}, {center_y:.4f}) as the exact user-selected point for removal."
    )


def _remove_target_area_list_prompt_snippet(targets: List[ScenePlacementBbox]) -> str:
    if not targets:
        return (
            "Treat the provided normalized coordinate as the exact user-selected point for removal."
        )
    centers = ", ".join(
        f"({target.x + (target.w / 2.0):.4f}, {target.y + (target.h / 2.0):.4f})" for target in targets
    )
    return f"Exact normalized removal target point(s): {centers}."


def _normalize_remove_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text == "object":
        return None
    return text


def _is_guided_remove_mode(params: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(params, dict):
        return False
    mode = str(params.get("mode") or "").strip().lower()
    return mode == "guidance-image"


def _extract_guided_remove_prompt_text(params: Optional[Dict[str, Any]]) -> str:
    if not isinstance(params, dict):
        return ""
    candidate = params.get("guidancePromptText")
    text = str(candidate or "").strip()
    return text


def _extract_guided_remove_prompt_override(params: Optional[Dict[str, Any]]) -> str:
    if not isinstance(params, dict):
        return ""
    candidates = (
        params.get("guidancePrompt"),
        params.get("prompt"),
        params.get("instruction"),
    )
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return ""


def _extract_guided_remove_image_data_url(params: Optional[Dict[str, Any]]) -> str:
    if not isinstance(params, dict):
        return ""
    return str(params.get("guidanceImageDataUrl") or "").strip()


def _looks_like_supported_guidance_data_url(image_ref: str) -> bool:
    value = (image_ref or "").strip().lower()
    return (
        value.startswith("data:image/png;base64,")
        or value.startswith("data:image/jpeg;base64,")
        or value.startswith("data:image/jpg;base64,")
    )


def _extract_guidance_manifest_target_counts(guidance_manifest: Any) -> Dict[str, int]:
    counts = {"manifest": 0, "detected": 0, "manual": 0}
    if not isinstance(guidance_manifest, dict):
        return counts

    targets = guidance_manifest.get("targets")
    if isinstance(targets, list):
        counts["manifest"] = len(targets)
    elif isinstance(guidance_manifest.get("targetCount"), int):
        counts["manifest"] = max(0, int(guidance_manifest.get("targetCount")))

    detected = guidance_manifest.get("detectedTargetCount")
    if isinstance(detected, int):
        counts["detected"] = max(0, int(detected))
    elif isinstance(guidance_manifest.get("detectedTargets"), list):
        counts["detected"] = len(guidance_manifest.get("detectedTargets"))

    manual = guidance_manifest.get("manualTargetCount")
    if isinstance(manual, int):
        counts["manual"] = max(0, int(manual))
    elif isinstance(guidance_manifest.get("manualTargets"), list):
        counts["manual"] = len(guidance_manifest.get("manualTargets"))

    return counts


def _sanitize_edit_run_remove_params_for_log(params: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(params or {})
    guidance_image_data_url = sanitized.get("guidanceImageDataUrl")
    if isinstance(guidance_image_data_url, str) and guidance_image_data_url:
        is_data_url = guidance_image_data_url.strip().startswith("data:image/")
        sanitized["guidanceImageDataUrl"] = (
            f"(data-url omitted, chars={len(guidance_image_data_url.strip())})"
            if is_data_url
            else "(non-data-url value omitted)"
        )
    guidance_manifest = sanitized.get("guidanceManifest")
    if isinstance(guidance_manifest, dict):
        sanitized["guidanceManifest"] = {
            "keys": sorted(list(guidance_manifest.keys())),
            "targetCounts": _extract_guidance_manifest_target_counts(guidance_manifest),
        }
    return sanitized


def _prompt_length(prompt: str) -> int:
    return len((prompt or "").strip())


def _url_kind_or_safe_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "(none)"
    if text.startswith("data:image/"):
        return "data-url"
    parsed = urlparse(text)
    if parsed.scheme in ("http", "https"):
        safe_path = parsed.path or "/"
        return f"remote-url:{safe_path}"
    return "other"


def _summarize_guided_remove_for_log(
    params: Dict[str, Any],
    base_image_url_kind: str,
    prompt: Optional[str] = None,
    guidance_png_bytes_len: Optional[int] = None,
) -> Dict[str, Any]:
    guidance_image_data_url = _extract_guided_remove_image_data_url(params)
    guidance_manifest = params.get("guidanceManifest")
    guidance_manifest_target_counts = _extract_guidance_manifest_target_counts(guidance_manifest)
    manifest_keys: List[str] = []
    if isinstance(guidance_manifest, dict):
        manifest_keys = sorted(str(key) for key in guidance_manifest.keys())

    source_version_id = str(
        params.get("sourceVersionId")
        or (
            guidance_manifest.get("sourceVersionId")
            if isinstance(guidance_manifest, dict)
            else ""
        )
        or ""
    ).strip()
    if not source_version_id:
        source_version_id = "(none)"

    remove_targets = params.get("removeTargets")
    request_target_count = len(remove_targets) if isinstance(remove_targets, list) else 0
    prompt_chars = _prompt_length(
        prompt or _build_guided_remove_prompt(params, guidance_manifest_target_counts)
    )
    prompt_present = prompt_chars > 0

    guidance_image_summary = "(none)"
    if guidance_image_data_url:
        is_data_url = guidance_image_data_url.startswith("data:image/")
        guidance_image_summary = (
            f"(data-url omitted, chars={len(guidance_image_data_url)})"
            if is_data_url
            else "(non-data-url value omitted)"
        )

    summary: Dict[str, Any] = {
        "mode": str(params.get("mode") or "").strip() or "(none)",
        "sourceImageUrlKind": _url_kind_or_safe_path(params.get("sourceImageUrl")),
        "baseImageUrlKind": base_image_url_kind,
        "sourceVersionId": source_version_id,
        "guidanceImageDataUrl": guidance_image_summary,
        "targetCount": (
            guidance_manifest_target_counts["manifest"]
            if guidance_manifest_target_counts["manifest"] > 0
            else request_target_count
        ),
        "detectedTargets": guidance_manifest_target_counts["detected"],
        "manualTargets": guidance_manifest_target_counts["manual"],
        "promptPresent": prompt_present,
        "promptChars": prompt_chars,
        "manifestKeys": manifest_keys,
    }
    if guidance_png_bytes_len is not None:
        summary["guidancePngBytes"] = guidance_png_bytes_len
    return summary


def _build_guided_remove_prompt(
    params: Dict[str, Any],
    guidance_manifest_target_counts: Dict[str, int],
) -> str:
    prompt_override = _extract_guided_remove_prompt_override(params)
    if prompt_override:
        return prompt_override

    guidance_prompt_text = _extract_guided_remove_prompt_text(params)
    lines = [
        "Use Image 1 as the source room image.",
        "Use Image 2 only to identify removal targets.",
        "Remove the numbered furniture targets listed below.",
        "If red X markers are present, also remove objects directly under red X markers.",
        "Do not keep numbers, circles, red Xs, labels, or markers in the final result.",
        "The final image must be a natural photorealistic room photo matching Image 1.",
        "Do not output a mask, silhouette, line drawing, segmentation map, colored region map, black-background image, or annotated guidance image.",
        "Preserve all unmarked furniture and decor.",
        "Preserve camera angle, room layout, lighting, materials, floor, walls, windows, and remaining objects.",
        "Reconstruct hidden background naturally.",
        "Do not redesign, restyle, refurnish, or add new objects.",
        "If a marker appears on empty wall/floor because of user error, ignore it unless there is a clear object directly under it.",
    ]

    detected_target_count = int(guidance_manifest_target_counts.get("detected", 0) or 0)
    manual_target_count = int(guidance_manifest_target_counts.get("manual", 0) or 0)
    if detected_target_count > 0 and manual_target_count == 0:
        lines.append(
            "There are no red X manual markers in Image 2. Remove only the numbered targets listed above. Ignore the black/transparent/annotation styling of Image 2; it is only a target-location guide."
        )

    if guidance_prompt_text:
        lines.extend(["", guidance_prompt_text])

    return "\n".join(lines)


def build_vibode_edit_run_prompt(
    action: Literal["add", "remove", "swap", "rotate"],
    target_placement: Optional[ScenePlacement],
    params: Optional[Dict[str, Any]],
    sku_label: Optional[str] = None,
    remove_target_bboxes: Optional[List[ScenePlacementBbox]] = None,
) -> str:
    params = params or {}
    bbox = target_placement.bbox if target_placement else None
    remove_target_bboxes = remove_target_bboxes or []
    rotate_x_norm = _coerce_finite_float(params.get("xNorm")) if action == "rotate" else None
    rotate_y_norm = _coerce_finite_float(params.get("yNorm")) if action == "rotate" else None
    if action == "remove":
        if remove_target_bboxes:
            target_guidance = _remove_target_area_list_prompt_snippet(remove_target_bboxes)
        else:
            target_guidance = _remove_target_area_prompt_snippet(bbox)
    elif (
        action == "rotate"
        and bbox is None
        and rotate_x_norm is not None
        and rotate_y_norm is not None
    ):
        target_guidance = _target_area_point_prompt_snippet(
            _clamp01(rotate_x_norm),
            _clamp01(rotate_y_norm),
        )
    else:
        target_guidance = _target_area_prompt_snippet(bbox)
    lines = [
        "You are a professional real-estate photo editor.",
        "",
        "Hard constraints:",
        "- Preserve camera position, room geometry, and architecture exactly.",
        "- Preserve perspective, lighting direction, white balance, and material realism.",
        "- Do not alter non-target objects.",
        "- Keep output photorealistic with clean natural edges and contact shadows.",
        "- Do not add text, logos, watermarks, or visible markup.",
        "",
        target_guidance,
        "",
    ]

    if action == "add":
        lines.extend(
            [
                "Task: Add exactly one object using the provided SKU asset image.",
                "Use only the isolated product from the SKU asset.",
                "Ignore and never reproduce any source border, frame, card, clipboard, or UI container artifact as an in-room object.",
                (
                    f"Place SKU '{sku_label}' roughly in the intended target area."
                    if sku_label
                    else "Place the provided SKU roughly in the intended target area."
                ),
                "Determine believable real-world scale automatically from room context, perspective, and nearby furniture.",
                "Prioritize realistic proportion over placeholder target area.",
                "Maintain realistic ground contact and occlusion.",
                "Do not move or modify existing furniture.",
            ]
        )
    elif action == "remove":
        remove_label = _normalize_remove_label(params.get("removeLabel"))
        remove_instruction = str(params.get("instruction") or "").strip()
        remove_prompt_override = str(params.get("prompt") or "").strip()
        has_remove_instruction = bool(remove_instruction or remove_prompt_override)
        remove_detail_line = remove_instruction or remove_prompt_override
        if remove_target_bboxes:
            if remove_label:
                lines.extend(
                    [
                        f"Task: Remove only the {remove_label} located at the target point(s).",
                        f"Use the normalized target coordinate(s) to identify which {remove_label} is intended.",
                        f"Remove the full visible {remove_label} as a coherent object, including connected parts and contact shadows, not merely a small patch around a coordinate.",
                        "Keep each edit bounded to only that targeted object and its contact shadows.",
                        "Preserve all surrounding objects, furniture, walls, layout, and overall scene composition.",
                        "Do not reposition, restyle, alter, improve, or otherwise change any non-target content.",
                        "Reconstruct the removed area naturally from nearby visual context.",
                        "Match original lighting, perspective, shadows, and textures.",
                        "Do not infer broader user intent, do not remove additional objects, and do not introduce new objects or design changes.",
                        "If no clear matching target object exists at a target coordinate, prefer no meaningful change rather than a broad or speculative edit.",
                    ]
                )
            else:
                lines.extend(
                    [
                        "Task: Perform precise, coordinate-first object removal at the target point(s).",
                        "For each target point, treat it independently and use the coordinate to identify only the single foreground object centered there.",
                        "When overlap is ambiguous, select only the centered foreground object nearest that target point.",
                        "Remove only that one object per provided target point.",
                        "Keep each edit bounded to the minimum area required to remove the target cleanly.",
                        "Preserve all surrounding objects, furniture, walls, layout, and overall scene composition.",
                        "Do not reposition, restyle, alter, improve, or otherwise change any non-target content.",
                        "Reconstruct the removed area naturally from nearby visual context.",
                        "Match original lighting, perspective, shadows, and textures.",
                        "Do not infer broader user intent, do not remove additional objects, and do not introduce new objects or design changes.",
                        "If no clear removable object exists exactly at a target coordinate, prefer no meaningful change rather than a broad or speculative edit.",
                    ]
                )
            if has_remove_instruction:
                lines.append(f"Additional user remove instruction: {remove_detail_line}")
        else:
            if remove_label:
                lines.extend(
                    [
                        f"Task: Remove only the {remove_label} located at the exact user-selected point.",
                        f"Use the normalized target coordinate to identify which {remove_label} is intended.",
                        f"Remove the full visible {remove_label} as a coherent object, including connected parts and contact shadows, not merely a small patch around the coordinate.",
                        "Keep the edit bounded to only that targeted object and its contact shadows.",
                        "Preserve all surrounding objects, furniture, walls, layout, and overall scene composition.",
                        "Do not reposition, restyle, alter, improve, or otherwise change any non-target content.",
                        "Reconstruct the removed area naturally from nearby visual context.",
                        "Match original lighting, perspective, shadows, and textures.",
                        "Do not infer broader user intent, do not remove additional objects, and do not introduce new objects or design changes.",
                        "If no clear matching target object exists at the coordinate, prefer no meaningful change rather than a broad or speculative edit.",
                    ]
                )
            else:
                lines.extend(
                    [
                        "Task: Perform precise, coordinate-first object removal at the exact user-selected point.",
                        "Use the coordinate to identify only the single foreground object centered there.",
                        "When overlap is ambiguous, select only the centered foreground object nearest that coordinate.",
                        "Remove only that single object at the target location.",
                        "Keep the edit bounded to the minimum area required to remove the target cleanly.",
                        "Preserve all surrounding objects, furniture, walls, layout, and overall scene composition.",
                        "Do not reposition, restyle, alter, improve, or otherwise change any non-target content.",
                        "Reconstruct the removed area naturally from nearby visual context.",
                        "Match original lighting, perspective, shadows, and textures.",
                        "Do not infer broader user intent, do not remove additional objects, and do not introduce new objects or design changes.",
                        "If no clear removable object exists at the coordinate, prefer no meaningful change rather than a broad or speculative edit.",
                    ]
                )
            if has_remove_instruction:
                lines.append(f"Additional user remove instruction: {remove_detail_line}")
    elif action == "swap":
        lines.extend(
            [
                "Task: Replace the target object in the intended target area using the provided SKU asset image.",
                "Use only the isolated product from the SKU asset.",
                "Ignore and never reproduce any source border, frame, card, clipboard, or UI container artifact as an in-room object.",
                (
                    f"Use SKU '{sku_label}' for the replacement."
                    if sku_label
                    else "Use the provided SKU for the replacement."
                ),
                "Determine believable scale automatically from room context and perspective.",
                "Prioritize realistic proportion over placeholder target area.",
                "Preserve placement intent without treating the target area as an exact size box.",
                "Do not change any other object.",
            ]
        )
    elif action == "rotate":
        rotation_deg = params.get("rotationDeg")
        if rotation_deg is None:
            rotation_deg = params.get("rotationDegrees")
        has_coordinate_target = rotate_x_norm is not None and rotate_y_norm is not None
        has_coordinate_only_target = has_coordinate_target and bbox is None
        if has_coordinate_only_target:
            lines.extend(
                [
                    "Task: Rotate only the single visible subject nearest the normalized target coordinate.",
                    f"Treat ({_clamp01(rotate_x_norm):.4f}, {_clamp01(rotate_y_norm):.4f}) as the center of the intended edit.",
                    "Keep the edit tightly localized around that point.",
                    (
                        f"Apply a {rotation_deg} degree rotation."
                        if rotation_deg is not None
                        else "Rotate to the requested orientation."
                    ),
                    "Do not rotate the full scene, walls, floor, or any unrelated nearby object.",
                    "Do not move the subject to a different location.",
                    "Do not change the subject scale.",
                    "Preserve contact with the floor or supporting surface if visible.",
                    "Preserve real-world perspective, shadows, lighting, and surrounding scene realism.",
                    "Keep all non-target content unchanged except minimal local adjustments needed for a believable rotation.",
                ]
            )
        else:
            lines.extend(
                [
                    "Task: Rotate the target object in place in the intended target area.",
                    (
                        f"Apply a {rotation_deg} degree rotation."
                        if rotation_deg is not None
                        else "Rotate to the requested orientation."
                    ),
                    "Keep the edit anchored on the resolved target object.",
                    "Preserve believable real-world scale and perspective.",
                    "Preserve surrounding scene realism and non-target objects.",
                    "Do not move any other object.",
                ]
            )
    else:
        raise ValueError(f"Unsupported vibode edit action: {action}")

    return "\n".join(lines)


def _debug_log_vibode_edit_run(
    action: str,
    model_name: str,
    requested_aspect_ratio: Optional[str],
    aspect_ratio_to_send: Optional[str],
    base_image_url_kind: str,
    target_placement_id: Optional[str],
    target_sku_id: Optional[str],
    params: Dict[str, Any],
    placements_count: int,
    eligible_skus_count: int,
    target_bbox: Optional[ScenePlacementBbox],
    prompt: str,
    sku_images_count: Optional[int] = None,
    guided_remove_summary: Optional[Dict[str, Any]] = None,
) -> None:
    log_event(
        "vibode_edit_run_ready",
        route="/api/vibode/edit-run",
        action=action,
        requested_aspect_ratio=requested_aspect_ratio if requested_aspect_ratio else "(none)",
        aspect_ratio_to_send=aspect_ratio_to_send if aspect_ratio_to_send else "(omitted)",
        model_name=model_name,
    )
    print("[/api/vibode/edit-run] request")
    print(f"  action={action}")
    print(f"  model={model_name}")
    print(f"  requested_aspect_ratio={requested_aspect_ratio if requested_aspect_ratio else '(none)'}")
    print(f"  aspect_ratio={aspect_ratio_to_send if aspect_ratio_to_send else '(omitted)'}")
    print(f"  baseImageUrl kind={base_image_url_kind}")
    print(f"  placements={placements_count}")
    print(f"  eligible_skus={eligible_skus_count}")
    print(f"  target_placement_id={target_placement_id if target_placement_id else '(none)'}")
    print(f"  target_sku_id={target_sku_id if target_sku_id else '(none)'}")
    if action == "remove" and guided_remove_summary is not None:
        print(f"  mode={guided_remove_summary.get('mode', '(none)')}")
        print(f"  sourceImageUrl kind={guided_remove_summary.get('sourceImageUrlKind', '(none)')}")
        print(f"  sourceVersionId={guided_remove_summary.get('sourceVersionId', '(none)')}")
        print(
            f"  guidanceImageDataUrl={guided_remove_summary.get('guidanceImageDataUrl', '(none)')}"
        )
        print(f"  targetCount={guided_remove_summary.get('targetCount', 0)}")
        print(f"  detectedTargets={guided_remove_summary.get('detectedTargets', 0)}")
        print(f"  manualTargets={guided_remove_summary.get('manualTargets', 0)}")
        print(f"  prompt_present={'yes' if guided_remove_summary.get('promptPresent') else 'no'}")
        print(f"  prompt_chars={guided_remove_summary.get('promptChars', 0)}")
        manifest_keys = guided_remove_summary.get("manifestKeys", [])
        print(
            "  manifestKeys="
            + (",".join(manifest_keys) if isinstance(manifest_keys, list) and manifest_keys else "(none)")
        )
        guidance_png_bytes = guided_remove_summary.get("guidancePngBytes")
        if guidance_png_bytes is not None:
            print(f"  guidance_png_bytes={guidance_png_bytes}")
    else:
        params_for_log = _sanitize_edit_run_remove_params_for_log(params) if action == "remove" else params
        print(f"  params={params_for_log}")
    if "removeTargets" in params:
        remove_targets = params.get("removeTargets")
        remove_targets_count = len(remove_targets) if isinstance(remove_targets, list) else 0
        print(f"  remove_targets={remove_targets_count}")
    if action == "remove":
        remove_label = _normalize_remove_label(params.get("removeLabel"))
        print(f"  remove_label={remove_label if remove_label else '(generic-object)'}")
        has_remove_instruction = bool(
            str(params.get("instruction") or "").strip() or str(params.get("prompt") or "").strip()
        )
        print(f"  remove_instruction_present={'yes' if has_remove_instruction else 'no'}")
    if target_bbox:
        print(
            "  bbox="
            + str(
                {
                    "x": target_bbox.x,
                    "y": target_bbox.y,
                    "w": target_bbox.w,
                    "h": target_bbox.h,
                }
            )
        )
    if sku_images_count is not None:
        print(f"  sku_images={sku_images_count}")
    if DEBUG_ROOMPRINTZ_PROMPT:
        request_id = get_request_id()
        mode = str(params.get("mode") or "").strip().lower()
        mode_fragment = f" mode={mode}" if mode else ""
        print(f"[roomprintz][prompt] BEGIN request_id={request_id} action={action}{mode_fragment}")
        print(prompt)
        print(f"[roomprintz][prompt] END request_id={request_id} action={action}{mode_fragment}")


def call_gemini_multimodal(
    prompt: str,
    room_png_bytes: bytes,
    room_overlay_png_bytes: bytes,
    sku_png_bytes_list: List[bytes],
    model_name: str,
    aspect_ratio: Optional[str] = None,
) -> bytes:
    started_at = time.perf_counter()
    provider_attempt_id = _next_provider_attempt_id()
    accounting_status = "failed"
    accounting_error_code: Optional[str] = None
    accounting_error_message: Optional[str] = None
    accounting_usage_metrics: Dict[str, Any] = {}
    response: Optional[Any] = None
    logged_terminal_failure = False
    log_event(
        "model_call_start",
        function="call_gemini_multimodal",
        model_name=model_name,
        modality="multimodal",
        aspect_ratio=aspect_ratio if aspect_ratio else "(omitted)",
        image_count=2 + len(sku_png_bytes_list),
        room_png_bytes=len(room_png_bytes),
        room_overlay_png_bytes=len(room_overlay_png_bytes),
        sku_count=len(sku_png_bytes_list),
    )
    try:
        config_kwargs = {"response_modalities": ["IMAGE"]}
        if aspect_ratio:
            config_kwargs["image_config"] = types.ImageConfig(aspect_ratio=aspect_ratio)
        contents = [
            types.Part(text=prompt),
            types.Part(
                inline_data=types.Blob(
                    data=room_png_bytes,
                    mime_type="image/png",
                )
            ),
            types.Part(
                inline_data=types.Blob(
                    data=room_overlay_png_bytes,
                    mime_type="image/png",
                )
            ),
        ]
        for sku_bytes in sku_png_bytes_list:
            contents.append(
                types.Part(
                    inline_data=types.Blob(
                        data=sku_bytes,
                        mime_type="image/png",
                    )
                )
            )
        _log_gemini_prompt_debug(
            function_name="call_gemini_multimodal",
            model_name=model_name,
            prompt=prompt,
        )
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        accounting_usage_metrics = _extract_provider_usage_metrics(response)
        try:
            candidate = response.candidates[0]
            part = candidate.content.parts[0]
            out_bytes = part.inline_data.data
        except Exception as e:
            candidates = getattr(response, "candidates", None)
            candidate_count = len(candidates) if isinstance(candidates, list) else 0
            first_candidate_finish_reason = "(none)"
            first_candidate_part_count = 0
            first_candidate_image_parts = 0
            first_candidate_text_parts = 0
            first_candidate_text_snippet = ""
            if candidate_count > 0:
                first_candidate = candidates[0]
                finish_reason = getattr(first_candidate, "finish_reason", None)
                if finish_reason is not None:
                    first_candidate_finish_reason = str(finish_reason)
                first_content = getattr(first_candidate, "content", None)
                first_parts = getattr(first_content, "parts", None) if first_content is not None else None
                if isinstance(first_parts, list):
                    first_candidate_part_count = len(first_parts)
                    for response_part in first_parts:
                        inline_data = getattr(response_part, "inline_data", None)
                        inline_bytes = getattr(inline_data, "data", None) if inline_data is not None else None
                        if inline_bytes:
                            first_candidate_image_parts += 1
                        text_part = getattr(response_part, "text", None)
                        if isinstance(text_part, str) and text_part.strip():
                            first_candidate_text_parts += 1
                            if not first_candidate_text_snippet:
                                first_candidate_text_snippet = text_part.strip()[:240]
            log_event(
                "model_call_extract_failed",
                function="call_gemini_multimodal",
                model_name=model_name,
                modality="multimodal",
                aspect_ratio=aspect_ratio if aspect_ratio else "(omitted)",
                sku_count=len(sku_png_bytes_list),
                error=str(e),
                candidate_count=candidate_count,
                first_candidate_finish_reason=first_candidate_finish_reason,
                first_candidate_part_count=first_candidate_part_count,
                first_candidate_image_parts=first_candidate_image_parts,
                first_candidate_text_parts=first_candidate_text_parts,
                first_candidate_text_snippet=first_candidate_text_snippet,
            )
            logged_terminal_failure = True
            raise RuntimeError("Could not extract generated image from Gemini response")
        if not out_bytes:
            logged_terminal_failure = True
            log_event(
                "model_call_failed",
                function="call_gemini_multimodal",
                model_name=model_name,
                modality="multimodal",
                aspect_ratio=aspect_ratio if aspect_ratio else "(omitted)",
                sku_count=len(sku_png_bytes_list),
                error="Gemini returned empty image bytes",
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )
            raise RuntimeError("Gemini returned empty image bytes")
        log_event(
            "model_call_success",
            function="call_gemini_multimodal",
            model_name=model_name,
            modality="multimodal",
            aspect_ratio=aspect_ratio if aspect_ratio else "(omitted)",
            sku_count=len(sku_png_bytes_list),
            output_png_bytes=len(out_bytes),
            latency_ms=int((time.perf_counter() - started_at) * 1000),
        )
        accounting_status = "success"
        return out_bytes
    except Exception as e:
        accounting_error_code = type(e).__name__
        accounting_error_message = str(e)
        if response is not None and not accounting_usage_metrics:
            accounting_usage_metrics = _extract_provider_usage_metrics(response)
        if not logged_terminal_failure:
            log_event(
                "model_call_failed",
                function="call_gemini_multimodal",
                model_name=model_name,
                modality="multimodal",
                aspect_ratio=aspect_ratio if aspect_ratio else "(omitted)",
                sku_count=len(sku_png_bytes_list),
                error=str(e),
                latency_ms=int((time.perf_counter() - started_at) * 1000),
            )
        raise
    finally:
        _write_gemini_usage_event_best_effort(
            attempt_id=provider_attempt_id,
            model_name=model_name,
            status=accounting_status,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
            error_code=accounting_error_code,
            error_message=accounting_error_message,
            usage_metrics=accounting_usage_metrics,
        )


# ---------- ROUTES ----------

@app.get("/", response_model=HealthResponse)
async def read_root():
    return HealthResponse(status="ok")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok")


@app.post("/api/vibode/paste-to-place/cancel")
async def cancel_paste_to_place_job(req: PasteToPlaceCancelRequest):
    scope_id = (req.scopeId or "").strip()
    job_id = (req.jobId or "").strip()
    if not scope_id or not job_id:
        raise HTTPException(status_code=400, detail="scopeId and jobId are required.")
    operation_id = _extract_operation_id_from_job_id(scope_id, job_id)
    state_before_cancel = _snapshot_paste_to_place_scope_state(scope_id, job_id)
    log_event(
        "paste_to_place_cancel_request",
        scope_id=scope_id,
        job_id=job_id,
        parsed_operation_id=operation_id,
        registry_state_before_cancel=state_before_cancel,
    )
    mark_cancelled(scope_id, job_id)
    state_after_cancel = _snapshot_paste_to_place_scope_state(scope_id, job_id)
    log_event(
        "paste_to_place_cancel_marked",
        scope_id=scope_id,
        job_id=job_id,
        parsed_operation_id=operation_id,
        registry_state_after_cancel=state_after_cancel,
    )
    return {"ok": True}


@app.post("/stage-room", response_model=StageRoomResponse)
async def stage_room(req: StageRoomRequest):
    wants_photo_tools = (
        req.enhancePhoto
        or req.cleanupRoom
        or req.repairDamage
        or req.emptyRoom
        or req.renovateRoom
        or req.repaintWalls
        or (req.flooringPreset is not None and req.flooringPreset != "")
    )
    wants_staging = bool(req.styleId and req.styleId.strip())

    if not wants_photo_tools and not wants_staging:
        raise HTTPException(status_code=400, detail="Nothing to do.")

    model_name = resolve_model_name_for_route("/stage-room", req.modelVersion)

    try:
        raw_bytes = base64.b64decode(req.imageBase64)
    except Exception as e:
        print("[/stage-room] Failed to decode base64:", e)
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    applied_ratio: Optional[str] = None
    aspect_ratio_to_send: Optional[str] = None
    continuation_field_set = getattr(req, "model_fields_set", None)
    if continuation_field_set is None:
        continuation_field_set = getattr(req, "__fields_set__", set())
    if "isContinuation" not in continuation_field_set:
        log_event(
            "stage_room_continuation_flag_defaulted",
            route="/stage-room",
            isContinuation=req.isContinuation,
        )

    try:
        if req.isContinuation:
            log_continuation_aspect_ratio_omitted("/stage-room")
            image_png_bytes = prepare_passthrough_png_bytes(raw_bytes)
            applied_ratio = None
            aspect_ratio_to_send = None  # OMIT aspect_ratio to avoid drift
        else:
            image_png_bytes, applied_ratio = normalize_image_bytes_for_ratio(
                raw_bytes,
                requested_ratio=req.aspectRatio,
                model_name=model_name,
            )
            aspect_ratio_to_send = applied_ratio
    except Exception as e:
        print("[/stage-room] Error preparing image:", e)
        raise HTTPException(status_code=400, detail="Could not process image")

    reference_image_urls_count = len(req.referenceImageUrls or [])
    reference_image_base64s_count = len(req.referenceImageBase64s or [])
    reference_images_count = len(req.referenceImages or [])
    reference_image_png_bytes_list = _collect_stage_room_reference_png_bytes(req)
    placement_intent = (req.placementIntent or "").strip().lower()
    reference_item_labels: List[str] = []
    for ref in req.referenceImages or []:
        candidate_label = (ref.label or ref.skuId or "").strip()
        if candidate_label:
            reference_item_labels.append(candidate_label)
    # Fallback label extraction for legacy-shaped reference image payloads.
    if not reference_item_labels:
        for ref in req.referenceImages or []:
            for field_name in ("name", "title"):
                raw_value = getattr(ref, field_name, None)
                if isinstance(raw_value, str) and raw_value.strip():
                    reference_item_labels.append(raw_value.strip())
                    break

    stage_room_prompt_override: Optional[str] = None
    prompt_intent = "default_roomprintz"
    prompt_version = "roomprintz_base_v1"
    if placement_intent == "model_decided" and len(reference_image_png_bytes_list) > 0:
        stage_room_prompt_override = build_stage_room_model_decided_staging_prompt(
            reference_image_count=len(reference_image_png_bytes_list),
            reference_item_labels=reference_item_labels,
            room_type=req.roomType,
        )
        prompt_intent = "model_decided_staging"
        prompt_version = "model_decided_staging_v1"

    resolved_stage_room_prompt = (
        stage_room_prompt_override
        if stage_room_prompt_override
        else build_roomprintz_prompt(
            enhance_photo=req.enhancePhoto,
            cleanup_room=req.cleanupRoom,
            repair_damage=req.repairDamage,
            empty_room=req.emptyRoom,
            renovate_room=req.renovateRoom,
            repaint_walls=req.repaintWalls,
            flooring_preset=req.flooringPreset,
            style_id=req.styleId,
            room_type=req.roomType,
        )
    )
    stage_room_prompt_summary = summarize_prompt(resolved_stage_room_prompt)
    final_multimodal_image_input_count = 1 + len(reference_image_png_bytes_list)
    print(
        "[/stage-room] Reference image debug:",
        {
            "referenceImageUrlsCount": reference_image_urls_count,
            "referenceImageBase64sCount": reference_image_base64s_count,
            "referenceImagesCount": reference_images_count,
            "referenceItemLabelCount": len(reference_item_labels),
            "maxAdditionalRefs": STAGE_ROOM_MAX_REFERENCE_IMAGES,
            "acceptedAdditionalRefs": len(reference_image_png_bytes_list),
            "finalMultimodalImageInputCount": final_multimodal_image_input_count,
            "placementIntent": placement_intent or "(none)",
            "promptIntent": prompt_intent,
            "promptVersion": prompt_version,
        },
    )
    if reference_image_urls_count > 0:
        print(
            "[/stage-room] referenceImageUrls are accepted by schema but ignored in adapter "
            "(embedded base64 refs are prioritized)."
        )

    print(
        "[/stage-room] Received request:",
        {
            "styleId": req.styleId,
            "raw_bytes_len": len(raw_bytes),
            "input_png_len": len(image_png_bytes),
            "isContinuation": req.isContinuation,
            "modelVersion": req.modelVersion,
            "modelName": model_name,
            "requestedAspectRatio": req.aspectRatio,
            "appliedAspectRatio": applied_ratio if applied_ratio else "(passthrough)",
            "sentAspectRatio": aspect_ratio_to_send if aspect_ratio_to_send else "(omitted)",
            "allowFlashNonSquare": ALLOW_FLASH_NON_SQUARE,
            "maxInputLongEdge": MAX_INPUT_LONG_EDGE_INT,
            "placementIntent": placement_intent or "(none)",
            "promptIntent": prompt_intent,
            "promptVersion": prompt_version,
        },
    )
    log_event(
        "stage_room_prompt_ready",
        route="/stage-room",
        placement_intent=placement_intent or "(none)",
        prompt_intent=prompt_intent,
        prompt_version=prompt_version,
        reference_image_count=len(reference_image_png_bytes_list),
        **stage_room_prompt_summary,
    )

    try:
        if wants_staging:
            out_bytes = run_fusion(
                image_png_bytes=image_png_bytes,
                style_id=req.styleId,
                enhance_photo=req.enhancePhoto,
                cleanup_room=req.cleanupRoom,
                repair_damage=req.repairDamage,
                empty_room=req.emptyRoom,
                renovate_room=req.renovateRoom,
                repaint_walls=req.repaintWalls,
                flooring_preset=req.flooringPreset,
                room_type=req.roomType,
                model_name=model_name,
                aspect_ratio=aspect_ratio_to_send,
                reference_image_png_bytes_list=reference_image_png_bytes_list,
                prompt_override=stage_room_prompt_override,
                prompt_intent=prompt_intent,
                prompt_version=prompt_version,
            )
        else:
            out_bytes = run_photo_tools(
                image_png_bytes=image_png_bytes,
                enhance_photo=req.enhancePhoto,
                cleanup_room=req.cleanupRoom,
                repair_damage=req.repairDamage,
                empty_room=req.emptyRoom,
                renovate_room=req.renovateRoom,
                repaint_walls=req.repaintWalls,
                flooring_preset=req.flooringPreset,
                room_type=req.roomType,
                model_name=model_name,
                aspect_ratio=aspect_ratio_to_send,
                reference_image_png_bytes_list=reference_image_png_bytes_list,
                prompt_override=stage_room_prompt_override,
                prompt_intent=prompt_intent,
                prompt_version=prompt_version,
            )
    except Exception as e:
        log_event("stage_room_processing_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Error during fusion")

    if not out_bytes:
        raise HTTPException(status_code=500, detail="Fusion returned empty image")

    data_url = make_data_url(out_bytes, mime_type="image/png")

    debug_ratio: Optional[str] = None
    if DEBUG_ROOMPRINTZ_RATIO:
        debug_ratio = "passthrough" if req.isContinuation else (applied_ratio or "auto")

    return StageRoomResponse(imageUrl=data_url, appliedAspectRatio=debug_ratio)


@app.post("/vibode/compose", response_model=VibodeComposeResponse)
async def vibode_compose(req: VibodeComposeRequest):
    _reject_if_vibode_strict_missing("/vibode/compose", _collect_vibode_compose_missing_fields(req))

    if not req.placements:
        raise HTTPException(status_code=400, detail="No placements provided.")

    model_name = resolve_model_name_for_route("/vibode/compose", req.modelVersion)
    placements_ordered = order_vibode_placements(req.placements)

    try:
        room_raw_bytes = _decode_base64_image(req.roomImageBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 room image data")

    try:
        room_orig_img = _safe_open_image(room_raw_bytes)
        orig_w, orig_h = room_orig_img.size
    except Exception as e:
        print("[/vibode/compose] Error decoding room image dimensions:", e)
        raise HTTPException(status_code=400, detail="Could not process room image")

    applied_ratio: Optional[str] = None
    aspect_ratio_to_send: Optional[str] = None
    try:
        room_png_bytes, applied_ratio = normalize_image_bytes_for_ratio(
            room_raw_bytes,
            requested_ratio=req.aspectRatio,
            model_name=model_name,
        )
        aspect_ratio_to_send = applied_ratio
    except Exception as e:
        print("[/vibode/compose] Error preparing room image:", e)
        raise HTTPException(status_code=400, detail="Could not process room image")

    try:
        room_pre_overlay_img = _safe_open_image(room_png_bytes)
        new_w, new_h = room_pre_overlay_img.size
    except Exception as e:
        print("[/vibode/compose] Error decoding prepared room image dimensions:", e)
        raise HTTPException(status_code=500, detail="Failed to prepare room overlay image")

    placements_for_overlay = scale_placements_for_resized_room_image(
        placements_ordered,
        original_size=(orig_w, orig_h),
        resized_size=(new_w, new_h),
    )

    try:
        room_overlay_png_bytes = draw_red_markers_overlay(room_png_bytes, placements_for_overlay)
    except Exception as e:
        print("[/vibode/compose] Error drawing markers:", e)
        raise HTTPException(status_code=500, detail="Failed to draw markers")

    try:
        clean_size = _safe_open_image(room_png_bytes).size
        marked_size = _safe_open_image(room_overlay_png_bytes).size
    except Exception as e:
        print("[/vibode/compose] Error decoding clean/marked image dimensions:", e)
        raise HTTPException(status_code=500, detail="Failed to validate prepared room images")

    sizes_match = clean_size == marked_size
    clean_size_str = f"{clean_size[0]}x{clean_size[1]}"
    marked_size_str = f"{marked_size[0]}x{marked_size[1]}"
    log_event(
        "compose_prepared_image_sizes",
        route="/vibode/compose",
        clean_size=clean_size_str,
        marked_size=marked_size_str,
        sizes_match=sizes_match,
    )
    if not sizes_match:
        print(
            "[/vibode/compose] WARNING prepared clean/marked dimension mismatch:",
            {"clean_size": clean_size_str, "marked_size": marked_size_str},
        )
        log_event(
            "compose_prepared_image_size_mismatch",
            route="/vibode/compose",
            clean_size=clean_size_str,
            marked_size=marked_size_str,
        )

    print(
        "[/vibode/compose] VIBODE_DUMP_ANNOTATED_IMAGE:",
        os.getenv("VIBODE_DUMP_ANNOTATED_IMAGE"),
    )
    maybe_dump_prepared_room_images(
        room_clean_png_bytes=room_png_bytes,
        room_marked_png_bytes=room_overlay_png_bytes,
    )

    sku_png_bytes_list: List[bytes] = []
    try:
        for placement in placements_ordered:
            sku_raw_bytes = _decode_base64_image(placement.skuImageBase64)
            sku_png_bytes_list.append(prepare_sku_png_bytes(sku_raw_bytes))
    except Exception as e:
        print("[/vibode/compose] Error preparing SKU images:", e)
        raise HTTPException(status_code=400, detail="Invalid base64 SKU image data")

    print(
        "[/vibode/compose] Received request:",
        {
            "placements": len(req.placements),
            "placements_ordered": len(placements_ordered),
            "sku_count": len(sku_png_bytes_list),
            "room_bytes_len": len(room_raw_bytes),
            "room_png_len": len(room_png_bytes),
            "modelVersion": req.modelVersion,
            "modelName": model_name,
            "requestedAspectRatio": req.aspectRatio,
            "appliedAspectRatio": applied_ratio,
            "sentAspectRatio": aspect_ratio_to_send if aspect_ratio_to_send else "(omitted)",
            "maxInputLongEdge": MAX_INPUT_LONG_EDGE_INT,
        },
    )

    prompt = build_vibode_compose_prompt(placements_ordered, enhance_photo=req.enhancePhoto)
    prompt_chars = len(prompt)
    prompt_hash = _short_prompt_hash(prompt)
    log_event(
        "compose_payload_sanity",
        route="/vibode/compose",
        clean_size=clean_size_str,
        marked_size=marked_size_str,
        sizes_match=sizes_match,
        marker_style=VIBODE_COMPOSE_MARKER_STYLE,
        marker_radius_px=VIBODE_COMPOSE_MARKER_RADIUS_PX,
        marker_color_family=VIBODE_COMPOSE_MARKER_COLOR_FAMILY,
        image_count=2 + len(sku_png_bytes_list),
        sku_count=len(sku_png_bytes_list),
        placements_count=len(placements_ordered),
        prompt_chars=prompt_chars,
        prompt_hash=prompt_hash,
        model_name=model_name,
        aspect_ratio=aspect_ratio_to_send if aspect_ratio_to_send else "(omitted)",
    )
    try:
        out_bytes = call_gemini_multimodal(
            prompt=prompt,
            room_png_bytes=room_png_bytes,
            room_overlay_png_bytes=room_overlay_png_bytes,
            sku_png_bytes_list=sku_png_bytes_list,
            model_name=model_name,
            aspect_ratio=aspect_ratio_to_send,
        )
    except Exception as e:
        log_event("vibode_compose_processing_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Error during compose")

    if not out_bytes:
        raise HTTPException(status_code=500, detail="Compose returned empty image")

    data_url = make_data_url(out_bytes, mime_type="image/png")

    debug_ratio: Optional[str] = None
    if DEBUG_ROOMPRINTZ_RATIO:
        debug_ratio = applied_ratio or "auto"

    return VibodeComposeResponse(imageUrl=data_url, appliedAspectRatio=debug_ratio)


@app.post("/vibode/vibe", response_model=VibodeComposeResponse)
async def vibode_vibe(req: VibodeVibeRequest):
    _reject_if_vibode_strict_missing("/vibode/vibe", _collect_vibode_vibe_missing_fields(req))

    model_name = resolve_model_name_for_route("/vibode/vibe", req.modelVersion)

    try:
        room_raw_bytes = _decode_base64_image(req.roomImageBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 room image data")

    applied_ratio: Optional[str] = None
    aspect_ratio_to_send: Optional[str] = None
    try:
        room_png_bytes, applied_ratio = normalize_image_bytes_for_ratio(
            room_raw_bytes,
            requested_ratio=req.aspectRatio,
            model_name=model_name,
        )
        aspect_ratio_to_send = applied_ratio
    except Exception as e:
        print("[/vibode/vibe] Error preparing room image:", e)
        raise HTTPException(status_code=400, detail="Could not process room image")

    max_target = min(len(req.eligibleSkus), 12)
    if req.targetCount is not None:
        target_count = max(1, min(req.targetCount, max_target))
    else:
        bundle_id_normalized = (req.bundleId or "").strip().lower()
        if "small" in bundle_id_normalized:
            default_target = 6
        elif "large" in bundle_id_normalized:
            default_target = 8
        else:
            default_target = 7
        target_count = max(1, min(default_target, max_target))

    selected_skus = req.eligibleSkus[:target_count]
    print(
        f"[vibode/vibe] selectedSkus={len(selected_skus)} targetCount={target_count} "
        f"collection={req.collectionId} bundle={req.bundleId}"
    )

    sku_png_bytes_list: List[bytes] = []
    for idx, sku in enumerate(selected_skus):
        image_ref: Optional[str] = None

        resolver = globals().get("resolveIkeaSkuImageUrl") or globals().get("resolve_ikea_sku_image_url")
        if callable(resolver):
            try:
                image_ref = _extract_vibode_vibe_sku_image_ref(resolver(sku.skuId))
            except Exception as e:
                print("[/vibode/vibe] SKU resolver failed:", {"skuId": sku.skuId, "error": str(e)})

        if not image_ref:
            image_ref = _extract_vibode_vibe_sku_image_ref(sku.variants or [])
        if not image_ref:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"eligibleSkus[{idx}] skuId={sku.skuId} is missing an image asset input. "
                    "Provide a variant image URL/data URL or add a skuId image resolver."
                ),
            )

        try:
            if image_ref.startswith("data:image/"):
                payload_b64 = image_ref.split(",", 1)[1] if "," in image_ref else image_ref
                sku_raw_bytes = _decode_base64_image(payload_b64)
            else:
                sku_raw_bytes = _fetch_image_bytes_from_url(image_ref)
            sku_png_bytes_list.append(prepare_sku_png_bytes(sku_raw_bytes))
        except HTTPException:
            raise
        except Exception as e:
            print("[/vibode/vibe] Error preparing SKU image:", {"skuId": sku.skuId, "error": str(e)})
            raise HTTPException(
                status_code=400,
                detail=f"Failed to fetch/prepare image for eligibleSkus[{idx}] skuId={sku.skuId}",
            )

    room_overlay_png_bytes = room_png_bytes
    prompt = build_vibode_vibe_prompt(
        eligible_skus=selected_skus,
        collection_id=req.collectionId,
        bundle_id=req.bundleId,
        target_count=target_count,
        enhance_photo=req.enhancePhoto,
    )

    try:
        out_bytes = call_gemini_multimodal(
            prompt=prompt,
            room_png_bytes=room_png_bytes,
            room_overlay_png_bytes=room_overlay_png_bytes,
            sku_png_bytes_list=sku_png_bytes_list,
            model_name=model_name,
            aspect_ratio=aspect_ratio_to_send,
        )
    except Exception as e:
        log_event("vibode_vibe_processing_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Error during vibe")

    if not out_bytes:
        raise HTTPException(status_code=500, detail="Vibe returned empty image")

    data_url = make_data_url(out_bytes, mime_type="image/png")

    debug_ratio: Optional[str] = None
    if DEBUG_ROOMPRINTZ_RATIO:
        debug_ratio = applied_ratio or "auto"

    return VibodeComposeResponse(imageUrl=data_url, appliedAspectRatio=debug_ratio)


@app.post("/vibode/full_vibe", response_model=VibodeComposeResponse)
async def vibode_full_vibe(req: VibodeFreezeRequest):
    _reject_if_vibode_strict_missing(
        "/vibode/full_vibe",
        _collect_vibode_full_vibe_missing_fields(req),
    )

    model_name = resolve_model_name_for_route("/vibode/full_vibe", req.modelVersion)

    base_image = req.freeze.baseImage
    try:
        if base_image.signedUrl and base_image.signedUrl.strip():
            room_raw_bytes = _fetch_image_bytes_from_url(base_image.signedUrl.strip())
        elif base_image.base64 and base_image.base64.strip():
            room_raw_bytes = _decode_base64_image(base_image.base64)
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide freeze.baseImage.signedUrl or freeze.baseImage.base64.",
            )
    except Exception as e:
        print("[/vibode/full_vibe] Failed to resolve base image:", e)
        raise HTTPException(status_code=400, detail="Failed to resolve freeze base image data")

    collection_id = req.collectionId
    bundle_id = req.bundleId
    enhance_photo = req.enhancePhoto
    heavy_declutter = req.heavyDeclutter
    vibode_intent_block = req.freeze.vibodeIntent
    if isinstance(vibode_intent_block, dict):
        vibode_collection = vibode_intent_block.get("collectionId")
        vibode_bundle = vibode_intent_block.get("bundleId")
        vibode_enhance = vibode_intent_block.get("enhancePhoto")
        vibode_heavy_declutter = vibode_intent_block.get("heavyDeclutter")
        if isinstance(vibode_collection, str) and vibode_collection.strip():
            collection_id = vibode_collection
        if isinstance(vibode_bundle, str) and vibode_bundle.strip():
            bundle_id = vibode_bundle
        if isinstance(vibode_enhance, bool):
            enhance_photo = vibode_enhance
        if isinstance(vibode_heavy_declutter, bool):
            heavy_declutter = vibode_heavy_declutter

    applied_ratio: Optional[str] = None
    aspect_ratio_to_send: Optional[str] = None
    try:
        room_png_bytes, applied_ratio = normalize_image_bytes_for_ratio(
            room_raw_bytes,
            requested_ratio=req.aspectRatio,
            model_name=model_name,
        )
        aspect_ratio_to_send = applied_ratio
    except Exception as e:
        print("[/vibode/full_vibe] Error preparing room image:", e)
        raise HTTPException(status_code=400, detail="Could not process room image")

    print(
        f"[vibode/full_vibe] collection={collection_id} bundle={bundle_id}"
    )

    prompt = build_full_vibe_prompt_sunlit_editorial_v1(
        collection_id=collection_id,
        bundle_id=bundle_id,
        enhance_photo=enhance_photo,
        heavy_declutter=heavy_declutter,
    )

    try:
        out_bytes = call_gemini_with_prompt(
            image_png_bytes=room_png_bytes,
            prompt=prompt,
            model_name=model_name,
            aspect_ratio=aspect_ratio_to_send,
        )
    except Exception as e:
        log_event("vibode_full_vibe_processing_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Error during full vibe")

    if not out_bytes:
        raise HTTPException(status_code=500, detail="Full vibe returned empty image")

    data_url = make_data_url(out_bytes, mime_type="image/png")

    debug_ratio: Optional[str] = None
    if DEBUG_ROOMPRINTZ_RATIO:
        debug_ratio = applied_ratio or "auto"

    return VibodeComposeResponse(imageUrl=data_url, appliedAspectRatio=debug_ratio)


@app.post("/api/vibode/stage-run", response_model=VibodeComposeResponse)
async def vibode_stage_run(req: VibodeStageRunRequest, http_request: Request):
    route = "/api/vibode/stage-run"
    paste_to_place_control = _extract_paste_to_place_control(
        http_request,
        route,
        log_missing_headers=True,
    )
    early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "arrival_post_mark_latest")
    if early_exit:
        return early_exit
    _reject_if_vibode_strict_missing(
        route,
        _collect_vibode_stage_run_missing_fields(req),
    )

    model_name = resolve_model_name_for_route(route, req.modelVersion)

    early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "base_image_resolve")
    if early_exit:
        return early_exit
    try:
        room_raw_bytes = _resolve_vibode_stage_run_room_raw_bytes(req)
    except HTTPException:
        raise
    except Exception as e:
        log_event("vibode_stage_run_base_image_resolve_failed", error=str(e))
        raise HTTPException(status_code=400, detail="Failed to resolve stage-run base image data")

    applied_ratio: Optional[str] = None
    aspect_ratio_to_send: Optional[str] = None
    # IMPORTANT:
    # Gemini image generation defaults to 1:1 when `image_config.aspect_ratio`
    # is omitted from the request. For Vibode continuation flows we preserve
    # passthrough image bytes (no crop/normalize), but we MUST still compute
    # and send an explicit `aspect_ratio_to_send` based on either the requested
    # aspectRatio or the source image dimensions.
    #
    # Do NOT omit aspect_ratio for continuation or edit flows, otherwise
    # Gemini will return square (1:1) outputs.
    is_continuation = bool(req.isContinuation)
    early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "room_prepare")
    if early_exit:
        return early_exit
    try:
        if is_continuation:
            room_png_bytes = prepare_passthrough_png_bytes(room_raw_bytes)
            applied_ratio = None
            requested_ratio = (req.aspectRatio or "").strip().lower().replace("x", ":")
            if requested_ratio and requested_ratio != "auto":
                aspect_ratio_to_send = requested_ratio
            else:
                source_w, source_h = _safe_open_image(room_raw_bytes).size
                aspect_ratio_to_send = choose_closest_aspect_ratio(source_w, source_h)
        else:
            room_png_bytes, applied_ratio = normalize_image_bytes_for_ratio(
                room_raw_bytes,
                requested_ratio=req.aspectRatio,
                model_name=model_name,
            )
            aspect_ratio_to_send = applied_ratio
    except Exception as e:
        log_event("vibode_stage_run_room_prepare_failed", error=str(e))
        raise HTTPException(status_code=400, detail="Could not process room image")

    prompt: str
    room_overlay_png_bytes = room_png_bytes
    sku_png_bytes_list: List[bytes] = []
    attached_sku_assets: List[Dict[str, str]] = []

    if req.stage == 1:
        stage1_mode = (req.stage1Mode or "").strip().lower()
        empty_room_mode = stage1_mode == "empty_room" or (not stage1_mode and req.emptyRoom)
        prompt = build_stage1_prestage_prompt_v1(
            enhance_photo=req.enhancePhoto,
            cleanup_room=req.cleanupRoom,
            empty_room_mode=empty_room_mode,
            room_type=req.roomType,
        )
    elif req.stage == 2:
        prompt = build_stage2_surfaces_prompt_v1(
            repair_damage=req.repairDamage,
            heavy_declutter=req.heavyDeclutter,
            renovate_room=req.renovateRoom,
            repaint_walls=req.repaintWalls,
            flooring_preset=req.flooringPreset,
            room_type=req.roomType,
        )
    elif req.stage == 3:
        eligible_skus = req.eligibleSkus or []
        if not eligible_skus:
            raise HTTPException(status_code=400, detail="eligibleSkus are required for stage 3")

        max_target = min(len(eligible_skus), 12)
        if req.targetCount is not None:
            target_count = max(1, min(req.targetCount, max_target))
        else:
            bundle_id_normalized = (req.bundleId or "").strip().lower()
            if "small" in bundle_id_normalized:
                default_target = 6
            elif "large" in bundle_id_normalized:
                default_target = 8
            else:
                default_target = 7
            target_count = max(1, min(default_target, max_target))

        def _is_user_sku(sku: VibodeEligibleSku) -> bool:
            if isinstance(sku.skuId, str) and sku.skuId.startswith("user_"):
                return True
            source = getattr(sku, "source", None)
            return isinstance(source, str) and source.strip().lower() == "user"

        user_skus = [sku for sku in eligible_skus if _is_user_sku(sku)]
        catalog_skus = [sku for sku in eligible_skus if not _is_user_sku(sku)]
        selected_skus = (user_skus + catalog_skus)[:target_count]
        for idx, sku in enumerate(selected_skus):
            early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "sku_prepare")
            if early_exit:
                return early_exit
            image_ref: Optional[str] = None

            resolver = globals().get("resolveIkeaSkuImageUrl") or globals().get("resolve_ikea_sku_image_url")
            if callable(resolver):
                try:
                    image_ref = _extract_vibode_vibe_sku_image_ref(resolver(sku.skuId))
                except Exception as e:
                    log_event("vibode_stage_run_sku_resolver_failed", sku_id=sku.skuId, error=str(e))

            if not image_ref:
                image_ref = _extract_vibode_vibe_sku_image_ref(sku.variants or [])
            if not image_ref:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"eligibleSkus[{idx}] skuId={sku.skuId} is missing an image asset input. "
                        "Provide a variant image URL/data URL or add a skuId image resolver."
                    ),
                )

            try:
                if image_ref.startswith("data:image/"):
                    payload_b64 = image_ref.split(",", 1)[1] if "," in image_ref else image_ref
                    sku_raw_bytes = _decode_base64_image(payload_b64)
                else:
                    sku_raw_bytes = _fetch_image_bytes_from_url(image_ref)
                sku_png_bytes_list.append(prepare_sku_png_bytes(sku_raw_bytes))
                attached_sku_assets.append({"skuId": sku.skuId, "imageRef": image_ref})
            except HTTPException:
                raise
            except Exception as e:
                log_event("vibode_stage_run_sku_prepare_failed", sku_id=sku.skuId, error=str(e))
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to fetch/prepare image for eligibleSkus[{idx}] skuId={sku.skuId}",
                )

        prompt = build_stage3_furniture_prompt_v2(
            eligible_skus=selected_skus,
            collection_id=req.collectionId,
            bundle_id=req.bundleId,
            target_count=target_count,
            enhance_photo=req.enhancePhoto,
        )
    elif req.stage == 4:
        stage4_mode: Stage4StyleMode = req.stage4Mode or "style_room"
        prompt = build_stage4_styling_prompt_v1(stage4_mode=stage4_mode)
    else:
        prompt = build_stage5_final_vibe_prompt_v3()

    prompt_summary = summarize_prompt(prompt)
    log_event(
        "vibode_stage_run_ready",
        route="/api/vibode/stage-run",
        stage=req.stage,
        continuation=is_continuation,
        requested_aspect_ratio=req.aspectRatio,
        applied_ratio=applied_ratio if applied_ratio else "(none)",
        aspect_ratio_to_send=aspect_ratio_to_send if aspect_ratio_to_send else "(omitted)",
        model_name=model_name,
        aspect_ratio=aspect_ratio_to_send if aspect_ratio_to_send else "(omitted)",
        sku_count=len(sku_png_bytes_list),
        options={
            "enhancePhoto": req.enhancePhoto,
            "cleanupRoom": req.cleanupRoom,
            "heavyDeclutter": req.heavyDeclutter,
            "repairDamage": req.repairDamage,
            "renovateRoom": req.renovateRoom,
            "repaintWalls": req.repaintWalls,
            "flooringPreset": req.flooringPreset,
            "emptyRoom": req.emptyRoom,
            "stage4Mode": req.stage4Mode or "style_room",
        },
        **prompt_summary,
    )
    if DEBUG_ROOMPRINTZ_PROMPT:
        request_id = get_request_id()
        print(
            f"[roomprintz][prompt] BEGIN request_id={request_id} stage={req.stage} model_name={model_name}"
        )
        print(prompt)
        print(f"[roomprintz][prompt] END request_id={request_id} stage={req.stage}")

    if req.stage == 3:
        eligible_skus = req.eligibleSkus or []
        eligible_sku_ids = [sku.skuId for sku in eligible_skus]
        user_sku_ids = [sku_id for sku_id in eligible_sku_ids if isinstance(sku_id, str) and sku_id.startswith("user_")]

        def _truncate_preview(value: Optional[str], max_len: int = 80) -> Optional[str]:
            if not isinstance(value, str):
                return None
            return value[:max_len]

        def _log_variant_shape(prefix: str, sku: VibodeEligibleSku) -> None:
            variants = sku.variants or []
            first_variant = variants[0] if variants else None
            first_variant_type = type(first_variant).__name__ if first_variant is not None else None
            payload: Dict[str, Any] = {
                "skuId": sku.skuId,
                "variants.length": len(variants),
                "typeof variants[0]": first_variant_type,
            }
            if isinstance(first_variant, dict):
                payload["variants[0].imageUrl"] = _truncate_preview(first_variant.get("imageUrl"))
            elif isinstance(first_variant, str):
                payload["variants[0]"] = _truncate_preview(first_variant)
            print(prefix, payload)

        print("[/api/vibode/stage-run][stage3-debug] summary", {
            "stage": req.stage,
            "targetCount": target_count,
            "eligibleSkus.length": len(eligible_skus),
            "eligibleSkuIds": eligible_sku_ids,
            "userSkuCount": len(user_sku_ids),
            "selectedSkuIds": [sku.skuId for sku in selected_skus],
            "selectedUserSkuCount": sum(1 for sku in selected_skus if _is_user_sku(sku)),
        })

        for idx, sku in enumerate(eligible_skus[:3], start=1):
            _log_variant_shape(f"[/api/vibode/stage-run][stage3-debug] parsed variants first3[{idx}]", sku)

        first_user_sku = next(
            (sku for sku in eligible_skus if isinstance(sku.skuId, str) and sku.skuId.startswith("user_")),
            None,
        )
        if first_user_sku is not None:
            _log_variant_shape("[/api/vibode/stage-run][stage3-debug] parsed variants first user sku", first_user_sku)
        else:
            print("[/api/vibode/stage-run][stage3-debug] parsed variants first user sku", {"found": False})

        attached_sku_ids = [asset["skuId"] for asset in attached_sku_assets]
        print("[/api/vibode/stage-run][stage3-debug] attached SKU assets", {
            "attachedSkuImageCount": len(attached_sku_assets),
            "attachedSkuIds": attached_sku_ids,
        })
        for idx, asset in enumerate(attached_sku_assets, start=1):
            print(
                f"[/api/vibode/stage-run][stage3-debug] attached SKU asset [{idx}]",
                {"skuId": asset["skuId"], "resolvedImageUrl": _truncate_preview(asset["imageRef"])},
            )

        if DEBUG_ROOMPRINTZ_STAGE3_PROMPT and not DEBUG_ROOMPRINTZ_PROMPT:
            print("[/api/vibode/stage-run][stage3-debug] full Stage 3 prompt text (exact) BEGIN")
            print(prompt)
            print("[/api/vibode/stage-run][stage3-debug] full Stage 3 prompt text (exact) END")

    early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "model_call")
    if early_exit:
        return early_exit
    try:
        if req.stage == 3:
            early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "before_provider_call")
            if early_exit:
                return early_exit
            out_bytes = call_gemini_multimodal(
                prompt=prompt,
                room_png_bytes=room_png_bytes,
                room_overlay_png_bytes=room_overlay_png_bytes,
                sku_png_bytes_list=sku_png_bytes_list,
                model_name=model_name,
                aspect_ratio=aspect_ratio_to_send,
            )
        else:
            early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "before_provider_call")
            if early_exit:
                return early_exit
            out_bytes = call_gemini_with_prompt(
                image_png_bytes=room_png_bytes,
                prompt=prompt,
                model_name=model_name,
                aspect_ratio=aspect_ratio_to_send,
            )
    except Exception as e:
        log_event("vibode_stage_run_processing_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Error during stage run")

    if not out_bytes:
        raise HTTPException(status_code=500, detail="Stage run returned empty image")

    early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "final_response")
    if early_exit:
        return early_exit
    data_url = make_data_url(out_bytes, mime_type="image/png")

    debug_ratio: Optional[str] = None
    if DEBUG_ROOMPRINTZ_RATIO:
        debug_ratio = "passthrough" if is_continuation else (applied_ratio or "auto")

    return VibodeComposeResponse(imageUrl=data_url, appliedAspectRatio=debug_ratio)


@app.post("/api/vibode/edit-run", response_model=VibodeEditRunResponse)
async def vibode_edit_run(req: VibodeEditRunRequest, http_request: Request):
    route = "/api/vibode/edit-run"
    paste_to_place_control = _extract_paste_to_place_control(
        http_request,
        route,
        log_missing_headers=True,
    )
    early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "arrival_post_mark_latest")
    if early_exit:
        return early_exit
    # v1 targeting is text-only using normalized bbox coordinates; this can be upgraded to mask/overlay targeting later.
    action = req.action
    params = req.params or {}
    target = req.target or VibodeEditRunTarget()

    if not req.baseImageUrl or not req.baseImageUrl.strip():
        return _vibode_edit_run_error(400, "baseImageUrl is required.")

    base_image_url_kind = "data-url" if req.baseImageUrl.strip().startswith("data:image/") else "remote-url"
    early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "base_image_fetch")
    if early_exit:
        return early_exit
    try:
        print("[/api/vibode/edit-run] baseImageUrl kind:", base_image_url_kind)
        room_raw_bytes = _load_image_ref_bytes(req.baseImageUrl.strip())
    except Exception:
        return _vibode_edit_run_error(400, "Failed to fetch baseImageUrl image data.")

    model_name = resolve_model_name_for_route(route, req.modelVersion)
    # IMPORTANT:
    # Gemini image generation defaults to 1:1 when `image_config.aspect_ratio`
    # is omitted from the request. For Vibode continuation flows we preserve
    # passthrough image bytes (no crop/normalize), but we MUST still compute
    # and send an explicit `aspect_ratio_to_send` based on either the requested
    # aspectRatio or the source image dimensions.
    #
    # Do NOT omit aspect_ratio for continuation or edit flows, otherwise
    # Gemini will return square (1:1) outputs.
    early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "room_prepare")
    if early_exit:
        return early_exit
    try:
        room_png_bytes = prepare_passthrough_png_bytes(room_raw_bytes)
        source_w, source_h = _safe_open_image(room_raw_bytes).size
        requested_ratio = (req.aspectRatio or "").strip().lower().replace("x", ":")
        if requested_ratio and requested_ratio != "auto" and requested_ratio in RATIO_MAP:
            aspect_ratio_to_send = requested_ratio
        else:
            aspect_ratio_to_send = choose_closest_aspect_ratio(source_w, source_h)
        log_event(
            "aspect_ratio_policy",
            route="/api/vibode/edit-run",
            continuation=True,
            requested_aspect_ratio=req.aspectRatio,
            source_canvas_width=source_w,
            source_canvas_height=source_h,
            mapped_aspect_ratio=aspect_ratio_to_send,
            aspect_ratio_applied=False,
        )
    except Exception as e:
        print("[/api/vibode/edit-run] Error preparing room image:", e)
        return _vibode_edit_run_error(400, "Could not process base image.")

    updated_placements = [placement.model_copy(deep=True) for placement in req.placements]
    prompt: str
    out_bytes: Optional[bytes] = None
    sku_png_bytes_list: List[bytes] = []
    target_placement_id: Optional[str] = None
    target_sku_id: Optional[str] = None
    target_bbox: Optional[ScenePlacementBbox] = None
    guided_remove_mode = False
    guided_remove_overlay_png_bytes: Optional[bytes] = None
    guided_remove_summary: Optional[Dict[str, Any]] = None

    if action == "add":
        target_sku_id = (target.skuId or "").strip()
        if not target_sku_id:
            return _vibode_edit_run_error(400, "target.skuId is required for add.")

        raw_x = params.get("x")
        raw_y = params.get("y")
        if raw_x is None or raw_y is None:
            return _vibode_edit_run_error(400, "params.x and params.y are required for add.")

        try:
            x = float(raw_x)
            y = float(raw_y)
        except Exception:
            return _vibode_edit_run_error(400, "params.x and params.y must be numbers.")
        if not (math.isfinite(x) and math.isfinite(y)):
            return _vibode_edit_run_error(400, "params.x and params.y must be finite numbers.")

        sku = _find_eligible_sku(req.eligibleSkus, target_sku_id)
        if sku is None:
            return _vibode_edit_run_error(400, f"Could not find eligibleSkus entry for skuId={target_sku_id}.")
        sku_image_ref = _select_eligible_sku_image_ref(sku)
        if not sku_image_ref:
            return _vibode_edit_run_error(400, f"eligibleSkus skuId={target_sku_id} is missing variants[].imageUrl.")

        early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "sku_prepare")
        if early_exit:
            return early_exit
        try:
            sku_raw_bytes = _load_image_ref_bytes(sku_image_ref)
            sku_png_bytes_list = [_convert_image_bytes_to_png(sku_raw_bytes)]
        except Exception:
            return _vibode_edit_run_error(400, f"Failed to fetch/prepare SKU image for skuId={target_sku_id}.")

        box_size = 0.18
        cx = _clamp01(x)
        cy = _clamp01(y)
        new_bbox = _normalized_bbox_for_storage(
            ScenePlacementBbox(
                x=cx - (box_size / 2.0),
                y=cy - (box_size / 2.0),
                w=box_size,
                h=box_size,
            )
        )
        new_placement = ScenePlacement(
            placementId="pl_" + uuid4().hex,
            skuId=target_sku_id,
            label=sku.label,
            source=sku.source,
            bbox=new_bbox,
            rotationDeg=0.0,
            stageAdded=3,
            locked=False,
        )
        updated_placements.append(new_placement)
        target_bbox = new_placement.bbox
        prompt = build_vibode_edit_run_prompt(
            action="add",
            target_placement=new_placement,
            params=params,
            sku_label=sku.label,
        )
    elif action == "remove":
        remove_targets = params.get("removeTargets")
        remove_target_bboxes: List[ScenePlacementBbox] = []
        remove_target_placement_ids: set[str] = set()
        target_placement_id = (target.placementId or "").strip() or None
        guided_remove_mode = _is_guided_remove_mode(params)
        guidance_image_data_url = _extract_guided_remove_image_data_url(params)
        guidance_manifest = params.get("guidanceManifest")
        guidance_manifest_target_counts = _extract_guidance_manifest_target_counts(guidance_manifest)
        remove_targets_count = len(remove_targets) if isinstance(remove_targets, list) else 0

        if guided_remove_mode:
            guided_remove_summary = _summarize_guided_remove_for_log(
                params=params,
                base_image_url_kind=base_image_url_kind,
            )
            print("[/api/vibode/edit-run][remove] guided payload summary:")
            print(f"  mode={guided_remove_summary.get('mode', '(none)')}")
            print(f"  sourceImageUrl kind={guided_remove_summary.get('sourceImageUrlKind', '(none)')}")
            print(f"  sourceVersionId={guided_remove_summary.get('sourceVersionId', '(none)')}")
            print(
                f"  guidanceImageDataUrl={guided_remove_summary.get('guidanceImageDataUrl', '(none)')}"
            )
            print(f"  targetCount={guided_remove_summary.get('targetCount', 0)}")
            print(f"  detectedTargets={guided_remove_summary.get('detectedTargets', 0)}")
            print(f"  manualTargets={guided_remove_summary.get('manualTargets', 0)}")
            print(f"  promptChars={guided_remove_summary.get('promptChars', 0)}")
            manifest_keys = guided_remove_summary.get("manifestKeys", [])
            print(
                "  manifestKeys="
                + (",".join(manifest_keys) if isinstance(manifest_keys, list) and manifest_keys else "(none)")
            )
            print(
                "[/api/vibode/edit-run][remove] mode:",
                {
                    "guided": True,
                    "hasGuidanceImage": bool(guidance_image_data_url),
                    "manifestTargetCount": guidance_manifest_target_counts["manifest"],
                    "detectedTargetCount": guidance_manifest_target_counts["detected"],
                    "manualTargetCount": guidance_manifest_target_counts["manual"],
                    "requestRemoveTargetsCount": remove_targets_count,
                    "modelVersion": model_name,
                },
            )
        else:
            print(
                "[/api/vibode/edit-run][remove] incoming payload:",
                {
                    "target": target.model_dump(exclude_none=True),
                    "params": _sanitize_edit_run_remove_params_for_log(params),
                },
            )
            print(
                "[/api/vibode/edit-run][remove] mode:",
                {
                    "guided": False,
                    "hasGuidanceImage": bool(guidance_image_data_url),
                    "manifestTargetCount": guidance_manifest_target_counts["manifest"],
                    "detectedTargetCount": guidance_manifest_target_counts["detected"],
                    "manualTargetCount": guidance_manifest_target_counts["manual"],
                    "requestRemoveTargetsCount": remove_targets_count,
                    "modelVersion": model_name,
                },
            )

        if isinstance(remove_targets, list):
            for remove_target in remove_targets:
                if not isinstance(remove_target, dict):
                    continue
                remove_target_placement_id = str(remove_target.get("placementId") or "").strip()
                if remove_target_placement_id:
                    remove_target_placement_ids.add(remove_target_placement_id)
                remove_target_bbox_raw = remove_target.get("bbox")
                if isinstance(remove_target_bbox_raw, dict):
                    try:
                        remove_target_bbox = ScenePlacementBbox.model_validate(remove_target_bbox_raw)
                    except Exception:
                        remove_target_bbox = None
                    if remove_target_bbox:
                        remove_target_bboxes.append(_normalized_bbox_for_storage(remove_target_bbox))
                        continue
                remove_target_x, remove_target_y, _ = _resolve_normalized_point(
                    [
                        ("removeTarget.xNorm/yNorm", remove_target.get("xNorm"), remove_target.get("yNorm")),
                        ("removeTarget.x/y", remove_target.get("x"), remove_target.get("y")),
                    ]
                )
                if remove_target_x is not None and remove_target_y is not None:
                    remove_target_bboxes.append(_target_bbox_from_point(remove_target_x, remove_target_y))

        if not remove_target_bboxes and remove_target_placement_ids:
            for placement in updated_placements:
                if placement.placementId in remove_target_placement_ids and placement.bbox:
                    remove_target_bboxes.append(_normalized_bbox_for_storage(placement.bbox))

        remove_x_norm, remove_y_norm, remove_coord_source = _resolve_normalized_point(
            [
                ("target.xNorm/yNorm", target.xNorm, target.yNorm),
                ("target.x/y", target.x, target.y),
                ("params.xNorm/yNorm", params.get("xNorm"), params.get("yNorm")),
                ("params.x/y", params.get("x"), params.get("y")),
            ]
        )
        if remove_x_norm is not None and remove_y_norm is not None:
            print(
                "[/api/vibode/edit-run][remove] normalized target:",
                {
                    "xNorm": round(remove_x_norm, 4),
                    "yNorm": round(remove_y_norm, 4),
                    "source": remove_coord_source,
                },
            )
        else:
            print("[/api/vibode/edit-run][remove] normalized target: (none)")

        if not remove_target_bboxes and remove_x_norm is not None and remove_y_norm is not None:
            remove_target_bboxes.append(_target_bbox_from_point(remove_x_norm, remove_y_norm))

        if guided_remove_mode:
            if not guidance_image_data_url:
                return _vibode_edit_run_error(400, "Guided remove requires guidanceImageDataUrl.")
            if not _looks_like_supported_guidance_data_url(guidance_image_data_url):
                return _vibode_edit_run_error(
                    400,
                    "Guided remove guidanceImageDataUrl must be a PNG or JPEG data URL.",
                )
            try:
                guidance_raw_bytes, guidance_hint_mime = _decode_base64_image_with_mime(
                    guidance_image_data_url
                )
                guidance_mime = _infer_image_mime_type(
                    guidance_raw_bytes,
                    fallback_mime=guidance_hint_mime,
                )
            except Exception:
                return _vibode_edit_run_error(400, "Guided remove guidanceImageDataUrl is invalid.")
            if guidance_mime not in ("image/png", "image/jpeg", "image/jpg"):
                return _vibode_edit_run_error(
                    400,
                    "Guided remove guidanceImageDataUrl must decode to PNG or JPEG image bytes.",
                )
            try:
                guided_remove_overlay_png_bytes = prepare_passthrough_png_bytes(guidance_raw_bytes)
            except Exception:
                return _vibode_edit_run_error(400, "Guided remove guidance image could not be processed.")

            prompt = _build_guided_remove_prompt(params, guidance_manifest_target_counts)
            target_bbox = remove_target_bboxes[0] if remove_target_bboxes else None
            print(
                "[/api/vibode/edit-run][remove] guided remove validated:",
                {
                    "guidanceMime": guidance_mime,
                    "guidancePngBytes": len(guided_remove_overlay_png_bytes),
                    "removeTargetsCount": len(remove_target_bboxes),
                    "modelVersion": model_name,
                    "promptChars": _prompt_length(prompt),
                },
            )
            guided_remove_summary = _summarize_guided_remove_for_log(
                params=params,
                base_image_url_kind=base_image_url_kind,
                prompt=prompt,
                guidance_png_bytes_len=len(guided_remove_overlay_png_bytes),
            )
        else:
            if remove_target_bboxes:
                prompt = build_vibode_edit_run_prompt(
                    action="remove",
                    target_placement=None,
                    params=params,
                    remove_target_bboxes=remove_target_bboxes,
                )
                target_bbox = remove_target_bboxes[0]
            elif target_placement_id:
                target_idx = _find_scene_placement_index(updated_placements, target_placement_id)
                if target_idx < 0:
                    return _vibode_edit_run_error(400, f"placementId={target_placement_id} was not found.")
                placement_to_remove = updated_placements[target_idx]
                prompt = build_vibode_edit_run_prompt(
                    action="remove",
                    target_placement=placement_to_remove,
                    params=params,
                )
                target_bbox = placement_to_remove.bbox
            elif remove_target_placement_ids:
                prompt = build_vibode_edit_run_prompt(
                    action="remove",
                    target_placement=None,
                    params=params,
                    remove_target_bboxes=[],
                )
            else:
                return _vibode_edit_run_error(
                    400,
                    "remove requires target.xNorm/target.yNorm, target.x/target.y, params.xNorm/params.yNorm, "
                    "target.placementId, or params.removeTargets.",
                )

        if remove_target_placement_ids:
            updated_placements = [
                placement
                for placement in updated_placements
                if placement.placementId not in remove_target_placement_ids
            ]
        elif target_placement_id:
            target_idx = _find_scene_placement_index(updated_placements, target_placement_id)
            if target_idx >= 0:
                updated_placements.pop(target_idx)
        elif remove_x_norm is not None and remove_y_norm is not None:
            hit_idx = _find_scene_placement_index_for_point(updated_placements, remove_x_norm, remove_y_norm)
            if hit_idx >= 0:
                target_placement_id = updated_placements[hit_idx].placementId
                hit_bbox = updated_placements[hit_idx].bbox
                if target_bbox is None and hit_bbox:
                    target_bbox = _normalized_bbox_for_storage(hit_bbox)
                updated_placements.pop(hit_idx)
    elif action == "swap":
        target_placement_id = (target.placementId or "").strip() or None
        target_sku_id = (target.skuId or "").strip()
        swap_error_message = (
            "swap requires target.placementId or target.skuId with finite params.x and params.y."
        )

        swap_x: Optional[float] = None
        swap_y: Optional[float] = None
        if not target_placement_id:
            raw_x = params.get("x")
            raw_y = params.get("y")
            if not target_sku_id or raw_x is None or raw_y is None:
                return _vibode_edit_run_error(400, swap_error_message)
            try:
                swap_x = float(raw_x)
                swap_y = float(raw_y)
            except Exception:
                return _vibode_edit_run_error(400, swap_error_message)
            if not (math.isfinite(swap_x) and math.isfinite(swap_y)):
                return _vibode_edit_run_error(400, swap_error_message)

        new_sku_id = (target_sku_id or str(params.get("newSkuId") or "").strip())
        if not new_sku_id:
            return _vibode_edit_run_error(400, "target.skuId or params.newSkuId is required for swap.")
        target_sku_id = new_sku_id

        sku = _find_eligible_sku(req.eligibleSkus, new_sku_id)
        if sku is None:
            return _vibode_edit_run_error(400, f"Could not find eligibleSkus entry for skuId={new_sku_id}.")
        sku_image_ref = _select_eligible_sku_image_ref(sku)
        if not sku_image_ref:
            return _vibode_edit_run_error(400, f"eligibleSkus skuId={new_sku_id} is missing variants[].imageUrl.")

        early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "sku_prepare")
        if early_exit:
            return early_exit
        try:
            sku_raw_bytes = _load_image_ref_bytes(sku_image_ref)
            sku_png_bytes_list = [_convert_image_bytes_to_png(sku_raw_bytes)]
        except Exception:
            return _vibode_edit_run_error(400, f"Failed to fetch/prepare SKU image for skuId={new_sku_id}.")

        if target_placement_id:
            target_idx = _find_scene_placement_index(updated_placements, target_placement_id)
            if target_idx < 0:
                return _vibode_edit_run_error(400, f"placementId={target_placement_id} was not found.")
            placement_before = updated_placements[target_idx]
            updated_placement = placement_before.model_copy(
                update={
                    "skuId": new_sku_id,
                    "label": sku.label,
                    "source": sku.source if sku.source is not None else placement_before.source,
                }
            )
            updated_placements[target_idx] = updated_placement
        else:
            box_size = 0.18
            click_x = _clamp01(swap_x)
            click_y = _clamp01(swap_y)
            click_bbox = _normalized_bbox_for_storage(
                ScenePlacementBbox(
                    x=click_x - (box_size / 2.0),
                    y=click_y - (box_size / 2.0),
                    w=box_size,
                    h=box_size,
                )
            )

            hit_idx = -1
            hit_area = float("inf")
            for idx, placement in enumerate(updated_placements):
                placement_bbox = placement.bbox
                if not placement_bbox:
                    continue
                within_x = placement_bbox.x <= click_x <= (placement_bbox.x + placement_bbox.w)
                within_y = placement_bbox.y <= click_y <= (placement_bbox.y + placement_bbox.h)
                if not (within_x and within_y):
                    continue
                area = placement_bbox.w * placement_bbox.h
                if area < hit_area:
                    hit_area = area
                    hit_idx = idx

            if hit_idx >= 0:
                placement_before = updated_placements[hit_idx]
                updated_placement = placement_before.model_copy(
                    update={
                        "skuId": new_sku_id,
                        "label": sku.label,
                        "source": sku.source if sku.source is not None else placement_before.source,
                    }
                )
                updated_placements[hit_idx] = updated_placement
                target_placement_id = updated_placement.placementId
            else:
                updated_placement = ScenePlacement(
                    placementId="pl_" + uuid4().hex,
                    skuId=new_sku_id,
                    label=sku.label,
                    source=sku.source,
                    bbox=click_bbox,
                    rotationDeg=0.0,
                    stageAdded=3,
                    locked=False,
                )
                updated_placements.append(updated_placement)
                target_placement_id = updated_placement.placementId

        target_bbox = updated_placement.bbox
        prompt = build_vibode_edit_run_prompt(
            action="swap",
            target_placement=updated_placement,
            params=params,
            sku_label=sku.label,
        )
    elif action == "rotate":
        rotate_target_placement_id = (target.placementId or "").strip()
        if not rotate_target_placement_id:
            rotate_target_placement_id = (req.placementId or "").strip()

        rotate_x_norm, rotate_y_norm, rotate_coord_source = _resolve_normalized_point(
            [
                ("xNorm/yNorm", req.xNorm, req.yNorm),
                ("x/y", req.x, req.y),
                ("target.xNorm/yNorm", target.xNorm, target.yNorm),
                ("target.x/y", target.x, target.y),
                ("params.xNorm/yNorm", params.get("xNorm"), params.get("yNorm")),
                ("params.x/y", params.get("x"), params.get("y")),
            ]
        )

        raw_rotation = req.rotationDegrees
        if raw_rotation is None:
            raw_rotation = params.get("rotationDegrees")
        if raw_rotation is None:
            raw_rotation = params.get("rotationDeg")
        if raw_rotation is None:
            return _vibode_edit_run_error(
                400,
                "rotate requires rotationDegrees (or params.rotationDegrees / params.rotationDeg).",
            )
        try:
            rotation_delta_deg = float(raw_rotation)
        except Exception:
            return _vibode_edit_run_error(400, "rotationDegrees must be a number.")
        if not math.isfinite(rotation_delta_deg):
            return _vibode_edit_run_error(400, "rotationDegrees must be a finite number.")

        target_idx = -1
        if rotate_x_norm is not None and rotate_y_norm is not None:
            rotate_hit_test_candidates: List[Dict[str, Any]] = []
            for idx, placement in enumerate(updated_placements):
                placement_bbox = placement.bbox
                candidate_summary: Dict[str, Any] = {
                    "idx": idx,
                    "placementId": placement.placementId,
                }
                if placement_bbox:
                    min_x = placement_bbox.x
                    min_y = placement_bbox.y
                    max_x = placement_bbox.x + placement_bbox.w
                    max_y = placement_bbox.y + placement_bbox.h
                    candidate_summary["bbox"] = {
                        "x": round(placement_bbox.x, 4),
                        "y": round(placement_bbox.y, 4),
                        "w": round(placement_bbox.w, 4),
                        "h": round(placement_bbox.h, 4),
                    }
                    candidate_summary["containsPoint"] = (
                        min_x <= rotate_x_norm <= max_x and min_y <= rotate_y_norm <= max_y
                    )
                else:
                    candidate_summary["bbox"] = None
                    candidate_summary["containsPoint"] = False
                rotate_hit_test_candidates.append(candidate_summary)
            target_idx = _find_scene_placement_index_for_point(updated_placements, rotate_x_norm, rotate_y_norm)
            if target_idx >= 0:
                rotate_target_placement_id = updated_placements[target_idx].placementId

        if target_idx < 0 and rotate_target_placement_id:
            target_idx = _find_scene_placement_index(updated_placements, rotate_target_placement_id)
        if target_idx >= 0:
            target_placement = updated_placements[target_idx]
            target_placement_id = target_placement.placementId
            base_rotation_deg = _coerce_finite_float(target_placement.rotationDeg)
            rotation_deg = _clamp_rotation_degrees((base_rotation_deg or 0.0) + rotation_delta_deg)

            updated_placement = target_placement.model_copy(update={"rotationDeg": rotation_deg})
            updated_placements[target_idx] = updated_placement
            target_bbox = updated_placement.bbox
            prompt_target_placement = updated_placement
        else:
            if rotate_x_norm is None or rotate_y_norm is None:
                return _vibode_edit_run_error(
                    400,
                    "rotate requires finite xNorm/yNorm coordinates when no placement match is found.",
                )
            target_placement_id = None
            target_bbox = None
            prompt_target_placement = None

        prompt = build_vibode_edit_run_prompt(
            action="rotate",
            target_placement=prompt_target_placement,
            params={
                "rotationDeg": rotation_delta_deg,
                "rotationDegrees": rotation_delta_deg,
                "xNorm": rotate_x_norm,
                "yNorm": rotate_y_norm,
            },
        )
    else:
        return _vibode_edit_run_error(400, f"Unsupported action: {action}")

    room_overlay_png_bytes = guided_remove_overlay_png_bytes or room_png_bytes
    _debug_log_vibode_edit_run(
        action=action,
        model_name=model_name,
        requested_aspect_ratio=req.aspectRatio,
        aspect_ratio_to_send=aspect_ratio_to_send,
        base_image_url_kind=base_image_url_kind,
        target_placement_id=target_placement_id,
        target_sku_id=target_sku_id,
        params=params,
        placements_count=len(req.placements),
        eligible_skus_count=len(req.eligibleSkus or []),
        target_bbox=target_bbox,
        prompt=prompt,
        sku_images_count=len(sku_png_bytes_list) if action in ("add", "swap") else None,
        guided_remove_summary=guided_remove_summary if (action == "remove" and guided_remove_mode) else None,
    )
    early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "model_call")
    if early_exit:
        return early_exit
    try:
        if action in ("add", "swap") or (action == "remove" and guided_remove_mode):
            gemini_multimodal_sig = inspect.signature(call_gemini_multimodal)
            gemini_multimodal_kwargs: Dict[str, Any] = {
                "prompt": prompt,
                "room_png_bytes": room_png_bytes,
                "sku_png_bytes_list": sku_png_bytes_list,
                "model_name": model_name,
            }
            if "room_overlay_png_bytes" in gemini_multimodal_sig.parameters:
                gemini_multimodal_kwargs["room_overlay_png_bytes"] = room_overlay_png_bytes
            if "aspect_ratio" in gemini_multimodal_sig.parameters:
                gemini_multimodal_kwargs["aspect_ratio"] = aspect_ratio_to_send
            early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "before_provider_call")
            if early_exit:
                return early_exit
            out_bytes = call_gemini_multimodal(**gemini_multimodal_kwargs)
        else:
            early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "before_provider_call")
            if early_exit:
                return early_exit
            out_bytes = call_gemini_with_prompt(
                image_png_bytes=room_png_bytes,
                prompt=prompt,
                model_name=model_name,
                aspect_ratio=aspect_ratio_to_send,
            )
    except Exception as e:
        log_event("vibode_edit_run_processing_failed", error=str(e))
        return _vibode_edit_run_error(500, "Error during edit run.")

    print(f"[/api/vibode/edit-run] output bytes={len(out_bytes) if out_bytes else 0}")
    if not out_bytes:
        return _vibode_edit_run_error(500, "Edit run returned empty image.")
    early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "output_prepare")
    if early_exit:
        return early_exit
    try:
        if _infer_image_mime_type(out_bytes) != "image/png":
            out_bytes = _convert_image_bytes_to_png(out_bytes)
    except Exception as e:
        print("[/api/vibode/edit-run] Failed to normalize output PNG bytes:", e)
        return _vibode_edit_run_error(500, "Edit run returned invalid image bytes.")

    scene_folder = _sanitize_storage_segment(req.sceneId, uuid4().hex)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    output_path = f"vibode-edits/{scene_folder}/{timestamp}_{action}.png"
    early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "upload")
    if early_exit:
        return early_exit
    try:
        _supabase_storage_upload_bytes(
            object_path=output_path,
            payload=out_bytes,
            mime_type="image/png",
        )
        signed_url = _supabase_storage_create_signed_url(output_path)
    except Exception as e:
        log_event("vibode_edit_run_upload_sign_failed", error=str(e))
        return _vibode_edit_run_error(500, "Failed to upload edited image.")

    print(f"[/api/vibode/edit-run] uploaded path={output_path}")
    early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "final_response")
    if early_exit:
        return early_exit
    return VibodeEditRunResponse(imageUrl=signed_url, placements=updated_placements)


@app.post("/api/vibode/user-skus/ingest", response_model=VibodeUserSkuIngestResponse)
async def ingest_user_sku(ingest_req: VibodeUserSkuIngestRequest, http_request: Request):
    route = "/api/vibode/user-skus/ingest"
    started_at = time.perf_counter()
    user_sku_id = "user_" + uuid4().hex
    body_model_raw = (ingest_req.model or "").strip()
    header_model_raw = (http_request.headers.get("x-roomprintz-ingest-image-model") or "").strip()
    requested_model_raw = body_model_raw or header_model_raw
    model_name = resolve_model_name_for_route(route, requested_model_raw or None)
    model_source = (
        "body:model"
        if body_model_raw
        else ("header:x-roomprintz-ingest-image-model" if header_model_raw else "default")
    )
    source_url = ingest_req.imageUrl.strip() if ingest_req.imageUrl and ingest_req.imageUrl.strip() else None
    resolved_label = (ingest_req.label or "").strip() or "User Upload"
    preview_bg_override_flags = _extract_user_sku_preview_bg_override_flags(ingest_req, http_request)

    def _duration_ms() -> int:
        return int((time.perf_counter() - started_at) * 1000)

    def _ingest_log(event: str, **fields: Any) -> None:
        log_event(
            event,
            route=route,
            sku_id=user_sku_id,
            duration_ms=_duration_ms(),
            **fields,
        )

    def _log_debug_checkpoint(
        checkpoint: str,
        image_png_bytes: bytes,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not DEBUG_USER_SKU_INGEST_METRICS:
            return
        try:
            metrics = _compute_user_sku_image_debug_metrics(image_png_bytes)
            if extra_metrics:
                metrics.update(extra_metrics)
            _ingest_log(
                "vibode_user_sku_ingest_debug_checkpoint",
                checkpoint=checkpoint,
                metrics=metrics,
            )
        except Exception as e:
            _ingest_log(
                "vibode_user_sku_ingest_debug_checkpoint_failed",
                checkpoint=checkpoint,
                error=str(e)[:160],
            )

    def _artifact_reason(metrics: Dict[str, Any]) -> str:
        if metrics.get("likely_screenshot_card_composition"):
            return "likely_screenshot_card_composition"
        if metrics.get("low_confidence_segmentation_fallback"):
            return "low_confidence_segmentation_fallback"
        return "none"

    _ingest_log(
        "vibode_user_sku_ingest_started",
        source="imageUrl" if source_url else "imageBase64",
        model_source=model_source,
        model_requested=requested_model_raw or None,
        model_name=model_name,
        status="started",
    )

    if source_url:
        normalized_source_url = source_url.lower()
        if not (
            normalized_source_url.startswith("http://")
            or normalized_source_url.startswith("https://")
        ):
            _ingest_log(
                "vibode_user_sku_ingest_failed",
                status="failed",
                error="Invalid imageUrl.",
            )
            raise HTTPException(status_code=400, detail="Invalid imageUrl.")
        if (
            "localhost" in normalized_source_url
            or "127.0.0.1" in normalized_source_url
            or "0.0.0.0" in normalized_source_url
        ):
            _ingest_log(
                "vibode_user_sku_ingest_failed",
                status="failed",
                error="Invalid imageUrl.",
            )
            raise HTTPException(status_code=400, detail="Invalid imageUrl.")

    try:
        if source_url:
            source_bytes, source_hint_mime = _fetch_image_bytes_from_url_limited(
                source_url,
                timeout_seconds=USER_SKU_INGEST_TIMEOUT_SECONDS,
                max_bytes=USER_SKU_MAX_INPUT_BYTES,
            )
        else:
            source_bytes, source_hint_mime = _decode_base64_image_with_mime(ingest_req.imageBase64 or "")
    except Exception as e:
        _ingest_log(
            "vibode_user_sku_ingest_failed",
            status="failed",
            error=f"Failed to acquire image bytes: {str(e)[:160]}",
        )
        raise HTTPException(status_code=400, detail="Failed to acquire image bytes.")

    source_mime = _infer_image_mime_type(source_bytes, fallback_mime=source_hint_mime)

    try:
        original_png_bytes = _convert_image_bytes_to_png(source_bytes)
    except Exception as e:
        _ingest_log(
            "vibode_user_sku_ingest_failed",
            status="failed",
            error=f"Invalid image payload: {str(e)[:160]}",
        )
        raise HTTPException(status_code=400, detail="Invalid image payload.")

    input_width: Optional[int] = None
    input_height: Optional[int] = None
    has_alpha = False
    try:
        with Image.open(io.BytesIO(original_png_bytes)) as input_img:
            input_width, input_height = input_img.size
            if "A" in input_img.getbands():
                alpha_lo, _ = input_img.getchannel("A").getextrema()
                has_alpha = alpha_lo < 255
    except Exception as e:
        _ingest_log(
            "vibode_user_sku_ingest_image_dimensions_alpha",
            input_width=None,
            input_height=None,
            has_alpha=False,
            error=str(e)[:160],
        )

    _ingest_log(
        "vibode_user_sku_ingest_source_image_loaded",
        source="imageUrl" if source_url else "imageBase64",
        input_bytes=len(source_bytes),
        decoded_png_bytes=len(original_png_bytes),
        mime_type=source_mime,
    )
    _ingest_log(
        "vibode_user_sku_ingest_image_dimensions_alpha",
        input_width=input_width,
        input_height=input_height,
        has_alpha=has_alpha,
    )
    _log_debug_checkpoint("original_png", original_png_bytes)

    original_path = f"user-skus/{user_sku_id}/original.png"
    try:
        _supabase_storage_upload_bytes(
            object_path=original_path,
            payload=original_png_bytes,
            mime_type="image/png",
        )
    except Exception as e:
        _ingest_log(
            "vibode_user_sku_ingest_failed",
            status="failed",
            error=f"Failed to upload original image: {str(e)[:160]}",
        )
        raise HTTPException(status_code=500, detail="Failed to upload original image.")
    try:
        bg_removed_bytes = _run_user_sku_background_removal(
            image_png_bytes=original_png_bytes,
            model_name=model_name,
        )
        if not bg_removed_bytes:
            raise RuntimeError("Background removal returned empty bytes.")
    except Exception as e:
        failed_reason = f"background removal failed: {str(e)[:160]}"
        failed_user_sku = VibodeUserSku(
            skuId=user_sku_id,
            label=resolved_label,
            variants=[],
            sourceUrl=source_url,
            status="failed",
            reason=failed_reason,
        )
        _ingest_log(
            "vibode_user_sku_ingest_failed",
            status="failed",
            error=failed_reason,
        )
        return VibodeUserSkuIngestResponse(userSku=failed_user_sku)

    candidate_product_bytes = bg_removed_bytes
    gate_metrics = _analyze_user_sku_candidate(original_png_bytes, candidate_product_bytes)
    _log_debug_checkpoint(
        "gemini_isolated",
        candidate_product_bytes,
        extra_metrics={
            "mask_foreground_coverage_ratio": round(
                float(gate_metrics.get("foreground_area_ratio", 0.0)),
                6,
            )
        },
    )
    _ingest_log(
        "vibode_user_sku_ingest_background_estimation_completed",
        dominant_bg_ratio=gate_metrics["uniform_bg_dominance_ratio"],
    )
    _ingest_log(
        "vibode_user_sku_ingest_foreground_mask_analysis_completed",
        foreground_area_ratio=gate_metrics["foreground_area_ratio"],
        bbox_fill_ratio=gate_metrics["fill_ratio"],
        edges_touched=gate_metrics["edge_touch_sides"],
        rectangular_frame_likelihood=gate_metrics["rectangular_outer_frame_likelihood"],
    )
    _ingest_log(
        "vibode_user_sku_ingest_artifact_suspicion_decision",
        artifact_retry_triggered=gate_metrics["likely_screenshot_card_composition"],
        artifact_reason=_artifact_reason(gate_metrics),
        status=(
            "retry"
            if gate_metrics["likely_screenshot_card_composition"]
            else ("failed" if gate_metrics["low_confidence_segmentation_fallback"] else "pass")
        ),
    )

    if gate_metrics["likely_screenshot_card_composition"]:
        _ingest_log(
            "vibode_user_sku_ingest_isolation_retry_started",
            artifact_retry_triggered=True,
            artifact_reason="likely_screenshot_card_composition",
        )
        try:
            candidate_product_bytes = _run_user_sku_clipboard_product_isolation(
                image_png_bytes=original_png_bytes,
                model_name=model_name,
            )
            gate_metrics = _analyze_user_sku_candidate(original_png_bytes, candidate_product_bytes)
            _log_debug_checkpoint(
                "gemini_isolated_retry",
                candidate_product_bytes,
                extra_metrics={
                    "mask_foreground_coverage_ratio": round(
                        float(gate_metrics.get("foreground_area_ratio", 0.0)),
                        6,
                    )
                },
            )
            _ingest_log(
                "vibode_user_sku_ingest_isolation_retry_completed",
                artifact_retry_triggered=True,
                artifact_reason=_artifact_reason(gate_metrics),
                status="completed",
                dominant_bg_ratio=gate_metrics["uniform_bg_dominance_ratio"],
                foreground_area_ratio=gate_metrics["foreground_area_ratio"],
                bbox_fill_ratio=gate_metrics["fill_ratio"],
                edges_touched=gate_metrics["edge_touch_sides"],
                rectangular_frame_likelihood=gate_metrics["rectangular_outer_frame_likelihood"],
            )
        except Exception as e:
            _ingest_log(
                "vibode_user_sku_ingest_isolation_retry_completed",
                artifact_retry_triggered=True,
                artifact_reason="clipboard_isolation_error",
                status="failed",
                error=str(e)[:160],
            )
            gate_metrics["low_confidence_segmentation_fallback"] = True

    if gate_metrics["low_confidence_segmentation_fallback"]:
        failed_reason = "low-confidence product isolation; screenshot/card artifacts likely"
        failed_user_sku = VibodeUserSku(
            skuId=user_sku_id,
            label=resolved_label,
            variants=[],
            sourceUrl=source_url,
            status="failed",
            reason=failed_reason,
        )
        _ingest_log(
            "vibode_user_sku_ingest_failed",
            status="failed",
            artifact_reason=_artifact_reason(gate_metrics),
            error=failed_reason,
        )
        return VibodeUserSkuIngestResponse(userSku=failed_user_sku)

    try:
        product_cutout_png_bytes, product_solid_crop_png_bytes, product_stats = _extract_user_sku_product_crops(
            candidate_product_bytes,
            source_png_bytes=original_png_bytes,
        )
        _log_debug_checkpoint(
            "masked_crop",
            product_cutout_png_bytes,
            extra_metrics={
                "mask_foreground_coverage_ratio": round(
                    float(product_stats.get("area_ratio", 0.0)),
                    6,
                )
            },
        )
    except Exception as e:
        failed_reason = f"product isolation failed: {str(e)[:160]}"
        failed_user_sku = VibodeUserSku(
            skuId=user_sku_id,
            label=resolved_label,
            variants=[],
            sourceUrl=source_url,
            status="failed",
            reason=failed_reason,
        )
        _ingest_log(
            "vibode_user_sku_ingest_failed",
            status="failed",
            error=failed_reason,
        )
        return VibodeUserSkuIngestResponse(userSku=failed_user_sku)

    normalization_mode = "transparent_cutout"
    try:
        prefer_transparent = _has_transparency(product_cutout_png_bytes) and product_stats["fill_ratio"] < 0.98
        forced_preview_bg = (
            USER_SKU_FORCED_PREVIEW_BG_RGB
            if preview_bg_override_flags["override_requested"]
            else None
        )
        if prefer_transparent:
            normalized_png_bytes, normalized_dims = _normalize_user_sku_transparent_png(
                product_cutout_png_bytes,
                max_dimension=USER_SKU_NORMALIZED_MAX_DIM,
                padding_ratio=USER_SKU_NORMALIZED_PADDING_RATIO,
                background_rgb=forced_preview_bg,
            )
            normalization_mode = (
                "transparent_cutout_forced_preview_bg"
                if forced_preview_bg is not None
                else "transparent_cutout"
            )
        else:
            with Image.open(io.BytesIO(product_solid_crop_png_bytes)) as solid_crop_img:
                solid_crop_rgb = solid_crop_img.convert("RGB")
            sampled_bg_rgb = (
                USER_SKU_FORCED_PREVIEW_BG_RGB
                if forced_preview_bg is not None
                else _estimate_corner_background_rgb(solid_crop_rgb)
            )
            normalized_png_bytes, normalized_dims = _normalize_user_sku_solid_bg_png(
                product_solid_crop_png_bytes,
                max_dimension=USER_SKU_NORMALIZED_MAX_DIM,
                padding_ratio=USER_SKU_NORMALIZED_PADDING_RATIO,
                bg_rgb=sampled_bg_rgb,
            )
            normalization_mode = (
                "solid_tight_crop_forced_preview_bg"
                if forced_preview_bg is not None
                else "solid_tight_crop"
            )
    except Exception as e:
        _ingest_log(
            "vibode_user_sku_ingest_failed",
            status="failed",
            error=f"Failed to normalize user SKU image: {str(e)[:160]}",
        )
        raise HTTPException(status_code=500, detail="Failed to normalize user SKU image.")
    _log_debug_checkpoint("normalized_output", normalized_png_bytes)

    normalized_path = f"user-skus/{user_sku_id}/normalized.png"
    try:
        _supabase_storage_upload_bytes(
            object_path=normalized_path,
            payload=normalized_png_bytes,
            mime_type="image/png",
        )
        normalized_url = _supabase_storage_create_signed_url(normalized_path)
    except Exception as e:
        _ingest_log(
            "vibode_user_sku_ingest_failed",
            status="failed",
            error=f"Failed to upload normalized user SKU image: {str(e)[:160]}",
        )
        raise HTTPException(status_code=500, detail="Failed to upload normalized user SKU image.")

    _ingest_log(
        "vibode_user_sku_ingest_normalization_output_completed",
        input_width=input_width,
        input_height=input_height,
        has_alpha=has_alpha,
        output_width=normalized_dims[0],
        output_height=normalized_dims[1],
        output_variant_count=1,
        normalization_mode=normalization_mode,
        forced_preview_bg_rgb=(
            list(USER_SKU_FORCED_PREVIEW_BG_RGB)
            if preview_bg_override_flags["override_requested"]
            else None
        ),
        preview_bg_override_flags=preview_bg_override_flags,
        status="completed",
    )
    _ingest_log(
        "vibode_user_sku_ingest_final_output_ready",
        input_width=input_width,
        input_height=input_height,
        has_alpha=has_alpha,
        output_variant_count=1,
        status="ready",
    )

    return VibodeUserSkuIngestResponse(
        userSku=VibodeUserSku(
            skuId=user_sku_id,
            label=resolved_label,
            variants=[normalized_url],
            sourceUrl=source_url,
            status="ready",
        )
    )


@app.post("/vibode/remove", response_model=VibodeComposeResponse)
async def vibode_remove(req: VibodeRemoveRequest):
    _reject_if_vibode_strict_missing("/vibode/remove", _collect_vibode_remove_missing_fields(req))

    if not req.marks:
        raise HTTPException(status_code=400, detail="No marks provided.")

    model_name = resolve_model_name_for_route("/vibode/remove", req.modelVersion)

    try:
        room_raw_bytes = _decode_base64_image(req.cleanBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 room image data")

    try:
        room_orig_img = _safe_open_image(room_raw_bytes)
        orig_w, orig_h = room_orig_img.size
    except Exception as e:
        print("[/vibode/remove] Error decoding room image dimensions:", e)
        raise HTTPException(status_code=400, detail="Could not process room image")

    applied_ratio: Optional[str] = None
    aspect_ratio_to_send: Optional[str] = None
    try:
        log_continuation_aspect_ratio_omitted("/vibode/remove")
        room_png_bytes = prepare_passthrough_png_bytes(room_raw_bytes)
        applied_ratio = None
        aspect_ratio_to_send = None
    except Exception as e:
        print("[/vibode/remove] Error preparing room image:", e)
        raise HTTPException(status_code=400, detail="Could not process room image")

    try:
        room_pre_overlay_img = _safe_open_image(room_png_bytes)
        new_w, new_h = room_pre_overlay_img.size
    except Exception as e:
        print("[/vibode/remove] Error decoding prepared room image dimensions:", e)
        raise HTTPException(status_code=500, detail="Failed to prepare room overlay image")

    marks_for_overlay = scale_marks_for_resized_image(
        req.marks,
        original_size=(orig_w, orig_h),
        resized_size=(new_w, new_h),
    )

    try:
        room_overlay_png_bytes = draw_red_x_overlay(room_png_bytes, marks_for_overlay)
    except Exception as e:
        print("[/vibode/remove] Error drawing markers:", e)
        raise HTTPException(status_code=500, detail="Failed to draw markers")

    maybe_dump_prepared_room_images(
        room_clean_png_bytes=room_png_bytes,
        room_marked_png_bytes=room_overlay_png_bytes,
    )

    print(
        "[/vibode/remove] Received request:",
        {
            "marks": len(req.marks),
            "marks_summary": _summarize_remove_marks(req.marks),
            "room_bytes_len": len(room_raw_bytes),
            "room_png_len": len(room_png_bytes),
            "modelVersion": req.modelVersion,
            "modelName": model_name,
            "requestedAspectRatio": req.aspectRatio,
            "appliedAspectRatio": applied_ratio,
            "sentAspectRatio": aspect_ratio_to_send if aspect_ratio_to_send else "(omitted)",
            "maxInputLongEdge": MAX_INPUT_LONG_EDGE_INT,
        },
    )

    remove_prompt_hash = _short_prompt_hash(VIBODE_REMOVE_PROMPT)
    remove_prompt_first_line = _prompt_first_line(VIBODE_REMOVE_PROMPT)
    print(
        "[/vibode/remove] Prompt summary:",
        {
            "prompt_hash": remove_prompt_hash,
            "prompt_first_line": remove_prompt_first_line,
            "marks_summary": _summarize_remove_marks(req.marks),
        },
    )
    if VIBODE_LOG_PROMPTS:
        print("\n===== VIBODE REMOVE PROMPT SENT TO GEMINI =====\n")
        print(VIBODE_REMOVE_PROMPT)
        print("\n================================================\n")

    try:
        out_bytes = call_gemini_multimodal(
            prompt=VIBODE_REMOVE_PROMPT,
            room_png_bytes=room_png_bytes,
            room_overlay_png_bytes=room_overlay_png_bytes,
            sku_png_bytes_list=[],
            model_name=model_name,
            aspect_ratio=aspect_ratio_to_send,
        )
    except Exception as e:
        log_event("vibode_remove_processing_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Error during remove")

    if not out_bytes:
        raise HTTPException(status_code=500, detail="Remove returned empty image")

    data_url = make_data_url(out_bytes, mime_type="image/png")

    debug_ratio: Optional[str] = None
    if DEBUG_ROOMPRINTZ_RATIO:
        debug_ratio = "passthrough"

    return VibodeComposeResponse(imageUrl=data_url, appliedAspectRatio=debug_ratio)

@app.post("/vibode/swap", response_model=VibodeSwapResponse)
async def vibode_swap(req: VibodeSwapRequest, http_request: Request):
    route = "/vibode/swap"
    paste_to_place_control = _extract_paste_to_place_control(http_request, route)
    _reject_if_vibode_strict_missing(route, _collect_vibode_swap_missing_fields(req))

    if not req.cleanBase64 or not req.cleanBase64.strip():
        raise HTTPException(status_code=400, detail="cleanBase64 is required.")
    if not req.marks:
        raise HTTPException(status_code=400, detail="marks must be non-empty.")
    if not req.replacementAssets:
        raise HTTPException(status_code=400, detail="replacementAssets must be non-empty.")
    if len(req.replacementAssets) < len(req.marks):
        raise HTTPException(
            status_code=400,
            detail=(
                "replacementAssets length must be >= marks length. "
                "Marker #1 uses replacementAssets[0], marker #2 uses replacementAssets[1], etc."
            ),
        )

    for idx, mark in enumerate(req.marks):
        if not math.isfinite(mark.x) or not math.isfinite(mark.y):
            raise HTTPException(
                status_code=400,
                detail=f"marks[{idx}] has invalid coordinates; x and y must be finite numbers.",
            )
        if not mark.replacement.imageUrl or not mark.replacement.imageUrl.strip():
            raise HTTPException(
                status_code=400,
                detail=f"marks[{idx}].replacement.imageUrl is required.",
            )

    for idx, asset in enumerate(req.replacementAssets):
        if not asset.imageUrl or not asset.imageUrl.strip():
            raise HTTPException(
                status_code=400,
                detail=f"replacementAssets[{idx}].imageUrl is required.",
            )

    model_name = resolve_model_name_for_route(route, req.modelVersion)
    marks_ordered = list(req.marks)  # Preserve request order: marker index -> replacement index.
    mapped_replacements = req.replacementAssets[: len(marks_ordered)]

    early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "base_image_decode")
    if early_exit:
        return early_exit
    try:
        room_raw_bytes = _decode_base64_image(req.cleanBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 room image data")

    try:
        room_orig_img = _safe_open_image(room_raw_bytes)
        orig_w, orig_h = room_orig_img.size
    except Exception as e:
        print("[/vibode/swap] Error decoding room image dimensions:", e)
        raise HTTPException(status_code=400, detail="Could not process room image")

    applied_ratio: Optional[str] = None
    aspect_ratio_to_send: Optional[str] = None
    early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "room_prepare")
    if early_exit:
        return early_exit
    try:
        log_continuation_aspect_ratio_omitted("/vibode/swap")
        room_png_bytes = prepare_passthrough_png_bytes(room_raw_bytes)
        applied_ratio = None
        aspect_ratio_to_send = None
    except Exception as e:
        print("[/vibode/swap] Error preparing room image:", e)
        raise HTTPException(status_code=400, detail="Could not process room image")

    try:
        room_pre_overlay_img = _safe_open_image(room_png_bytes)
        new_w, new_h = room_pre_overlay_img.size
    except Exception as e:
        print("[/vibode/swap] Error decoding prepared room image dimensions:", e)
        raise HTTPException(status_code=500, detail="Failed to prepare room overlay image")

    marks_for_overlay = scale_swap_marks_for_resized_image(
        marks_ordered,
        original_size=(orig_w, orig_h),
        resized_size=(new_w, new_h),
    )

    early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "overlay_prepare")
    if early_exit:
        return early_exit
    try:
        room_overlay_png_bytes = render_vibode_swap_overlay(room_png_bytes, marks_for_overlay)
    except Exception as e:
        print("[/vibode/swap] Error drawing swap markers:", e)
        raise HTTPException(status_code=500, detail="Failed to draw swap markers")

    maybe_dump_vibode_swap_images(
        room_clean_png_bytes=room_png_bytes,
        room_swap_overlay_png_bytes=room_overlay_png_bytes,
    )

    replacement_png_bytes_list: List[bytes] = []
    try:
        for asset in mapped_replacements:
            early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "replacement_prepare")
            if early_exit:
                return early_exit
            replacement_raw_bytes = _fetch_image_bytes_from_url(asset.imageUrl)
            replacement_png_bytes_list.append(prepare_sku_png_bytes(replacement_raw_bytes))
    except Exception as e:
        print("[/vibode/swap] Error preparing replacement asset images:", e)
        raise HTTPException(status_code=400, detail="Failed to fetch replacement asset image data")

    print(
        "[/vibode/swap] Received request:",
        {
            "marks": len(req.marks),
            "replacementAssets": len(req.replacementAssets),
            "mappedReplacementAssets": len(mapped_replacements),
            "marks_summary": _summarize_swap_marks(marks_ordered, mapped_replacements),
            "room_bytes_len": len(room_raw_bytes),
            "room_png_len": len(room_png_bytes),
            "modelVersion": req.modelVersion,
            "modelName": model_name,
            "requestedAspectRatio": "auto",
            "appliedAspectRatio": applied_ratio,
            "sentAspectRatio": aspect_ratio_to_send if aspect_ratio_to_send else "(omitted)",
            "maxInputLongEdge": MAX_INPUT_LONG_EDGE_INT,
        },
    )

    swap_prompt_hash = _short_prompt_hash(VIBODE_SWAP_PROMPT)
    swap_prompt_first_line = _prompt_first_line(VIBODE_SWAP_PROMPT)
    print(
        "[/vibode/swap] Prompt summary:",
        {
            "prompt_hash": swap_prompt_hash,
            "prompt_first_line": swap_prompt_first_line,
            "marks_summary": _summarize_swap_marks(marks_ordered, mapped_replacements),
        },
    )
    if VIBODE_LOG_PROMPTS:
        print("\n===== VIBODE SWAP PROMPT SENT TO GEMINI =====\n")
        print(VIBODE_SWAP_PROMPT)
        print("\n==============================================\n")

    early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "model_call")
    if early_exit:
        return early_exit
    try:
        out_bytes = call_gemini_multimodal(
            prompt=VIBODE_SWAP_PROMPT,
            room_png_bytes=room_png_bytes,
            room_overlay_png_bytes=room_overlay_png_bytes,
            sku_png_bytes_list=replacement_png_bytes_list,
            model_name=model_name,
            aspect_ratio=aspect_ratio_to_send,
        )
    except Exception as e:
        log_event("vibode_swap_processing_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Error during swap")

    if not out_bytes:
        raise HTTPException(status_code=500, detail="Swap returned empty image")

    early_exit = _ensure_paste_to_place_job_active(paste_to_place_control, route, "final_response")
    if early_exit:
        return early_exit
    data_url = make_data_url(out_bytes, mime_type="image/png")
    return VibodeSwapResponse(imageUrl=data_url)


@app.post("/vibode/rotate", response_model=VibodeComposeResponse)
async def vibode_rotate(req: VibodeRotateRequest):
    marks_ordered = _extract_rotate_marks(
        freeze_payload=req.freezePayload,
        request_marks=req.marks,
    )

    image_source_kind = "cleanBase64"
    if req.cleanBase64 and req.cleanBase64.strip():
        try:
            room_raw_bytes = _decode_base64_image(req.cleanBase64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 room image data")
    elif req.baseImageUrl and req.baseImageUrl.strip():
        image_source_kind = "baseImageUrl"
        try:
            room_raw_bytes = _fetch_image_bytes_from_url(req.baseImageUrl)
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to fetch baseImageUrl image data")
    else:
        raise HTTPException(status_code=400, detail="Provide either cleanBase64 or baseImageUrl.")

    model_name = resolve_model_name_for_route("/vibode/rotate", req.modelVersion)

    try:
        room_orig_img = _safe_open_image(room_raw_bytes)
        orig_w, orig_h = room_orig_img.size
    except Exception as e:
        print("[/vibode/rotate] Error decoding room image dimensions:", e)
        raise HTTPException(status_code=400, detail="Could not process room image")

    applied_ratio: Optional[str] = None
    aspect_ratio_to_send: Optional[str] = None
    try:
        log_continuation_aspect_ratio_omitted("/vibode/rotate")
        room_png_bytes = prepare_passthrough_png_bytes(room_raw_bytes)
        applied_ratio = None
        aspect_ratio_to_send = None
    except Exception as e:
        print("[/vibode/rotate] Error preparing room image:", e)
        raise HTTPException(status_code=400, detail="Could not process room image")

    try:
        room_pre_overlay_img = _safe_open_image(room_png_bytes)
        new_w, new_h = room_pre_overlay_img.size
    except Exception as e:
        print("[/vibode/rotate] Error decoding prepared room image dimensions:", e)
        raise HTTPException(status_code=500, detail="Failed to prepare room overlay image")

    try:
        room_overlay_png_bytes = render_vibode_rotate_overlay(room_png_bytes, marks_ordered)
    except Exception as e:
        print("[/vibode/rotate] Error drawing rotate markers:", e)
        raise HTTPException(status_code=500, detail="Failed to draw rotate markers")

    rotate_prompt = build_vibode_rotate_prompt(marks_ordered)
    rotate_prompt_hash = _short_prompt_hash(rotate_prompt)
    rotate_prompt_first_line = _prompt_first_line(rotate_prompt)

    maybe_dump_vibode_rotate_images(
        room_clean_png_bytes=room_png_bytes,
        room_rotate_overlay_png_bytes=room_overlay_png_bytes,
        prompt_hash=rotate_prompt_hash,
    )

    print(
        "[/vibode/rotate] Received request:",
        {
            "marks": len(marks_ordered),
            "marks_summary": _summarize_rotate_marks(marks_ordered),
            "freezePayloadKeys": sorted(req.freezePayload.keys()),
            "sourceImage": image_source_kind,
            "baseImageUrlProvided": bool(req.baseImageUrl and req.baseImageUrl.strip()),
            "room_bytes_len": len(room_raw_bytes),
            "room_png_len": len(room_png_bytes),
            "originalDims": (orig_w, orig_h),
            "preparedDims": (new_w, new_h),
            "modelVersion": req.modelVersion,
            "modelName": model_name,
            "requestedAspectRatio": req.aspectRatio,
            "appliedAspectRatio": applied_ratio,
            "sentAspectRatio": aspect_ratio_to_send if aspect_ratio_to_send else "(omitted)",
            "maxInputLongEdge": MAX_INPUT_LONG_EDGE_INT,
        },
    )
    print(
        "[/vibode/rotate] Prompt summary:",
        {
            "prompt_hash": rotate_prompt_hash,
            "prompt_first_line": rotate_prompt_first_line,
            "marks_summary": _summarize_rotate_marks(marks_ordered),
        },
    )
    if VIBODE_LOG_PROMPTS:
        print("\n===== VIBODE ROTATE PROMPT SENT TO GEMINI =====\n")
        print(rotate_prompt)
        print("\n================================================\n")

    try:
        out_bytes = call_gemini_multimodal(
            prompt=rotate_prompt,
            room_png_bytes=room_png_bytes,
            room_overlay_png_bytes=room_overlay_png_bytes,
            sku_png_bytes_list=[],
            model_name=model_name,
            aspect_ratio=aspect_ratio_to_send,
        )
    except Exception as e:
        log_event("vibode_rotate_processing_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Error during rotate")

    if not out_bytes:
        raise HTTPException(status_code=500, detail="Rotate returned empty image")

    data_url = make_data_url(out_bytes, mime_type="image/png")

    debug_ratio: Optional[str] = None
    if DEBUG_ROOMPRINTZ_RATIO:
        debug_ratio = "passthrough"

    return VibodeComposeResponse(imageUrl=data_url, appliedAspectRatio=debug_ratio)

# Quick test:
# curl -sS -X POST "http://localhost:8000/api/vibode/user-skus/ingest" -H "Content-Type: application/json" -d '{"label":"Demo SKU","imageUrl":"https://example.com/product.png"}'
