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
from uuid import uuid4
from urllib.parse import quote, urlparse
from typing import Any, Literal, Optional, Tuple, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator
from PIL import Image
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

# Toggle prompt logging with env var: DEBUG_ROOMPRINTZ_PROMPT=1
DEBUG_ROOMPRINTZ_PROMPT = os.getenv("DEBUG_ROOMPRINTZ_PROMPT", "0") == "1"

# Strict Stage 3 prompt dump flag (route-level, exact prompt text).
DEBUG_ROOMPRINTZ_STAGE3_PROMPT = os.getenv("DEBUG_ROOMPRINTZ_STAGE3_PROMPT", "0") == "1"

# Toggle ratio debug return
DEBUG_ROOMPRINTZ_RATIO = os.getenv("DEBUG_ROOMPRINTZ_RATIO", "0") == "1"

# Optional: cap input size to keep cost + latency down (resize down only; never upscale)
# Set to "" to disable.
MAX_INPUT_LONG_EDGE = os.getenv("ROOMPRINTZ_MAX_INPUT_LONG_EDGE", "2048").strip()
MAX_INPUT_LONG_EDGE_INT = (
    int(MAX_INPUT_LONG_EDGE) if MAX_INPUT_LONG_EDGE.isdigit() else None
)

# ✅ CHANGE: beta testing wants ALL ratios for Gemini 2.5 Flash.
# We keep the env var, but default it to "1" so Flash is NOT forced to 1:1.
ALLOW_FLASH_NON_SQUARE = os.getenv("ROOMPRINTZ_ALLOW_FLASH_NON_SQUARE", "1") == "1"

# User SKU ingest config
USER_SKU_MAX_INPUT_BYTES = 12 * 1024 * 1024  # 12 MB
USER_SKU_NORMALIZED_MAX_DIM = 1536
USER_SKU_NORMALIZED_PADDING_RATIO = 0.03
USER_SKU_INGEST_TIMEOUT_SECONDS = 10.0
SUPABASE_SIGNED_URL_TTL_SECONDS = 7 * 24 * 60 * 60
SUPABASE_STORAGE_UPLOAD_TIMEOUT_SECONDS = 20.0
SUPABASE_URL = (os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or "").strip()
SUPABASE_SERVICE_KEY = (os.getenv("SUPABASE_SERVICE_KEY") or "").strip()
SUPABASE_STORAGE_BUCKET = (
    os.getenv("SUPABASE_STORAGE_BUCKET")
    or os.getenv("NEXT_PUBLIC_SUPABASE_STORAGE_BUCKET")
    or ""
).strip()
_SUPABASE_STORAGE_BUCKET_CACHE: Optional[str] = None

USER_SKU_BG_REMOVAL_PROMPT = (
    "Isolate the product and replace the entire background with a flat, uniform, solid light grey color (#F2F2F2).\n"
    "The background must be a single solid color.\n"
    "No gradients.\n"
    "No shadows.\n"
    "No floor.\n"
    "No texture.\n"
    "No checkerboard pattern.\n"
    "Do not simulate transparency.\n"
    "Preserve the product's exact shape, scale, and materials.\n"
    "Keep clean edges.\n"
    "Include a small margin around the product."
)


def resolve_model_name(model_version: Optional[str]) -> str:
    """
    Map a simple modelVersion string from the frontend into a concrete Gemini model ID.

    Expected values from the frontend:
    - "gemini-3"   -> "gemini-3-pro-image-preview" (Nano Banana Pro, default)
    - "gemini-2.5" -> "gemini-2.5-flash-image"    (OG Nano Banana)

    If a full model ID is passed, we just use it as-is.
    """
    if not model_version or model_version.strip() == "":
        return DEFAULT_MODEL_NAME

    v = model_version.strip().lower()

    if v in ("gemini-3", "gemini-3-pro", "gemini-3-pro-image-preview"):
        return "gemini-3-pro-image-preview"

    if v in ("gemini-2.5", "gemini-2.5-flash-image"):
        return "gemini-2.5-flash-image"

    return model_version


# ---------- ASPECT RATIO NORMALIZATION ----------

AspectRatio = Literal["auto", "4:3", "3:2", "16:9", "1:1"]
Stage4StyleMode = Literal[
    "style_room",
    "accessories",
    "wall_art",
    "shelves",
    "curtains",
    "ceiling_light",
]

RATIO_MAP: Dict[str, float] = {
    "4:3": 4 / 3,
    "3:2": 3 / 2,
    "16:9": 16 / 9,
    "1:1": 1.0,
}

SUPPORTED_RATIOS_ORDERED = ["4:3", "3:2", "16:9", "1:1"]

_REQUEST_ID_CTX: ContextVar[str] = ContextVar("roomprintz_request_id", default="-")


def get_request_id() -> str:
    return _REQUEST_ID_CTX.get()


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


# ---------- FASTAPI APP ----------

app = FastAPI()


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    request_id = uuid4().hex[:12]
    token = _REQUEST_ID_CTX.set(request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response
    finally:
        _REQUEST_ID_CTX.reset(token)


# ---------- MODELS ----------

class HealthResponse(BaseModel):
    status: str


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


class VibodeEditRunRequest(BaseModel):
    baseImageUrl: Optional[str] = None
    action: Literal["add", "remove", "swap", "rotate", "move"]
    placements: List[ScenePlacement]
    target: Optional[VibodeEditRunTarget] = None
    params: Optional[Dict[str, Any]] = None
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


class VibodeMoveMark(BaseModel):
    id: str
    xNorm: float
    yNorm: float
    dxNorm: float
    dyNorm: float


class VibodeMoveRequest(BaseModel):
    imageUrl: Optional[str] = None
    imageBase64: Optional[str] = None
    marks: List[VibodeMoveMark]
    modelVersion: Optional[str] = None
    aspectRatio: Optional[AspectRatio] = "auto"


class VibodeUserSkuIngestRequest(BaseModel):
    imageUrl: Optional[str] = None
    imageBase64: Optional[str] = None
    label: Optional[str] = None

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
) -> bytes:
    """
    If aspect_ratio is None, we OMIT image_config.aspect_ratio entirely
    (premium continuation behavior).
    """
    started_at = time.perf_counter()
    logged_terminal_failure = False
    log_event(
        "model_call_start",
        function="call_gemini_with_prompt",
        model_name=model_name,
        modality="image+text",
        aspect_ratio=aspect_ratio if aspect_ratio else "(omitted)",
        image_count=1,
        input_png_bytes=len(image_png_bytes),
    )
    try:
        config_kwargs = {"response_modalities": ["IMAGE"]}

        if aspect_ratio:
            config_kwargs["image_config"] = types.ImageConfig(aspect_ratio=aspect_ratio)

        response = client.models.generate_content(
            model=model_name,
            contents=[
                prompt,
                types.Part(
                    inline_data=types.Blob(
                        data=image_png_bytes,
                        mime_type="image/png",
                    )
                ),
            ],
            config=types.GenerateContentConfig(**config_kwargs),
        )

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
        return out_bytes

    except Exception as e:
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
) -> bytes:
    prompt = build_roomprintz_prompt(
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

    return call_gemini_with_prompt(image_png_bytes, prompt, model_name, aspect_ratio)


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
    )


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
            print(
                "[_supabase_storage_create_signed_url] Warning: unexpected absolute signedURL path prefix",
                {"path": parsed.path},
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
        print(
            "[_supabase_storage_create_signed_url] signedURL normalization",
            {"original": signed_url_value, "normalized": normalized_signed_url},
        )
    return normalized_signed_url


def _run_user_sku_background_removal(image_png_bytes: bytes, model_name: str) -> bytes:
    return call_gemini_with_prompt(
        image_png_bytes=image_png_bytes,
        prompt=USER_SKU_BG_REMOVAL_PROMPT,
        model_name=model_name,
        aspect_ratio=None,
    )


def _normalize_user_sku_transparent_png(
    image_bytes: bytes,
    max_dimension: int = USER_SKU_NORMALIZED_MAX_DIM,
    padding_ratio: float = USER_SKU_NORMALIZED_PADDING_RATIO,
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
    thickness = 3
    min_x = max(0, cx - r - thickness)
    max_x = min(width - 1, cx + r + thickness)
    min_y = max(0, cy - r - thickness)
    max_y = min(height - 1, cy + r + thickness)
    r_inner = max(0, r - thickness)
    r_outer = r + thickness
    r_inner_sq = r_inner * r_inner
    r_outer_sq = r_outer * r_outer
    for y in range(min_y, max_y + 1):
        dy = y - cy
        dy_sq = dy * dy
        for x in range(min_x, max_x + 1):
            dx = x - cx
            dist_sq = dx * dx + dy_sq
            if r_inner_sq <= dist_sq <= r_outer_sq:
                pixels[x, y] = (255, 0, 0)


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
) -> None:
    font_size = max(14, int(radius * 0.9))
    font = _load_marker_label_font(font_size)
    stroke_width = max(1, int(round(radius * 0.12)))
    try:
        draw.text(
            (cx, cy),
            marker_label,
            fill=(255, 255, 255),
            font=font,
            anchor="mm",
            stroke_width=stroke_width,
            stroke_fill=(255, 0, 0),
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
        fill=(255, 255, 255),
        font=font,
        stroke_width=stroke_width,
        stroke_fill=(0, 0, 0),
    )


def draw_red_markers_overlay(image_png_bytes: bytes, placements: List[VibodePlacement]) -> bytes:
    img = _safe_open_image(image_png_bytes)
    if img.mode != "RGB":
        img = img.convert("RGB")
    width, height = img.size
    max_radius = max(1, min(width, height) // 4)
    if _HAS_IMAGE_DRAW and ImageDraw is not None:
        draw = ImageDraw.Draw(img)
        for idx, placement in enumerate(placements):
            radius = int(round(placement.rPx)) if placement.rPx else 60
            radius = max(20, radius)
            radius = min(radius, max_radius)
            cx = int(round(placement.cxPx))
            cy = int(round(placement.cyPx))
            cx = max(0, min(cx, width - 1))
            cy = max(0, min(cy, height - 1))
            bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
            draw.ellipse(bbox, outline=(255, 0, 0), width=6)
            marker_label = str(idx + 1)
            _draw_marker_label(draw, cx, cy, marker_label, radius)
    else:
        for placement in placements:
            radius = int(round(placement.rPx)) if placement.rPx else 60
            radius = max(20, radius)
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


def _clamp_move_delta(delta: float) -> float:
    if not math.isfinite(delta):
        return 0.0
    return max(-1.0, min(1.0, float(delta)))


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


def render_vibode_move_overlay(base_img: Image.Image, marks: List[VibodeMoveMark]) -> Image.Image:
    img = base_img.copy()
    if img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    min_dim = min(width, height)
    anchor_radius = max(8, int(round(min_dim * 0.016)))
    anchor_radius = min(anchor_radius, 28)
    anchor_ring_width = max(2, int(round(anchor_radius * 0.28)))
    line_width = max(3, int(round(anchor_radius * 0.52)))
    head_len = max(8, int(round(anchor_radius * 1.05)))

    teal_anchor = (25, 184, 174)
    cyan_arrow = (35, 218, 255)
    white = (255, 255, 255)
    label_bg = (16, 22, 28)

    if _HAS_IMAGE_DRAW and ImageDraw is not None:
        draw = ImageDraw.Draw(img)
        font = _load_marker_label_font(max(10, int(round(anchor_radius * 0.95))))

        for idx, mark in enumerate(marks):
            start_x = _normalized_to_pixel(mark.xNorm, width)
            start_y = _normalized_to_pixel(mark.yNorm, height)
            end_x_norm = max(0.0, min(1.0, float(mark.xNorm) + _clamp_move_delta(mark.dxNorm)))
            end_y_norm = max(0.0, min(1.0, float(mark.yNorm) + _clamp_move_delta(mark.dyNorm)))
            end_x = _normalized_to_pixel(end_x_norm, width)
            end_y = _normalized_to_pixel(end_y_norm, height)

            draw.line((start_x, start_y, end_x, end_y), fill=cyan_arrow, width=line_width)

            cap_radius = max(2, int(round(line_width * 0.5)))
            draw.ellipse(
                (
                    start_x - cap_radius,
                    start_y - cap_radius,
                    start_x + cap_radius,
                    start_y + cap_radius,
                ),
                fill=cyan_arrow,
            )
            draw.ellipse(
                (
                    end_x - cap_radius,
                    end_y - cap_radius,
                    end_x + cap_radius,
                    end_y + cap_radius,
                ),
                fill=cyan_arrow,
            )

            theta = math.atan2((end_y - start_y), (end_x - start_x))
            left_theta = theta + math.radians(152)
            right_theta = theta - math.radians(152)
            left_x = int(round(end_x + (math.cos(left_theta) * head_len)))
            left_y = int(round(end_y + (math.sin(left_theta) * head_len)))
            right_x = int(round(end_x + (math.cos(right_theta) * head_len)))
            right_y = int(round(end_y + (math.sin(right_theta) * head_len)))

            draw.line((end_x, end_y, left_x, left_y), fill=cyan_arrow, width=max(2, line_width - 1))
            draw.line((end_x, end_y, right_x, right_y), fill=cyan_arrow, width=max(2, line_width - 1))

            anchor_outer_bbox = (
                start_x - anchor_radius,
                start_y - anchor_radius,
                start_x + anchor_radius,
                start_y + anchor_radius,
            )
            draw.ellipse(anchor_outer_bbox, fill=white)

            anchor_inner_radius = max(2, anchor_radius - anchor_ring_width)
            anchor_inner_bbox = (
                start_x - anchor_inner_radius,
                start_y - anchor_inner_radius,
                start_x + anchor_inner_radius,
                start_y + anchor_inner_radius,
            )
            draw.ellipse(anchor_inner_bbox, fill=teal_anchor)

            label_radius = max(8, int(round(anchor_radius * 0.74)))
            label_cx = start_x + anchor_radius + label_radius
            label_cy = start_y - anchor_radius - label_radius
            label_cx = max(label_radius, min(label_cx, width - 1 - label_radius))
            label_cy = max(label_radius, min(label_cy, height - 1 - label_radius))
            label_bbox = (
                label_cx - label_radius,
                label_cy - label_radius,
                label_cx + label_radius,
                label_cy + label_radius,
            )
            draw.ellipse(label_bbox, fill=label_bg, outline=white, width=1)
            marker_label = str(idx + 1)
            try:
                draw.text(
                    (label_cx, label_cy),
                    marker_label,
                    fill=white,
                    font=font,
                    anchor="mm",
                    stroke_width=1,
                    stroke_fill=(0, 0, 0),
                )
            except Exception:
                draw.text((label_cx - 3, label_cy - 6), marker_label, fill=white, font=font)
    else:
        for idx, mark in enumerate(marks):
            start_x = _normalized_to_pixel(mark.xNorm, width)
            start_y = _normalized_to_pixel(mark.yNorm, height)
            end_x_norm = max(0.0, min(1.0, float(mark.xNorm) + _clamp_move_delta(mark.dxNorm)))
            end_y_norm = max(0.0, min(1.0, float(mark.yNorm) + _clamp_move_delta(mark.dyNorm)))
            end_x = _normalized_to_pixel(end_x_norm, width)
            end_y = _normalized_to_pixel(end_y_norm, height)

            _draw_colored_line_pixels(
                img=img,
                x1=start_x,
                y1=start_y,
                x2=end_x,
                y2=end_y,
                color=cyan_arrow,
                thickness=line_width,
            )
            theta = math.atan2((end_y - start_y), (end_x - start_x))
            left_theta = theta + math.radians(152)
            right_theta = theta - math.radians(152)
            left_x = int(round(end_x + (math.cos(left_theta) * head_len)))
            left_y = int(round(end_y + (math.sin(left_theta) * head_len)))
            right_x = int(round(end_x + (math.cos(right_theta) * head_len)))
            right_y = int(round(end_y + (math.sin(right_theta) * head_len)))
            _draw_colored_line_pixels(
                img=img,
                x1=end_x,
                y1=end_y,
                x2=left_x,
                y2=left_y,
                color=cyan_arrow,
                thickness=max(2, line_width - 1),
            )
            _draw_colored_line_pixels(
                img=img,
                x1=end_x,
                y1=end_y,
                x2=right_x,
                y2=right_y,
                color=cyan_arrow,
                thickness=max(2, line_width - 1),
            )

            _draw_colored_disc_pixels(img=img, cx=start_x, cy=start_y, radius=anchor_radius, color=white)
            _draw_colored_disc_pixels(
                img=img,
                cx=start_x,
                cy=start_y,
                radius=max(2, anchor_radius - anchor_ring_width),
                color=teal_anchor,
            )

    return img


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


def _summarize_move_marks(
    marks: List[VibodeMoveMark],
    limit: int = 5,
) -> Dict[str, object]:
    preview: List[Dict[str, object]] = []
    for idx, mark in enumerate(marks[:limit]):
        clamped_x = max(0.0, min(1.0, float(mark.xNorm)))
        clamped_y = max(0.0, min(1.0, float(mark.yNorm)))
        clamped_dx = _clamp_move_delta(mark.dxNorm)
        clamped_dy = _clamp_move_delta(mark.dyNorm)
        preview.append(
            {
                "markerIndex": idx + 1,
                "markId": mark.id,
                "xNorm": round(clamped_x, 4),
                "yNorm": round(clamped_y, 4),
                "dxNorm": round(clamped_dx, 4),
                "dyNorm": round(clamped_dy, 4),
                "endXNorm": round(max(0.0, min(1.0, clamped_x + clamped_dx)), 4),
                "endYNorm": round(max(0.0, min(1.0, clamped_y + clamped_dy)), 4),
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


def maybe_dump_vibode_move_images(
    room_clean_png_bytes: bytes,
    room_move_overlay_png_bytes: bytes,
    output_png_bytes: bytes,
    stable_seed: str,
    output_dir: Optional[str] = None,
) -> None:
    if not _env_truthy("VIBODE_DUMP_ANNOTATED_IMAGE"):
        return

    try:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        resolved_dir = output_dir or os.getenv("VIBODE_DEBUG_DIR") or "tmp/vibode_debug"
        abs_dir = os.path.abspath(resolved_dir)
        os.makedirs(abs_dir, exist_ok=True)

        normalized_seed = (stable_seed or "").strip() or timestamp
        short_id = hashlib.sha256(normalized_seed.encode("utf-8")).hexdigest()[:8]
        file_prefix = f"vibode_move_{timestamp}_{short_id}"

        clean_path = os.path.join(abs_dir, f"{file_prefix}_clean.png")
        overlay_path = os.path.join(abs_dir, f"{file_prefix}_overlay.png")
        output_path = os.path.join(abs_dir, f"{file_prefix}_output.png")

        with open(clean_path, "wb") as handle:
            handle.write(room_clean_png_bytes)
        with open(overlay_path, "wb") as handle:
            handle.write(room_move_overlay_png_bytes)
        with open(output_path, "wb") as handle:
            handle.write(output_png_bytes)

        print(
            "[maybe_dump_vibode_move_images] wrote:",
            {
                "clean": clean_path,
                "overlay": overlay_path,
                "output": output_path,
                "stableSeed": normalized_seed,
            },
        )
    except Exception as e:
        print("[maybe_dump_vibode_move_images] failed:", e)


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
    lines = [
        "You are a professional real-estate photo editor.",
        "",
        "You are given multiple images in this exact order:",
        "1) Image 1 is the clean prepared room photo (background reference).",
        "   Final output must match Image 1's room exactly.",
        f"2) Image 2 is the same prepared room with numbered red circle markers (1..{placements_count}) for guidance only.",
        "   Use Image 2 only to locate targets.",
        "3) Each subsequent image is a single SKU item to insert, ordered by marker number.",
        "Place SKU image i at marker i. Do not swap indices. Do not invent additional furniture.",
        "",
        "Strict rules:",
        "- Place each SKU item at its corresponding numbered red marker location on the floor.",
        "- Use Image 1 as the source of truth for room appearance; keep its background unchanged.",
        "- Use Image 2 only as placement guidance; do not keep any markers in the final result.",
        "- Do not move, resize, or rotate the room camera perspective.",
        "- Do not change existing walls, windows, doors, floors, or lighting.",
        "- Do not invent extra furniture or decor; only insert the provided SKU items.",
        "- Add realistic contact shadows so items sit naturally on the floor.",
        "- Remove all markup in the final image: no circles, numbers, or any text.",
        "- Do not add text, logos, or watermarks.",
    ]
    if enhance_photo:
        lines += [
            "",
            "Enhance photo quality subtly:",
            "- Correct white balance and exposure without changing room appearance.",
            "- Improve clarity and reduce noise while staying photorealistic.",
        ]
    if DEBUG_ROOMPRINTZ_PROMPT:
        print("\n===== VIBODE COMPOSE PROMPT SENT TO GEMINI =====\n")
        print("\n".join(lines))
        print("\n=================================================\n")
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
        "- Do not rearrange existing furniture unless necessary to place the selected SKUs.",
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


def build_vibode_move_prompt(marks: List[VibodeMoveMark]) -> str:
    marker_count = len(marks)
    lines = [
        "You are a professional real-estate photo editor.",
        "",
        "You are given exactly two images in this order:",
        "1) Image 1 is the clean/base room image and is the source of truth.",
        f"2) Image 2 is the same room with numbered move overlays (anchor + arrow) for markers 1..{marker_count}.",
        "",
        "Critical interpretation:",
        "- Each marker's anchor identifies exactly one object to edit.",
        "- Each marker's arrow is a translation vector only.",
        "- Translation only: move the object in-place by the exact vector shown.",
        "- Do not interpret the arrow as rotation.",
        "",
        "Hard constraints to preserve:",
        "- Preserve lighting and shadows (direction, softness, and intensity).",
        "- Preserve camera perspective.",
        "- Preserve object scale.",
        "- Preserve object rotation.",
        "- Keep all non-target objects unchanged.",
        "",
        "Strict prohibitions:",
        "- Do not restage the room.",
        "- Do not relight the scene.",
        "- Do not change materials or colors.",
        "- Do not add or remove items.",
        "- Do not leave markers, arrows, numbers, text, logos, or watermarks in the final image.",
        "",
        "Apply these translations in marker order:",
    ]

    for idx, mark in enumerate(marks):
        marker_index = idx + 1
        clamped_x = max(0.0, min(1.0, float(mark.xNorm)))
        clamped_y = max(0.0, min(1.0, float(mark.yNorm)))
        clamped_dx = _clamp_move_delta(mark.dxNorm)
        clamped_dy = _clamp_move_delta(mark.dyNorm)
        end_x = max(0.0, min(1.0, clamped_x + clamped_dx))
        end_y = max(0.0, min(1.0, clamped_y + clamped_dy))
        lines.extend(
            [
                (
                    f"- Marker #{marker_index}: Move ONLY the object at anchor #{marker_index} "
                    f"from ({clamped_x:.4f}, {clamped_y:.4f}) to ({end_x:.4f}, {end_y:.4f}) "
                    f"using translation vector ({clamped_dx:.4f}, {clamped_dy:.4f})."
                ),
                "  Translation only. Keep rotation, scale, perspective, lighting, and shadows unchanged.",
            ]
        )

    return "\n".join(lines)


def _parse_and_clamp_move_marks(marks: List[VibodeMoveMark]) -> List[VibodeMoveMark]:
    if not marks:
        raise HTTPException(status_code=400, detail="marks must be non-empty.")

    parsed_marks: List[VibodeMoveMark] = []
    for idx, mark in enumerate(marks):
        raw_x = float(mark.xNorm)
        raw_y = float(mark.yNorm)
        raw_dx = float(mark.dxNorm)
        raw_dy = float(mark.dyNorm)
        if (
            not math.isfinite(raw_x)
            or not math.isfinite(raw_y)
            or not math.isfinite(raw_dx)
            or not math.isfinite(raw_dy)
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"marks[{idx}] has invalid values; xNorm, yNorm, dxNorm, and dyNorm "
                    "must be finite numbers."
                ),
            )

        clamped_x = max(0.0, min(1.0, raw_x))
        clamped_y = max(0.0, min(1.0, raw_y))
        clamped_dx = _clamp_move_delta(raw_dx)
        clamped_dy = _clamp_move_delta(raw_dy)
        parsed_marks.append(
            mark.model_copy(
                update={
                    "xNorm": clamped_x,
                    "yNorm": clamped_y,
                    "dxNorm": clamped_dx,
                    "dyNorm": clamped_dy,
                }
            )
        )

    return parsed_marks


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


def _target_area_list_prompt_snippet(targets: List[ScenePlacementBbox]) -> str:
    if not targets:
        return "The target edits should occur roughly in the intended target areas."
    centers = ", ".join(
        f"({target.x + (target.w / 2.0):.4f}, {target.y + (target.h / 2.0):.4f})" for target in targets
    )
    return f"The target edits should occur roughly near normalized locations: {centers}."


def build_vibode_edit_run_prompt(
    action: Literal["add", "remove", "swap", "rotate", "move"],
    target_placement: Optional[ScenePlacement],
    params: Optional[Dict[str, Any]],
    sku_label: Optional[str] = None,
    remove_target_bboxes: Optional[List[ScenePlacementBbox]] = None,
) -> str:
    params = params or {}
    bbox = target_placement.bbox if target_placement else None
    remove_target_bboxes = remove_target_bboxes or []
    if action == "remove" and remove_target_bboxes:
        target_guidance = _target_area_list_prompt_snippet(remove_target_bboxes)
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
        if remove_target_bboxes:
            lines.extend(
                [
                    "Task: Remove only the target objects in the intended target areas.",
                    "Fill background naturally.",
                    "Do not remove surrounding room content.",
                    "Do not alter other objects.",
                ]
            )
        else:
            lines.extend(
                [
                    "Task: Remove only the target object in the intended target area.",
                    "Fill background naturally.",
                    "Do not remove surrounding room content.",
                ]
            )
    elif action == "swap":
        lines.extend(
            [
                "Task: Replace the target object in the intended target area using the provided SKU asset image.",
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
        lines.extend(
            [
                "Task: Rotate the target object in place in the intended target area.",
                (
                    f"Apply a {rotation_deg} degree rotation."
                    if rotation_deg is not None
                    else "Rotate to the requested orientation."
                ),
                "Preserve believable real-world scale and perspective.",
                "Do not move any other object.",
            ]
        )
    else:
        dx = params.get("dx")
        dy = params.get("dy")
        lines.extend(
            [
                "Task: Move the target object to the updated intended target area.",
                (
                    f"Translation vector normalized is (dx={dx}, dy={dy})."
                    if dx is not None and dy is not None
                    else "Move to the requested updated target area."
                ),
                "Preserve believable real-world scale and perspective.",
                "Do not alter any other object.",
            ]
        )

    return "\n".join(lines)


def _debug_log_vibode_edit_run(
    action: str,
    model_name: str,
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
) -> None:
    print("[/api/vibode/edit-run] request")
    print(f"  action={action}")
    print(f"  model={model_name}")
    print(f"  aspect_ratio={aspect_ratio_to_send if aspect_ratio_to_send else '(omitted)'}")
    print(f"  baseImageUrl kind={base_image_url_kind}")
    print(f"  placements={placements_count}")
    print(f"  eligible_skus={eligible_skus_count}")
    print(f"  target_placement_id={target_placement_id if target_placement_id else '(none)'}")
    print(f"  target_sku_id={target_sku_id if target_sku_id else '(none)'}")
    print(f"  params={params}")
    if "removeTargets" in params:
        remove_targets = params.get("removeTargets")
        remove_targets_count = len(remove_targets) if isinstance(remove_targets, list) else 0
        print(f"  remove_targets={remove_targets_count}")
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
    print("  prompt:")
    print(prompt)


def call_gemini_multimodal(
    prompt: str,
    room_png_bytes: bytes,
    room_overlay_png_bytes: bytes,
    sku_png_bytes_list: List[bytes],
    model_name: str,
    aspect_ratio: Optional[str] = None,
) -> bytes:
    started_at = time.perf_counter()
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
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        try:
            candidate = response.candidates[0]
            part = candidate.content.parts[0]
            out_bytes = part.inline_data.data
        except Exception as e:
            log_event(
                "model_call_extract_failed",
                function="call_gemini_multimodal",
                model_name=model_name,
                modality="multimodal",
                aspect_ratio=aspect_ratio if aspect_ratio else "(omitted)",
                sku_count=len(sku_png_bytes_list),
                error=str(e),
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
        return out_bytes
    except Exception as e:
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


# ---------- ROUTES ----------

@app.get("/", response_model=HealthResponse)
async def read_root():
    return HealthResponse(status="ok")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok")


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

    model_name = resolve_model_name(req.modelVersion)

    try:
        raw_bytes = base64.b64decode(req.imageBase64)
    except Exception as e:
        print("[/stage-room] Failed to decode base64:", e)
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    applied_ratio: Optional[str] = None
    aspect_ratio_to_send: Optional[str] = None

    try:
        if req.isContinuation:
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
        },
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

    model_name = resolve_model_name(req.modelVersion)
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

    model_name = resolve_model_name(req.modelVersion)

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

    model_name = resolve_model_name(req.modelVersion)

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
async def vibode_stage_run(req: VibodeStageRunRequest):
    _reject_if_vibode_strict_missing(
        "/api/vibode/stage-run",
        _collect_vibode_stage_run_missing_fields(req),
    )

    model_name = resolve_model_name(req.modelVersion)

    try:
        room_raw_bytes = _resolve_vibode_stage_run_room_raw_bytes(req)
    except HTTPException:
        raise
    except Exception as e:
        log_event("vibode_stage_run_base_image_resolve_failed", error=str(e))
        raise HTTPException(status_code=400, detail="Failed to resolve stage-run base image data")

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
        stage=req.stage,
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

    try:
        if req.stage == 3:
            out_bytes = call_gemini_multimodal(
                prompt=prompt,
                room_png_bytes=room_png_bytes,
                room_overlay_png_bytes=room_overlay_png_bytes,
                sku_png_bytes_list=sku_png_bytes_list,
                model_name=model_name,
                aspect_ratio=aspect_ratio_to_send,
            )
        else:
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

    data_url = make_data_url(out_bytes, mime_type="image/png")

    debug_ratio: Optional[str] = None
    if DEBUG_ROOMPRINTZ_RATIO:
        debug_ratio = applied_ratio or "auto"

    return VibodeComposeResponse(imageUrl=data_url, appliedAspectRatio=debug_ratio)


@app.post("/api/vibode/edit-run", response_model=VibodeEditRunResponse)
async def vibode_edit_run(req: VibodeEditRunRequest):
    # v1 targeting is text-only using normalized bbox coordinates; this can be upgraded to mask/overlay targeting later.
    action = req.action
    params = req.params or {}
    target = req.target or VibodeEditRunTarget()

    if not req.baseImageUrl or not req.baseImageUrl.strip():
        return _vibode_edit_run_error(400, "baseImageUrl is required.")

    base_image_url_kind = "data-url" if req.baseImageUrl.strip().startswith("data:image/") else "remote-url"
    try:
        print("[/api/vibode/edit-run] baseImageUrl kind:", base_image_url_kind)
        room_raw_bytes = _load_image_ref_bytes(req.baseImageUrl.strip())
    except Exception:
        return _vibode_edit_run_error(400, "Failed to fetch baseImageUrl image data.")

    model_name = resolve_model_name(req.modelVersion)
    try:
        room_png_bytes, applied_ratio = normalize_image_bytes_for_ratio(
            room_raw_bytes,
            requested_ratio=req.aspectRatio,
            model_name=model_name,
        )
        aspect_ratio_to_send = applied_ratio
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
        if isinstance(remove_targets, list) and remove_targets:
            remove_target_bboxes: List[ScenePlacementBbox] = []
            remove_target_placement_ids: set[str] = set()
            target_placement_id = (target.placementId or "").strip() or None

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

            if not remove_target_bboxes and remove_target_placement_ids:
                for placement in updated_placements:
                    if placement.placementId in remove_target_placement_ids and placement.bbox:
                        remove_target_bboxes.append(_normalized_bbox_for_storage(placement.bbox))

            if not remove_target_bboxes:
                return _vibode_edit_run_error(400, "removeTargets did not include any valid target areas.")

            prompt = build_vibode_edit_run_prompt(
                action="remove",
                target_placement=None,
                params=params,
                remove_target_bboxes=remove_target_bboxes,
            )
            target_bbox = remove_target_bboxes[0] if remove_target_bboxes else None
            if remove_target_placement_ids:
                updated_placements = [
                    placement
                    for placement in updated_placements
                    if placement.placementId not in remove_target_placement_ids
                ]
        else:
            target_placement_id = (target.placementId or "").strip()
            if not target_placement_id:
                return _vibode_edit_run_error(400, "target.placementId is required for remove.")
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
            updated_placements.pop(target_idx)
    elif action == "swap":
        target_placement_id = (target.placementId or "").strip()
        if not target_placement_id:
            return _vibode_edit_run_error(400, "target.placementId is required for swap.")
        new_sku_id = str(params.get("newSkuId") or "").strip()
        if not new_sku_id:
            return _vibode_edit_run_error(400, "params.newSkuId is required for swap.")
        target_idx = _find_scene_placement_index(updated_placements, target_placement_id)
        if target_idx < 0:
            return _vibode_edit_run_error(400, f"placementId={target_placement_id} was not found.")

        sku = _find_eligible_sku(req.eligibleSkus, new_sku_id)
        if sku is None:
            return _vibode_edit_run_error(400, f"Could not find eligibleSkus entry for skuId={new_sku_id}.")
        sku_image_ref = _select_eligible_sku_image_ref(sku)
        if not sku_image_ref:
            return _vibode_edit_run_error(400, f"eligibleSkus skuId={new_sku_id} is missing variants[].imageUrl.")

        try:
            sku_raw_bytes = _load_image_ref_bytes(sku_image_ref)
            sku_png_bytes_list = [_convert_image_bytes_to_png(sku_raw_bytes)]
        except Exception:
            return _vibode_edit_run_error(400, f"Failed to fetch/prepare SKU image for skuId={new_sku_id}.")

        placement_before = updated_placements[target_idx]
        updated_placement = placement_before.model_copy(
            update={
                "skuId": new_sku_id,
                "label": sku.label,
                "source": sku.source if sku.source is not None else placement_before.source,
            }
        )
        updated_placements[target_idx] = updated_placement
        target_bbox = updated_placement.bbox
        prompt = build_vibode_edit_run_prompt(
            action="swap",
            target_placement=updated_placement,
            params=params,
            sku_label=sku.label,
        )
    elif action == "rotate":
        target_placement_id = (target.placementId or "").strip()
        if not target_placement_id:
            return _vibode_edit_run_error(400, "target.placementId is required for rotate.")
        raw_rotation = params.get("rotationDeg")
        if raw_rotation is None:
            return _vibode_edit_run_error(400, "params.rotationDeg is required for rotate.")
        try:
            rotation_deg = float(raw_rotation)
        except Exception:
            return _vibode_edit_run_error(400, "params.rotationDeg must be a number.")
        if not math.isfinite(rotation_deg):
            return _vibode_edit_run_error(400, "params.rotationDeg must be a finite number.")

        target_idx = _find_scene_placement_index(updated_placements, target_placement_id)
        if target_idx < 0:
            return _vibode_edit_run_error(400, f"placementId={target_placement_id} was not found.")

        updated_placement = updated_placements[target_idx].model_copy(update={"rotationDeg": rotation_deg})
        updated_placements[target_idx] = updated_placement
        target_bbox = updated_placement.bbox
        prompt = build_vibode_edit_run_prompt(
            action="rotate",
            target_placement=updated_placement,
            params={"rotationDeg": rotation_deg},
        )
    else:
        target_placement_id = (target.placementId or "").strip()
        if not target_placement_id:
            return _vibode_edit_run_error(400, "target.placementId is required for move.")
        raw_dx = params.get("dx")
        raw_dy = params.get("dy")
        if raw_dx is None or raw_dy is None:
            return _vibode_edit_run_error(400, "params.dx and params.dy are required for move.")
        try:
            dx = float(raw_dx)
            dy = float(raw_dy)
        except Exception:
            return _vibode_edit_run_error(400, "params.dx and params.dy must be numbers.")
        if not (math.isfinite(dx) and math.isfinite(dy)):
            return _vibode_edit_run_error(400, "params.dx and params.dy must be finite numbers.")

        target_idx = _find_scene_placement_index(updated_placements, target_placement_id)
        if target_idx < 0:
            return _vibode_edit_run_error(400, f"placementId={target_placement_id} was not found.")
        target_placement = updated_placements[target_idx]
        if not target_placement.bbox:
            return _vibode_edit_run_error(400, f"placementId={target_placement_id} is missing bbox for move.")

        normalized_bbox = _normalized_bbox_for_storage(target_placement.bbox)
        moved_bbox = _normalized_bbox_for_storage(
            ScenePlacementBbox(
                x=normalized_bbox.x + dx,
                y=normalized_bbox.y + dy,
                w=normalized_bbox.w,
                h=normalized_bbox.h,
            )
        )
        updated_placement = target_placement.model_copy(update={"bbox": moved_bbox})
        updated_placements[target_idx] = updated_placement
        target_bbox = updated_placement.bbox
        prompt = build_vibode_edit_run_prompt(
            action="move",
            target_placement=updated_placement,
            params={"dx": dx, "dy": dy},
        )

    room_overlay_png_bytes = room_png_bytes
    _debug_log_vibode_edit_run(
        action=action,
        model_name=model_name,
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
    )
    try:
        if action in ("add", "swap"):
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
            out_bytes = call_gemini_multimodal(**gemini_multimodal_kwargs)
        else:
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
    try:
        if _infer_image_mime_type(out_bytes) != "image/png":
            out_bytes = _convert_image_bytes_to_png(out_bytes)
    except Exception as e:
        print("[/api/vibode/edit-run] Failed to normalize output PNG bytes:", e)
        return _vibode_edit_run_error(500, "Edit run returned invalid image bytes.")

    scene_folder = _sanitize_storage_segment(req.sceneId, uuid4().hex)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    output_path = f"vibode-edits/{scene_folder}/{timestamp}_{action}.png"
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
    return VibodeEditRunResponse(imageUrl=signed_url, placements=updated_placements)


@app.post("/api/vibode/user-skus/ingest", response_model=VibodeUserSkuIngestResponse)
async def ingest_user_sku(request: VibodeUserSkuIngestRequest):
    user_sku_id = "user_" + uuid4().hex
    source_url = request.imageUrl.strip() if request.imageUrl and request.imageUrl.strip() else None
    resolved_label = (request.label or "").strip() or "User Upload"

    if source_url:
        normalized_source_url = source_url.lower()
        if not (
            normalized_source_url.startswith("http://")
            or normalized_source_url.startswith("https://")
        ):
            raise HTTPException(status_code=400, detail="Invalid imageUrl.")
        if (
            "localhost" in normalized_source_url
            or "127.0.0.1" in normalized_source_url
            or "0.0.0.0" in normalized_source_url
        ):
            raise HTTPException(status_code=400, detail="Invalid imageUrl.")

    print(
        "[/api/vibode/user-skus/ingest] image acquisition start",
        {
            "skuId": user_sku_id,
            "source": "imageUrl" if source_url else "imageBase64",
        },
    )

    try:
        if source_url:
            source_bytes, source_hint_mime = _fetch_image_bytes_from_url_limited(
                source_url,
                timeout_seconds=USER_SKU_INGEST_TIMEOUT_SECONDS,
                max_bytes=USER_SKU_MAX_INPUT_BYTES,
            )
        else:
            source_bytes, source_hint_mime = _decode_base64_image_with_mime(request.imageBase64 or "")
    except Exception as e:
        print("[/api/vibode/user-skus/ingest] image acquisition failed:", e)
        raise HTTPException(status_code=400, detail="Failed to acquire image bytes.")

    source_mime = _infer_image_mime_type(source_bytes, fallback_mime=source_hint_mime)
    print(
        "[/api/vibode/user-skus/ingest] image acquisition complete",
        {
            "skuId": user_sku_id,
            "bytes": len(source_bytes),
            "mimeType": source_mime,
        },
    )

    try:
        original_png_bytes = _convert_image_bytes_to_png(source_bytes)
    except Exception as e:
        print("[/api/vibode/user-skus/ingest] failed to decode input image:", e)
        raise HTTPException(status_code=400, detail="Invalid image payload.")

    original_path = f"user-skus/{user_sku_id}/original.png"
    try:
        _supabase_storage_upload_bytes(
            object_path=original_path,
            payload=original_png_bytes,
            mime_type="image/png",
        )
    except Exception as e:
        print("[/api/vibode/user-skus/ingest] original upload failed:", e)
        raise HTTPException(status_code=500, detail="Failed to upload original image.")

    print(
        "[/api/vibode/user-skus/ingest] original upload complete",
        {
            "skuId": user_sku_id,
            "path": original_path,
            "bytes": len(original_png_bytes),
            "mimeType": "image/png",
        },
    )

    print("[/api/vibode/user-skus/ingest] background removal start", {"skuId": user_sku_id})
    try:
        bg_removed_bytes = _run_user_sku_background_removal(
            image_png_bytes=original_png_bytes,
            model_name=DEFAULT_MODEL_NAME,
        )
        if not bg_removed_bytes:
            raise RuntimeError("Background removal returned empty bytes.")
    except Exception as e:
        print("[/api/vibode/user-skus/ingest] background removal failed:", e)
        failed_reason = f"background removal failed: {str(e)[:160]}"
        failed_user_sku = VibodeUserSku(
            skuId=user_sku_id,
            label=resolved_label,
            variants=[],
            sourceUrl=source_url,
            status="failed",
            reason=failed_reason,
        )
        print(
            "[/api/vibode/user-skus/ingest] final status",
            {"skuId": user_sku_id, "status": "failed", "reason": failed_reason},
        )
        return VibodeUserSkuIngestResponse(userSku=failed_user_sku)

    print(
        "[/api/vibode/user-skus/ingest] background removal complete",
        {"skuId": user_sku_id, "bytes": len(bg_removed_bytes)},
    )

    normalization_mode = "transparent"
    try:
        has_alpha = _has_transparency(bg_removed_bytes)
        if has_alpha:
            normalized_png_bytes, normalized_dims = _normalize_user_sku_transparent_png(
                bg_removed_bytes,
                max_dimension=USER_SKU_NORMALIZED_MAX_DIM,
                padding_ratio=USER_SKU_NORMALIZED_PADDING_RATIO,
            )
            normalization_mode = "transparent"
        else:
            normalized_png_bytes, normalized_dims = _normalize_user_sku_solid_bg_png(
                bg_removed_bytes,
                max_dimension=USER_SKU_NORMALIZED_MAX_DIM,
                padding_ratio=USER_SKU_NORMALIZED_PADDING_RATIO,
            )
            normalization_mode = "solid_bg"
    except Exception as e:
        print("[/api/vibode/user-skus/ingest] normalization failed:", e)
        raise HTTPException(status_code=500, detail="Failed to normalize user SKU image.")

    print(
        "[/api/vibode/user-skus/ingest] normalized dimensions",
        {
            "skuId": user_sku_id,
            "width": normalized_dims[0],
            "height": normalized_dims[1],
            "bytes": len(normalized_png_bytes),
        },
    )
    print(
        "[/api/vibode/user-skus/ingest] normalization mode",
        {"skuId": user_sku_id, "normalizationMode": normalization_mode},
    )

    normalized_path = f"user-skus/{user_sku_id}/normalized.png"
    try:
        _supabase_storage_upload_bytes(
            object_path=normalized_path,
            payload=normalized_png_bytes,
            mime_type="image/png",
        )
        normalized_url = _supabase_storage_create_signed_url(normalized_path)
    except Exception as e:
        print("[/api/vibode/user-skus/ingest] normalized upload/sign failed:", e)
        raise HTTPException(status_code=500, detail="Failed to upload normalized user SKU image.")

    print(
        "[/api/vibode/user-skus/ingest] final status",
        {
            "skuId": user_sku_id,
            "status": "ready",
            "normalizedPath": normalized_path,
        },
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

    model_name = resolve_model_name(req.modelVersion)

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
        room_png_bytes, applied_ratio = normalize_image_bytes_for_ratio(
            room_raw_bytes,
            requested_ratio=req.aspectRatio,
            model_name=model_name,
        )
        aspect_ratio_to_send = applied_ratio
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
        debug_ratio = applied_ratio or "auto"

    return VibodeComposeResponse(imageUrl=data_url, appliedAspectRatio=debug_ratio)

@app.post("/vibode/swap", response_model=VibodeSwapResponse)
async def vibode_swap(req: VibodeSwapRequest):
    _reject_if_vibode_strict_missing("/vibode/swap", _collect_vibode_swap_missing_fields(req))

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

    model_name = resolve_model_name(req.modelVersion)
    marks_ordered = list(req.marks)  # Preserve request order: marker index -> replacement index.
    mapped_replacements = req.replacementAssets[: len(marks_ordered)]

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
    try:
        room_png_bytes, applied_ratio = normalize_image_bytes_for_ratio(
            room_raw_bytes,
            requested_ratio="auto",
            model_name=model_name,
        )
        aspect_ratio_to_send = applied_ratio
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

    model_name = resolve_model_name(req.modelVersion)

    try:
        room_orig_img = _safe_open_image(room_raw_bytes)
        orig_w, orig_h = room_orig_img.size
    except Exception as e:
        print("[/vibode/rotate] Error decoding room image dimensions:", e)
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
        debug_ratio = applied_ratio or "auto"

    return VibodeComposeResponse(imageUrl=data_url, appliedAspectRatio=debug_ratio)

@app.post("/vibode/move", response_model=VibodeComposeResponse)
async def vibode_move(req: VibodeMoveRequest):
    marks_ordered = _parse_and_clamp_move_marks(req.marks)

    image_source_kind = "imageBase64"
    if req.imageBase64 and req.imageBase64.strip():
        try:
            room_raw_bytes = _decode_base64_image(req.imageBase64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 room image data")
    elif req.imageUrl and req.imageUrl.strip():
        image_source_kind = "imageUrl"
        try:
            room_raw_bytes = _fetch_image_bytes_from_url(req.imageUrl)
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to fetch imageUrl image data")
    else:
        raise HTTPException(status_code=400, detail="Provide either imageBase64 or imageUrl.")

    model_name = resolve_model_name(req.modelVersion)

    try:
        room_orig_img = _safe_open_image(room_raw_bytes)
        orig_w, orig_h = room_orig_img.size
    except Exception as e:
        print("[/vibode/move] Error decoding room image dimensions:", e)
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
        print("[/vibode/move] Error preparing room image:", e)
        raise HTTPException(status_code=400, detail="Could not process room image")

    try:
        room_pre_overlay_img = _safe_open_image(room_png_bytes)
        new_w, new_h = room_pre_overlay_img.size
    except Exception as e:
        print("[/vibode/move] Error decoding prepared room image dimensions:", e)
        raise HTTPException(status_code=500, detail="Failed to prepare room overlay image")

    try:
        overlay_img = render_vibode_move_overlay(room_pre_overlay_img, marks_ordered)
        room_overlay_png_bytes = image_to_png_bytes(overlay_img)
    except Exception as e:
        print("[/vibode/move] Error drawing move markers:", e)
        raise HTTPException(status_code=500, detail="Failed to draw move markers")

    move_prompt = build_vibode_move_prompt(marks_ordered)
    move_prompt_hash = _short_prompt_hash(move_prompt)
    move_prompt_first_line = _prompt_first_line(move_prompt)

    print(
        "[/vibode/move] Received request:",
        {
            "marks": len(marks_ordered),
            "marks_summary": _summarize_move_marks(marks_ordered),
            "sourceImage": image_source_kind,
            "imageUrlProvided": bool(req.imageUrl and req.imageUrl.strip()),
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
        "[/vibode/move] Prompt summary:",
        {
            "prompt_hash": move_prompt_hash,
            "prompt_first_line": move_prompt_first_line,
            "marks_summary": _summarize_move_marks(marks_ordered),
        },
    )
    if VIBODE_LOG_PROMPTS:
        print("\n===== VIBODE MOVE PROMPT SENT TO GEMINI =====\n")
        print(move_prompt)
        print("\n==============================================\n")

    try:
        out_bytes = call_gemini_multimodal(
            prompt=move_prompt,
            room_png_bytes=room_png_bytes,
            room_overlay_png_bytes=room_overlay_png_bytes,
            sku_png_bytes_list=[],
            model_name=model_name,
            aspect_ratio=aspect_ratio_to_send,
        )
    except Exception as e:
        log_event("vibode_move_processing_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Error during move")

    if not out_bytes:
        raise HTTPException(status_code=500, detail="Move returned empty image")

    move_dump_seed = (
        move_prompt_hash
        + "|"
        + "|".join(mark.id for mark in marks_ordered)
        + f"|{len(room_png_bytes)}|{len(room_overlay_png_bytes)}"
    )
    maybe_dump_vibode_move_images(
        room_clean_png_bytes=room_png_bytes,
        room_move_overlay_png_bytes=room_overlay_png_bytes,
        output_png_bytes=out_bytes,
        stable_seed=move_dump_seed,
    )

    data_url = make_data_url(out_bytes, mime_type="image/png")

    debug_ratio: Optional[str] = None
    if DEBUG_ROOMPRINTZ_RATIO:
        debug_ratio = applied_ratio or "auto"

    return VibodeComposeResponse(imageUrl=data_url, appliedAspectRatio=debug_ratio)


# Quick test:
# curl -sS -X POST "http://localhost:8000/api/vibode/user-skus/ingest" -H "Content-Type: application/json" -d '{"label":"Demo SKU","imageUrl":"https://example.com/product.png"}'
