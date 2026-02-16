import os
import io
import base64
import hashlib
import math
from datetime import datetime
from typing import Literal, Optional, Tuple, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
DEBUG_ROOMPRINTZ_PROMPT = os.getenv("DEBUG_ROOMPRINTZ_PROMPT", "1") == "1"

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

RATIO_MAP: Dict[str, float] = {
    "4:3": 4 / 3,
    "3:2": 3 / 2,
    "16:9": 16 / 9,
    "1:1": 1.0,
}

SUPPORTED_RATIOS_ORDERED = ["4:3", "3:2", "16:9", "1:1"]


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

    if DEBUG_ROOMPRINTZ_PROMPT:
        print("\n===== ROOMPRINTZ PROMPT SENT TO NANO BANANA =====\n")
        print(final_prompt)
        print("\n=================================================\n")

    return final_prompt


# ---------- FASTAPI APP ----------

app = FastAPI()


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
    try:
        if DEBUG_ROOMPRINTZ_PROMPT:
            print(
                "[call_gemini_with_prompt] Calling model:",
                model_name,
                "| Input PNG bytes:",
                len(image_png_bytes),
                "| aspect_ratio:",
                aspect_ratio if aspect_ratio else "(omitted)",
            )

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
            print("[call_gemini_with_prompt] Failed to extract image bytes:", e)
            raise RuntimeError("Could not extract generated image from Gemini response")

        if not out_bytes:
            raise RuntimeError("Gemini returned empty image bytes")

        return out_bytes

    except Exception as e:
        print("[call_gemini_with_prompt] Error calling Gemini:", e)
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


def prepare_sku_png_bytes(image_bytes: bytes) -> bytes:
    img = _safe_open_image(image_bytes)
    img = resize_down_if_needed(img, MAX_INPUT_LONG_EDGE_INT)
    return image_to_png_bytes(img)


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


def call_gemini_multimodal(
    prompt: str,
    room_png_bytes: bytes,
    room_overlay_png_bytes: bytes,
    sku_png_bytes_list: List[bytes],
    model_name: str,
    aspect_ratio: Optional[str] = None,
) -> bytes:
    try:
        if DEBUG_ROOMPRINTZ_PROMPT:
            print(
                "[call_gemini_multimodal] Calling model:",
                model_name,
                "| Room clean bytes:",
                len(room_png_bytes),
                "| Room overlay bytes:",
                len(room_overlay_png_bytes),
                "| sku_count:",
                len(sku_png_bytes_list),
                "| aspect_ratio:",
                aspect_ratio if aspect_ratio else "(omitted)",
            )
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
            print("[call_gemini_multimodal] Failed to extract image bytes:", e)
            raise RuntimeError("Could not extract generated image from Gemini response")
        if not out_bytes:
            raise RuntimeError("Gemini returned empty image bytes")
        return out_bytes
    except Exception as e:
        print("[call_gemini_multimodal] Error calling Gemini:", e)
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
        print("[/stage-room] Error in processing:", e)
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
        print("[/vibode/compose] Error in processing:", e)
        raise HTTPException(status_code=500, detail="Error during compose")

    if not out_bytes:
        raise HTTPException(status_code=500, detail="Compose returned empty image")

    data_url = make_data_url(out_bytes, mime_type="image/png")

    debug_ratio: Optional[str] = None
    if DEBUG_ROOMPRINTZ_RATIO:
        debug_ratio = applied_ratio or "auto"

    return VibodeComposeResponse(imageUrl=data_url, appliedAspectRatio=debug_ratio)


@app.post("/vibode/remove", response_model=VibodeComposeResponse)
async def vibode_remove(req: VibodeRemoveRequest):
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
        print("[/vibode/remove] Error in processing:", e)
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
        print("[/vibode/swap] Error in processing:", e)
        raise HTTPException(status_code=500, detail="Error during swap")

    if not out_bytes:
        raise HTTPException(status_code=500, detail="Swap returned empty image")

    data_url = make_data_url(out_bytes, mime_type="image/png")
    return VibodeSwapResponse(imageUrl=data_url)
