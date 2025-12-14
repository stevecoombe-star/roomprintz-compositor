import os
import io
import base64
from typing import Literal, Optional, Tuple, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

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

DEBUG_ROOMPRINTZ_RATIO = os.getenv("DEBUG_ROOMPRINTZ_RATIO", "0") == "1"

# Optional: cap input size to keep cost + latency down (resize down only; never upscale)
# Set to "" to disable.
MAX_INPUT_LONG_EDGE = os.getenv("ROOMPRINTZ_MAX_INPUT_LONG_EDGE", "2048").strip()
MAX_INPUT_LONG_EDGE_INT = int(MAX_INPUT_LONG_EDGE) if MAX_INPUT_LONG_EDGE.isdigit() else None

# Gemini 2.5 Flash historically defaults to 1:1. By default, we enforce 1:1 for that model.
# Set ROOMPRINTZ_ALLOW_FLASH_NON_SQUARE=1 if you want to allow other ratios (not recommended).
ALLOW_FLASH_NON_SQUARE = os.getenv("ROOMPRINTZ_ALLOW_FLASH_NON_SQUARE", "0") == "1"


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

    # Fallback: assume caller passed a valid full model name
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
        return "4:3"  # safe default
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

    # Already close enough (tiny floating error tolerance)
    if abs(current_ratio - target_ratio) < 1e-6:
        return img

    if current_ratio > target_ratio:
        # Too wide -> crop width
        new_w = int(round(h * target_ratio))
        new_w = min(new_w, w)
        left = (w - new_w) // 2
        right = left + new_w
        top = 0
        bottom = h
        return img.crop((left, top, right, bottom))

    # Too tall -> crop height (upward bias)
    new_h = int(round(w / target_ratio))
    new_h = min(new_h, h)

    # Upward bias: keep more ceiling by cropping more from the bottom.
    # bias=0.65 means the crop window starts a bit higher than center.
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

    Behavior:
    - requested_ratio None/"auto" => choose closest preset from uploaded image ratio.
    - then Fill/Crop to that ratio with upward bias.
    - optionally resize down to keep cost reasonable.
    - For gemini-2.5-flash-image, optionally enforce 1:1 unless ALLOW_FLASH_NON_SQUARE=1.
    """
    img = _safe_open_image(image_bytes)
    w, h = img.size

    # Determine target ratio
    ratio_choice = (requested_ratio or "auto").strip().lower()

    if ratio_choice == "auto":
        chosen = choose_closest_aspect_ratio(w, h)
    else:
        # normalize variants like "16x9"
        normalized = ratio_choice.replace("x", ":")
        if normalized not in RATIO_MAP:
            print(f"[normalize_image_bytes_for_ratio] Unknown aspectRatio '{requested_ratio}', defaulting to auto.")
            chosen = choose_closest_aspect_ratio(w, h)
        else:
            chosen = normalized

    # Guardrails for Gemini 2.5 Flash (optional)
    if (model_name.strip().lower() == "gemini-2.5-flash-image") and (chosen != "1:1") and (not ALLOW_FLASH_NON_SQUARE):
        print("[normalize_image_bytes_for_ratio] Forcing aspect ratio to 1:1 for gemini-2.5-flash-image (reliability).")
        chosen = "1:1"

    # Fill/Crop
    target_ratio = RATIO_MAP[chosen]
    cropped = crop_to_aspect_ratio_fill(img, target_ratio)

    # Resize down (optional) AFTER crop
    cropped = resize_down_if_needed(cropped, MAX_INPUT_LONG_EDGE_INT)

    return image_to_png_bytes(cropped), chosen


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
    imageBase64: str  # base64-encoded JPEG/PNG
    styleId: Optional[str] = None

    # Phase 1 tools
    enhancePhoto: bool = False
    cleanupRoom: bool = False
    repairDamage: bool = False
    emptyRoom: bool = False
    renovateRoom: bool = False

    # Phase 2 surfaces
    repaintWalls: bool = False
    flooringPreset: Optional[str] = None

    # Room type
    roomType: Optional[str] = None

    # Model selector
    modelVersion: Optional[str] = None

    # NEW: aspect ratio control (default: "auto")
    aspectRatio: Optional[AspectRatio] = "auto"


class StageRoomResponse(BaseModel):
    imageUrl: str
    appliedAspectRatio: Optional[str] = None  # DEBUG ONLY


StyleId = Literal[
    "modern-luxury",
    "japandi",
    "scandinavian",
    "coastal",
    "urban-loft",
    "farmhouse",
]


def call_gemini_with_prompt(
    image_png_bytes: bytes,
    prompt: str,
    model_name: str,
    aspect_ratio: str,
) -> bytes:
    """
    Core helper that calls Gemini / Nano Banana with a prompt + a single image.
    Returns raw bytes of the generated image (PNG).
    """
    try:
        if DEBUG_ROOMPRINTZ_PROMPT:
            print(
                "[call_gemini_with_prompt] Calling model:",
                model_name,
                "| Input PNG bytes:",
                len(image_png_bytes),
                "| aspect_ratio:",
                aspect_ratio,
            )

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
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                ),
            ),
        )

        try:
            candidate = response.candidates[0]
            part = candidate.content.parts[0]
            out_bytes = part.inline_data.data
        except Exception as e:
            print("[call_gemini_with_prompt] Failed to extract image bytes:", e)
            raise RuntimeError(
                "Could not extract generated image from Nano Banana / Gemini response"
            )

        if not out_bytes:
            raise RuntimeError("Nano Banana / Gemini returned empty image bytes")

        return out_bytes

    except Exception as e:
        print("[call_gemini_with_prompt] Error calling Nano Banana / Gemini:", e)
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
    aspect_ratio: str,
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
    aspect_ratio: str,
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
        raise HTTPException(
            status_code=400,
            detail="No photo tools or styleId provided; nothing to do.",
        )

    # Decide model
    model_name = resolve_model_name(req.modelVersion)

    # 1) Decode base64 input
    try:
        raw_bytes = base64.b64decode(req.imageBase64)
    except Exception as e:
        print("[/stage-room] Failed to decode base64:", e)
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    # 2) Normalize to chosen aspect ratio (Fill/Crop) AND return applied ratio
    try:
        normalized_png_bytes, applied_ratio = normalize_image_bytes_for_ratio(
            raw_bytes,
            requested_ratio=req.aspectRatio,
            model_name=model_name,
        )
    except Exception as e:
        print("[/stage-room] Error normalizing image:", e)
        raise HTTPException(status_code=400, detail="Could not process uploaded image")

    print(
        "[/stage-room] Received request:",
        {
            "styleId": req.styleId,
            "raw_bytes_len": len(raw_bytes),
            "normalized_png_len": len(normalized_png_bytes),
            "enhancePhoto": req.enhancePhoto,
            "cleanupRoom": req.cleanupRoom,
            "repairDamage": req.repairDamage,
            "emptyRoom": req.emptyRoom,
            "renovateRoom": req.renovateRoom,
            "repaintWalls": req.repaintWalls,
            "flooringPreset": req.flooringPreset,
            "roomType": req.roomType,
            "modelVersion": req.modelVersion,
            "modelName": model_name,
            "requestedAspectRatio": req.aspectRatio,
            "appliedAspectRatio": applied_ratio,
            "allowFlashNonSquare": ALLOW_FLASH_NON_SQUARE,
            "maxInputLongEdge": MAX_INPUT_LONG_EDGE_INT,
        },
    )

    # 3) Run processing
    try:
        if wants_staging:
            out_bytes = run_fusion(
                image_png_bytes=normalized_png_bytes,
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
                aspect_ratio=applied_ratio,
            )
        else:
            out_bytes = run_photo_tools(
                image_png_bytes=normalized_png_bytes,
                enhance_photo=req.enhancePhoto,
                cleanup_room=req.cleanupRoom,
                repair_damage=req.repairDamage,
                empty_room=req.emptyRoom,
                renovate_room=req.renovateRoom,
                repaint_walls=req.repaintWalls,
                flooring_preset=req.flooringPreset,
                room_type=req.roomType,
                model_name=model_name,
                aspect_ratio=applied_ratio,
            )
    except Exception as e:
        print("[/stage-room] Error in processing:", e)
        raise HTTPException(status_code=500, detail="Error during fusion")

    if not out_bytes:
        raise HTTPException(status_code=500, detail="Fusion returned empty image")

    # 4) Return as data URL
    data_url = make_data_url(out_bytes, mime_type="image/png")
    return StageRoomResponse(
        imageUrl=data_url,
        appliedAspectRatio=applied_ratio if DEBUG_ROOMPRINTZ_RATIO else None,
    )
