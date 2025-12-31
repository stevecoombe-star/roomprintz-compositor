import base64
import json
from pathlib import Path

HERE = Path(__file__).parent

IMAGE_PATH = HERE / "room_test.jpg"
CALIBRATION_PATH = HERE / "calibration_payload.json"
OUT_PATH = HERE / "stage_room_payload.json"

def main():
    # --- load image ---
    image_bytes = IMAGE_PATH.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # --- load calibration ---
    calibration = json.loads(CALIBRATION_PATH.read_text())

    payload = {
        "imageBase64": image_b64,
        "styleId": "modern-luxury",
        "enhancePhoto": False,
        "cleanupRoom": False,
        "repairDamage": False,
        "emptyRoom": False,
        "renovateRoom": False,
        "repaintWalls": False,
        "flooringPreset": None,
        "roomType": "living-room",
        "modelVersion": "gemini-3",
        "aspectRatio": "auto",
        "isContinuation": False,
        "calibration": calibration,
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"✅ Wrote payload to {OUT_PATH}")

if __name__ == "__main__":
    main()
