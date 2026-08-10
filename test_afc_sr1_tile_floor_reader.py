import copy
import unittest

import cv2
import numpy as np

from research.afc_sr1_tile_floor_reader import POLICY_VERSION, read_floor_vanishing_line


FULL_IMAGE_ROI = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def encoded_png(width: int, height: int, value: int = 0) -> bytes:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise AssertionError("failed to create synthetic PNG")
    return bytes(buffer)


class AfcSr1TileFloorReaderTests(unittest.TestCase):
    def test_malformed_bytes_fail_closed(self):
        result = read_floor_vanishing_line(b"not an encoded image", FULL_IMAGE_ROI)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "invalid_input_image")

    def test_wrong_reference_grid_fails_closed_without_resizing(self):
        result = read_floor_vanishing_line(encoded_png(640, 480), FULL_IMAGE_ROI)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "unsupported_analysis_grid")

    def test_invalid_and_eroded_away_rois_fail_closed(self):
        image = encoded_png(1264, 848)
        invalid = read_floor_vanishing_line(image, ((0.0, 0.0), (float("nan"), 0.0), (0.0, 1.0)))
        tiny = read_floor_vanishing_line(
            image,
            ((0.5, 0.5), (0.501, 0.5), (0.5, 0.501)),
        )
        self.assertEqual((invalid["status"], invalid["reason"]), ("rejected", "invalid_roi"))
        self.assertEqual((tiny["status"], tiny["reason"]), ("rejected", "impossible_eroded_roi"))

    def test_blank_reference_image_rejects_and_replays_identically(self):
        image = encoded_png(1264, 848)
        first = read_floor_vanishing_line(image, FULL_IMAGE_ROI, POLICY_VERSION)
        second = read_floor_vanishing_line(image, FULL_IMAGE_ROI, POLICY_VERSION)
        self.assertEqual(first["status"], "rejected")
        self.assertEqual(first["reason"], "insufficient_segments")
        self.assertEqual(first, second)
        self.assertEqual(copy.deepcopy(FULL_IMAGE_ROI), FULL_IMAGE_ROI)

    def test_unsupported_policy_rejects_before_decode(self):
        result = read_floor_vanishing_line(b"", FULL_IMAGE_ROI, "future-policy/v2")
        self.assertEqual((result["status"], result["reason"]), ("rejected", "unsupported_policy_version"))

    def test_reader_contains_no_downstream_afc_geometry(self):
        with open("research/afc_sr1_tile_floor_reader.py", encoding="utf-8") as source_file:
            source = source_file.read().lower()
        for forbidden in ("truncatedanchor", "oppositeendpoint", "seamt", "vp_width", "tile_grid_scaffold", "gemini"):
            self.assertNotIn(forbidden, source, forbidden)


if __name__ == "__main__":
    unittest.main()
