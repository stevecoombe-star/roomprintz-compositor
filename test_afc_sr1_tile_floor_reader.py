import copy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from research.afc_sr1_tile_floor_reader import (
    POLICY_VERSION,
    V2_POLICY_VERSION,
    _analysis_dimensions,
    _analysis_identity,
    _map_analysis_line_to_input,
    _normalize_line,
    read_floor_vanishing_line,
)


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

    def test_v2_dimension_rule_preserves_aspect_and_fixes_long_edge(self):
        self.assertEqual(_analysis_dimensions(1264, 848), (1264, 848))
        self.assertEqual(_analysis_dimensions(2528, 1696), (1264, 848))
        self.assertEqual(_analysis_dimensions(1600, 1201), (1264, 949))
        self.assertEqual(_analysis_dimensions(1000, 2000), (632, 1264))
        with self.assertRaisesRegex(ValueError, "below_reference_analysis_long_edge"):
            _analysis_dimensions(1263, 848)

    def test_v2_dual_map_handles_identity_anisotropy_and_infinite_line_geometry(self):
        line = _normalize_line(np.asarray([3.0, -4.0, 12.0]))
        self.assertIsNotNone(line)
        assert line is not None
        self.assertTrue(np.allclose(_map_analysis_line_to_input(line, 1.0, 1.0), line))
        anisotropic = _map_analysis_line_to_input(line, 2.0, 3.0)
        expected = _normalize_line(np.asarray([line[0] / 2.0, line[1] / 3.0, line[2]]))
        self.assertTrue(np.allclose(anisotropic, expected))
        horizontal = _map_analysis_line_to_input(np.asarray([0.0, 1.0, -10.0]), 2.0, 3.0)
        vertical = _map_analysis_line_to_input(np.asarray([1.0, 0.0, -10.0]), 2.0, 3.0)
        self.assertTrue(np.allclose(horizontal, np.asarray([0.0, 1.0, -30.0])))
        self.assertTrue(np.allclose(vertical, np.asarray([1.0, 0.0, -20.0])))

    def test_v2_pixel_identity_hashes_contiguous_bgr_pixels(self):
        image = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        identity = _analysis_identity(image[:, ::2], "identity", 2, 2)
        expected = np.ascontiguousarray(image[:, ::2], dtype=np.uint8).tobytes(order="C")
        self.assertEqual(identity["pixelBufferSha256"], __import__("hashlib").sha256(expected).hexdigest())
        self.assertNotEqual(identity["pixelBufferSha256"], __import__("hashlib").sha256(image.tobytes()).hexdigest())

    def test_v2_below_reference_rejects_before_lsd(self):
        image = encoded_png(1000, 800)
        with patch("research.afc_sr1_tile_floor_reader.cv2.createLineSegmentDetector") as lsd:
            result = read_floor_vanishing_line(image, FULL_IMAGE_ROI, V2_POLICY_VERSION)
        self.assertEqual((result["status"], result["reason"]), ("rejected", "below_reference_analysis_long_edge"))
        lsd.assert_not_called()

    def test_v2_source_raster_limit_rejects_before_lsd(self):
        image = encoded_png(8193, 1)
        with patch("research.afc_sr1_tile_floor_reader.cv2.createLineSegmentDetector") as lsd:
            result = read_floor_vanishing_line(image, FULL_IMAGE_ROI, V2_POLICY_VERSION)
        self.assertEqual((result["status"], result["reason"]), ("rejected", "source_raster_too_large"))
        lsd.assert_not_called()

    def test_v2_identity_and_area_downscale_bind_analysis_raster(self):
        identity = read_floor_vanishing_line(encoded_png(1264, 848), FULL_IMAGE_ROI, V2_POLICY_VERSION)
        downscaled = read_floor_vanishing_line(encoded_png(2528, 1696), FULL_IMAGE_ROI, V2_POLICY_VERSION)
        for result, mode, resampler in (
            (identity, "identity", "identity"),
            (downscaled, "downscale_long_edge", "opencv-inter-area/v1"),
        ):
            self.assertEqual((result["status"], result["reason"]), ("rejected", "insufficient_segments"))
            identity_data = result["diagnostics"]["analysisIdentity"]
            self.assertEqual((identity_data["analysisWidth"], identity_data["analysisHeight"]), (1264, 848))
            self.assertEqual((identity_data["mode"], identity_data["resampler"]), (mode, resampler))
            self.assertRegex(identity_data["pixelBufferSha256"], r"^[0-9a-f]{64}$")

    def test_reader_contains_no_downstream_afc_geometry(self):
        with open("research/afc_sr1_tile_floor_reader.py", encoding="utf-8") as source_file:
            source = source_file.read().lower()
        for forbidden in ("truncatedanchor", "oppositeendpoint", "seamt", "vp_width", "tile_grid_scaffold", "gemini"):
            self.assertNotIn(forbidden, source, forbidden)


if __name__ == "__main__":
    unittest.main()
