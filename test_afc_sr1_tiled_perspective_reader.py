import ast
import hashlib
from pathlib import Path
import unittest

import cv2
import numpy as np

from research.afc_sr1_tiled_perspective_reader import (
    _BORDER_MARGIN_FRACTION,
    _MIN_BORDER_MARGIN_PX,
    _result_from_ordered_pixel_corners,
    read_tiled_perspective_single_tile,
)


C_T1_SHA256 = "93c4c40764c863246cd58976c5bc242689f6dfb0e71cb952248872daac76da92"
C_T1_BYTE_COUNT = 1_178_891
C_T1_DIMENSIONS = (1264, 848)
C_T1_FIXTURE = (
    Path(__file__).resolve().parent / "research/fixtures" / f"c-t1.{C_T1_SHA256}.png"
)


def load_frozen_c_t1() -> bytes:
    """Read the compositor-local, SHA-bound immutable C-T1 fixture."""
    if not C_T1_FIXTURE.is_file():
        raise AssertionError(f"missing vendored C-T1 bytes: {C_T1_FIXTURE}")
    encoded = C_T1_FIXTURE.read_bytes()
    assert len(encoded) == C_T1_BYTE_COUNT
    assert hashlib.sha256(encoded).hexdigest() == C_T1_SHA256
    return encoded


def map_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    mapped = (homography @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def assert_vp_incident(
    testcase: unittest.TestCase,
    vp: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> None:
    line = np.cross(np.r_[first, 1.0], np.r_[second, 1.0])
    residual = abs(float(line @ vp)) / np.linalg.norm(line[:2])
    testcase.assertLessEqual(residual, 1e-6)


class AfcSr1TiledPerspectiveReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.encoded = load_frozen_c_t1()

    def test_c_t1_fixture_bytes_load_unchanged_with_verified_dimensions(self):
        decoded = cv2.imdecode(np.frombuffer(self.encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertEqual((decoded.shape[1], decoded.shape[0]), C_T1_DIMENSIONS)

    def test_c_t1_detects_one_complete_cell_with_deterministic_order(self):
        first = read_tiled_perspective_single_tile(self.encoded)
        second = read_tiled_perspective_single_tile(self.encoded)
        self.assertEqual(first, second)
        self.assertGreater(first.candidate_count, 0)
        self.assertGreater(first.accepted_complete_tile_count, 0)

        corners = np.asarray(first.selected_tile.pixel_corners, dtype=np.float64)
        self.assertEqual(corners.shape, (4, 2))
        self.assertTrue(np.all(np.isfinite(corners)))
        self.assertEqual(
            {tuple(point) for point in np.rint(corners).astype(int)},
            {(648, 681), (724, 643), (843, 687), (775, 743)},
        )
        cross_products = [
            (
                (corners[(index + 1) % 4, 0] - corners[index, 0])
                * (corners[(index + 2) % 4, 1] - corners[(index + 1) % 4, 1])
                - (corners[(index + 1) % 4, 1] - corners[index, 1])
                * (corners[(index + 2) % 4, 0] - corners[(index + 1) % 4, 0])
            )
            for index in range(4)
        ]
        self.assertTrue(all(value > 0 for value in cross_products) or all(value < 0 for value in cross_products))
        self.assertGreater(
            abs(
                0.5
                * (
                    np.dot(corners[:, 0], np.roll(corners[:, 1], -1))
                    - np.dot(corners[:, 1], np.roll(corners[:, 0], -1))
                )
            ),
            0.0,
        )
        border_margin = max(
            _MIN_BORDER_MARGIN_PX,
            min(first.decoded_width, first.decoded_height) * _BORDER_MARGIN_FRACTION,
        )
        self.assertTrue(np.all(corners[:, 0] > border_margin))
        self.assertTrue(np.all(corners[:, 0] < first.decoded_width - border_margin))
        self.assertTrue(np.all(corners[:, 1] > border_margin))
        self.assertTrue(np.all(corners[:, 1] < first.decoded_height - border_margin))

        # Public order is NL -> NR -> FR -> FL.  Thus 0/1 are the complete
        # near edge and 2/3 are the complete far edge.
        self.assertGreater(min(corners[0, 1], corners[1, 1]), max(corners[2, 1], corners[3, 1]))
        self.assertLess(corners[0, 0], corners[1, 0])
        self.assertLess(corners[3, 0], corners[2, 0])

    def test_homography_reprojects_the_selected_cell_and_authoritative_quad(self):
        result = read_tiled_perspective_single_tile(self.encoded)
        homography = np.asarray(result.h_lattice_to_tiled, dtype=np.float64)
        unit_square = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
        reprojected = map_points(homography, unit_square)
        selected = np.asarray(result.selected_tile.pixel_corners, dtype=np.float64)
        authoritative = np.asarray(result.authoritative_quad_pixel, dtype=np.float64)
        np.testing.assert_allclose(reprojected, selected, rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(authoritative, selected, rtol=0.0, atol=1e-5)

    def test_homogeneous_vps_are_finite_and_incident_to_their_opposite_edges(self):
        result = read_tiled_perspective_single_tile(self.encoded)
        corners = np.asarray(result.authoritative_quad_pixel, dtype=np.float64)
        vp_a = np.asarray(result.vp_a_homogeneous, dtype=np.float64)
        vp_b = np.asarray(result.vp_b_homogeneous, dtype=np.float64)
        self.assertEqual(vp_a.shape, (3,))
        self.assertEqual(vp_b.shape, (3,))
        self.assertTrue(np.all(np.isfinite(vp_a)))
        self.assertTrue(np.all(np.isfinite(vp_b)))
        assert_vp_incident(self, vp_a, corners[0], corners[1])
        assert_vp_incident(self, vp_a, corners[3], corners[2])
        assert_vp_incident(self, vp_b, corners[1], corners[2])
        assert_vp_incident(self, vp_b, corners[0], corners[3])

    def test_affine_tile_preserves_valid_vanishing_directions_at_infinity(self):
        result = _result_from_ordered_pixel_corners(
            np.asarray(((10.0, 30.0), (50.0, 30.0), (50.0, 10.0), (10.0, 10.0))),
            width=100,
            height=50,
            candidate_count=1,
            accepted_complete_tile_count=1,
        )
        self.assertEqual(result.vp_a_homogeneous[2], 0.0)
        self.assertEqual(result.vp_b_homogeneous[2], 0.0)

    def test_non_tile_readers_are_not_imported(self):
        source_path = Path("research/afc_sr1_tiled_perspective_reader.py")
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_modules = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        ]
        for forbidden in (
            "research.afc_sr1_tile_floor_reader",
            "research.afc_sr1_tile_floor_reader_v3",
            "research.afc_sr1_tr2_tile_floor_reader_http",
            "research.afc_sr1_ts0_child_projective_placement",
            "main",
        ):
            self.assertNotIn(forbidden, imported_modules)


if __name__ == "__main__":
    unittest.main()
