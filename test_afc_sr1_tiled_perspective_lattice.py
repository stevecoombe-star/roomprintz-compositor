import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from research.afc_sr1_tiled_perspective_reader import (
    _normalize_component,
    _semantic_core_quad,
    TiledPerspectiveLatticeFailure,
    build_tiled_perspective_lattice,
    read_tiled_perspective_lattice,
    read_tiled_perspective_single_tile,
)


C_T1_FIXTURE = (
    Path(__file__).resolve().parent
    / "research/fixtures"
    / "c-t1.93c4c40764c863246cd58976c5bc242689f6dfb0e71cb952248872daac76da92.png"
)
C_T2_FIXTURE = (
    Path(__file__).resolve().parent
    / "research/fixtures"
    / "c-t2.66aa1803fa3cb45516ae639d0bce22eec0afa4fd89ca92e64d412901ce930056.png"
)
C_T3_FIXTURE = (
    Path(__file__).resolve().parent
    / "research/fixtures"
    / "c-t3.c81e678c1e908d901fea256f56983fcd253e6bb3dbc421c5a492aad5d056d053.png"
)
ROOM_C_FIXTURES = (
    (
        C_T1_FIXTURE,
        "93c4c40764c863246cd58976c5bc242689f6dfb0e71cb952248872daac76da92",
        1_178_891,
    ),
    (
        C_T2_FIXTURE,
        "66aa1803fa3cb45516ae639d0bce22eec0afa4fd89ca92e64d412901ce930056",
        1_181_262,
    ),
    (
        C_T3_FIXTURE,
        "c81e678c1e908d901fea256f56983fcd253e6bb3dbc421c5a492aad5d056d053",
        1_209_805,
    ),
)
IMAGE_SIZE = (1200, 900)
PERSPECTIVE_H = np.asarray(
    (
        (100.0, 18.0, 180.0),
        (12.0, 82.0, 160.0),
        (0.025, 0.012, 1.0),
    )
)


def project(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    mapped = (homography @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def lattice_quads(
    coordinates: tuple[tuple[int, int], ...],
    homography: np.ndarray = PERSPECTIVE_H,
) -> list[np.ndarray]:
    return [
        project(
            homography,
            np.asarray(
                (
                    (float(j), float(i)),
                    (float(j + 1), float(i)),
                    (float(j + 1), float(i + 1)),
                    (float(j), float(i + 1)),
                )
            ),
        )
        for j, i in coordinates
    ]


class AfcSr1TiledPerspectiveLatticeTests(unittest.TestCase):
    def test_room_c_fixtures_are_local_immutable_pngs(self):
        for fixture, expected_sha, expected_size in ROOM_C_FIXTURES:
            encoded = fixture.read_bytes()
            self.assertEqual((len(encoded), hashlib.sha256(encoded).hexdigest()), (expected_size, expected_sha))
            decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(decoded)
            assert decoded is not None
            self.assertEqual((decoded.shape[1], decoded.shape[0]), (1264, 848))

    def test_morphological_close_is_used_only_by_s1_enumeration(self):
        with patch(
            "research.afc_sr1_tiled_perspective_reader.cv2.morphologyEx",
            wraps=cv2.morphologyEx,
        ) as close:
            read_tiled_perspective_single_tile(C_T1_FIXTURE.read_bytes())
            close.assert_not_called()
            read_tiled_perspective_lattice(C_T1_FIXTURE.read_bytes())
            close.assert_called_once()
            self.assertEqual(close.call_args.args[1], cv2.MORPH_CLOSE)

    def test_empty_observation_set_uses_s1_failure_contract(self):
        with self.assertRaises(TiledPerspectiveLatticeFailure) as failure:
            build_tiled_perspective_lattice(
                width=IMAGE_SIZE[0],
                height=IMAGE_SIZE[1],
                quadrilaterals=(),
            )
        self.assertEqual(failure.exception.reason, "no_complete_tile")

    def test_deduplicates_near_identical_contours(self):
        tile = lattice_quads(((0, 0),))[0]
        duplicate = tile + np.asarray(((0.5, -0.5), (0.5, -0.5), (0.5, -0.5), (0.5, -0.5)))
        result = build_tiled_perspective_lattice(
            width=IMAGE_SIZE[0],
            height=IMAGE_SIZE[1],
            quadrilaterals=(tile, duplicate),
        )
        self.assertEqual((result.raw_quadrilateral_count, result.deduplicated_cell_count), (2, 1))
        self.assertEqual(result.selected_component.tile_count, 1)
        self.assertEqual((result.authoritative_core.rows, result.authoritative_core.columns), (1, 1))

    def test_shared_edge_normalizes_rotated_and_reversed_cell_orders(self):
        first, second = lattice_quads(((0, 0), (1, 0)))
        result = build_tiled_perspective_lattice(
            width=IMAGE_SIZE[0],
            height=IMAGE_SIZE[1],
            quadrilaterals=(first[[2, 1, 0, 3]], second[::-1]),
        )
        self.assertEqual(result.selected_component.tile_count, 2)
        self.assertEqual((result.authoritative_core.rows, result.authoritative_core.columns), (1, 2))
        selected = {
            cell.id: cell
            for cell in result.cells
            if cell.id in result.selected_component.cell_ids
        }
        self.assertEqual(len({cell.lattice_coordinate for cell in selected.values()}), 2)
        self.assertTrue(
            any(
                neighbor in selected
                for cell in selected.values()
                for neighbor in cell.neighbor_ids_by_local_edge
                if neighbor is not None
            )
        )

    def test_one_by_n_and_n_by_one_strips_are_valid(self):
        horizontal = build_tiled_perspective_lattice(
            width=IMAGE_SIZE[0],
            height=IMAGE_SIZE[1],
            quadrilaterals=lattice_quads(((0, 0), (1, 0), (2, 0))),
        )
        vertical = build_tiled_perspective_lattice(
            width=IMAGE_SIZE[0],
            height=IMAGE_SIZE[1],
            quadrilaterals=lattice_quads(((0, 0), (0, 1), (0, 2))),
        )
        self.assertEqual(
            (horizontal.authoritative_core.rows, horizontal.authoritative_core.columns),
            (1, 3),
        )
        self.assertEqual(
            (vertical.authoritative_core.rows, vertical.authoritative_core.columns),
            (3, 1),
        )

    def test_irregular_component_selects_largest_full_rectangular_core(self):
        result = build_tiled_perspective_lattice(
            width=IMAGE_SIZE[0],
            height=IMAGE_SIZE[1],
            quadrilaterals=lattice_quads(((0, 0), (1, 0), (2, 0), (0, 1), (1, 1))),
        )
        self.assertEqual(result.selected_component.tile_count, 5)
        self.assertEqual((result.authoritative_core.rows, result.authoritative_core.columns), (2, 2))
        self.assertEqual(len(result.authoritative_core.cell_ids), 4)

    def test_shared_overdetermined_homography_reprojects_all_cells(self):
        result = build_tiled_perspective_lattice(
            width=IMAGE_SIZE[0],
            height=IMAGE_SIZE[1],
            quadrilaterals=lattice_quads(((0, 0), (1, 0), (0, 1), (1, 1))),
        )
        self.assertEqual(result.selected_component.tile_count, 4)
        self.assertLess(result.reprojection_max_px, 1e-5)
        self.assertLess(result.reprojection_mean_px, 1e-5)

    def test_shared_homography_preserves_infinite_vanishing_directions(self):
        affine = np.asarray(((80.0, 0.0, 100.0), (0.0, 80.0, 100.0), (0.0, 0.0, 1.0)))
        result = build_tiled_perspective_lattice(
            width=IMAGE_SIZE[0],
            height=IMAGE_SIZE[1],
            quadrilaterals=lattice_quads(((0, 0), (1, 0), (2, 0)), affine),
        )
        self.assertEqual(result.vp_a_homogeneous[2], 0.0)
        self.assertEqual(result.vp_b_homogeneous[2], 0.0)

    def test_semantic_order_selects_i_boundaries_when_i_has_larger_depth_span(self):
        raw_quad = np.asarray(((100.0, 100.0), (300.0, 120.0), (280.0, 300.0), (90.0, 280.0)))
        semantic = _semantic_core_quad(raw_quad, 1.0)
        np.testing.assert_allclose(
            semantic,
            np.asarray(((90.0, 280.0), (280.0, 300.0), (300.0, 120.0), (100.0, 100.0))),
        )

    def test_semantic_order_selects_j_boundaries_when_j_has_larger_depth_span(self):
        raw_quad = np.asarray(((100.0, 100.0), (300.0, 280.0), (280.0, 300.0), (90.0, 120.0)))
        semantic = _semantic_core_quad(raw_quad, 1.0)
        np.testing.assert_allclose(
            semantic,
            np.asarray(((280.0, 300.0), (300.0, 280.0), (100.0, 100.0), (90.0, 120.0))),
        )

    def test_semantic_axis_tie_retains_i_boundary_pair(self):
        raw_quad = np.asarray(((0.0, 0.0), (100.0, 100.0), (0.0, 200.0), (-100.0, 100.0)))
        semantic = _semantic_core_quad(raw_quad, 1.0)
        np.testing.assert_allclose(
            semantic,
            np.asarray(((-100.0, 100.0), (0.0, 200.0), (100.0, 100.0), (0.0, 0.0))),
        )

    def test_semantic_fails_when_near_far_is_genuinely_unresolved(self):
        with self.assertRaises(TiledPerspectiveLatticeFailure) as failure:
            _semantic_core_quad(
                np.asarray(((0.0, 10.0), (10.0, 10.0), (20.0, 10.0), (30.0, 10.0))),
                0.0,
            )
        self.assertEqual(failure.exception.reason, "semantic_ordering_failure")

    def test_contradictory_cycle_is_dropped_deterministically(self):
        unit = np.asarray(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
        local_vertex_ids = (
            (15, 10, 11, 14),
            (11, 12, 13, 10),
            (13, 14, 15, 12),
        )
        neighbors = (
            (None, 1, None, 2),
            (None, 2, None, 0),
            (None, 0, None, 1),
        )
        self.assertIsNone(
            _normalize_component(
                (0, 1, 2),
                (unit, unit, unit),
                local_vertex_ids,
                neighbors,
            )
        )

    def test_real_c_t1_floor_lattice_beats_isolated_quadrilateral_distractors(self):
        result = read_tiled_perspective_lattice(C_T1_FIXTURE.read_bytes())
        self.assertGreaterEqual(result.raw_quadrilateral_count, result.deduplicated_cell_count)
        self.assertGreater(result.selected_component.tile_count, 1)
        self.assertGreaterEqual(result.authoritative_core.rows * result.authoritative_core.columns, 4)
        self.assertLessEqual(
            result.authoritative_core.rows * result.authoritative_core.columns,
            result.selected_component.tile_count,
        )
        quad = np.asarray(result.authoritative_quad_pixel)
        self.assertGreater(np.mean(quad[:2, 1]), np.mean(quad[2:, 1]))
        self.assertLess(quad[0, 0], quad[1, 0])
        self.assertLess(quad[3, 0], quad[2, 0])

    def test_real_c_t2_hardened_detector_recovers_multicell_floor_core(self):
        result = read_tiled_perspective_lattice(C_T2_FIXTURE.read_bytes())
        self.assertGreater(result.selected_component.tile_count, 1)
        self.assertGreaterEqual(result.authoritative_core.rows * result.authoritative_core.columns, 4)
        self.assertTrue(np.all(np.isfinite(result.h_lattice_to_tiled)))
        self.assertTrue(np.all(np.isfinite(result.vp_a_homogeneous)))
        self.assertTrue(np.all(np.isfinite(result.vp_b_homogeneous)))

    def test_real_c_t3_hardened_detector_keeps_floor_component_topologically_separate(self):
        result = read_tiled_perspective_lattice(C_T3_FIXTURE.read_bytes())
        self.assertGreater(result.selected_component.tile_count, 1)
        self.assertGreaterEqual(result.authoritative_core.rows * result.authoritative_core.columns, 4)
        selected = [
            cell
            for cell in result.cells
            if cell.id in result.selected_component.cell_ids
        ]
        self.assertTrue(all(cell.centroid_pixel[1] > 500.0 for cell in selected))


if __name__ == "__main__":
    unittest.main()
