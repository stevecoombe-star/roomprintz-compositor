import copy
import math
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from research.afc_sr1_tile_floor_reader import Segment, _normalize_line, _normalize_point
from research.afc_sr1_tile_floor_reader_v3 import (
    CHORDAL_DELTA_MIN,
    DIRECTIONAL_SEPARATION_DEGREES,
    Family,
    POLICY_VERSION,
    STABILITY_MAX_PX,
    _distinct,
    _direction_fit,
    _family_passes,
    _family_sort_key,
    _pair_key,
    _refine,
    _residuals,
    _select_pair,
    _stability,
    classify_vp,
    line_distance,
    read_floor_vanishing_line,
)

WIDTH, HEIGHT = 1264, 848
DIAGONAL = math.hypot(WIDTH, HEIGHT)


def segment(index, first, second, *, length=None):
    p1, p2 = np.asarray([*first, 1.0]), np.asarray([*second, 1.0])
    line = _normalize_line(np.cross(p1, p2))
    assert line is not None
    return Segment(index, p1, p2, line, length or math.dist(first, second), ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2))


def segment_from_line(line, index, center):
    normalized = _normalize_line(np.asarray(line, dtype=np.float64))
    assert normalized is not None
    tangent = np.asarray([-normalized[1], normalized[0]])
    midpoint = np.asarray(center, dtype=np.float64)
    p1xy, p2xy = midpoint - 35.0 * tangent, midpoint + 35.0 * tangent
    p1, p2 = np.r_[p1xy, 1.0], np.r_[p2xy, 1.0]
    return Segment(index, p1, p2, normalized, 70.0, (float(midpoint[0]), float(midpoint[1])))


def finite_segments(vp, offset=0):
    items = []
    for index, point in enumerate((150 + 100 * i, 600 + 11 * (i % 4)) for i in range(8)):
        line = np.cross(np.asarray([*vp, 1.0]), np.asarray([*point, 1.0]))
        items.append(segment_from_line(line, offset + index, point))
    return items


def directional_segments(direction, offset=0):
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    normal = np.asarray([-direction[1], direction[0]])
    return [
        segment_from_line(
            [normal[0], normal[1], -float(normal @ np.asarray((140 + i * 120, 500 + (i % 3) * 55)))],
            offset + i,
            (140 + i * 120, 500 + (i % 3) * 55),
        )
        for i in range(8)
    ]


def fitted_family(items, model):
    source = np.asarray([1.0, 0.0, 0.0]) if model == "directional" else np.asarray([650.0, 408.0, 1.0])
    family = _refine(items, items, model, DIAGONAL, source)
    assert family is not None and _family_passes(family)
    return family


def synthetic_family(vp, model, offset, residuals=None, support_length=70.0):
    normalized = _normalize_point(np.asarray(vp, dtype=np.float64))
    assert normalized is not None
    supporters = tuple(
        segment(offset + i, (0.0, 20.0 * i + offset), (100.0, 20.0 * i + offset), length=support_length)
        for i in range(8)
    )
    values = np.asarray(residuals if residuals is not None else [1.0] * 8, dtype=np.float64)
    return Family(model, normalized, supporters, values, 0.0, normalized)


def stable_result(distance=1.0):
    return {
        "stable": distance <= STABILITY_MAX_PX,
        "classPreserving": True,
        "splitFloorLines": [[0.0, 1.0, -400.0], [0.0, 1.0, -400.0]],
        "splitVsFullProbeDistancesPx": [[distance] * 4, [distance] * 4],
        "maxSplitVsFullProbeDistancePx": distance,
    }


class AfcSr1TileFloorReaderV3GeometryTests(unittest.TestCase):
    def test_finite_plus_finite_end_to_end_pair(self):
        winner, diagnostics = _select_pair(
            [*finite_segments((650.0, 408.0)), *finite_segments((1100.0, 320.0), 20)],
            WIDTH,
            HEIGHT,
        )
        self.assertIsNotNone(winner)
        self.assertEqual({winner["first"].model, winner["second"].model}, {"finite"})
        self.assertEqual(diagnostics["validPairCount"], 1)

    def test_finite_plus_directional_horizontal_end_to_end_pair(self):
        winner, _ = _select_pair(
            [*finite_segments((650.0, 408.0)), *directional_segments((1.0, 0.0), 20)],
            WIDTH,
            HEIGHT,
        )
        self.assertIsNotNone(winner)
        self.assertEqual({winner["first"].model, winner["second"].model}, {"finite", "directional"})

    def test_finite_plus_directional_oblique_end_to_end_pair(self):
        winner, _ = _select_pair(
            [*finite_segments((650.0, 408.0)), *directional_segments((1.0, 0.4), 20)],
            WIDTH,
            HEIGHT,
        )
        self.assertIsNotNone(winner)
        self.assertEqual({winner["first"].model, winner["second"].model}, {"finite", "directional"})

    def test_near_parallel_family_classifies_directional(self):
        self.assertEqual(classify_vp(np.asarray([20_000.0, 0.0, 1.0]), DIAGONAL)[0], "directional")

    def test_tiny_nonzero_z_is_directional_when_rho_is_large(self):
        model, point, rho = classify_vp(np.asarray([1.0, 0.0, 1e-12]), DIAGONAL)
        self.assertEqual(model, "directional")
        self.assertEqual(float(point[2]), 0.0)
        self.assertGreater(rho, 8.0)

    def test_classification_is_homogeneous_scale_and_sign_invariant(self):
        original = classify_vp(np.asarray([4.0, -2.0, 1.0]), 1500)
        for scale in (-17.0, 0.5, 11.0):
            actual = classify_vp(np.asarray([4.0, -2.0, 1.0]) * scale, 1500)
            self.assertEqual(actual[0], original[0])
            self.assertTrue(np.allclose(actual[1], original[1]))

    def test_directional_refinement_is_exactly_at_infinity(self):
        items = [segment(i, (0, 100 + i * 20), (120, 100 + i * 20)) for i in range(6)]
        direction = _direction_fit(items)
        self.assertIsNotNone(direction)
        assert direction is not None
        self.assertEqual(float(direction[2]), 0.0)
        self.assertLess(float(np.max(_residuals(items, "directional", direction, 1500))), 1e-10)

    def test_refinement_does_not_grow_a_below_minimum_seed(self):
        candidates = directional_segments((1.0, 0.4))
        self.assertIsNone(_refine(
            candidates,
            candidates[:5],
            "directional",
            DIAGONAL,
            np.asarray([1.0, 0.4, 0.0]),
        ))

    def test_well_conditioned_finite_stays_finite(self):
        model, point, rho = classify_vp(np.asarray([650.0, 408.0, 1.0]), DIAGONAL)
        self.assertEqual(model, "finite")
        self.assertIsNotNone(point)
        self.assertLess(rho, 8.0)

    def test_finite_directional_cross_is_incident_and_canonical(self):
        finite = np.asarray([200.0, 100.0, 1.0])
        directional = np.asarray([1.0, 0.0, 0.0])
        line = _normalize_line(np.cross(finite, directional))
        self.assertIsNotNone(line)
        assert line is not None
        self.assertAlmostEqual(float(line[0]), 0.0)
        self.assertGreaterEqual(float(line[1]), 0.0)
        self.assertAlmostEqual(float(line @ finite), 0.0)
        self.assertAlmostEqual(float(line @ directional), 0.0)

    def test_duplicate_family_rejection_and_canonical_dedupe_order(self):
        left = synthetic_family([650.0, 408.0, 1.0], "finite", 0, [2.0] * 8)
        right = synthetic_family([651.0, 408.0, 1.0], "finite", 20, [1.0] * 8)
        self.assertFalse(_distinct(left, right, DIAGONAL)[0])
        self.assertEqual(_family_sort_key(left)[:2], ("finite", tuple(float(value) for value in left.vp)))
        self.assertNotEqual(_family_sort_key(left), _family_sort_key(right))
        self.assertEqual(CHORDAL_DELTA_MIN, 0.15)

    def test_directional_distinctness_uses_rp1_separation(self):
        self.assertFalse(_distinct(np.asarray([1.0, 0.0, 0.0]), np.asarray([-1.0, 0.0, 0.0]), DIAGONAL)[0])
        five_degrees = np.asarray([math.cos(math.radians(5)), math.sin(math.radians(5)), 0.0])
        self.assertFalse(_distinct(np.asarray([1.0, 0.0, 0.0]), five_degrees, DIAGONAL)[0])
        self.assertTrue(_distinct(np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0]), DIAGONAL)[0])
        self.assertEqual(DIRECTIONAL_SEPARATION_DEGREES, 15.0)

    def test_all_unordered_projectively_distinct_pairs_are_enumerated(self):
        families = [
            synthetic_family([1.0, 0.0, 0.0], "directional", 0),
            synthetic_family([0.0, 1.0, 0.0], "directional", 20),
            synthetic_family([0.0, 0.0, 1.0], "finite", 40),
            synthetic_family([1.0, 1.0, 1.0], "finite", 60),
        ]
        with patch(
            "research.afc_sr1_tile_floor_reader_v3._discover_families",
            return_value=(families, {"finalFamilies": []}),
        ), patch("research.afc_sr1_tile_floor_reader_v3._stability", return_value=stable_result()):
            _, diagnostics = _select_pair([], WIDTH, HEIGHT)
        self.assertEqual(diagnostics["candidateUnorderedPairCount"], 6)
        self.assertEqual(diagnostics["validPairCount"] + len(diagnostics["invalidPairs"]), 6)
        self.assertEqual(len(diagnostics["validPairUniverse"]), diagnostics["validPairCount"])

    def test_pair_stability_over_18_is_a_hard_invalidity(self):
        families = [
            synthetic_family([1.0, 0.0, 0.0], "directional", 0),
            synthetic_family([0.0, 0.0, 1.0], "finite", 20),
        ]
        with patch(
            "research.afc_sr1_tile_floor_reader_v3._discover_families",
            return_value=(families, {"finalFamilies": []}),
        ), patch("research.afc_sr1_tile_floor_reader_v3._stability", return_value=stable_result(18.000001)):
            winner, diagnostics = _select_pair([], WIDTH, HEIGHT)
        self.assertIsNone(winner)
        self.assertEqual(diagnostics["validPairCount"], 0)
        self.assertEqual(diagnostics["invalidPairs"][0]["reason"], "unstable_vanishing_line")

    def test_class_preserving_split_refit_is_stable_for_exact_families(self):
        first = fitted_family(finite_segments((650.0, 408.0)), "finite")
        second = fitted_family(directional_segments((1.0, 0.0), 20), "directional")
        line = _normalize_line(np.cross(first.vp, second.vp))
        assert line is not None
        stability = _stability(first, second, line, WIDTH, HEIGHT)
        self.assertTrue(stability["stable"])
        self.assertLessEqual(stability["maxSplitVsFullProbeDistancePx"], STABILITY_MAX_PX)

    def test_floor_line_normalization_has_unit_normal_and_deterministic_sign(self):
        line = _normalize_line(np.asarray([-2.0, -3.0, 14.0]))
        assert line is not None
        self.assertAlmostEqual(math.hypot(float(line[0]), float(line[1])), 1.0)
        self.assertGreater(float(line[1]), 0.0)

    def test_floor_line_distance_is_sign_and_scale_invariant(self):
        line = np.asarray([2.0, -3.0, 14.0])
        self.assertEqual(line_distance(line, -9.0 * line, 1264, 848), 0.0)
        displaced = np.asarray([2.0, -3.0, 32.0])
        self.assertGreater(line_distance(line, displaced, 1264, 848), 0.0)

    def test_known_basin_support_counts_use_equal_votes(self):
        lines = [
            np.asarray([0.0, 1.0, 0.0]),
            np.asarray([0.0, 1.0, -10.0]),
            np.asarray([0.0, 1.0, -30.0]),
        ]
        counts = [
            sum(line_distance(line, other, WIDTH, HEIGHT) <= STABILITY_MAX_PX for other in lines)
            for line in lines
        ]
        self.assertEqual(counts, [2, 2, 1])

    def test_unstable_pairs_cast_zero_basin_votes(self):
        families = [
            synthetic_family([1.0, 0.0, 0.0], "directional", 0),
            synthetic_family([0.0, 0.0, 1.0], "finite", 20),
            synthetic_family([0.0, 1.0, 1.0], "finite", 40),
        ]
        with patch(
            "research.afc_sr1_tile_floor_reader_v3._discover_families",
            return_value=(families, {"finalFamilies": []}),
        ), patch(
            "research.afc_sr1_tile_floor_reader_v3._stability",
            side_effect=[stable_result(), stable_result(19.0), stable_result()],
        ):
            _, diagnostics = _select_pair([], WIDTH, HEIGHT)
        self.assertEqual(diagnostics["validPairCount"], 2)
        self.assertEqual(len(diagnostics["validPairUniverse"]), 2)
        self.assertEqual(len(diagnostics["invalidPairs"]), 1)

    def test_basin_support_outranks_own_stability(self):
        family_a = synthetic_family([1.0, 0.0, 0.0], "directional", 0)
        family_b = synthetic_family([0.0, 0.0, 1.0], "finite", 20)
        low_basin = {"basinSupport": 1, "stability": stable_result(0.01), "first": family_a,
                     "second": family_b, "line": np.asarray([0.0, 1.0, -400.0])}
        high_basin = {**low_basin, "basinSupport": 2, "stability": stable_result(17.9)}
        self.assertIs(sorted([low_basin, high_basin], key=_pair_key)[0], high_basin)

    def test_residual_support_and_canonical_pair_tie_break_hierarchy(self):
        first = synthetic_family([1.0, 0.0, 0.0], "directional", 0, [1.0] * 8, 80.0)
        second = synthetic_family([0.0, 0.0, 1.0], "finite", 20, [2.0] * 8, 70.0)
        pair = {"basinSupport": 3, "stability": stable_result(2.0), "first": first,
                "second": second, "line": np.asarray([0.0, 1.0, -400.0])}
        key = _pair_key(pair)
        self.assertEqual(key[:6], (-3, 2.0, 3.0, 3.0, -1200.0, -16))
        reverse = {**pair, "first": second, "second": first}
        self.assertEqual(_pair_key(pair), _pair_key(reverse))

    def test_exactly_one_authoritative_line_is_returned(self):
        candidates = [*finite_segments((650.0, 408.0)), *directional_segments((1.0, 0.0), 20)]
        raster = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        diagnostics = {
            "analysisImage": {"decodedWidth": WIDTH, "decodedHeight": HEIGHT},
            "analysisIdentity": {"scaleX": 1.0, "scaleY": 1.0},
        }
        winner = {"line": np.asarray([0.0, 1.0, -400.0])}
        with patch(
            "research.afc_sr1_tile_floor_reader_v3._admit_v2_segments",
            return_value=(diagnostics, candidates, raster),
        ), patch(
            "research.afc_sr1_tile_floor_reader_v3._select_pair",
            return_value=(winner, {"validPairCount": 1, "winningPair": {}}),
        ):
            result = read_floor_vanishing_line(b"synthetic", ((0, 0), (1, 0), (1, 1)))
        self.assertEqual(result["status"], "usable")
        self.assertEqual(set(result["floorVanishingLinePixel"]), {"a", "b", "c"})
        self.assertNotIn("floorVanishingLineCandidates", result)

    def test_malformed_and_nonfinite_evidence_fail_closed(self):
        ok, encoded = cv2.imencode(".png", np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
        self.assertTrue(ok)
        invalid_roi = read_floor_vanishing_line(
            bytes(encoded), ((0.0, 0.0), (1.0, 0.0), (float("nan"), 1.0))
        )
        self.assertEqual((invalid_roi["status"], invalid_roi["reason"]), ("rejected", "invalid_roi"))
        self.assertEqual(read_floor_vanishing_line(b"not-image", ((0, 0), (1, 0), (0, 1)))["reason"], "invalid_input_image")
        self.assertIsNone(classify_vp(np.asarray([math.nan, 0.0, 1.0]), DIAGONAL)[0])

    def test_deterministic_repeat_output(self):
        candidates = [*finite_segments((650.0, 408.0)), *directional_segments((1.0, 0.4), 20)]
        raster = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        base = {
            "analysisImage": {"decodedWidth": WIDTH, "decodedHeight": HEIGHT},
            "analysisIdentity": {"scaleX": 1.0, "scaleY": 1.0},
            "segmentCounts": {"raw": len(candidates), "admittedAllNineInside": len(candidates)},
        }
        with patch(
            "research.afc_sr1_tile_floor_reader_v3._admit_v2_segments",
            side_effect=lambda *_: (copy.deepcopy(base), candidates, raster),
        ):
            first = read_floor_vanishing_line(b"synthetic", ((0, 0), (1, 0), (1, 1)))
            second = read_floor_vanishing_line(b"synthetic", ((0, 0), (1, 0), (1, 1)))
        self.assertEqual(first, second)

    def test_unsupported_v3_policy_rejects_before_decode(self):
        result = read_floor_vanishing_line(b"", (), "afc-sr1-ts2-extractor-policy/v2")
        self.assertEqual((result["status"], result["reason"]), ("rejected", "unsupported_policy_version"))
        self.assertEqual(POLICY_VERSION, "afc-sr1-ts2-extractor-policy/v3")

    def test_basin_threshold_is_frozen(self):
        self.assertEqual(STABILITY_MAX_PX, 18.0)


if __name__ == "__main__":
    unittest.main()
