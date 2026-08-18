import copy
import json
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
    INDEPENDENT_DIRECTION_FIELD_MIN_MEDIAN_DEGREES,
    POLICY_VERSION,
    READER_MODULE_VERSION,
    STABILITY_MAX_PX,
    _distinct,
    _direction_fit,
    _direction_field_disagreement,
    _family_pair_independence_diagnostics,
    _family_diag,
    _family_passes,
    _family_support_geometry,
    _family_sort_key,
    _axial_disagreement_degrees,
    _axial_summary,
    _pair_key,
    _predicted_direction,
    _refine,
    _residual_summary,
    _residuals,
    _select_pair,
    _stability,
    classify_vp,
    evaluate_independent_direction_eligibility,
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


def directional_segments_at(direction, points, offset=0):
    """Exact directional segments through explicit analysis-pixel points."""
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    normal = np.asarray([-direction[1], direction[0]])
    return [
        segment_from_line(
            [normal[0], normal[1], -float(normal @ np.asarray(point, dtype=np.float64))],
            offset + index,
            point,
        )
        for index, point in enumerate(points)
    ]


def finite_segments_at(vp, points, offset=0):
    """Exact finite-pencil segments through explicit analysis-pixel points."""
    return [
        segment_from_line(
            np.cross(np.asarray([*vp, 1.0]), np.asarray([*point, 1.0])),
            offset + index,
            point,
        )
        for index, point in enumerate(points)
    ]


def shared_baseline_segments(first_vp, second_vp, points, offset=0):
    """Segments on the line incident to both finite VPs, shared by construction."""
    line = np.cross(np.asarray([*first_vp, 1.0]), np.asarray([*second_vp, 1.0]))
    return [
        segment_from_line(line, offset + index, point)
        for index, point in enumerate(points)
    ]


def _physical_predicted_direction(segment_item, family):
    """Independent test-only local-field calculation, not the sidecar helper."""
    if family.model == "finite":
        point = family.vp / family.vp[2]
        dx, dy = point[0] - segment_item.midpoint[0], point[1] - segment_item.midpoint[1]
    else:
        dx, dy = family.vp[0], family.vp[1]
    return math.atan2(float(dy), float(dx)) % math.pi


def field_disagreement_distribution(segments, first, second):
    """Test-only distribution summary for a development-corpus field sample."""
    values = []
    for item in segments:
        delta = abs(
            (_physical_predicted_direction(item, first) - _physical_predicted_direction(item, second))
            % math.pi
        )
        values.append(math.degrees(min(delta, math.pi - delta)))
    values.sort()
    assert values
    samples = np.asarray(values, dtype=np.float64)
    return {
        "minDegrees": float(samples.min()),
        "p10Degrees": float(np.percentile(samples, 10)),
        "p25Degrees": float(np.percentile(samples, 25)),
        "p50Degrees": float(np.percentile(samples, 50)),
        "p75Degrees": float(np.percentile(samples, 75)),
        "p90Degrees": float(np.percentile(samples, 90)),
        "maxDegrees": float(samples.max()),
        "fractionBelow0_5Degrees": float(np.mean(samples < 0.5)),
        "fractionBelow1Degrees": float(np.mean(samples < 1.0)),
        "fractionBelow2Degrees": float(np.mean(samples < 2.0)),
        "fractionBelow5Degrees": float(np.mean(samples < 5.0)),
        "fractionAbove10Degrees": float(np.mean(samples > 10.0)),
        "fractionAbove20Degrees": float(np.mean(samples > 20.0)),
        "fractionAbove45Degrees": float(np.mean(samples > 45.0)),
    }


def _negative_a_segments():
    first_vp, second_vp = (-25.0, 475.0), (-596.0, 404.0)
    # Coextensive right-side samples make each pencil locally interchangeable.
    points = [(900 + 40 * index, 600 + 11 * (index % 4)) for index in range(8)]
    return [
        *finite_segments_at(first_vp, points),
        *finite_segments_at(second_vp, points, 20),
    ]


def _negative_b_segments():
    first_vp, second_vp = (-25.0, 475.0), (-596.0, 404.0)
    # A wider lattice supplies separate support, plus four physically shared
    # baseline segments; this is not a production duplicate classification.
    points = [(200 + 90 * index, 600 + 11 * (index % 4)) for index in range(8)]
    baseline_points = [
        (150 + 180 * index, 404 + (150 + 180 * index + 596) * 71 / 571)
        for index in range(4)
    ]
    return [
        *finite_segments_at(first_vp, points),
        *finite_segments_at(second_vp, points, 20),
        *shared_baseline_segments(first_vp, second_vp, baseline_points, 40),
    ]


ASYMMETRIC_H1_FINITE_VP = (-200.0, 400.0)
ASYMMETRIC_H1_DIRECTION = (1.0, 0.0)
ASYMMETRIC_H1_FINITE_POINTS = (
    (220, 430), (360, 418), (500, 436), (640, 422),
    (780, 432), (920, 416), (1060, 440), (1180, 424),
)
ASYMMETRIC_H1_DIRECTIONAL_POINTS = (
    (420, 660), (540, 700), (660, 680), (780, 730),
    (900, 690), (1020, 720), (1140, 705), (620, 750),
)


def _asymmetric_h1_segments():
    """Candidate 1: a true finite×directional pair with partitioned support."""
    return [
        *finite_segments_at(ASYMMETRIC_H1_FINITE_VP, ASYMMETRIC_H1_FINITE_POINTS),
        *directional_segments_at(
            ASYMMETRIC_H1_DIRECTION, ASYMMETRIC_H1_DIRECTIONAL_POINTS, 20
        ),
    ]


def generated_finite_directional_field_distribution(finite_vp, directional_vector, points):
    """Independent physical field calculation from known fixture generators."""
    finite_x, finite_y = finite_vp
    direction_x, direction_y = directional_vector
    directional_angle = math.atan2(direction_y, direction_x) % math.pi
    values = []
    for x, y in points:
        finite_angle = math.atan2(finite_y - y, finite_x - x) % math.pi
        delta = abs(finite_angle - directional_angle)
        values.append(math.degrees(min(delta, math.pi - delta)))
    samples = np.asarray(sorted(values), dtype=np.float64)
    return {
        "minDegrees": float(samples.min()),
        "p25Degrees": float(np.percentile(samples, 25)),
        "p50Degrees": float(np.percentile(samples, 50)),
        "p90Degrees": float(np.percentile(samples, 90)),
        "maxDegrees": float(samples.max()),
        "fractionBelow5Degrees": float(np.mean(samples < 5.0)),
        "fractionAbove10Degrees": float(np.mean(samples > 10.0)),
    }


def development_corpus_summary():
    """Returns a silent, deterministic comparison table for test/research use."""
    cases = (
        ("P1 finite×finite", [*finite_segments((650.0, 408.0)), *finite_segments((1100.0, 320.0), 20)]),
        ("P2 finite×directional horizontal", [*finite_segments((650.0, 408.0)), *directional_segments((1.0, 0.0), 20)]),
        ("P3 finite×directional oblique", [*finite_segments((650.0, 408.0)), *directional_segments((1.0, 0.4), 20)]),
        ("H2 modest finite×directional", [*finite_segments((-200.0, 400.0)), *directional_segments((1.0, 0.0), 20)]),
        ("ASYM-H1 partitioned finite×directional", _asymmetric_h1_segments()),
        ("N-A duplicate same-side", _negative_a_segments()),
        ("N-B fragmented same-side", _negative_b_segments()),
    )
    rows = []
    for fixture_id, candidates in cases:
        winner, diagnostics = _select_pair(candidates, WIDTH, HEIGHT)
        pair = diagnostics["familyPairIndependenceDiagnostics"]["pairs"][0]
        field = pair["predictedDirectionFieldDisagreement"]
        eligible = diagnostics["validPairUniverse"][0] if winner is not None else None
        rejected = diagnostics["independentDirectionEligibilityRejectedPairs"]
        rows.append({
            "fixtureId": fixture_id,
            "usable": winner is not None,
            "supportCounts": (
                pair["overlap"]["familyASupporterCount"],
                pair["overlap"]["familyBSupporterCount"],
            ),
            "jaccard": pair["overlap"]["jaccard"],
            "overlapFractionOfSmaller": pair["overlap"]["overlapFractionOfSmaller"],
            "crossFitMediansPx": (
                pair["crossFit"]["firstSupportersAgainstSecond"]["medianResidualPx"],
                pair["crossFit"]["secondSupportersAgainstFirst"]["medianResidualPx"],
            ),
            "firstFieldMedianP90Degrees": (
                field["onFirstSupporterMidpoints"]["medianDegrees"],
                field["onFirstSupporterMidpoints"]["p90Degrees"],
            ),
            "secondFieldMedianP90Degrees": (
                field["onSecondSupporterMidpoints"]["medianDegrees"],
                field["onSecondSupporterMidpoints"]["p90Degrees"],
            ),
            "unionFieldMedianP90Degrees": (
                field["onUnionSupporterMidpoints"]["medianDegrees"],
                field["onUnionSupporterMidpoints"]["p90Degrees"],
            ),
            "sharedFieldMedianP90Degrees": (
                field["onSharedSupporterMidpoints"]["medianDegrees"],
                field["onSharedSupporterMidpoints"]["p90Degrees"],
            ),
            "stableProjectivelyValid": diagnostics["stableProjectivelyValidPairCount"] == 1,
            "eligibility": (
                eligible["independentDirectionEligibility"]
                if eligible is not None
                else rejected[0]
            ),
        })
    return rows


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
    def test_family_support_geometry_projects_only_final_supporters_with_stable_identity(self):
        shared = segment(17, (10.25, 20.5), (30.75, 40.125))
        first_only = segment(9, (1.0, 2.0), (3.0, 4.0))
        second_only = segment(31, (50.0, 60.0), (70.0, 80.0))
        first = Family(
            "finite", np.asarray([1.0, 0.0, 1.0]), (shared, first_only),
            np.asarray([1.0, 2.0]), 0.0, np.asarray([1.0, 0.0, 1.0])
        )
        second = Family(
            "directional", np.asarray([0.0, 1.0, 0.0]), (second_only, shared),
            np.asarray([3.0, 4.0]), 0.0, np.asarray([0.0, 1.0, 0.0])
        )

        projection = _family_support_geometry((first, second))

        self.assertEqual(projection["coordinateSpace"], "analysis-pixel/v1")
        self.assertEqual(projection["authority"], "none")
        self.assertEqual(projection["role"], "observation_only")
        self.assertTrue(projection["excludedFromCanonicalEvidence"])
        self.assertEqual(
            projection["families"],
            [
                {"familyIndex": 0, "supporterDetectorIndices": [17, 9]},
                {"familyIndex": 1, "supporterDetectorIndices": [31, 17]},
            ],
        )
        self.assertEqual([item["detectorIndex"] for item in projection["segments"]], [9, 17, 31])
        self.assertEqual(
            projection["segments"][1],
            {"detectorIndex": 17, "x1": 10.25, "y1": 20.5, "x2": 30.75, "y2": 40.125},
        )
        self.assertEqual(len(projection["families"][0]["supporterDetectorIndices"]), _family_diag(first)["supportCount"])
        self.assertEqual(len(projection["families"][1]["supporterDetectorIndices"]), _family_diag(second)["supportCount"])
        json.dumps(projection, allow_nan=False)
        self.assertTrue(
            all(
                not isinstance(value, (np.ndarray, np.generic))
                for item in projection["segments"]
                for value in item.values()
            )
        )

    def test_finite_plus_finite_end_to_end_pair(self):
        winner, diagnostics = _select_pair(
            [*finite_segments((650.0, 408.0)), *finite_segments((1100.0, 320.0), 20)],
            WIDTH,
            HEIGHT,
        )
        self.assertIsNotNone(winner)
        self.assertEqual({winner["first"].model, winner["second"].model}, {"finite"})
        self.assertEqual(diagnostics["validPairCount"], 1)
        self.assertIn("familySupportGeometry", diagnostics)
        pair = diagnostics["familyPairIndependenceDiagnostics"]["pairs"][0]
        self.assertEqual(pair["overlap"]["jaccard"], 0.0)
        self.assertEqual(pair["overlap"]["overlapFractionOfSmaller"], 0.0)
        self.assertEqual(pair["overlap"]["sharedSupporterCount"], 0)
        self.assertEqual(
            pair["crossFit"]["firstSupportersAgainstSecond"]["withinExistingInlierBandCount"], 0
        )
        self.assertEqual(
            pair["crossFit"]["secondSupportersAgainstFirst"]["withinExistingInlierBandCount"], 0
        )
        self.assertAlmostEqual(
            pair["crossFit"]["firstSupportersAgainstSecond"]["medianResidualPx"], 318.1656423040751
        )
        self.assertAlmostEqual(
            pair["crossFit"]["secondSupportersAgainstFirst"]["medianResidualPx"], 120.84220786415665
        )
        field = pair["predictedDirectionFieldDisagreement"]
        self.assertAlmostEqual(field["onFirstSupporterMidpoints"]["medianDegrees"], 29.080811644692808)
        self.assertAlmostEqual(field["onFirstSupporterMidpoints"]["p90Degrees"], 76.0542911864474)
        self.assertAlmostEqual(field["onSecondSupporterMidpoints"]["medianDegrees"], 29.080811644692808)
        self.assertAlmostEqual(field["onSecondSupporterMidpoints"]["p90Degrees"], 76.0542911864474)
        self.assertAlmostEqual(field["onUnionSupporterMidpoints"]["medianDegrees"], 29.080811644692808)
        self.assertAlmostEqual(field["onUnionSupporterMidpoints"]["p90Degrees"], 77.25268585545307)

    def test_finite_plus_directional_horizontal_end_to_end_pair(self):
        winner, diagnostics = _select_pair(
            [*finite_segments((650.0, 408.0)), *directional_segments((1.0, 0.0), 20)],
            WIDTH,
            HEIGHT,
        )
        self.assertIsNotNone(winner)
        self.assertEqual({winner["first"].model, winner["second"].model}, {"finite", "directional"})
        self.assertEqual(diagnostics["validPairCount"], 1)
        pair = diagnostics["familyPairIndependenceDiagnostics"]["pairs"][0]
        self.assertEqual(pair["overlap"]["jaccard"], 0.0)
        self.assertEqual(
            pair["crossFit"]["firstSupportersAgainstSecond"]["withinExistingInlierBandCount"], 0
        )
        self.assertEqual(
            pair["crossFit"]["secondSupportersAgainstFirst"]["withinExistingInlierBandCount"], 0
        )
        field = pair["predictedDirectionFieldDisagreement"]
        self.assertAlmostEqual(field["onFirstSupporterMidpoints"]["medianDegrees"], 27.76648815532568)
        self.assertAlmostEqual(field["onFirstSupporterMidpoints"]["p90Degrees"], 69.7290422890257)
        self.assertAlmostEqual(field["onSecondSupporterMidpoints"]["medianDegrees"], 48.366460663429024)
        self.assertAlmostEqual(field["onSecondSupporterMidpoints"]["p90Degrees"], 72.46765945425243)
        self.assertAlmostEqual(field["onUnionSupporterMidpoints"]["medianDegrees"], 36.151730096977225)
        self.assertAlmostEqual(field["onUnionSupporterMidpoints"]["p90Degrees"], 72.22513859111538)

    def test_finite_plus_directional_oblique_end_to_end_pair(self):
        winner, diagnostics = _select_pair(
            [*finite_segments((650.0, 408.0)), *directional_segments((1.0, 0.4), 20)],
            WIDTH,
            HEIGHT,
        )
        self.assertIsNotNone(winner)
        self.assertEqual({winner["first"].model, winner["second"].model}, {"finite", "directional"})
        self.assertEqual(diagnostics["validPairCount"], 1)
        pair = diagnostics["familyPairIndependenceDiagnostics"]["pairs"][0]
        self.assertEqual(pair["overlap"]["jaccard"], 0.0)
        self.assertEqual(
            pair["crossFit"]["firstSupportersAgainstSecond"]["withinExistingInlierBandCount"], 0
        )
        self.assertEqual(
            pair["crossFit"]["secondSupportersAgainstFirst"]["withinExistingInlierBandCount"], 0
        )
        field = pair["predictedDirectionFieldDisagreement"]
        self.assertAlmostEqual(field["onFirstSupporterMidpoints"]["medianDegrees"], 43.318740883500986)
        self.assertAlmostEqual(field["onFirstSupporterMidpoints"]["p90Degrees"], 64.94230179842393)
        self.assertAlmostEqual(field["onSecondSupporterMidpoints"]["medianDegrees"], 53.00607076960853)
        self.assertAlmostEqual(field["onSecondSupporterMidpoints"]["p90Degrees"], 74.4043311635895)
        self.assertAlmostEqual(field["onUnionSupporterMidpoints"]["medianDegrees"], 46.44634568121448)
        self.assertAlmostEqual(field["onUnionSupporterMidpoints"]["p90Degrees"], 74.95054065854183)

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
        self.assertEqual(
            diagnostics["stableProjectivelyValidPairCount"] + len(diagnostics["invalidPairs"]),
            6,
        )
        self.assertEqual(
            diagnostics["eligiblePairCount"]
            + len(diagnostics["independentDirectionEligibilityRejectedPairs"]),
            diagnostics["stableProjectivelyValidPairCount"],
        )
        self.assertEqual(len(diagnostics["validPairUniverse"]), diagnostics["validPairCount"])
        self.assertEqual(
            [pair["familyIndices"] for pair in diagnostics["familyPairIndependenceDiagnostics"]["pairs"]],
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        )

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
        self.assertEqual(diagnostics["stableProjectivelyValidPairCount"], 2)
        self.assertEqual(
            diagnostics["eligiblePairCount"]
            + len(diagnostics["independentDirectionEligibilityRejectedPairs"]),
            2,
        )
        self.assertEqual(len(diagnostics["validPairUniverse"]), diagnostics["eligiblePairCount"])
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

    def test_unsupported_v4_policy_rejects_before_decode(self):
        result = read_floor_vanishing_line(b"", (), "afc-sr1-ts2-extractor-policy/v3")
        self.assertEqual((result["status"], result["reason"]), ("rejected", "unsupported_policy_version"))
        self.assertEqual(POLICY_VERSION, "afc-sr1-ts2-extractor-policy/v4")
        self.assertEqual(READER_MODULE_VERSION, "afc-sr1-tile-floor-reader/v4")

    def test_basin_threshold_is_frozen(self):
        self.assertEqual(STABILITY_MAX_PX, 18.0)


class AfcSr1IndependentDirectionEligibilityTests(unittest.TestCase):
    def evaluate(self, **overrides):
        return evaluate_independent_direction_eligibility(
            overlap_fraction_of_smaller=overrides.get("overlap", 0.6),
            first_support_count=overrides.get("first_support_count", 10),
            second_support_count=overrides.get("second_support_count", 10),
            first_inlier_band_count=overrides.get("first_inlier_band_count", 6),
            second_inlier_band_count=overrides.get("second_inlier_band_count", 6),
            first_region_median_degrees=overrides.get("first_median", 10.0),
            second_region_median_degrees=overrides.get("second_median", 10.0),
        )

    def test_stage_one_half_boundaries_are_inclusive_rejections(self):
        cases = (
            {"overlap": 0.5},
            {"first_inlier_band_count": 5},
            {"second_inlier_band_count": 5},
            {
                "overlap": 0.5,
                "first_inlier_band_count": 5,
                "second_inlier_band_count": 5,
            },
        )
        for case in cases:
            with self.subTest(case=case):
                result = self.evaluate(**case)
                self.assertFalse(result["eligible"])
                self.assertEqual(result["failedStage"], 1)
                self.assertEqual(
                    result["rejectionReason"],
                    "duplicate_or_interchangeable_families",
                )

    def test_stage_one_one_value_below_half_survives_to_stage_two(self):
        result = self.evaluate(first_inlier_band_count=4)
        self.assertTrue(result["eligible"])
        self.assertIsNone(result["failedStage"])
        self.assertIsNone(result["rejectionReason"])
        self.assertEqual(result["firstInlierBandFraction"], 0.4)

    def test_stage_two_ten_degree_comparator_is_inclusive(self):
        for median, expected in ((9.999, False), (10.0, True), (10.001, True)):
            with self.subTest(median=median):
                result = self.evaluate(
                    overlap=0.0,
                    first_inlier_band_count=0,
                    second_inlier_band_count=0,
                    first_median=median,
                    second_median=0.0,
                )
                self.assertEqual(result["eligible"], expected)
                self.assertEqual(
                    result["failedStage"],
                    None if expected else 2,
                )
                self.assertEqual(
                    result["rejectionReason"],
                    None if expected else "insufficient_direction_field_separation",
                )
        self.assertEqual(INDEPENDENT_DIRECTION_FIELD_MIN_MEDIAN_DEGREES, 10.0)


class AfcSr1V3IndependenceDevelopmentCorpusTests(unittest.TestCase):
    def test_h2_modest_finite_directional_is_a_true_independent_hard_positive(self):
        winner, diagnostics = _select_pair(
            [*finite_segments((-200.0, 400.0)), *directional_segments((1.0, 0.0), 20)],
            WIDTH,
            HEIGHT,
        )

        self.assertIsNotNone(winner)
        self.assertEqual({winner["first"].model, winner["second"].model}, {"finite", "directional"})
        self.assertEqual(diagnostics["validFamilyCount"], 2)
        self.assertEqual(diagnostics["validPairCount"], 1)
        pair = diagnostics["familyPairIndependenceDiagnostics"]["pairs"][0]
        self.assertEqual(pair["overlap"]["jaccard"], 0.0)
        self.assertEqual(pair["overlap"]["sharedSupporterCount"], 0)
        self.assertAlmostEqual(
            pair["crossFit"]["firstSupportersAgainstSecond"]["medianResidualPx"], 155.0000000000104
        )
        self.assertAlmostEqual(
            pair["crossFit"]["secondSupportersAgainstFirst"]["medianResidualPx"], 452.9017846461923
        )
        field = pair["predictedDirectionFieldDisagreement"]
        self.assertAlmostEqual(field["onFirstSupporterMidpoints"]["medianDegrees"], 11.648668352691022)
        self.assertAlmostEqual(field["onFirstSupporterMidpoints"]["p90Degrees"], 19.006230417734265)
        self.assertAlmostEqual(field["onSecondSupporterMidpoints"]["medianDegrees"], 17.32611058599424)
        self.assertAlmostEqual(field["onSecondSupporterMidpoints"]["p90Degrees"], 26.508415509695766)
        self.assertAlmostEqual(field["onUnionSupporterMidpoints"]["medianDegrees"], 14.436229331519664)
        self.assertAlmostEqual(field["onUnionSupporterMidpoints"]["p90Degrees"], 23.551076651522052)
        eligibility = diagnostics["validPairUniverse"][0]["independentDirectionEligibility"]
        self.assertTrue(eligibility["eligible"])
        self.assertIsNone(eligibility["rejectionReason"])
        self.assertAlmostEqual(eligibility["strongRegionMedianDegrees"], 17.32611058599424)

    def test_asymmetric_h1_partitioned_support_is_a_true_independent_pair(self):
        # This is a true positive by construction: the finite pencil exactly
        # concurs at (-200, 400), the other family is exactly horizontal, and
        # their disjoint regions intentionally sample weak versus strong local
        # field separation. The weak region is sampling geometry, not shared
        # semantics or an interchangeable-support construction.
        first_winner, first = _select_pair(_asymmetric_h1_segments(), WIDTH, HEIGHT)
        second_winner, second = _select_pair(_asymmetric_h1_segments(), WIDTH, HEIGHT)

        self.assertIsNotNone(first_winner)
        self.assertIsNotNone(second_winner)
        self.assertEqual(first, second)
        assert first_winner is not None and second_winner is not None
        self.assertTrue(np.array_equal(first_winner["line"], second_winner["line"]))
        for key in ("first", "second"):
            self.assertEqual(first_winner[key].model, second_winner[key].model)
            self.assertTrue(np.array_equal(first_winner[key].vp, second_winner[key].vp))
            self.assertEqual(
                [item.detector_index for item in first_winner[key].supporters],
                [item.detector_index for item in second_winner[key].supporters],
            )
        winner = first_winner
        diagnostics = first

        self.assertEqual(diagnostics["validFamilyCount"], 2)
        self.assertEqual(diagnostics["candidateUnorderedPairCount"], 1)
        self.assertEqual(diagnostics["validPairCount"], 1)
        self.assertEqual(len(diagnostics["validPairUniverse"]), 1)
        final_families = diagnostics["candidateDiscovery"]["finalFamilies"]
        self.assertEqual([family["vpClass"] for family in final_families], ["directional", "finite"])
        self.assertEqual({family.model for family in (winner["first"], winner["second"])}, {"finite", "directional"})

        # Associate all pair diagnostics with their recovered model classes,
        # rather than assuming the directional family always sorts first.
        pair = diagnostics["familyPairIndependenceDiagnostics"]["pairs"][0]
        pair_models = [final_families[index]["vpClass"] for index in pair["familyIndices"]]
        self.assertEqual(set(pair_models), {"finite", "directional"})
        field = pair["predictedDirectionFieldDisagreement"]
        field_by_model = {
            pair_models[0]: field["onFirstSupporterMidpoints"],
            pair_models[1]: field["onSecondSupporterMidpoints"],
        }
        cross_fit_by_source_model = {
            pair_models[0]: pair["crossFit"]["firstSupportersAgainstSecond"],
            pair_models[1]: pair["crossFit"]["secondSupportersAgainstFirst"],
        }

        families_by_model = {family.model: family for family in (winner["first"], winner["second"])}
        finite = families_by_model["finite"]
        directional = families_by_model["directional"]
        np.testing.assert_allclose(
            finite.vp / finite.vp[2],
            np.asarray([*ASYMMETRIC_H1_FINITE_VP, 1.0]),
            atol=1e-8,
        )
        self.assertAlmostEqual(float(directional.vp[2]), 0.0, places=12)
        self.assertAlmostEqual(abs(float(directional.vp[0])), 1.0, places=12)
        self.assertAlmostEqual(float(directional.vp[1]), 0.0, places=10)
        self.assertEqual(
            [item.detector_index for item in finite.supporters],
            list(range(8)),
        )
        self.assertEqual(
            [item.detector_index for item in directional.supporters],
            list(range(20, 28)),
        )

        overlap = pair["overlap"]
        self.assertEqual(
            (overlap["familyASupporterCount"], overlap["familyBSupporterCount"]),
            (8, 8),
        )
        self.assertEqual(overlap["sharedSupporterCount"], 0)
        self.assertEqual(overlap["jaccard"], 0.0)
        self.assertEqual(overlap["overlapFractionOfSmaller"], 0.0)
        self.assertEqual(pair["exclusiveSupport"]["firstOnlySupporterCount"], 8)
        self.assertEqual(pair["exclusiveSupport"]["secondOnlySupporterCount"], 8)
        self.assertAlmostEqual(
            cross_fit_by_source_model["directional"]["medianResidualPx"],
            302.50000000001705,
        )
        self.assertAlmostEqual(
            cross_fit_by_source_model["finite"]["medianResidualPx"],
            48.59796812770969,
        )
        self.assertEqual(
            cross_fit_by_source_model["directional"]["withinExistingInlierBandCount"],
            0,
        )
        self.assertEqual(
            cross_fit_by_source_model["finite"]["withinExistingInlierBandCount"],
            0,
        )

        # Independently compute expected local fields from the known finite VP,
        # horizontal direction, and exact supporter midpoints, not V3's
        # _direction_field_disagreement helper or fitted-family outputs.
        weak_expected = generated_finite_directional_field_distribution(
            ASYMMETRIC_H1_FINITE_VP,
            ASYMMETRIC_H1_DIRECTION,
            ASYMMETRIC_H1_FINITE_POINTS,
        )
        strong_expected = generated_finite_directional_field_distribution(
            ASYMMETRIC_H1_FINITE_VP,
            ASYMMETRIC_H1_DIRECTION,
            ASYMMETRIC_H1_DIRECTIONAL_POINTS,
        )
        union_expected = generated_finite_directional_field_distribution(
            ASYMMETRIC_H1_FINITE_VP,
            ASYMMETRIC_H1_DIRECTION,
            [*ASYMMETRIC_H1_FINITE_POINTS, *ASYMMETRIC_H1_DIRECTIONAL_POINTS],
        )
        self.assertAlmostEqual(weak_expected["p50Degrees"], 1.8296595856881535)
        self.assertAlmostEqual(weak_expected["p90Degrees"], 3.286517478983792)
        self.assertAlmostEqual(strong_expected["p50Degrees"], 18.32222514747859)
        self.assertAlmostEqual(strong_expected["p90Degrees"], 22.8599458349494)
        self.assertAlmostEqual(union_expected["minDegrees"], 0.8184554616886027)
        self.assertAlmostEqual(union_expected["p25Degrees"], 1.835337896306317)
        self.assertAlmostEqual(union_expected["p50Degrees"], 8.454181186786002)
        self.assertAlmostEqual(union_expected["p90Degrees"], 22.409437952598932)
        self.assertAlmostEqual(union_expected["maxDegrees"], 23.114207983326832)
        self.assertEqual(union_expected["fractionBelow5Degrees"], 0.5)
        self.assertEqual(union_expected["fractionAbove10Degrees"], 0.5)

        weak_actual = field_by_model["finite"]
        strong_actual = field_by_model["directional"]
        self.assertLess(weak_actual["medianDegrees"], 5.0)
        self.assertGreater(strong_actual["medianDegrees"], 10.0)
        self.assertAlmostEqual(weak_actual["medianDegrees"], weak_expected["p50Degrees"], places=8)
        self.assertAlmostEqual(weak_actual["p90Degrees"], weak_expected["p90Degrees"], places=8)
        self.assertAlmostEqual(strong_actual["medianDegrees"], strong_expected["p50Degrees"], places=8)
        self.assertAlmostEqual(strong_actual["p90Degrees"], strong_expected["p90Degrees"], places=8)
        self.assertAlmostEqual(
            field["onUnionSupporterMidpoints"]["medianDegrees"],
            union_expected["p50Degrees"],
            places=8,
        )
        self.assertAlmostEqual(
            field["onUnionSupporterMidpoints"]["p90Degrees"],
            union_expected["p90Degrees"],
            places=8,
        )

        valid_pair = diagnostics["validPairUniverse"][0]
        self.assertTrue(_distinct(winner["first"], winner["second"], DIAGONAL)[0])
        self.assertAlmostEqual(valid_pair["distinctness"]["chordal"], 1.0514632875484748)
        self.assertTrue(valid_pair["stability"]["stable"])
        self.assertLess(valid_pair["stability"]["maxSplitVsFullProbeDistancePx"], 1e-8)
        eligibility = valid_pair["independentDirectionEligibility"]
        self.assertTrue(eligibility["eligible"])
        self.assertIsNone(eligibility["failedStage"])
        self.assertAlmostEqual(eligibility["firstRegionMedianDegrees"], 18.32222514747859)
        self.assertAlmostEqual(eligibility["secondRegionMedianDegrees"], 1.8296595856881535)
        self.assertAlmostEqual(eligibility["strongRegionMedianDegrees"], 18.32222514747859)

    def test_negative_a_duplicate_same_side_pencils_fails_stage_one(self):
        first_winner, first = _select_pair(_negative_a_segments(), WIDTH, HEIGHT)
        second_winner, second = _select_pair(_negative_a_segments(), WIDTH, HEIGHT)

        self.assertIsNone(first_winner)
        self.assertIsNone(second_winner)
        self.assertEqual(first, second)
        self.assertEqual(first["validFamilyCount"], 2)
        self.assertEqual(first["stableProjectivelyValidPairCount"], 1)
        self.assertEqual(first["eligiblePairCount"], 0)
        self.assertEqual(first["validPairUniverse"], [])
        eligibility = first["independentDirectionEligibilityRejectedPairs"][0]
        self.assertEqual(eligibility["familyIndices"], [0, 1])
        self.assertFalse(eligibility["eligible"])
        self.assertEqual(eligibility["failedStage"], 1)
        self.assertEqual(eligibility["rejectionReason"], "duplicate_or_interchangeable_families")
        self.assertGreaterEqual(eligibility["overlapFractionOfSmaller"], 0.5)
        self.assertGreaterEqual(eligibility["firstInlierBandFraction"], 0.5)
        self.assertGreaterEqual(eligibility["secondInlierBandFraction"], 0.5)
        pair = first["familyPairIndependenceDiagnostics"]["pairs"][0]
        self.assertEqual({family["vpClass"] for family in first["candidateDiscovery"]["finalFamilies"]}, {"finite"})
        self.assertEqual(pair["overlap"]["sharedSupporterCount"], 8)
        self.assertAlmostEqual(pair["overlap"]["jaccard"], 0.5)
        self.assertAlmostEqual(pair["overlap"]["overlapFractionOfSmaller"], 8 / 11)
        self.assertAlmostEqual(
            pair["crossFit"]["firstSupportersAgainstSecond"]["medianResidualPx"], 2.202986759524549
        )
        self.assertAlmostEqual(
            pair["crossFit"]["secondSupportersAgainstFirst"]["medianResidualPx"], 2.4577233697967245
        )
        field = pair["predictedDirectionFieldDisagreement"]
        self.assertAlmostEqual(field["onUnionSupporterMidpoints"]["medianDegrees"], 0.18537407387996394)
        self.assertAlmostEqual(field["onUnionSupporterMidpoints"]["p90Degrees"], 0.4354327099230883)
        self.assertAlmostEqual(field["onSharedSupporterMidpoints"]["medianDegrees"], 0.08601980434926773)
        self.assertAlmostEqual(field["onSharedSupporterMidpoints"]["p90Degrees"], 0.17926260934426788)

    def test_negative_b_fragmented_same_side_pencils_fails_stage_two(self):
        first_winner, first = _select_pair(_negative_b_segments(), WIDTH, HEIGHT)
        second_winner, second = _select_pair(_negative_b_segments(), WIDTH, HEIGHT)

        self.assertIsNone(first_winner)
        self.assertIsNone(second_winner)
        self.assertEqual(first, second)
        self.assertEqual(first["validFamilyCount"], 2)
        self.assertEqual(first["stableProjectivelyValidPairCount"], 1)
        self.assertEqual(first["eligiblePairCount"], 0)
        eligibility = first["independentDirectionEligibilityRejectedPairs"][0]
        self.assertEqual(eligibility["failedStage"], 2)
        self.assertEqual(eligibility["rejectionReason"], "insufficient_direction_field_separation")
        self.assertLess(eligibility["overlapFractionOfSmaller"], 0.5)
        self.assertLess(eligibility["firstInlierBandFraction"], 0.5)
        self.assertLess(eligibility["secondInlierBandFraction"], 0.5)
        self.assertAlmostEqual(eligibility["strongRegionMedianDegrees"], 1.7843706062334603)
        pair = first["familyPairIndependenceDiagnostics"]["pairs"][0]
        self.assertEqual({family["vpClass"] for family in first["candidateDiscovery"]["finalFamilies"]}, {"finite"})
        self.assertEqual(pair["overlap"]["sharedSupporterCount"], 4)
        self.assertAlmostEqual(pair["overlap"]["jaccard"], 0.2)
        self.assertAlmostEqual(pair["overlap"]["overlapFractionOfSmaller"], 1 / 3)
        self.assertEqual(pair["exclusiveSupport"]["firstOnlySupporterCount"], 8)
        self.assertEqual(pair["exclusiveSupport"]["secondOnlySupporterCount"], 8)
        self.assertAlmostEqual(
            pair["crossFit"]["firstSupportersAgainstSecond"]["medianResidualPx"], 40.60206588803783
        )
        self.assertAlmostEqual(
            pair["crossFit"]["secondSupportersAgainstFirst"]["medianResidualPx"], 22.710643227432627
        )
        field = pair["predictedDirectionFieldDisagreement"]
        self.assertAlmostEqual(field["onUnionSupporterMidpoints"]["medianDegrees"], 2.198675023182687)
        self.assertAlmostEqual(field["onUnionSupporterMidpoints"]["p90Degrees"], 10.70371214841603)
        self.assertAlmostEqual(field["onSharedSupporterMidpoints"]["medianDegrees"], 7.82416451640025e-12)
        self.assertAlmostEqual(field["onSharedSupporterMidpoints"]["p90Degrees"], 1.3521174061837704e-11)

    def test_all_independent_direction_rejections_use_the_new_top_level_reason(self):
        raster = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        with patch(
            "research.afc_sr1_tile_floor_reader_v3._admit_v2_segments",
            return_value=(
                {
                    "analysisImage": {"decodedWidth": WIDTH, "decodedHeight": HEIGHT},
                    "analysisIdentity": {"scaleX": 1.0, "scaleY": 1.0},
                },
                _negative_a_segments(),
                raster,
            ),
        ):
            result = read_floor_vanishing_line(
                b"synthetic", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
            )
        self.assertEqual((result["status"], result["reason"]), (
            "rejected", "no_independent_direction_pair",
        ))
        self.assertEqual(result["diagnostics"]["stableProjectivelyValidPairCount"], 1)
        self.assertEqual(result["diagnostics"]["eligiblePairCount"], 0)

    def test_silent_development_corpus_summary_preserves_the_development_regimes(self):
        rows = development_corpus_summary()
        by_id = {row["fixtureId"]: row for row in rows}

        self.assertEqual(
            list(by_id),
            [
                "P1 finite×finite",
                "P2 finite×directional horizontal",
                "P3 finite×directional oblique",
                "H2 modest finite×directional",
                "ASYM-H1 partitioned finite×directional",
                "N-A duplicate same-side",
                "N-B fragmented same-side",
            ],
        )
        self.assertTrue(all(row["stableProjectivelyValid"] for row in rows))
        for fixture_id in (
            "P1 finite×finite",
            "P2 finite×directional horizontal",
            "P3 finite×directional oblique",
            "H2 modest finite×directional",
            "ASYM-H1 partitioned finite×directional",
        ):
            with self.subTest(fixture_id=fixture_id):
                self.assertTrue(by_id[fixture_id]["usable"])
                self.assertTrue(by_id[fixture_id]["eligibility"]["eligible"])
                self.assertIsNone(by_id[fixture_id]["eligibility"]["rejectionReason"])
        self.assertFalse(by_id["N-A duplicate same-side"]["usable"])
        self.assertEqual(
            by_id["N-A duplicate same-side"]["eligibility"]["rejectionReason"],
            "duplicate_or_interchangeable_families",
        )
        self.assertFalse(by_id["N-B fragmented same-side"]["usable"])
        self.assertEqual(
            by_id["N-B fragmented same-side"]["eligibility"]["rejectionReason"],
            "insufficient_direction_field_separation",
        )


class AfcSr1FamilyPairIndependenceDiagnosticsTests(unittest.TestCase):
    def family(self, model, vp, supporters):
        return Family(
            model,
            np.asarray(vp, dtype=np.float64),
            tuple(supporters),
            np.asarray([1.0] * len(supporters)),
            0.0,
            np.asarray(vp, dtype=np.float64),
        )

    def test_final_family_pair_projection_preserves_indices_order_overlap_and_lengths(self):
        shared = segment(7, (0, 1), (2, 1), length=2.0)
        first_only = segment(3, (0, 2), (3, 2), length=3.0)
        second_only = segment(11, (0, 3), (4, 3), length=4.0)
        first = self.family("finite", [0, 0, 1], [shared, first_only])
        second = self.family("finite", [0, 4, 1], [second_only, shared])
        third = self.family("directional", [1, 0, 0], [segment(15, (1, 5), (5, 5), length=4.0)])

        projection = _family_pair_independence_diagnostics((first, second, third), DIAGONAL)

        self.assertEqual(
            [item["familyIndices"] for item in projection["pairs"]],
            [[0, 1], [0, 2], [1, 2]],
        )
        pair = projection["pairs"][0]
        self.assertEqual(pair["overlap"], {
            "sharedSupporterCount": 1,
            "unionSupporterCount": 3,
            "jaccard": 1 / 3,
            "overlapFractionOfSmaller": 1 / 2,
            "familyASupporterCount": 2,
            "familyBSupporterCount": 2,
        })
        self.assertEqual(pair["exclusiveSupport"], {
            "sharedSupportLengthPx": 2.0,
            "firstOnlySupporterCount": 1,
            "secondOnlySupporterCount": 1,
            "firstOnlySupportLengthPx": 3.0,
            "secondOnlySupportLengthPx": 4.0,
        })
        self.assertEqual(
            pair["predictedDirectionFieldDisagreement"]["onUnionSupporterMidpoints"]["supporterCount"],
            3,
        )
        self.assertEqual(
            pair["predictedDirectionFieldDisagreement"]["onSharedSupporterMidpoints"]["supporterCount"],
            1,
        )
        self.assertEqual(
            [item["familyIndex"] for item in projection["familyOrientationSummaries"]],
            [0, 1, 2],
        )
        json.dumps(projection, allow_nan=False)

    def test_cross_fit_uses_authoritative_v3_residuals_without_changing_own_residuals(self):
        first_supporters = [segment(1, (0, 1), (2, 1), length=2.0)]
        second_supporters = [segment(2, (0, 3), (2, 3), length=2.0)]
        first = self.family("finite", [0, 0, 1], first_supporters)
        second = self.family("finite", [0, 4, 1], second_supporters)
        own_residuals = first.residuals.copy(), second.residuals.copy()

        pair = _family_pair_independence_diagnostics((first, second), DIAGONAL)["pairs"][0]
        expected_first = _residuals(first_supporters, second.model, second.vp, DIAGONAL)
        expected_second = _residuals(second_supporters, first.model, first.vp, DIAGONAL)

        self.assertEqual(
            pair["crossFit"]["firstSupportersAgainstSecond"],
            _residual_summary(first_supporters, second.model, second.vp, DIAGONAL),
        )
        self.assertEqual(
            pair["crossFit"]["secondSupportersAgainstFirst"],
            _residual_summary(second_supporters, first.model, first.vp, DIAGONAL),
        )
        self.assertEqual(pair["crossFit"]["firstSupportersAgainstSecond"]["medianResidualPx"], float(np.median(expected_first)))
        self.assertEqual(pair["crossFit"]["secondSupportersAgainstFirst"]["medianResidualPx"], float(np.median(expected_second)))
        self.assertTrue(np.array_equal(first.residuals, own_residuals[0]))
        self.assertTrue(np.array_equal(second.residuals, own_residuals[1]))

    def test_axial_statistics_and_undirected_wrap_are_rp1_correct(self):
        near_zero = [
            segment(1, (0, 0), (math.cos(math.radians(179)), math.sin(math.radians(179)))),
            segment(2, (0, 0), (math.cos(math.radians(1)), math.sin(math.radians(1)))),
        ]
        summary = _axial_summary(near_zero)
        self.assertAlmostEqual(summary["axialMeanDegrees"], 0.0, places=10)
        self.assertLess(summary["axialIqrDegrees"], 3.0)
        self.assertAlmostEqual(
            _axial_disagreement_degrees(math.radians(179), math.radians(1)),
            2.0,
            places=10,
        )

    def test_direction_field_handles_finite_and_directional_models_projectively(self):
        midpoint = segment(1, (-1, 0), (1, 0))
        finite_a = self.family("finite", [1, 0, 1], [midpoint])
        finite_same = self.family("finite", [1, 0, 1], [midpoint])
        finite_b = self.family("finite", [0, 1, 1], [midpoint])
        directional = self.family("directional", [1, 0, 0], [midpoint])
        directional_opposite = self.family("directional", [-1, 0, 0], [midpoint])
        finite_diagonal = self.family("finite", [1, 1, 1], [midpoint])

        self.assertEqual(_direction_field_disagreement([midpoint], finite_a, finite_same)["medianDegrees"], 0.0)
        self.assertEqual(_direction_field_disagreement([midpoint], directional, directional)["medianDegrees"], 0.0)
        self.assertEqual(_direction_field_disagreement([midpoint], directional, directional_opposite)["medianDegrees"], 0.0)
        self.assertAlmostEqual(_direction_field_disagreement([midpoint], finite_a, finite_b)["medianDegrees"], 90.0)
        self.assertAlmostEqual(_direction_field_disagreement([midpoint], finite_diagonal, directional)["medianDegrees"], 45.0)
        self.assertAlmostEqual(_predicted_direction(midpoint, finite_a), 0.0)
        self.assertAlmostEqual(_predicted_direction(midpoint, directional), 0.0)

    def test_current_style_left_vps_have_small_field_disagreement_at_right_side_evidence(self):
        support = segment(1, (990, 600), (1010, 600))
        first = self.family("finite", [-25.0, 475.0, 1.0], [support])
        second = self.family("finite", [-596.0, 404.0, 1.0], [support])

        # Independent analytic expectation from midpoint-to-VP vectors, not the
        # diagnostic helper under test.
        midpoint = (1000.0, 600.0)
        first_angle = math.atan2(475.0 - midpoint[1], -25.0 - midpoint[0]) % math.pi
        second_angle = math.atan2(404.0 - midpoint[1], -596.0 - midpoint[0]) % math.pi
        expected = math.degrees(min(abs(first_angle - second_angle), math.pi - abs(first_angle - second_angle)))
        actual = _direction_field_disagreement([support], first, second)
        self.assertIsNotNone(actual)
        self.assertAlmostEqual(actual["medianDegrees"], expected, places=12)
        self.assertLess(actual["medianDegrees"], 1.0)

    def test_degenerate_diagnostic_geometry_is_null_without_affecting_projection_serialization(self):
        support = segment(1, (0, 0), (2, 0))
        finite_at_midpoint = self.family("finite", [1, 0, 1], [support])
        directional = self.family("directional", [1, 0, 0], [support])
        # The segment midpoint equals the finite VP, so field direction is undefined.
        self.assertIsNone(_direction_field_disagreement([support], finite_at_midpoint, directional))
        degenerate_directional = self.family("directional", [0, 0, 0], [support])
        self.assertIsNone(_residual_summary([support], "directional", degenerate_directional.vp, DIAGONAL))


if __name__ == "__main__":
    unittest.main()
