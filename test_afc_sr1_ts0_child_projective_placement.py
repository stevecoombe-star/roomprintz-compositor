import copy
import hashlib
import inspect
import json
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import research.afc_sr1_ts0_child_projective_placement as placement


MASK_LABEL = "STRICT_EMPTY_POLYGON_USED_AS_REGISTRATION_EXCLUSION_MASK_ONLY"
POLYGON = [[0.0, 0.9], [1.0, 0.9], [1.0, 1.0], [0.0, 1.0]]


def png_bytes(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise AssertionError("failed to encode PNG")
    return bytes(encoded)


def basis(data: bytes) -> dict:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return placement.image_basis(data, image)


def textured_pair(tx: int = 4, ty: int = -3) -> tuple[bytes, bytes]:
    rng = np.random.default_rng(1234)
    parent = rng.integers(0, 256, (600, 800, 3), dtype=np.uint8)
    for y in range(20, 580, 40):
        cv2.line(parent, (10, y), (790, y), (255, 255, 255), 2)
    child = cv2.warpAffine(
        parent,
        np.float32([[1, 0, tx], [0, 1, ty]]),
        (800, 600),
        flags=cv2.INTER_NEAREST,
    )
    return png_bytes(parent), png_bytes(child)


def lineage(parent: bytes, child: bytes) -> dict:
    return {"parent": basis(parent), "child": basis(child)}


class AfcSr1Ts0ChildPlacementPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent, cls.child = textured_pair()
        cls.lineage = lineage(cls.parent, cls.child)
        cls.usable = placement.place_ts0_child(
            cls.parent, cls.child, POLYGON, MASK_LABEL, cls.lineage
        )

    def run_mocked_pipeline(self, **overrides):
        image = png_bytes(np.zeros((100, 100, 3), dtype=np.uint8))
        bound = lineage(image, image)
        src = np.asarray(
            [[x, y] for y in np.linspace(10, 90, 8) for x in np.linspace(10, 90, 8)],
            dtype=np.float64,
        )
        dst = src + np.asarray([1.0, 2.0])
        defaults = {
            "sift_correspondences": (src, dst, 64, 64),
            "akaze_correspondences": (src, dst, 64, 64),
            "edge_validation": {
                "supportCount": 2000,
                "hitCount": 2000,
                "hitRate": 1.0,
                "refitApplied": False,
            },
        }
        defaults.update(overrides)
        with ExitStack() as stack:
            for name, configured in defaults.items():
                if isinstance(configured, dict) and name != "edge_validation":
                    stack.enter_context(patch.object(placement, name, **configured))
                else:
                    stack.enter_context(
                        patch.object(placement, name, return_value=configured)
                    )
            return placement.place_ts0_child(
                image, image, POLYGON, MASK_LABEL, bound
            )

    def test_frozen_identity_and_translation_only_containment(self):
        self.assertEqual(
            placement.POLICY_VERSION,
            "afc-sr1-ts0-child-projective-placement-policy/v1",
        )
        self.assertEqual(
            placement.SCHEMA_VERSION, "afc-sr1-ts0-child-projective-placement/v1"
        )
        self.assertEqual(placement.TRANSFORM_TYPE, "translation")
        self.assertEqual(placement.TRANSFORM_DIRECTION, "parent_to_child")
        source = Path(
            "research/afc_sr1_ts0_child_projective_placement.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "findHomography",
            "estimateAffine",
            "estimateAffinePartial",
            "RANSAC",
            "maxTranslationMagnitude",
            "cycleValidation",
        ):
            self.assertNotIn(forbidden, source)

    def test_exact_detector_settings_are_observable(self):
        self.assertEqual(
            placement.SIFT_PARAMETERS,
            {
                "nfeatures": 4000,
                "nOctaveLayers": 3,
                "contrastThreshold": 0.04,
                "edgeThreshold": 10,
                "sigma": 1.6,
            },
        )
        self.assertEqual(
            placement.AKAZE_PARAMETERS,
            {
                "descriptor_type": cv2.AKAZE_DESCRIPTOR_MLDB,
                "descriptor_size": 0,
                "descriptor_channels": 3,
                "threshold": 0.001,
                "nOctaves": 4,
                "nOctaveLayers": 4,
                "diffusivity": cv2.KAZE_DIFF_PM_G2,
            },
        )
        source = inspect.getsource(placement._ratio_matches)
        self.assertIn("match.distance < 0.75 * next_match.distance", source)
        self.assertIn("(item.queryIdx, item.trainIdx, item.distance)", source)

    def test_mask_exact_rasterization_and_off_frame_clipping(self):
        polygon = [[-0.1, 0.5], [0.5, 0.5], [1.1, 1.1]]
        actual = placement.usable_structure_mask(20, 30, polygon)
        points = np.asarray(
            [[int(round(x * 30)), int(round(y * 20))] for x, y in polygon],
            dtype=np.int32,
        )
        excluded = np.zeros((20, 30), dtype=np.uint8)
        cv2.fillPoly(excluded, [points], 255)
        excluded = cv2.dilate(
            excluded, np.ones((9, 9), np.uint8), iterations=2
        )
        expected = np.where(excluded > 0, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(actual, expected)

    def test_componentwise_median_and_strict_three_pixel_inlier(self):
        src = np.zeros((5, 2), dtype=np.float64)
        dst = np.asarray([[1, 2], [1, 2], [1, 2], [20, -4], [-10, 30]], dtype=np.float64)
        self.assertEqual(placement.translation_from(src, dst), (1.0, 2.0))
        residuals = placement.translation_residuals(
            np.asarray([[0.0, 0.0], [0.0, 0.0]]),
            np.asarray([[3.0, 0.0], [2.999999, 0.0]]),
            0.0,
            0.0,
        )
        self.assertEqual((residuals < 3.0).tolist(), [False, True])

    def test_holdout_primary_fallback_and_degenerate_paths(self):
        primary = np.asarray(
            [
                [x, y]
                for y in (10.0, 30.0, 55.0, 80.0)
                for x in (10.0, 30.0, 55.0, 80.0)
            ],
            dtype=np.float64,
        )
        fit, validation, method = placement.holdout_split(primary, 100, 100)
        self.assertEqual(method, "4x4_even_odd")
        self.assertEqual((len(fit), len(validation)), (8, 8))
        fallback = np.asarray([[float(x), 10.0] for x in range(20)], dtype=np.float64)
        fit, validation, method = placement.holdout_split(fallback, 100, 100)
        self.assertEqual(method, "median_x_fallback")
        self.assertEqual((len(fit), len(validation)), (10, 10))
        fit, validation, method = placement.holdout_split(fallback[:7], 100, 100)
        self.assertEqual(method, "median_x_fallback")
        self.assertIsNone(fit)
        self.assertIsNone(validation)

    def test_coverage_and_cell_percentiles_match_frozen_definitions(self):
        points = np.asarray(
            [[5, 5], [45, 5], [5, 45], [45, 45], [80, 80]], dtype=np.float64
        )
        coverage = placement.spatial_coverage(points, 100, 100)
        self.assertEqual(coverage["occupiedCells"], 5)
        self.assertEqual(coverage["quadrants"], 2)
        self.assertEqual(coverage["xExtentFraction"], 0.75)
        self.assertEqual(coverage["yExtentFraction"], 0.75)
        residuals = np.asarray([1.0, 2.0, 3.0, 4.0, 9.0])
        self.assertEqual(
            placement.maximum_cell_p90(points, residuals, 100, 100), 9.0
        )
        self.assertEqual(
            placement.percentile90(np.asarray([0.0, 10.0], dtype=np.float64)),
            9.0,
        )

    def test_unequal_dimension_normalized_matrix_is_derived(self):
        self.assertEqual(
            placement.h_norm_of(8.0, -6.0, 800, 600, 400, 300),
            [[2.0, 0.0, 0.02], [0.0, 2.0, -0.02], [0.0, 0.0, 1.0]],
        )

    def test_synthetic_known_translation_passes_independently_of_corpus(self):
        receipt = self.usable
        self.assertEqual((receipt["status"], receipt["reason"]), ("usable", None))
        self.assertEqual(receipt["translationPx"], {"tx": 4.0, "ty": -3.0})
        self.assertEqual(receipt["transformType"], "translation")
        self.assertGreaterEqual(receipt["diagnostics"]["sift"]["finalInliers"], 40)
        self.assertLessEqual(receipt["diagnostics"]["holdout"]["validationP90Px"], 3.5)
        self.assertFalse(receipt["diagnostics"]["akaze"]["refitApplied"])
        self.assertFalse(receipt["diagnostics"]["canny"]["refitApplied"])

    def test_akaze_validates_without_refitting_sift_translation(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        parent = png_bytes(image)
        child = png_bytes(image)
        bound = lineage(parent, child)
        src = np.asarray(
            [[10.0 + 15.0 * (index % 6), 10.0 + 15.0 * (index // 6)] for index in range(48)]
        )
        sift_dst = src + np.asarray([1.0, 0.0])
        akaze_dst = src + np.asarray([2.0, 0.0])
        edge = {"supportCount": 2000, "hitCount": 2000, "hitRate": 1.0, "refitApplied": False}
        with (
            patch.object(
                placement,
                "sift_correspondences",
                return_value=(src, sift_dst, 48, 48),
            ),
            patch.object(
                placement,
                "akaze_correspondences",
                return_value=(src, akaze_dst, 48, 48),
            ),
            patch.object(placement, "edge_validation", return_value=edge),
        ):
            receipt = placement.place_ts0_child(
                parent, child, [[0, 0.95], [1, 0.95], [1, 1]], MASK_LABEL, bound
            )
        self.assertEqual(receipt["status"], "usable")
        self.assertEqual(receipt["translationPx"], {"tx": 1.0, "ty": 0.0})
        self.assertEqual(receipt["diagnostics"]["akaze"]["transferP90Px"], 1.0)

    def test_edge_validation_uses_final_translation_and_mask(self):
        image = np.zeros((120, 160), dtype=np.uint8)
        for x in range(5, 155, 5):
            cv2.line(image, (x, 0), (x, 119), 255, 1)
        child = cv2.warpAffine(
            image, np.float32([[1, 0, 2], [0, 1, 1]]), (160, 120)
        )
        result = placement.edge_validation(
            image, child, 2.0, 1.0, np.full_like(image, 255)
        )
        self.assertGreater(result["supportCount"], 2000)
        self.assertGreaterEqual(result["hitRate"], 0.99)
        self.assertFalse(result["refitApplied"])

    def test_image_lineage_and_mask_are_bound(self):
        receipt = self.usable
        self.assertEqual(receipt["sourceImageBasis"], self.lineage["parent"])
        self.assertEqual(receipt["targetImageBasis"], self.lineage["child"])
        self.assertEqual(receipt["ts0Lineage"], self.lineage)
        mask = receipt["registrationMaskIdentity"]
        self.assertEqual(mask["role"], placement.MASK_ROLE)
        self.assertRegex(mask["maskDigest"], r"^[a-f0-9]{64}$")
        self.assertRegex(mask["parentUsableMaskSha256"], r"^[a-f0-9]{64}$")

    def test_wrong_declared_parent_identity_is_invalid_source(self):
        wrong = copy.deepcopy(self.lineage)
        wrong["parent"]["sha256"] = "0" * 64
        receipt = placement.place_ts0_child(
            self.parent, self.child, POLYGON, MASK_LABEL, wrong
        )
        self.assertEqual(
            (receipt["status"], receipt["reason"]),
            ("rejected", "invalid_source_image"),
        )

    def test_wrong_declared_child_identity_is_invalid_target(self):
        wrong = copy.deepcopy(self.lineage)
        wrong["child"]["byteCount"] += 1
        receipt = placement.place_ts0_child(
            self.parent, self.child, POLYGON, MASK_LABEL, wrong
        )
        self.assertEqual(
            (receipt["status"], receipt["reason"]),
            ("rejected", "invalid_target_image"),
        )

    def test_malformed_or_extra_lineage_structure_is_lineage_mismatch(self):
        for malformed in (
            {},
            {"parent": self.lineage["parent"]},
            {**self.lineage, "extra": True},
            {
                **self.lineage,
                "parent": {**self.lineage["parent"], "extra": True},
            },
        ):
            with self.subTest(lineage=malformed):
                receipt = placement.place_ts0_child(
                    self.parent, self.child, POLYGON, MASK_LABEL, malformed
                )
                self.assertEqual(
                    (receipt["status"], receipt["reason"]),
                    ("rejected", "lineage_mismatch"),
                )

    def test_early_rejection_reasons_are_deterministic_and_digested(self):
        cases = [
            (
                placement.place_ts0_child(
                    b"bad", self.child, POLYGON, MASK_LABEL, self.lineage
                ),
                "invalid_source_image",
            ),
            (
                placement.place_ts0_child(
                    self.parent, b"bad", POLYGON, MASK_LABEL, self.lineage
                ),
                "invalid_target_image",
            ),
            (
                placement.place_ts0_child(
                    self.parent, self.child, None, None, self.lineage
                ),
                "registration_mask_missing",
            ),
        ]
        for receipt, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual((receipt["status"], receipt["reason"]), ("rejected", reason))
                self.assertTrue(placement.receipt_digest_is_valid(receipt))

    def test_runtime_version_mismatch_fails_closed(self):
        with patch.object(placement, "runtime_is_supported", return_value=False):
            receipt = placement.place_ts0_child(
                self.parent, self.child, POLYGON, MASK_LABEL, self.lineage
            )
        self.assertEqual(
            (receipt["status"], receipt["reason"]),
            ("rejected", "deterministic_replay_failed"),
        )
        self.assertIn("opencvVersion", receipt["runtimeIdentity"])
        self.assertIn("numpyVersion", receipt["runtimeIdentity"])

    def test_unsupported_direct_policy_does_not_enter_canonical_evidence(self):
        receipt = placement.place_ts0_child(
            self.parent,
            self.child,
            POLYGON,
            MASK_LABEL,
            self.lineage,
            "unsupported-policy/v9",
        )
        self.assertEqual(
            (receipt["status"], receipt["reason"]),
            ("rejected", "deterministic_replay_failed"),
        )
        preimage = json.loads(receipt["evidenceCanonicalJson"])
        self.assertEqual(preimage["policyVersion"], placement.POLICY_VERSION)
        self.assertNotIn("unsupported-policy/v9", receipt["evidenceCanonicalJson"])

    def test_mocked_degenerate_holdout_reason(self):
        receipt = self.run_mocked_pipeline(
            holdout_split=(None, None, "median_x_fallback")
        )
        self.assertEqual(receipt["reason"], "degenerate_correspondence_geometry")

    def test_mocked_nonfinite_fit_reason(self):
        receipt = self.run_mocked_pipeline(
            translation_from=(float("nan"), 0.0)
        )
        self.assertEqual(receipt["reason"], "translation_not_finite")

    def test_mocked_insufficient_final_inliers_reason(self):
        receipt = self.run_mocked_pipeline(
            translation_residuals={
                "side_effect": [
                    np.zeros(32, dtype=np.float64),
                    np.full(64, 3.0, dtype=np.float64),
                ]
            }
        )
        self.assertEqual(receipt["reason"], "insufficient_correspondence")
        self.assertEqual(receipt["diagnostics"]["sift"]["finalInliers"], 0)

    def test_mocked_fit_residual_reason(self):
        receipt = self.run_mocked_pipeline(
            percentile90={"side_effect": [0.0, 2.000001]}
        )
        self.assertEqual(receipt["reason"], "fit_residual_exceeds_limit")

    def test_mocked_spatial_coverage_reason(self):
        receipt = self.run_mocked_pipeline(
            spatial_coverage={
                "return_value": {
                    "occupiedCells": 3,
                    "quadrants": 2,
                    "xExtentFraction": 0.5,
                    "yExtentFraction": 0.5,
                    "collinearityScore": 0.2,
                }
            }
        )
        self.assertEqual(receipt["reason"], "insufficient_spatial_coverage")

    def test_mocked_cell_drift_reason(self):
        receipt = self.run_mocked_pipeline(maximum_cell_p90=5.000001)
        self.assertEqual(receipt["reason"], "nonprojective_drift_detected")

    def test_mocked_akaze_shortage_and_residual_reasons(self):
        empty = np.zeros((0, 2), dtype=np.float64)
        shortage = self.run_mocked_pipeline(
            akaze_correspondences=(empty, empty.copy(), 64, 64)
        )
        self.assertEqual(shortage["reason"], "insufficient_correspondence")
        residual = self.run_mocked_pipeline(
            translation_residuals={
                "side_effect": [
                    np.zeros(32, dtype=np.float64),
                    np.zeros(64, dtype=np.float64),
                    np.full(64, 3.500001, dtype=np.float64),
                ]
            }
        )
        self.assertEqual(residual["reason"], "validation_residual_exceeds_limit")

    def test_mocked_canny_support_and_hit_reasons(self):
        for edge in (
            {
                "supportCount": 1999,
                "hitCount": 1999,
                "hitRate": 1.0,
                "refitApplied": False,
            },
            {
                "supportCount": 2000,
                "hitCount": 1099,
                "hitRate": 0.5495,
                "refitApplied": False,
            },
        ):
            with self.subTest(edge=edge):
                receipt = self.run_mocked_pipeline(edge_validation=edge)
                self.assertEqual(
                    receipt["reason"], "validation_residual_exceeds_limit"
                )

    def test_reason_taxonomy_is_exact(self):
        self.assertEqual(
            placement.REJECTION_REASONS,
            {
                "invalid_source_image",
                "invalid_target_image",
                "registration_mask_missing",
                "lineage_mismatch",
                "insufficient_correspondence",
                "degenerate_correspondence_geometry",
                "insufficient_spatial_coverage",
                "fit_residual_exceeds_limit",
                "validation_residual_exceeds_limit",
                "nonprojective_drift_detected",
                "translation_not_finite",
                "deterministic_replay_failed",
            },
        )

    def test_canonical_receipt_digest_tamper_and_replay(self):
        first = self.usable
        replay = placement.place_ts0_child(
            self.parent, self.child, POLYGON, MASK_LABEL, self.lineage
        )
        self.assertTrue(placement.receipt_digest_is_valid(first))
        self.assertEqual(first["evidenceCanonicalJson"], replay["evidenceCanonicalJson"])
        self.assertEqual(first["evidenceDigest"], replay["evidenceDigest"])
        preimage = json.loads(first["evidenceCanonicalJson"])
        self.assertNotIn("elapsedMs", preimage)
        self.assertEqual(
            hashlib.sha256(first["evidenceCanonicalJson"].encode()).hexdigest(),
            first["evidenceDigest"]["value"],
        )
        for mutate in (
            lambda value: value["translationPx"].__setitem__("tx", 99.0),
            lambda value: value["H_norm"][0].__setitem__(2, 99.0),
            lambda value: value["diagnostics"]["sift"].__setitem__("finalInliers", 0),
            lambda value: value["sourceImageBasis"].__setitem__("sha256", "0" * 64),
            lambda value: value["evidenceDigest"].__setitem__("value", "0" * 64),
        ):
            tampered = copy.deepcopy(first)
            mutate(tampered)
            self.assertFalse(placement.receipt_digest_is_valid(tampered))

    def test_canonical_numbers_follow_jcs_thresholds(self):
        self.assertEqual(
            placement.canonical_json(
                {"a": 1.0, "b": -0.0, "c": 1e-6, "d": 1e-7, "e": 1e20}
            ),
            '{"a":1,"b":0,"c":0.000001,"d":1e-7,"e":100000000000000000000}',
        )


if __name__ == "__main__":
    unittest.main()
