import base64
import copy
import hashlib
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("GEMINI_API_KEY", "test-no-provider-call")

import cv2
import numpy as np
from fastapi.testclient import TestClient

from main import app
from research.afc_sr1_tr2_tile_floor_reader_http import (
    MAX_BASE64_PAYLOAD_CHARS,
    POLICY_VERSION,
    RESEARCH_PROFILE,
    RESULT_SCHEMA_VERSION,
    V2_POLICY_VERSION,
    V2_RESEARCH_PROFILE,
    V2_RESULT_SCHEMA_VERSION,
    V4_POLICY_VERSION,
    V4_RESEARCH_PROFILE,
    V4_RESULT_SCHEMA_VERSION,
)


def encoded_png(width: int, height: int) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise AssertionError("failed to encode test image")
    return bytes(buffer)


class AfcSr1Tr2TileFloorReaderHttpTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.image = encoded_png(1264, 848)
        self.payload = {
            "researchProfile": RESEARCH_PROFILE,
            "policyVersion": POLICY_VERSION,
            "imageBase64": base64.b64encode(self.image).decode("ascii"),
            "roi": {
                "coordinateSpace": "source-normalized/v1",
                "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            },
        }

    def post_enabled(self, payload=None):
        with patch.dict(os.environ, {"AFC_SR1_TR2_READER_ENABLED": "true"}, clear=False):
            return self.client.post(
                "/api/research/afc-sr1/tile-floor-vanishing-line",
                json=self.payload if payload is None else payload,
            )

    def test_disabled_route_is_not_advertised(self):
        with patch.dict(os.environ, {"AFC_SR1_TR2_READER_ENABLED": "false"}, clear=False):
            response = self.client.post("/api/research/afc-sr1/tile-floor-vanishing-line", json=self.payload)
        self.assertEqual(response.status_code, 404)

    def test_forbidden_semantic_fields_fail_at_request_contract(self):
        for name, value in (("truncatedAnchor", "NL"), ("seamT", 0.5)):
            payload = {**self.payload, name: value}
            response = self.post_enabled(payload)
            self.assertEqual(response.status_code, 422, name)

    def test_profile_and_policy_contracts_have_no_fallback(self):
        for name, value in (
            ("researchProfile", "latest"),
            ("policyVersion", "latest"),
        ):
            payload = {**self.payload, name: value}
            self.assertEqual(self.post_enabled(payload).status_code, 422, name)
        for name in ("researchProfile", "policyVersion"):
            payload = dict(self.payload)
            del payload[name]
            self.assertEqual(self.post_enabled(payload).status_code, 422, name)

    def test_v2_pair_rejects_cross_pair_and_below_reference_is_bound(self):
        cross_pair = {**self.payload, "researchProfile": V2_RESEARCH_PROFILE}
        self.assertEqual(self.post_enabled(cross_pair).status_code, 422)
        v2 = {
            **self.payload,
            "researchProfile": V2_RESEARCH_PROFILE,
            "policyVersion": V2_POLICY_VERSION,
            "imageBase64": base64.b64encode(encoded_png(1000, 800)).decode("ascii"),
        }
        response = self.post_enabled(v2)
        self.assertEqual(response.status_code, 200)
        receipt = response.json()
        self.assertEqual(receipt["schemaVersion"], V2_RESULT_SCHEMA_VERSION)
        self.assertEqual(
            (receipt["status"], receipt["reason"]), ("rejected", "below_reference_analysis_long_edge")
        )
        self.assertEqual(receipt["runtimeIdentity"]["readerModuleVersion"], "afc-sr1-tile-floor-reader/v2")
        self.assertIn("analysisIdentity", receipt)
        self.assertEqual(receipt["analysisIdentity"]["resampler"], "identity")
        self.assert_receipt_is_bound(receipt)

    def test_v4_pair_is_exact_and_binds_projective_selection_evidence(self):
        v4 = {**self.payload, "researchProfile": V4_RESEARCH_PROFILE, "policyVersion": V4_POLICY_VERSION}
        self.assertEqual(self.post_enabled({**v4, "researchProfile": V2_RESEARCH_PROFILE}).status_code, 422)
        self.assertEqual(self.post_enabled({**self.payload, "policyVersion": V4_POLICY_VERSION}).status_code, 422)
        reader_result = {
            "status": "usable",
            "policyVersion": V4_POLICY_VERSION,
            "analysisImage": {"decodedWidth": 1264, "decodedHeight": 848},
            "analysisIdentity": {
                "mode": "identity", "analysisWidth": 1264, "analysisHeight": 848,
                "scaleX": 1.0, "scaleY": 1.0, "referenceLongEdge": 1264,
                "resampler": "identity", "pixelFormat": "bgr8", "pixelBufferSha256": "a" * 64,
            },
            "floorVanishingLinePixel": {"a": 0.0, "b": 1.0, "c": -400.0},
            "diagnostics": {
                "segmentCounts": {"raw": 20, "admittedAllNineInside": 12},
                "candidateDiscovery": {"hypothesisStrategy": "exhaustive", "finalFamilies": []},
                "validFamilyCount": 2,
                "candidateUnorderedPairCount": 1,
                "stableProjectivelyValidPairCount": 1,
                "eligiblePairCount": 1,
                "validPairCount": 1,
                "validPairUniverse": [{
                    "familyIndices": [0, 1],
                    "independentDirectionEligibility": {
                        "eligible": True,
                        "failedStage": None,
                        "rejectionReason": None,
                        "overlapFractionOfSmaller": 0.0,
                        "firstSupportCount": 8,
                        "secondSupportCount": 8,
                        "firstInlierBandCount": 0,
                        "secondInlierBandCount": 0,
                        "firstInlierBandFraction": 0.0,
                        "secondInlierBandFraction": 0.0,
                        "firstRegionMedianDegrees": 11.0,
                        "secondRegionMedianDegrees": 17.0,
                        "strongRegionMedianDegrees": 17.0,
                    },
                }],
                "independentDirectionEligibilityRejectedPairs": [],
                "winningPair": {
                    "familyIndices": [0, 1],
                    "basinSupport": 1,
                    "families": [],
                    "stability": {"maxSplitVsFullProbeDistancePx": 1.0},
                    "independentDirectionEligibility": {
                        "eligible": True,
                        "failedStage": None,
                        "rejectionReason": None,
                        "overlapFractionOfSmaller": 0.0,
                        "firstSupportCount": 8,
                        "secondSupportCount": 8,
                        "firstInlierBandCount": 0,
                        "secondInlierBandCount": 0,
                        "firstInlierBandFraction": 0.0,
                        "secondInlierBandFraction": 0.0,
                        "firstRegionMedianDegrees": 11.0,
                        "secondRegionMedianDegrees": 17.0,
                        "strongRegionMedianDegrees": 17.0,
                    },
                },
            },
        }
        with patch("research.afc_sr1_tr2_tile_floor_reader_http.read_floor_vanishing_line_v3", return_value=reader_result):
            receipt = self.post_enabled(v4).json()
        reader_with_support_geometry = {
            **reader_result,
            "diagnostics": {
                **reader_result["diagnostics"],
                "familySupportGeometry": {
                    "coordinateSpace": "analysis-pixel/v1",
                    "authority": "none",
                    "role": "observation_only",
                    "excludedFromCanonicalEvidence": True,
                    "segments": [
                        {"detectorIndex": 17, "x1": 10.25, "y1": 20.5, "x2": 30.75, "y2": 40.125},
                    ],
                    "families": [{"familyIndex": 0, "supporterDetectorIndices": [17]}],
                },
            },
        }
        reader_with_pair_independence = {
            **reader_with_support_geometry,
            "diagnostics": {
                **reader_with_support_geometry["diagnostics"],
                "familyPairIndependenceDiagnostics": {
                    "contractVersion": "afc-sr1-family-pair-independence-diagnostics/v1",
                    "coordinateSpace": "analysis-pixel/v1",
                    "authority": "none",
                    "role": "observation_only",
                    "excludedFromCanonicalEvidence": True,
                    "familyOrientationSummaries": [{
                        "familyIndex": 0,
                        "supporterCount": 1,
                        "axialMeanDegrees": 8.0,
                        "axialMedianDegrees": 8.0,
                        "axialCircularStdDevDegrees": 0.0,
                        "axialIqrDegrees": 0.0,
                    }],
                    "pairs": [],
                },
            },
        }
        with patch(
            "research.afc_sr1_tr2_tile_floor_reader_http.read_floor_vanishing_line_v3",
            return_value=reader_with_pair_independence,
        ):
            with_support_geometry = self.post_enabled(v4).json()
        self.assertEqual(receipt["schemaVersion"], V4_RESULT_SCHEMA_VERSION)
        self.assertEqual(receipt["runtimeIdentity"]["readerModuleVersion"], "afc-sr1-tile-floor-reader/v4")
        self.assertIn("winningPair", json.loads(receipt["evidenceCanonicalJson"])["diagnostics"])
        self.assert_receipt_is_bound(receipt)
        self.assertEqual(
            with_support_geometry["diagnostics"]["familySupportGeometry"],
            reader_with_support_geometry["diagnostics"]["familySupportGeometry"],
        )
        canonical = json.loads(with_support_geometry["evidenceCanonicalJson"])
        self.assertNotIn("familySupportGeometry", canonical["diagnostics"])
        self.assertNotIn("familyPairIndependenceDiagnostics", canonical["diagnostics"])
        self.assertEqual(canonical["diagnostics"]["stableProjectivelyValidPairCount"], 1)
        self.assertEqual(canonical["diagnostics"]["eligiblePairCount"], 1)
        self.assertEqual(
            canonical["diagnostics"]["validPairUniverse"][0]["independentDirectionEligibility"],
            reader_result["diagnostics"]["validPairUniverse"][0]["independentDirectionEligibility"],
        )
        self.assertEqual(with_support_geometry["evidenceCanonicalJson"], receipt["evidenceCanonicalJson"])
        self.assertEqual(with_support_geometry["evidenceDigest"], receipt["evidenceDigest"])
        self.assertEqual(with_support_geometry["floorVanishingLinePixel"], receipt["floorVanishingLinePixel"])
        self.assertEqual(
            with_support_geometry["diagnostics"]["familyPairIndependenceDiagnostics"],
            reader_with_pair_independence["diagnostics"]["familyPairIndependenceDiagnostics"],
        )
        self.assertEqual(
            with_support_geometry["diagnostics"]["winningPair"],
            receipt["diagnostics"]["winningPair"],
        )
        self.assertEqual(
            with_support_geometry["diagnostics"].get("validPairUniverse"),
            receipt["diagnostics"].get("validPairUniverse"),
        )
        changed_sidecar = copy.deepcopy(reader_with_pair_independence)
        changed_sidecar["diagnostics"]["familySupportGeometry"]["segments"][0]["x1"] = 999.0
        with patch(
            "research.afc_sr1_tr2_tile_floor_reader_http.read_floor_vanishing_line_v3",
            return_value=changed_sidecar,
        ):
            changed_sidecar_receipt = self.post_enabled(v4).json()
        self.assertEqual(
            changed_sidecar_receipt["evidenceCanonicalJson"],
            receipt["evidenceCanonicalJson"],
        )
        changed_eligibility = copy.deepcopy(reader_result)
        for pair in (
            changed_eligibility["diagnostics"]["validPairUniverse"][0],
            changed_eligibility["diagnostics"]["winningPair"],
        ):
            pair["independentDirectionEligibility"]["strongRegionMedianDegrees"] = 18.0
        with patch(
            "research.afc_sr1_tr2_tile_floor_reader_http.read_floor_vanishing_line_v3",
            return_value=changed_eligibility,
        ):
            changed_eligibility_receipt = self.post_enabled(v4).json()
        self.assertNotEqual(
            changed_eligibility_receipt["evidenceCanonicalJson"],
            receipt["evidenceCanonicalJson"],
        )
        self.assertNotEqual(
            changed_eligibility_receipt["evidenceDigest"],
            receipt["evidenceDigest"],
        )

    def test_v4_invalid_image_early_rejection_may_omit_analysis_identity(self):
        v4 = {
            **self.payload,
            "researchProfile": V4_RESEARCH_PROFILE,
            "policyVersion": V4_POLICY_VERSION,
            "imageBase64": base64.b64encode(b"not an image").decode("ascii"),
        }
        receipt = self.post_enabled(v4).json()
        self.assertEqual((receipt["status"], receipt["reason"]), ("rejected", "invalid_input_image"))
        self.assertNotIn("analysisIdentity", receipt)
        self.assertNotIn("analysisIdentity", json.loads(receipt["evidenceCanonicalJson"]))
        self.assert_receipt_is_bound(receipt)

    def test_v4_no_independent_direction_pair_is_canonical_bound(self):
        v4 = {
            **self.payload,
            "researchProfile": V4_RESEARCH_PROFILE,
            "policyVersion": V4_POLICY_VERSION,
        }
        eligibility = {
            "familyIndices": [0, 1],
            "eligible": False,
            "failedStage": 2,
            "rejectionReason": "insufficient_direction_field_separation",
            "overlapFractionOfSmaller": 0.0,
            "firstSupportCount": 8,
            "secondSupportCount": 8,
            "firstInlierBandCount": 0,
            "secondInlierBandCount": 0,
            "firstInlierBandFraction": 0.0,
            "secondInlierBandFraction": 0.0,
            "firstRegionMedianDegrees": 1.0,
            "secondRegionMedianDegrees": 2.0,
            "strongRegionMedianDegrees": 2.0,
        }
        reader_result = {
            "status": "rejected",
            "policyVersion": V4_POLICY_VERSION,
            "reason": "no_independent_direction_pair",
            "diagnostics": {
                "segmentCounts": {"raw": 20, "admittedAllNineInside": 16},
                "candidateDiscovery": {"hypothesisStrategy": "exhaustive", "finalFamilies": []},
                "validFamilyCount": 2,
                "candidateUnorderedPairCount": 1,
                "stableProjectivelyValidPairCount": 1,
                "eligiblePairCount": 0,
                "validPairCount": 0,
                "invalidPairs": [],
                "independentDirectionEligibilityRejectedPairs": [eligibility],
                "validPairUniverse": [],
            },
        }
        with patch(
            "research.afc_sr1_tr2_tile_floor_reader_http.read_floor_vanishing_line_v3",
            return_value=reader_result,
        ):
            receipt = self.post_enabled(v4).json()
        canonical = json.loads(receipt["evidenceCanonicalJson"])
        self.assertEqual((receipt["status"], receipt["reason"]), (
            "rejected", "no_independent_direction_pair",
        ))
        self.assertEqual(canonical["reason"], "no_independent_direction_pair")
        self.assertEqual(
            canonical["diagnostics"]["independentDirectionEligibilityRejectedPairs"],
            [eligibility],
        )
        self.assert_receipt_is_bound(receipt)

    def test_v4_below_reference_and_source_too_large_rejections_bind_analysis_identity(self):
        cases = (
            (encoded_png(1000, 800), "below_reference_analysis_long_edge"),
            (encoded_png(8193, 2), "source_raster_too_large"),
        )
        for image, reason in cases:
            with self.subTest(reason=reason):
                v4 = {
                    **self.payload,
                    "researchProfile": V4_RESEARCH_PROFILE,
                    "policyVersion": V4_POLICY_VERSION,
                    "imageBase64": base64.b64encode(image).decode("ascii"),
                }
                receipt = self.post_enabled(v4).json()
                self.assertEqual((receipt["status"], receipt["reason"]), ("rejected", reason))
                self.assertIn("analysisIdentity", receipt)
                self.assertEqual(receipt["analysisIdentity"]["resampler"], "identity")
                self.assert_receipt_is_bound(receipt)

    def test_v4_rejected_receipt_with_analysis_raster_replays_deterministically(self):
        v4 = {
            **self.payload,
            "researchProfile": V4_RESEARCH_PROFILE,
            "policyVersion": V4_POLICY_VERSION,
        }
        first = self.post_enabled(v4).json()
        second = self.post_enabled(v4).json()
        self.assertEqual((first["status"], first["reason"]), ("rejected", "insufficient_segments"))
        self.assertIn("analysisIdentity", first)
        self.assertEqual(first["evidenceCanonicalJson"], second["evidenceCanonicalJson"])
        self.assertEqual(first["evidenceDigest"], second["evidenceDigest"])
        self.assert_receipt_is_bound(first)

    def test_v4_unsupported_policy_and_malformed_input_fail_closed(self):
        v4 = {
            **self.payload,
            "researchProfile": V4_RESEARCH_PROFILE,
            "policyVersion": V4_POLICY_VERSION,
        }
        self.assertEqual(self.post_enabled({**v4, "policyVersion": "afc-sr1-ts2-extractor-policy/v3"}).status_code, 422)
        self.assertEqual(self.post_enabled({**v4, "imageBase64": ""}).status_code, 422)
        self.assertEqual(self.post_enabled({**v4, "unexpected": True}).status_code, 422)

    def test_v4_endpoint_executes_reader_once_without_provider_authority(self):
        v4 = {
            **self.payload,
            "researchProfile": V4_RESEARCH_PROFILE,
            "policyVersion": V4_POLICY_VERSION,
        }
        reader_result = {
            "status": "rejected",
            "policyVersion": V4_POLICY_VERSION,
            "reason": "invalid_roi",
            "diagnostics": {
                "analysisImage": {"decodedWidth": 1264, "decodedHeight": 848},
            },
        }
        with patch(
            "research.afc_sr1_tr2_tile_floor_reader_http.read_floor_vanishing_line_v3",
            return_value=reader_result,
        ) as reader:
            response = self.post_enabled(v4)
        self.assertEqual(response.status_code, 200)
        reader.assert_called_once()
        self.assertEqual(response.json()["status"], "rejected")

    def test_invalid_image_is_a_reader_rejection(self):
        payload = {**self.payload, "imageBase64": base64.b64encode(b"not an image").decode("ascii")}
        response = self.post_enabled(payload)
        self.assertEqual(response.status_code, 200)
        receipt = response.json()
        self.assertEqual((receipt["status"], receipt["reason"]), ("rejected", "invalid_input_image"))
        self.assertIsNone(receipt["imageIdentity"]["decodedWidth"])
        self.assertNotIn("floorVanishingLinePixel", receipt)
        self.assert_receipt_is_bound(receipt)

    def test_malformed_base64_is_a_transport_failure(self):
        response = self.post_enabled({**self.payload, "imageBase64": "not base64!"})
        self.assertEqual(response.status_code, 400)

    def test_unsupported_grid_remains_a_reader_rejection(self):
        wrong_grid = encoded_png(640, 480)
        response = self.post_enabled(
            {**self.payload, "imageBase64": base64.b64encode(wrong_grid).decode("ascii")}
        )
        self.assertEqual(response.status_code, 200)
        receipt = response.json()
        self.assertEqual((receipt["status"], receipt["reason"]), ("rejected", "unsupported_analysis_grid"))
        self.assertEqual(receipt["imageIdentity"]["decodedWidth"], 640)
        self.assertEqual(receipt["imageIdentity"]["decodedHeight"], 480)
        self.assert_receipt_is_bound(receipt)

    def test_data_url_and_raw_base64_preserve_identical_evidence(self):
        raw = self.post_enabled().json()
        data_url = "data:image/png;base64," + self.payload["imageBase64"]
        data = self.post_enabled({**self.payload, "imageBase64": data_url}).json()
        self.assertEqual(raw["evidenceCanonicalJson"], data["evidenceCanonicalJson"])
        self.assertEqual(raw["evidenceDigest"], data["evidenceDigest"])

    def test_oversized_base64_is_rejected_before_reader_execution(self):
        payload = {**self.payload, "imageBase64": "A" * (MAX_BASE64_PAYLOAD_CHARS + 1)}
        with patch(
            "research.afc_sr1_tr2_tile_floor_reader_http.read_floor_vanishing_line"
        ) as reader:
            response = self.post_enabled(payload)
        self.assertEqual(response.status_code, 413)
        reader.assert_not_called()

    def test_usable_receipt_replays_and_preserves_python_float_values(self):
        expected_line = {
            "a": 0.004509195231282803,
            "b": 0.9999898335275046,
            "c": -427.2113127729756,
        }
        reader_result = {
            "status": "usable",
            "policyVersion": POLICY_VERSION,
            "analysisImage": {"decodedWidth": 1264, "decodedHeight": 848},
            "floorVanishingLinePixel": expected_line,
            "diagnostics": {
                "segmentCounts": {"raw": 823, "admittedAllNineInside": 87},
                "firstFamily": {
                    "support_count": 28,
                    "median_residual_px": 1.4694300267944982,
                    "p90_residual_px": 2.6624173779097644,
                    "hypothesis_strategy": "exhaustive",
                },
                "secondFamily": {
                    "support_count": 27,
                    "median_residual_px": 1.0754893568541775,
                    "p90_residual_px": 3.1425278556526566,
                    "hypothesis_strategy": "exhaustive",
                },
                "stability": {"max_split_vs_full_probe_distance_px": 1.4396801707106306},
            },
        }
        with patch(
            "research.afc_sr1_tr2_tile_floor_reader_http.read_floor_vanishing_line",
            return_value=reader_result,
        ):
            first = self.post_enabled().json()
            second = self.post_enabled().json()
        self.assertEqual(first["floorVanishingLinePixel"], expected_line)
        self.assertEqual(first["evidenceCanonicalJson"], second["evidenceCanonicalJson"])
        self.assertEqual(first["evidenceDigest"], second["evidenceDigest"])
        self.assert_receipt_is_bound(first)

    def test_adapter_containment_has_no_provider_generation_or_storage_calls(self):
        source = Path("research/afc_sr1_tr2_tile_floor_reader_http.py").read_text(encoding="utf-8").lower()
        for forbidden in (
            "google.genai",
            "genai.client",
            "stage-run",
            "supabase",
            "usage_record",
            "tile_grid_scaffold",
            "truncatedanchor",
            "seamt",
            "oppositeendpoint",
        ):
            self.assertNotIn(forbidden, source, forbidden)

    def assert_receipt_is_bound(self, receipt):
        expected_schema = (
            V4_RESULT_SCHEMA_VERSION if receipt["policyVersion"] == V4_POLICY_VERSION
            else V2_RESULT_SCHEMA_VERSION if receipt["policyVersion"] == V2_POLICY_VERSION
            else RESULT_SCHEMA_VERSION
        )
        self.assertEqual(receipt["schemaVersion"], expected_schema)
        canonical = receipt["evidenceCanonicalJson"]
        self.assertEqual(
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            receipt["evidenceDigest"]["value"],
        )
        preimage = json.loads(canonical)
        self.assertEqual(preimage["status"], receipt["status"])
        self.assertEqual(preimage["image"], receipt["imageIdentity"])
        self.assertEqual(preimage["roi"], receipt["roiIdentity"])
        self.assertEqual(preimage["runtime"], receipt["runtimeIdentity"])
        if "analysisIdentity" in receipt:
            self.assertEqual(preimage["analysisIdentity"], receipt["analysisIdentity"])
        else:
            self.assertNotIn("analysisIdentity", preimage)
        if receipt["status"] == "usable":
            self.assertEqual(preimage["floorVanishingLinePixel"], receipt["floorVanishingLinePixel"])
        else:
            self.assertEqual(preimage["reason"], receipt["reason"])


if __name__ == "__main__":
    unittest.main()
