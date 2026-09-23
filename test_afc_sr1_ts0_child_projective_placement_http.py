import base64
import copy
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("GEMINI_API_KEY", "test-no-provider-call")

import cv2
import numpy as np
from fastapi.testclient import TestClient

from main import app
from research.afc_sr1_ts0_child_projective_placement import (
    MASK_ROLE,
    POLICY_VERSION,
    image_basis,
)


ROUTE = "/api/research/afc-sr1/ts0-child-projective-placement"


def encoded_png(value: int = 0) -> bytes:
    image = np.full((80, 120, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise AssertionError("failed to encode test image")
    return bytes(encoded)


def identity(data: bytes) -> dict:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    return image_basis(data, image)


class AfcSr1Ts0ChildPlacementHttpTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.parent = encoded_png(0)
        self.child = encoded_png(1)
        self.payload = {
            "policyVersion": POLICY_VERSION,
            "parentImageBase64": base64.b64encode(self.parent).decode("ascii"),
            "childImageBase64": base64.b64encode(self.child).decode("ascii"),
            "registrationExclusion": {
                "coordinateSpace": "source-normalized/v1",
                "role": MASK_ROLE,
                "evidenceLabel": "STRICT_EMPTY_POLYGON_USED_AS_REGISTRATION_EXCLUSION_MASK_ONLY",
                "polygon": [[0.0, 0.8], [1.0, 0.8], [1.0, 1.0]],
            },
            "ts0Lineage": {
                "parent": identity(self.parent),
                "child": identity(self.child),
            },
        }

    def post_enabled(self, payload=None):
        with patch.dict(
            os.environ, {"AFC_SR1_TS0_CHILD_PLACEMENT_ENABLED": "true"}, clear=False
        ):
            return self.client.post(ROUTE, json=self.payload if payload is None else payload)

    def test_route_is_env_gated_and_not_advertised_when_disabled(self):
        with patch.dict(
            os.environ, {"AFC_SR1_TS0_CHILD_PLACEMENT_ENABLED": "false"}, clear=False
        ):
            response = self.client.post(ROUTE, json=self.payload)
        self.assertEqual(response.status_code, 404)

    def test_request_accepts_exact_fields_and_returns_placement_receipt_only(self):
        expected = {
            "schemaVersion": "afc-sr1-ts0-child-projective-placement/v1",
            "status": "rejected",
            "reason": "insufficient_correspondence",
        }
        with patch(
            "research.afc_sr1_ts0_child_projective_placement_http.place_ts0_child",
            return_value=expected,
        ) as estimator:
            response = self.post_enabled()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), expected)
        estimator.assert_called_once()
        kwargs = estimator.call_args.kwargs
        self.assertEqual(kwargs["parent_bytes"], self.parent)
        self.assertEqual(kwargs["child_bytes"], self.child)
        self.assertEqual(kwargs["policy_version"], POLICY_VERSION)

    def test_extra_and_forbidden_semantic_fields_fail_contract(self):
        forbidden = (
            ("line", {"a": 0, "b": 1, "c": 0}),
            ("seamT", 0.5),
            ("GT0", 0.5),
            ("anchor", "NL"),
            ("FOV", 60),
            ("width", 4),
            ("depth", 5),
            ("Track1a", {}),
            ("transformType", "similarity"),
            ("homography", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        )
        for name, value in forbidden:
            with self.subTest(name=name):
                self.assertEqual(
                    self.post_enabled({**self.payload, name: value}).status_code, 422
                )

    def test_missing_required_top_level_fields_fail_contract(self):
        for name in tuple(self.payload):
            with self.subTest(name=name):
                payload = dict(self.payload)
                del payload[name]
                self.assertEqual(self.post_enabled(payload).status_code, 422)

    def test_wrong_policy_role_label_and_nested_extras_fail_contract(self):
        cases = []
        cases.append({**self.payload, "policyVersion": "latest"})
        wrong_role = copy.deepcopy(self.payload)
        wrong_role["registrationExclusion"]["role"] = "placement_authority"
        cases.append(wrong_role)
        wrong_label = copy.deepcopy(self.payload)
        wrong_label["registrationExclusion"]["evidenceLabel"] = "authoritative"
        cases.append(wrong_label)
        extra_lineage = copy.deepcopy(self.payload)
        extra_lineage["ts0Lineage"]["runId"] = "x"
        cases.append(extra_lineage)
        for payload in cases:
            self.assertEqual(self.post_enabled(payload).status_code, 422)

    def test_off_frame_polygon_is_accepted_without_semantic_clamping(self):
        payload = copy.deepcopy(self.payload)
        payload["registrationExclusion"]["polygon"] = [
            [-0.2, 0.8],
            [1.2, 0.8],
            [1.2, 1.1],
        ]
        with patch(
            "research.afc_sr1_ts0_child_projective_placement_http.place_ts0_child",
            return_value={"status": "rejected", "reason": "insufficient_correspondence"},
        ) as estimator:
            response = self.post_enabled(payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(estimator.call_args.kwargs["polygon_norm"][0], (-0.2, 0.8))

    def test_malformed_base64_is_transport_failure(self):
        response = self.post_enabled({**self.payload, "parentImageBase64": "not base64"})
        self.assertEqual(response.status_code, 400)

    def test_raw_bytes_must_match_expected_ts0_lineage(self):
        payload = copy.deepcopy(self.payload)
        payload["ts0Lineage"]["child"]["sha256"] = "0" * 64
        response = self.post_enabled(payload)
        self.assertEqual(response.status_code, 200)
        receipt = response.json()
        self.assertEqual(
            (receipt["status"], receipt["reason"]),
            ("rejected", "invalid_target_image"),
        )

    def test_numeric_orientation_one_is_required(self):
        self.assertEqual(self.payload["ts0Lineage"]["parent"]["orientation"], 1)
        payload = copy.deepcopy(self.payload)
        payload["ts0Lineage"]["parent"]["orientation"] = 2
        self.assertEqual(self.post_enabled(payload).status_code, 422)

    def test_lineage_identity_requires_dimensions_bytes_and_orientation(self):
        for name in (
            "sha256",
            "byteCount",
            "decodedWidth",
            "decodedHeight",
            "orientation",
        ):
            with self.subTest(name=name):
                payload = copy.deepcopy(self.payload)
                del payload["ts0Lineage"]["parent"][name]
                self.assertEqual(self.post_enabled(payload).status_code, 422)


if __name__ == "__main__":
    unittest.main()
