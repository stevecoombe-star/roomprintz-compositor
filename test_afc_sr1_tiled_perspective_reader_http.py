import ast
import base64
import hashlib
import os
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("GEMINI_API_KEY", "test-no-provider-call")

from fastapi.testclient import TestClient

from main import app
from research.afc_sr1_tiled_perspective_reader import TiledPerspectiveLatticeFailure
from research.afc_sr1_tiled_perspective_reader_http import RESEARCH_PROFILE


FIXTURES = (
    "c-t1.93c4c40764c863246cd58976c5bc242689f6dfb0e71cb952248872daac76da92.png",
    "c-t2.66aa1803fa3cb45516ae639d0bce22eec0afa4fd89ca92e64d412901ce930056.png",
    "c-t3.c81e678c1e908d901fea256f56983fcd253e6bb3dbc421c5a492aad5d056d053.png",
)
ROUTE = "/api/research/afc-sr1/tiled-perspective-reader"


def payload(encoded: bytes) -> dict[str, object]:
    return {
        "researchProfile": RESEARCH_PROFILE,
        "imageBase64": base64.b64encode(encoded).decode("ascii"),
        "claimedIdentity": {
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "byteCount": len(encoded),
            "decodedWidth": 1264,
            "decodedHeight": 848,
            "mimeType": "image/png",
            "orientation": 1,
        },
    }


class AfcSr1TiledPerspectiveReaderHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)
        root = Path(__file__).resolve().parent / "research/fixtures"
        cls.fixture_bytes = tuple((root / fixture).read_bytes() for fixture in FIXTURES)

    def test_c_t1_c_t2_c_t3_are_usable_and_identity_bound(self):
        for encoded in self.fixture_bytes:
            with self.subTest(sha256=hashlib.sha256(encoded).hexdigest()):
                response = self.client.post(ROUTE, json=payload(encoded))
                self.assertEqual(response.status_code, 200)
                result = response.json()
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["readerVersion"], RESEARCH_PROFILE)
                self.assertEqual(result["decodedIdentity"], payload(encoded)["claimedIdentity"])
                self.assertEqual(len(result["authoritativeQuadSourceNormalized"]), 4)
                self.assertEqual(len(result["authoritativeQuadPixel"]), 4)

    def test_sha_mismatch_rejected_before_reader_science(self):
        request = payload(self.fixture_bytes[0])
        request["claimedIdentity"]["sha256"] = "0" * 64
        self._assert_identity_rejected_before_reader(request, "sha256_mismatch")

    def test_byte_count_mismatch_rejected_before_reader_science(self):
        request = payload(self.fixture_bytes[0])
        request["claimedIdentity"]["byteCount"] += 1
        self._assert_identity_rejected_before_reader(request, "byte_count_mismatch")

    def test_decoded_dimensions_mismatch_rejected_before_reader_science(self):
        request = payload(self.fixture_bytes[0])
        request["claimedIdentity"]["decodedWidth"] += 1
        self._assert_identity_rejected_before_reader(request, "decoded_dimensions_mismatch")

    def test_mime_mismatch_rejected_before_reader_science(self):
        request = payload(self.fixture_bytes[0])
        request["claimedIdentity"]["mimeType"] = "image/jpeg"
        self._assert_identity_rejected_before_reader(request, "mime_magic_mismatch")

    def test_lattice_failure_reason_round_trips_with_verified_identity(self):
        request = payload(self.fixture_bytes[0])
        with patch(
            "research.afc_sr1_tiled_perspective_reader_http.read_tiled_perspective_lattice",
            side_effect=TiledPerspectiveLatticeFailure("no_coherent_lattice"),
        ) as reader:
            response = self.client.post(ROUTE, json=request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "failed",
                "reason": "no_coherent_lattice",
                "decodedIdentity": request["claimedIdentity"],
                "readerVersion": RESEARCH_PROFILE,
            },
        )
        reader.assert_called_once()

    def test_adapter_has_no_v3_v4_tr2_or_provider_dependency(self):
        source_path = Path("research/afc_sr1_tiled_perspective_reader_http.py")
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        for forbidden in (
            "research.afc_sr1_tile_floor_reader",
            "research.afc_sr1_tile_floor_reader_v3",
            "research.afc_sr1_tr2_tile_floor_reader_http",
            "research.afc_sr1_ts0_child_projective_placement",
            "google.genai",
            "main",
        ):
            self.assertNotIn(forbidden, imported_modules)

    def _assert_identity_rejected_before_reader(self, request, reason: str):
        with patch(
            "research.afc_sr1_tiled_perspective_reader_http.read_tiled_perspective_lattice"
        ) as reader:
            response = self.client.post(ROUTE, json=request)
        self.assertEqual(response.status_code, 400)
        self.assertIn(reason, response.json()["detail"])
        reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
