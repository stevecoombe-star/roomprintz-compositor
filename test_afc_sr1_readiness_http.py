import os
import unittest
from unittest.mock import patch

os.environ.setdefault("GEMINI_API_KEY", "test-no-provider-call")

from fastapi.testclient import TestClient

from main import app
from research.afc_sr1_readiness import AFC_SR1_READINESS_SCHEMA_VERSION


ROUTE = "/api/research/afc-sr1/readiness"
TRUTHY_VALUES = ("1", "true", "yes", "on")


class AfcSr1ReadinessHttpTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_missing_gates_report_false(self):
        with patch.dict(os.environ, {}, clear=True):
            response = self.client.get(ROUTE)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "schemaVersion": AFC_SR1_READINESS_SCHEMA_VERSION,
                "readerEnabled": False,
                "placementEnabled": False,
            },
        )

    def test_each_frozen_truthy_value_enables_both_gates(self):
        for value in TRUTHY_VALUES:
            with self.subTest(value=value):
                with patch.dict(
                    os.environ,
                    {
                        "AFC_SR1_TR2_READER_ENABLED": value,
                        "AFC_SR1_TS0_CHILD_PLACEMENT_ENABLED": value.upper(),
                    },
                    clear=True,
                ):
                    readiness = self.client.get(ROUTE).json()
                self.assertTrue(readiness["readerEnabled"])
                self.assertTrue(readiness["placementEnabled"])

    def test_falsy_values_and_gate_states_are_independent(self):
        for value in ("", "0", "false", "no", "off", "enabled"):
            with self.subTest(value=value):
                with patch.dict(
                    os.environ,
                    {
                        "AFC_SR1_TR2_READER_ENABLED": value,
                        "AFC_SR1_TS0_CHILD_PLACEMENT_ENABLED": "true",
                    },
                    clear=True,
                ):
                    readiness = self.client.get(ROUTE).json()
                self.assertFalse(readiness["readerEnabled"])
                self.assertTrue(readiness["placementEnabled"])

    def test_readiness_has_no_scientific_side_effects(self):
        with (
            patch.dict(
                os.environ,
                {
                    "AFC_SR1_TR2_READER_ENABLED": "true",
                    "AFC_SR1_TS0_CHILD_PLACEMENT_ENABLED": "true",
                },
                clear=True,
            ),
            patch("main.execute_tile_floor_reader") as reader,
            patch("main.execute_ts0_child_placement") as placement,
        ):
            response = self.client.get(ROUTE)
        self.assertEqual(response.status_code, 200)
        reader.assert_not_called()
        placement.assert_not_called()


if __name__ == "__main__":
    unittest.main()
