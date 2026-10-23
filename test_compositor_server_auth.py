import logging
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("GEMINI_API_KEY", "test-no-provider-call")

from fastapi.testclient import TestClient

import main
from compositor_server_auth import (
    PUBLIC_PATHS,
    SECRET_ENV,
    authorize,
    configured_secret,
    credentials_match,
)
from compositor_test_auth import FIXTURE_SECRET, auth_headers

PAID_PATHS = (
    "/stage-room",
    "/vibode/compose",
    "/vibode/vibe",
    "/vibode/full_vibe",
    "/api/vibode/stage-run",
    "/api/vibode/edit-run",
    "/api/vibode/user-skus/ingest",
    "/vibode/remove",
    "/vibode/swap",
    "/vibode/rotate",
)
OTHER_GATED_PATHS = (
    "/api/research/afc-sr1/readiness",
    "/api/research/afc-sr1/tile-floor-vanishing-line",
    "/api/research/afc-sr1/tiled-perspective-reader",
    "/api/research/afc-sr1/ts0-child-projective-placement",
    "/api/vibode/paste-to-place/cancel",
    "/openapi.json",
)


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class CompositorServerAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ[SECRET_ENV] = FIXTURE_SECRET
        self.client = TestClient(main.app)
        self.provider = patch.object(
            main.client.models,
            "generate_content",
            side_effect=AssertionError("provider call"),
        )
        self.generate_content = self.provider.start()
        self.logs = _Capture()
        logging.getLogger().addHandler(self.logs)

    def tearDown(self) -> None:
        self.provider.stop()
        logging.getLogger().removeHandler(self.logs)

    def test_public_paths_are_only_root_and_health(self) -> None:
        self.assertEqual(PUBLIC_PATHS, frozenset({"/", "/health"}))

    def test_credentials_match_is_constant_shape(self) -> None:
        secret = configured_secret()
        self.assertTrue(credentials_match(secret, secret))
        self.assertFalse(credentials_match("", secret))
        self.assertFalse(credentials_match("incorrect", secret))
        self.assertFalse(credentials_match(secret, ""))
        self.assertTrue(authorize("/health", "", ""))
        self.assertFalse(authorize("/stage-room", "", secret))

    def test_health_and_root_stay_open_without_a_header(self) -> None:
        health = self.client.get("/health")
        root = self.client.get("/")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(root.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok"})
        self.generate_content.assert_not_called()

    def test_missing_header_rejects_paid_routes_before_the_provider(self) -> None:
        for path in PAID_PATHS:
            with self.subTest(path=path):
                response = self.client.post(path, json={})
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.json(), {"detail": "Unauthorized"})
        self.generate_content.assert_not_called()

    def test_incorrect_header_rejects_before_the_provider(self) -> None:
        response = self.client.post(
            "/stage-room",
            headers={"X-Vibode-Compositor-Auth": "incorrect"},
            json={"imageBase64": "aaaa", "styleId": "modern"},
        )
        self.assertEqual(response.status_code, 401)
        self.generate_content.assert_not_called()

    def test_correct_header_passes_authentication(self) -> None:
        response = self.client.post(
            "/stage-room",
            headers=auth_headers(),
            json={},
        )
        self.assertEqual(response.status_code, 422)
        self.generate_content.assert_not_called()
        readiness = self.client.get(
            "/api/research/afc-sr1/readiness",
            headers=auth_headers(),
        )
        self.assertEqual(readiness.status_code, 200)
        self.generate_content.assert_not_called()

    def test_missing_server_secret_fails_closed(self) -> None:
        with patch.dict(os.environ, {SECRET_ENV: "   "}, clear=False):
            response = self.client.post(
                "/vibode/remove",
                headers={"X-Vibode-Compositor-Auth": FIXTURE_SECRET},
                json={},
            )
        self.assertEqual(response.status_code, 401)
        self.generate_content.assert_not_called()

    def test_other_routes_are_gated_and_do_not_call_the_provider(self) -> None:
        for path in OTHER_GATED_PATHS:
            with self.subTest(path=path):
                response = self.client.request("POST" if path != "/openapi.json" and path != "/api/research/afc-sr1/readiness" else "GET", path)
                self.assertEqual(response.status_code, 401)
        self.generate_content.assert_not_called()

    def test_logs_and_responses_do_not_contain_the_fixture(self) -> None:
        self.client.post("/stage-room", json={})
        rendered = "\n".join(self.logs.messages)
        self.assertIs(FIXTURE_SECRET in rendered, False)
        self.assertIs(SECRET_ENV in rendered, False)


if __name__ == "__main__":
    unittest.main()
