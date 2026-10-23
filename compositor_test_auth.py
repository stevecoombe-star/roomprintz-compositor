"""Shared fixture for compositor HTTP tests. Not a production secret."""

import os

FIXTURE_SECRET = "sec19b-local-fixture"
os.environ.setdefault("VIBODE_COMPOSITOR_AUTH_SECRET", FIXTURE_SECRET)


def auth_headers() -> dict[str, str]:
    return {"X-Vibode-Compositor-Auth": os.environ["VIBODE_COMPOSITOR_AUTH_SECRET"]}


def with_auth_env(values: dict[str, str] | None = None) -> dict[str, str]:
    merged = {
        "VIBODE_COMPOSITOR_AUTH_SECRET": os.environ["VIBODE_COMPOSITOR_AUTH_SECRET"],
    }
    if values:
        merged.update(values)
    return merged
