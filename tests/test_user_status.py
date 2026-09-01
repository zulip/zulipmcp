"""Tests for authenticated-user status operations."""

import importlib

from zulipmcp import core

mcp_module = importlib.import_module("zulipmcp.mcp")


class FakeClient:
    def __init__(self):
        self.calls = []

    def get_profile(self):
        return {
            "result": "success",
            "msg": "",
            "user_id": 42,
            "email": "user@example.com",
        }

    def call_endpoint(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("method") == "GET":
            return {
                "result": "success",
                "msg": "",
                "status": {
                    "status_text": "long break",
                    "emoji_name": "coffee",
                },
            }
        return {"result": "success", "msg": ""}


def test_get_own_status_uses_authenticated_user(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(core, "get_client", lambda: client)

    result = core.get_own_status()

    assert result["result"] == "success"
    assert result["status"]["status_text"] == "long break"
    assert client.calls == [
        {
            "url": "/users/42/status",
            "method": "GET",
        }
    ]


def test_get_own_status_handles_missing_user_id(monkeypatch):
    class ClientWithoutUserId:
        def get_profile(self):
            return {"result": "success", "msg": ""}

    monkeypatch.setattr(core, "get_client", lambda: ClientWithoutUserId())

    result = core.get_own_status()

    assert result["result"] == "error"
    assert "user_id" in result["msg"]


def test_update_own_status_only_sends_requested_fields(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(core, "get_client", lambda: client)

    result = core.update_own_status("long break")

    assert result["result"] == "success"
    assert client.calls == [
        {
            "url": "/users/me/status",
            "method": "POST",
            "request": {
                "status_text": "long break",
            },
        }
    ]


def test_update_own_status_supports_emoji(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(core, "get_client", lambda: client)

    result = core.update_own_status(
        status_text="driving",
        emoji_name="car",
        emoji_code="1f697",
        reaction_type="unicode_emoji",
    )

    assert result["result"] == "success"
    assert client.calls[-1]["request"] == {
        "status_text": "driving",
        "emoji_name": "car",
        "emoji_code": "1f697",
        "reaction_type": "unicode_emoji",
    }


def test_clear_own_status_clears_text_and_emoji(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(core, "get_client", lambda: client)

    result = core.clear_own_status()

    assert result["result"] == "success"
    assert client.calls[-1] == {
        "url": "/users/me/status",
        "method": "POST",
        "request": {
            "status_text": "",
            "emoji_name": "",
            "emoji_code": "",
            "reaction_type": "unicode_emoji",
        },
    }


def test_mcp_get_my_status(monkeypatch):
    monkeypatch.setattr(
        mcp_module.zulip_core,
        "get_own_status",
        lambda: {
            "result": "success",
            "status": {
                "status_text": "long break",
                "emoji_name": "coffee",
            },
        },
    )

    result = mcp_module.get_my_status()

    assert "long break" in result
    assert ":coffee:" in result


def test_mcp_set_my_status(monkeypatch):
    captured = {}

    def fake_update(**kwargs):
        captured.update(kwargs)
        return {"result": "success", "msg": ""}

    monkeypatch.setattr(
        mcp_module.zulip_core,
        "update_own_status",
        fake_update,
    )

    result = mcp_module.set_my_status("long break")

    assert result == "Status updated: long break"
    assert captured == {
        "status_text": "long break",
        "emoji_name": None,
        "emoji_code": None,
        "reaction_type": None,
    }


def test_mcp_rejects_status_over_60_codepoints(monkeypatch):
    called = False

    def fake_update(**kwargs):
        nonlocal called
        called = True
        return {"result": "success"}

    monkeypatch.setattr(
        mcp_module.zulip_core,
        "update_own_status",
        fake_update,
    )

    result = mcp_module.set_my_status("x" * 61)

    assert result.startswith("Error:")
    assert called is False


def test_mcp_clear_my_status(monkeypatch):
    monkeypatch.setattr(
        mcp_module.zulip_core,
        "clear_own_status",
        lambda: {"result": "success", "msg": ""},
    )

    assert mcp_module.clear_my_status() == "Custom status cleared."
