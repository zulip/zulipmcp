"""Contract tests for the optional Hermes Zulip platform plugin."""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@dataclass
class FakePlatform:
    value: str


@dataclass
class FakePlatformConfig:
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeSendResult:
    success: bool
    message_id: str | None = None
    error: str | None = None
    continuation_message_ids: tuple[str, ...] = ()


@dataclass
class FakeMessageEvent:
    text: str
    message_type: Any
    source: Any
    raw_message: Any = None
    message_id: str | None = None
    channel_context: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FakeBasePlatformAdapter:
    def __init__(self, config: FakePlatformConfig, platform: FakePlatform):
        self.config = config
        self.platform = platform
        self.handled: list[FakeMessageEvent] = []
        self._session_store = None
        self._reaction_handler = None
        self.connected = False
        self.cancelled_sessions: list[str] = []

    def build_source(self, **kwargs: Any) -> Any:
        return SimpleNamespace(platform=self.platform, **kwargs)

    async def handle_message(self, event: FakeMessageEvent) -> None:
        self.handled.append(event)

    def _mark_connected(self) -> None:
        self.connected = True

    def _mark_disconnected(self) -> None:
        self.connected = False

    async def cancel_session_processing(self, session_key: str, **_: Any) -> None:
        self.cancelled_sessions.append(session_key)


def _install_gateway_fakes() -> None:
    gateway = types.ModuleType("gateway")
    config = types.ModuleType("gateway.config")
    platforms = types.ModuleType("gateway.platforms")
    base = types.ModuleType("gateway.platforms.base")
    session = types.ModuleType("gateway.session")

    config.Platform = FakePlatform
    config.PlatformConfig = FakePlatformConfig
    base.BasePlatformAdapter = FakeBasePlatformAdapter
    base.MessageEvent = FakeMessageEvent
    base.MessageType = SimpleNamespace(TEXT="text")
    base.SendResult = FakeSendResult

    def build_session_key(source: Any, **_: Any) -> str:
        return ":".join(
            [
                source.platform.value,
                source.chat_type,
                source.chat_id,
                source.thread_id or "",
            ]
        )

    session.build_session_key = build_session_key
    sys.modules["gateway"] = gateway
    sys.modules["gateway.config"] = config
    sys.modules["gateway.platforms"] = platforms
    sys.modules["gateway.platforms.base"] = base
    sys.modules["gateway.session"] = session


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_install_gateway_fakes()

from hermes_plugin.zulip import adapter as zulip_adapter


@pytest.fixture(autouse=True)
def clean_gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ZULIP_RC_PATH",
        "ZULIP_ALLOWED_USERS",
        "ZULIP_ALLOW_ALL_USERS",
        "ZULIP_ALLOWED_STREAMS",
        "ZULIP_REQUIRE_MENTION",
        "ZULIP_LISTEN_REACTION",
        "ZULIP_STOPPED_REACTION",
        "ZULIP_DISMISS_REACTION",
        "ZULIP_INITIAL_HISTORY_MESSAGES",
        "ZULIP_MAX_MESSAGE_LENGTH",
    ):
        monkeypatch.delenv(name, raising=False)


def _adapter(**extra: Any) -> zulip_adapter.ZulipAdapter:
    config = FakePlatformConfig(
        {
            "zuliprc": "/missing/.zuliprc",
            "allowed_users": ["42"],
            **extra,
        }
    )
    result = zulip_adapter.ZulipAdapter(config)
    result._bot_user_id = "99"
    result._bot_email = "hermes@example.com"
    result._bot_full_name = "Hermes"
    return result


def _stream_message(
    *,
    message_id: int = 1,
    topic: str = "Kiki setup",
    content: str = "@**Hermes** start",
    sender_id: int = 42,
    flags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "message",
        "flags": ["mentioned"] if flags is None else flags,
        "message": {
            "id": message_id,
            "type": "stream",
            "stream_id": 7,
            "display_recipient": "agents",
            "subject": topic,
            "content": content,
            "sender_id": sender_id,
            "sender_email": "john@example.com",
            "sender_full_name": "John",
        },
    }


def test_mention_activates_topic_and_followups_do_not_need_mention() -> None:
    adapter = _adapter()

    async def scenario() -> None:
        await adapter._handle_message_event(_stream_message())
        await adapter._handle_message_event(
            _stream_message(
                message_id=2,
                content="continue with the install",
                flags=[],
            )
        )

    asyncio.run(scenario())

    assert [event.text for event in adapter.handled] == [
        "start",
        "continue with the install",
    ]
    assert adapter.handled[0].source.thread_id == adapter.handled[1].source.thread_id
    assert all(event.metadata == {} for event in adapter.handled)
    assert adapter._stream_ids_by_name == {"agents": 7}


def test_unmentioned_new_topic_is_independent_and_ignored() -> None:
    adapter = _adapter()

    async def scenario() -> None:
        await adapter._handle_message_event(_stream_message())
        await adapter._handle_message_event(
            _stream_message(
                message_id=2,
                topic="Networking",
                content="this belongs to another topic",
                flags=[],
            )
        )

    asyncio.run(scenario())

    assert len(adapter.handled) == 1


def test_topic_session_ids_are_safe_and_distinct() -> None:
    adapter = _adapter(require_mention=False)
    first = _stream_message(topic="../danger", content="one", flags=[])
    second = _stream_message(message_id=2, topic="danger", content="two", flags=[])

    async def scenario() -> None:
        await adapter._handle_message_event(first)
        await adapter._handle_message_event(second)

    asyncio.run(scenario())

    first_id = adapter.handled[0].source.thread_id
    second_id = adapter.handled[1].source.thread_id
    assert first_id != second_id
    assert ".." not in first_id
    assert "/" not in first_id
    assert zulip_adapter._topic_from_thread_id(first_id) == "../danger"


def test_dm_does_not_require_mention() -> None:
    adapter = _adapter()
    event = {
        "type": "message",
        "message": {
            "id": 5,
            "type": "direct",
            "content": "hello",
            "sender_id": 42,
            "sender_email": "john@example.com",
            "sender_full_name": "John",
            "display_recipient": [
                {"email": "john@example.com"},
                {"email": "hermes@example.com"},
            ],
        },
    }

    asyncio.run(adapter._handle_message_event(event))

    assert len(adapter.handled) == 1
    assert adapter.handled[0].source.chat_type == "dm"
    assert adapter.handled[0].metadata == {}
    assert zulip_adapter._dm_recipients(adapter.handled[0].source.chat_id) == [
        "john@example.com"
    ]


def test_unauthorized_and_self_messages_are_ignored() -> None:
    adapter = _adapter()
    own = _stream_message(sender_id=99)
    own["message"]["sender_email"] = "hermes@example.com"

    async def scenario() -> None:
        await adapter._handle_message_event(_stream_message(sender_id=7))
        await adapter._handle_message_event(own)

    asyncio.run(scenario())
    assert adapter.handled == []


class FakeApiClient:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.added_reactions: list[dict[str, Any]] = []
        self.removed_reactions: list[dict[str, Any]] = []
        self.endpoints: list[dict[str, Any]] = []
        self.typing_payloads: list[dict[str, Any]] = []
        self.messages_by_id: dict[str, dict[str, Any]] = {}
        self.history: list[dict[str, Any]] = []

    def send_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.sent.append(payload)
        return {"result": "success", "id": len(self.sent)}

    def add_reaction(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.added_reactions.append(payload)
        return {"result": "success"}

    def remove_reaction(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.removed_reactions.append(payload)
        return {"result": "success"}

    def set_typing_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.typing_payloads.append(payload)
        return {"result": "success"}

    def get_stream_id(self, stream: str) -> dict[str, Any]:
        assert stream == "agents"
        return {"result": "success", "stream_id": 17}

    def call_endpoint(self, **kwargs: Any) -> dict[str, Any]:
        self.endpoints.append(kwargs)
        if str(kwargs.get("url") or "").startswith("/messages/"):
            message_id = str(kwargs["url"]).rsplit("/", 1)[-1]
            return {
                "result": "success",
                "message": self.messages_by_id[message_id],
            }
        return {"result": "success"}

    def get_users(self) -> dict[str, Any]:
        return {
            "result": "success",
            "members": [
                {"user_id": 42, "email": "john@example.com"},
                {"user_id": 99, "email": "someone-else@example.com"},
            ],
        }

    def get_messages(self, request: dict[str, Any]) -> dict[str, Any]:
        if request["include_anchor"]:
            return {
                "result": "success",
                "messages": [self.messages_by_id[str(request["anchor"])]],
            }
        return {"result": "success", "messages": self.history}


class FakeEventClient:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self.events = events
        self.register_kwargs: dict[str, Any] = {}

    def register(self, **kwargs: Any) -> dict[str, Any]:
        self.register_kwargs = kwargs
        return {"result": "success", "queue_id": "queue-1", "last_event_id": 3}

    def get_events(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs == {"queue_id": "queue-1", "last_event_id": 3}
        return {"result": "success", "events": self.events}


def test_event_queue_registration_and_polling_are_plugin_owned() -> None:
    adapter = _adapter()
    client = FakeEventClient([{"id": 4, **_stream_message()}])
    adapter._event_client = client

    async def scenario() -> None:
        await adapter._register_queue()
        await adapter._poll_once()

    asyncio.run(scenario())

    assert client.register_kwargs == {
        "event_types": ["message", "reaction"],
        "apply_markdown": False,
    }
    assert adapter._last_event_id == 4
    assert [event.text for event in adapter.handled] == ["start"]


def test_listening_reaction_moves_from_bot_reply_on_next_user_message() -> None:
    adapter = _adapter(listen_reaction="ear")
    api = FakeApiClient()
    adapter._api_client = api
    chat_id = zulip_adapter._stream_chat_id("agents")
    thread_id = zulip_adapter._topic_thread_id("Kiki setup")

    async def scenario() -> None:
        await adapter.send(
            chat_id,
            "I am listening.",
            metadata={"thread_id": thread_id},
        )
        await adapter._handle_message_event(
            _stream_message(message_id=42, content="@**Hermes** next question")
        )

    asyncio.run(scenario())

    assert api.added_reactions == [{"message_id": "1", "emoji_name": "ear"}]
    assert api.removed_reactions == [
        {"message_id": "1", "emoji_name": "ear", "reaction_type": "unicode_emoji"}
    ]
    assert adapter._status_reactions == {}


def test_send_routes_and_splits_to_original_stream_topic() -> None:
    adapter = _adapter(max_message_length=8, listen_reaction="")
    api = FakeApiClient()
    adapter._api_client = api
    chat_id = zulip_adapter._stream_chat_id("agents")
    thread_id = zulip_adapter._topic_thread_id("Kiki setup")

    result = asyncio.run(
        adapter.send(
            chat_id,
            "first paragraph\n\nsecond",
            metadata={"thread_id": thread_id},
        )
    )

    assert result.success
    assert len(api.sent) > 1
    assert {message["to"] for message in api.sent} == {"agents"}
    assert {message["topic"] for message in api.sent} == {"Kiki setup"}
    assert "".join(message["content"] for message in api.sent).replace("\n", "") == (
        "first paragraphsecond".replace("\n", "")
    )


def test_typing_indicator_targets_stream_and_topic() -> None:
    adapter = _adapter()
    api = FakeApiClient()
    adapter._api_client = api
    adapter._stream_ids_by_name["agents"] = 17

    asyncio.run(
        adapter.send_typing(
            zulip_adapter._stream_chat_id("agents"),
            {"thread_id": zulip_adapter._topic_thread_id("Kiki setup")},
        )
    )

    assert api.typing_payloads == [
        {
            "type": "stream",
            "op": "start",
            "stream_id": 17,
            "topic": "Kiki setup",
        }
    ]


def test_first_mention_hydrates_only_that_topics_recent_context() -> None:
    adapter = _adapter(initial_history_messages=20)
    api = FakeApiClient()
    api.history = [
        {
            "id": 3,
            "timestamp": 1,
            "sender_full_name": "Alice",
            "sender_email": "alice@example.com",
            "content": "Earlier topic detail",
            "type": "stream",
            "display_recipient": "agents",
            "subject": "Kiki setup",
            "reactions": [],
        }
    ]
    adapter._api_client = api

    asyncio.run(adapter._handle_message_event(_stream_message(message_id=4)))

    assert "Earlier topic detail" in (adapter.handled[0].channel_context or "")
    assert "Recent Zulip topic context" in adapter.handled[0].channel_context


def test_stop_reaction_interrupts_and_requires_a_new_mention() -> None:
    adapter = _adapter(dismiss_reaction="stop_sign", stopped_reaction="zzz")
    api = FakeApiClient()
    api.messages_by_id["50"] = {
        "id": 50,
        "type": "stream",
        "display_recipient": "agents",
        "subject": "Kiki setup",
        "content": "Hermes response",
        "sender_id": 99,
        "sender_email": "hermes@example.com",
        "sender_full_name": "Hermes",
    }
    adapter._api_client = api

    async def scenario() -> None:
        await adapter._handle_message_event(_stream_message())
        source = adapter.handled[0].source
        topic_key = (source.chat_id, source.thread_id or "")
        adapter._status_reactions[topic_key] = ("50", "ear")
        await adapter._handle_reaction_event(
            {
                "id": 10,
                "type": "reaction",
                "op": "add",
                "message_id": 50,
                "emoji_name": "stop_sign",
                "user_id": 42,
            }
        )
        await adapter._handle_message_event(
            _stream_message(message_id=2, content="ignored for now", flags=[])
        )
        assert adapter._status_reactions[topic_key] == ("50", "zzz")
        await adapter._handle_message_event(
            _stream_message(message_id=3, content="@**Hermes** come back")
        )

    asyncio.run(scenario())

    assert [event.text for event in adapter.handled] == ["start", "come back"]
    assert adapter.cancelled_sessions
    assert {"message_id": "50", "emoji_name": "ear", "reaction_type": "unicode_emoji"} in api.removed_reactions
    assert {"message_id": "50", "emoji_name": "zzz"} in api.added_reactions
    assert {"message_id": "50", "emoji_name": "zzz", "reaction_type": "unicode_emoji"} in api.removed_reactions


@pytest.mark.parametrize(
    ("reaction", "expected"),
    [("thumbs_up", "once"), ("infinity", "session"), ("thumbs_down", "deny")],
)
def test_approval_reactions(
    reaction: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    choices: list[tuple[str, str]] = []
    approval = types.ModuleType("tools.approval")
    approval.resolve_gateway_approval = (
        lambda session_key, choice: choices.append((session_key, choice)) or 1
    )
    monkeypatch.setitem(sys.modules, "tools.approval", approval)

    adapter = _adapter()
    adapter._approval_messages["50"] = "session-key"
    asyncio.run(
        adapter._handle_reaction_event(
            {
                "type": "reaction",
                "op": "add",
                "message_id": 50,
                "emoji_name": reaction,
                "user_id": 42,
            }
        )
    )

    assert choices == [("session-key", expected)]


def test_plugin_registration_exposes_native_platform_contract() -> None:
    captured: dict[str, Any] = {}

    class Context:
        def register_platform(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    zulip_adapter.register(Context())

    assert captured["name"] == "zulip"
    assert captured["required_env"] == ["ZULIP_RC_PATH"]
    assert captured["allowed_users_env"] == "ZULIP_ALLOWED_USERS"
    assert captured["allow_all_env"] == "ZULIP_ALLOW_ALL_USERS"
    assert captured["max_message_length"] == 10_000
    assert callable(captured["adapter_factory"])


def test_plugin_manifest_and_init_are_installable() -> None:
    plugin_dir = Path(zulip_adapter.__file__).parent
    manifest = (plugin_dir / "plugin.yaml").read_text()
    init = (plugin_dir / "__init__.py").read_text()

    assert "kind: platform" in manifest
    assert "name: zulip-platform" in manifest
    assert "from .adapter import register" in init


def test_dm_and_topic_routing_tokens_round_trip_unicode() -> None:
    topic = "Kiki / deployment: café"
    thread_id = zulip_adapter._topic_thread_id(topic)
    chat_id = zulip_adapter._dm_chat_id(["john+bot@example.com"])

    assert zulip_adapter._topic_from_thread_id(thread_id) == topic
    assert zulip_adapter._dm_recipients(chat_id) == ["john+bot@example.com"]
