"""Native Hermes gateway adapter for Zulip.

The adapter owns transport and conversation routing.  zulipmcp remains the
reusable API/MCP layer; its session-oriented ``reply``/``listen`` tools are not
used by gateway sessions.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests
import zulip
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_MESSAGE_LENGTH = 10_000
DEFAULT_LISTEN_REACTION = "ear"
DEFAULT_STOPPED_REACTION = "zzz"
DEFAULT_DISMISS_REACTION = "stop_sign"
DEFAULT_INITIAL_HISTORY_MESSAGES = 20
POLL_RETRY_SECONDS = 5
APPROVAL_REACTIONS = {
    "thumbs_up": "once",
    "infinity": "session",
    "thumbs_down": "deny",
}
DIRECT_MESSAGE_TYPES = {"direct", "private", "dm"}
MENTION_FLAGS = {"mentioned", "wildcard_mentioned"}
NOBOTS_MARKERS = ("/nobots", "/nb")
LEADING_MENTION_RE = re.compile(r"^\s*@\*\*(?P<name>[^*]+)\*\*\s*:?\s*")


def _platform() -> Any:
    return Platform("zulip")


def _extra(config: Any) -> dict[str, Any]:
    value = getattr(config, "extra", None)
    return value if isinstance(value, dict) else {}


def _setting(config: Any, env_name: str, extra_name: str, default: Any = None) -> Any:
    if env_name in os.environ:
        return os.environ[env_name]
    return _extra(config).get(extra_name, default)


def _bool_setting(
    config: Any,
    env_name: str,
    extra_name: str,
    default: bool,
) -> bool:
    value = _setting(config, env_name, extra_name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _values(value: Any, *, lowercase: bool = False) -> set[str]:
    if value is None:
        return set()
    items: Iterable[Any]
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = value
    else:
        items = [value]
    result = {str(item).strip() for item in items if str(item).strip()}
    return {item.lower() for item in result} if lowercase else result


def _token(value: str) -> str:
    """Encode user-controlled routing text for a safe Hermes session key."""
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def _untoken(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding).decode("utf-8")


def _stream_chat_id(stream: str) -> str:
    return f"stream:{_token(stream)}"


def _stream_from_chat_id(chat_id: str) -> str:
    prefix = "stream:"
    if not chat_id.startswith(prefix):
        raise ValueError(f"Not a Zulip stream chat_id: {chat_id}")
    return _untoken(chat_id[len(prefix) :])


def _topic_thread_id(topic: str) -> str:
    return f"topic:{_token(topic)}"


def _topic_from_thread_id(thread_id: str | None) -> str:
    prefix = "topic:"
    if not thread_id or not str(thread_id).startswith(prefix):
        raise ValueError("Zulip stream sends require topic thread metadata")
    return _untoken(str(thread_id)[len(prefix) :])


def _dm_chat_id(recipients: list[str]) -> str:
    return f"dm:{_token(json.dumps(sorted(recipients), separators=(',', ':')))}"


def _dm_recipients(chat_id: str) -> list[str]:
    prefix = "dm:"
    if not chat_id.startswith(prefix):
        raise ValueError(f"Not a Zulip DM chat_id: {chat_id}")
    value = json.loads(_untoken(chat_id[len(prefix) :]))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Invalid Zulip DM chat_id")
    return value


def _split_message(content: str, limit: int) -> list[str]:
    """Split long Markdown at useful boundaries while preserving all text."""
    if len(content) <= limit:
        return [content]
    chunks: list[str] = []
    remaining = content
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        boundary = max(
            remaining.rfind("\n\n", 0, limit + 1),
            remaining.rfind("\n", 0, limit + 1),
            remaining.rfind(" ", 0, limit + 1),
        )
        if boundary <= 0:
            boundary = limit
        chunks.append(remaining[:boundary])
        remaining = remaining[boundary:].lstrip("\n")
    return [chunk for chunk in chunks if chunk]


def _reaction_name(event: dict[str, Any]) -> str:
    return str(
        event.get("emoji_name")
        or event.get("emoji_code")
        or event.get("reaction")
        or ""
    ).strip().lower()


def _format_topic_history(messages: list[dict[str, Any]]) -> str:
    parts = ["[Recent Zulip topic context before the activating mention]"]
    for message in messages:
        sender = message.get("sender_full_name") or message.get("sender_email") or "Unknown"
        parts.append(
            f"{sender} (message {message.get('id', '')}):\n"
            f"{message.get('content', '')}"
        )
    return "\n\n".join(parts)


class ZulipAdapter(BasePlatformAdapter):
    """Hermes platform adapter backed by Zulip's Events and message APIs."""

    supports_code_blocks = True
    splits_long_messages = True

    def __init__(self, config: PlatformConfig):
        super().__init__(config, _platform())
        self.config = config
        self.zuliprc = Path(
            str(_setting(config, "ZULIP_RC_PATH", "zuliprc", "") or "")
        ).expanduser()
        self.allowed_users = _values(
            _setting(config, "ZULIP_ALLOWED_USERS", "allowed_users"),
            lowercase=True,
        )
        self.allow_all_users = _bool_setting(
            config, "ZULIP_ALLOW_ALL_USERS", "allow_all_users", False
        )
        self.allowed_streams = _values(
            _setting(config, "ZULIP_ALLOWED_STREAMS", "allowed_streams"),
            lowercase=True,
        )
        self.require_mention = _bool_setting(
            config, "ZULIP_REQUIRE_MENTION", "require_mention", True
        )
        listen = _setting(
            config,
            "ZULIP_LISTEN_REACTION",
            "listen_reaction",
            DEFAULT_LISTEN_REACTION,
        )
        self.listen_reaction = str(listen).strip()
        stopped = _setting(
            config,
            "ZULIP_STOPPED_REACTION",
            "stopped_reaction",
            DEFAULT_STOPPED_REACTION,
        )
        self.stopped_reaction = str(stopped).strip()
        dismiss = _setting(
            config,
            "ZULIP_DISMISS_REACTION",
            "dismiss_reaction",
            DEFAULT_DISMISS_REACTION,
        )
        self.dismiss_reaction = str(dismiss).strip().lower()
        try:
            self.initial_history_messages = max(
                0,
                int(
                    _setting(
                        config,
                        "ZULIP_INITIAL_HISTORY_MESSAGES",
                        "initial_history_messages",
                        DEFAULT_INITIAL_HISTORY_MESSAGES,
                    )
                ),
            )
        except (TypeError, ValueError):
            self.initial_history_messages = DEFAULT_INITIAL_HISTORY_MESSAGES
        try:
            self.max_message_length = int(
                _setting(
                    config,
                    "ZULIP_MAX_MESSAGE_LENGTH",
                    "max_message_length",
                    DEFAULT_MAX_MESSAGE_LENGTH,
                )
            )
        except (TypeError, ValueError):
            self.max_message_length = DEFAULT_MAX_MESSAGE_LENGTH

        self._event_client: Any | None = None
        self._api_client: Any | None = None
        self._queue_id: str | None = None
        self._last_event_id = -1
        self._poll_task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        self._api_lock = asyncio.Lock()
        self._activated_topics: set[tuple[str, str]] = set()
        self._deactivated_topics: set[tuple[str, str]] = set()
        self._status_reactions: dict[tuple[str, str], tuple[str, str]] = {}
        self._approval_messages: dict[str, str] = {}
        self._stream_ids_by_name: dict[str, int] = {}
        self._user_emails_by_id: dict[str, str] = {}
        self._realm_emoji_names: set[str] = set()
        self._user_ids_by_email: dict[str, int] = {}
        self._bot_user_id: str | None = None
        self._bot_email = ""
        self._bot_full_name = ""

    @property
    def enforces_own_access_policy(self) -> bool:
        return True

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Authenticate, register a fresh event queue, and start polling."""
        if not self.zuliprc.is_file():
            logger.error("Zulip config file not found: %s", self.zuliprc)
            return False

        try:
            self._event_client = await asyncio.to_thread(
                zulip.Client, config_file=str(self.zuliprc)
            )
            self._api_client = await asyncio.to_thread(
                zulip.Client, config_file=str(self.zuliprc)
            )
            profile = await asyncio.to_thread(self._api_client.get_profile)
            if profile.get("result") != "success":
                raise RuntimeError(profile.get("msg") or "get_profile failed")
            self._bot_user_id = str(profile.get("user_id") or "") or None
            self._bot_email = str(profile.get("email") or "").lower()
            self._bot_full_name = str(profile.get("full_name") or "")
            users = await asyncio.to_thread(self._api_client.get_users)
            if users.get("result") == "success":
                self._user_emails_by_id = {
                    str(user["user_id"]): str(user.get("email") or "").lower()
                    for user in users.get("members", [])
                    if isinstance(user, dict) and user.get("user_id") is not None
                }
                self._user_ids_by_email = {
                    email: int(user_id)
                    for user_id, email in self._user_emails_by_id.items()
                    if email
                }

            realm_emoji = await asyncio.to_thread(
                self._api_client.get_realm_emoji
            )
            if realm_emoji.get("result") == "success":
                self._realm_emoji_names = {
                    info["name"]
                    for info in realm_emoji.get("emoji", {}).values()
                    if not info.get("deactivated", False)
                }

            identity = f"{getattr(self._api_client, 'base_url', '')}:{self._bot_email}"
            acquire = getattr(self, "_acquire_platform_lock", None)
            if callable(acquire) and not acquire(
                "zulip_bot", identity, "Zulip bot event queue"
            ):
                await self._close_clients()
                return False

            self._stopping = asyncio.Event()
            await self._register_queue()
        except Exception as exc:  # noqa: BLE001 - platform boundary must fail closed
            logger.error("Failed to connect Zulip gateway adapter: %s", exc)
            await self._close_clients()
            release = getattr(self, "_release_platform_lock", None)
            if callable(release):
                release()
            return False

        self._poll_task = asyncio.create_task(
            self._poll_loop(), name="hermes-zulip-events"
        )
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        """Stop polling, remove the event queue, and close HTTP sessions."""
        self._stopping.set()
        if self._api_client is not None and self._queue_id:
            with contextlib.suppress(Exception):
                await self._api_call(
                    self._api_client.call_endpoint,
                    url="/events",
                    method="DELETE",
                    request={"queue_id": self._queue_id},
                )
        self._queue_id = None

        task = self._poll_task
        self._poll_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        await self._clear_all_status_reactions()
        await self._close_clients()
        release = getattr(self, "_release_platform_lock", None)
        if callable(release):
            release()
        self._mark_disconnected()

    async def _close_clients(self) -> None:
        clients = [self._event_client, self._api_client]
        self._event_client = None
        self._api_client = None
        for client in clients:
            session = getattr(client, "session", None)
            if session is not None:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(session.close)

    async def _api_call(self, method: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Serialize access to the shared outbound Zulip requests session."""
        async with self._api_lock:
            return await asyncio.to_thread(method, *args, **kwargs)

    async def _register_queue(self) -> None:
        if self._event_client is None:
            raise RuntimeError("Zulip event client is not initialized")
        result = await asyncio.to_thread(
            self._event_client.register,
            event_types=["message", "reaction"],
            apply_markdown=False,
        )
        if result.get("result") != "success":
            raise RuntimeError(result.get("msg") or "Zulip queue registration failed")
        self._queue_id = str(result["queue_id"])
        self._last_event_id = int(result["last_event_id"])

    async def _poll_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except ValueError as exc:
                if str(exc) == "BAD_EVENT_QUEUE_ID":
                    logger.warning("Zulip event queue expired; registering a new queue")
                    await self._register_queue()
                    continue
                logger.warning("Zulip event polling failed: %s", exc)
                await asyncio.sleep(POLL_RETRY_SECONDS)
            except Exception as exc:  # noqa: BLE001 - keep the gateway poller alive
                logger.warning("Zulip event polling failed: %s", exc)
                await asyncio.sleep(POLL_RETRY_SECONDS)

    async def _poll_once(self) -> None:
        if self._event_client is None or not self._queue_id:
            raise RuntimeError("Zulip event queue is not initialized")
        try:
            result = await asyncio.to_thread(
                self._event_client.get_events,
                queue_id=self._queue_id,
                last_event_id=self._last_event_id,
            )
        except requests.exceptions.ReadTimeout:
            return
        if result.get("result") != "success":
            if result.get("code") == "BAD_EVENT_QUEUE_ID":
                raise ValueError("BAD_EVENT_QUEUE_ID")
            raise RuntimeError(result.get("msg") or "Zulip event polling failed")
        events = result.get("events") or []
        for event in events:
            self._last_event_id = max(
                self._last_event_id, int(event.get("id", self._last_event_id))
            )
            event_type = event.get("type")
            if event_type == "message":
                await self._handle_message_event(event)
            elif event_type == "reaction":
                await self._handle_reaction_event(event)

    async def _handle_message_event(self, event: dict[str, Any]) -> None:
        message = event.get("message")
        if not isinstance(message, dict) or self._is_self_message(message):
            return

        kind = str(message.get("type") or "").lower()
        is_dm = kind in DIRECT_MESSAGE_TYPES
        is_stream = kind == "stream"
        if not (is_dm or is_stream) or not self._is_authorized_message(message):
            return

        stream = str(message.get("display_recipient") or "") if is_stream else ""
        topic = str(message.get("subject") or message.get("topic") or "") if is_stream else ""
        if is_stream and message.get("stream_id") is not None:
            self._stream_ids_by_name[stream.lower()] = int(message["stream_id"])
        content = str(message.get("content") or "").strip()
        if self._hidden_from_bot(topic, content):
            return
        if is_stream and self.allowed_streams and stream.lower() not in self.allowed_streams:
            return

        source = self._source_for_message(message, is_dm=is_dm)
        channel_context: str | None = None
        if is_stream:
            mentioned = self._is_mentioned(event, message, content)
            topic_key = (source.chat_id, source.thread_id or "")
            active = topic_key in self._activated_topics
            if (
                self.require_mention
                and not mentioned
                and (topic_key in self._deactivated_topics or not active)
            ):
                return
            if mentioned:
                if not active:
                    channel_context = await self._initial_topic_context(
                        stream,
                        topic,
                        before_message_id=message.get("id"),
                    )
                content = self._strip_leading_mention(content)
                self._deactivated_topics.discard(topic_key)
                self._activated_topics.add(topic_key)

        if not content:
            content = "[The user sent a message without text.]"

        await self._clear_status_reaction(
            source.chat_id,
            getattr(source, "thread_id", None),
        )
        message_event = MessageEvent(
            text=content,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=event,
            message_id=str(message.get("id") or "") or None,
            channel_context=channel_context,
        )
        await self.handle_message(message_event)

    async def _initial_topic_context(
        self,
        stream: str,
        topic: str,
        *,
        before_message_id: Any,
    ) -> str | None:
        if self._api_client is None or self.initial_history_messages <= 0:
            return None
        try:
            result = await self._api_call(
                self._api_client.get_messages,
                {
                    "narrow": [
                        {"operator": "stream", "operand": stream},
                        {"operator": "topic", "operand": topic},
                    ],
                    "anchor": before_message_id or "newest",
                    "num_before": min(self.initial_history_messages, 100),
                    "num_after": 0,
                    "include_anchor": False,
                    "apply_markdown": False,
                },
            )
            if result.get("result") != "success":
                return None
            messages = result.get("messages") or []
            if not messages:
                return None
            return _format_topic_history(messages)
        except Exception:
            logger.debug("Could not hydrate initial Zulip topic context", exc_info=True)
            return None

    def _source_for_message(self, message: dict[str, Any], *, is_dm: bool) -> Any:
        sender_id = str(message.get("sender_id") or "")
        sender_email = str(message.get("sender_email") or "").lower()
        sender_name = str(message.get("sender_full_name") or sender_email)
        message_id = str(message.get("id") or "") or None
        if is_dm:
            recipients = self._dm_human_recipients(message)
            return self.build_source(
                chat_id=_dm_chat_id(recipients),
                chat_name=", ".join(recipients) or sender_name,
                chat_type="dm",
                user_id=sender_id or sender_email,
                user_name=sender_name,
                user_id_alt=sender_email or None,
                message_id=message_id,
            )

        stream = str(message.get("display_recipient") or "")
        topic = str(message.get("subject") or message.get("topic") or "")
        return self.build_source(
            chat_id=_stream_chat_id(stream),
            chat_name=stream,
            chat_type="channel",
            user_id=sender_id or sender_email,
            user_name=sender_name,
            user_id_alt=sender_email or None,
            thread_id=_topic_thread_id(topic),
            chat_topic=topic,
            message_id=message_id,
        )

    def _dm_human_recipients(self, message: dict[str, Any]) -> list[str]:
        display = message.get("display_recipient")
        recipients = (
            [
                str(item.get("email") or "").lower()
                for item in display
                if isinstance(item, dict)
                and item.get("email")
                and str(item.get("email")).lower() != self._bot_email
            ]
            if isinstance(display, list)
            else []
        )
        sender = str(message.get("sender_email") or "").lower()
        if not recipients and sender:
            recipients = [sender]
        return sorted(set(recipients))

    def _is_self_message(self, message: dict[str, Any]) -> bool:
        sender_id = str(message.get("sender_id") or "")
        sender_email = str(message.get("sender_email") or "").lower()
        return bool(
            (self._bot_user_id and sender_id == self._bot_user_id)
            or (self._bot_email and sender_email == self._bot_email)
        )

    def _is_authorized_message(self, message: dict[str, Any]) -> bool:
        if self.allow_all_users:
            return True
        sender_id = str(message.get("sender_id") or "").lower()
        sender_email = str(message.get("sender_email") or "").lower()
        return sender_id in self.allowed_users or sender_email in self.allowed_users

    @staticmethod
    def _hidden_from_bot(topic: str, content: str) -> bool:
        topic_lower = topic.lower()
        content_lower = content.lstrip().lower()
        return any(marker in topic_lower for marker in NOBOTS_MARKERS) or any(
            content_lower.startswith(marker) for marker in NOBOTS_MARKERS
        )

    def _is_mentioned(
        self,
        event: dict[str, Any],
        message: dict[str, Any],
        content: str,
    ) -> bool:
        flags = {
            str(flag).lower()
            for flag in [*(event.get("flags") or []), *(message.get("flags") or [])]
        }
        if flags & MENTION_FLAGS:
            return True
        match = LEADING_MENTION_RE.match(content)
        if not match:
            return False
        names = {
            self._bot_full_name.lower(),
            self._bot_email.split("@", 1)[0].lower(),
        }
        return match.group("name").strip().lower() in names

    def _strip_leading_mention(self, content: str) -> str:
        match = LEADING_MENTION_RE.match(content)
        if match:
            return content[match.end() :].lstrip()
        return content

    async def _set_message_reaction(
        self,
        message_id: Any,
        emoji_name: str,
        *,
        add: bool,
    ) -> bool:
        if not emoji_name or self._api_client is None or not message_id:
            return False
        method = (
            self._api_client.add_reaction
            if add
            else self._api_client.remove_reaction
        )
        payload: dict[str, Any] = {"message_id": message_id, "emoji_name": emoji_name}
        if not add:
            payload["reaction_type"] = (
                "realm_emoji"
                if emoji_name in self._realm_emoji_names
                else "unicode_emoji"
            )
        try:
            result = await self._api_call(method, payload)
            return result.get("result") == "success"
        except Exception:  # noqa: BLE001 - reaction updates are best effort
            return False

    @staticmethod
    def _conversation_key(
        chat_id: str,
        thread_id: str | None,
    ) -> tuple[str, str]:
        return chat_id, str(thread_id or "")

    async def _clear_status_reaction(
        self,
        chat_id: str,
        thread_id: str | None,
    ) -> None:
        key = self._conversation_key(chat_id, thread_id)
        status = self._status_reactions.pop(key, None)
        if status:
            message_id, emoji_name = status
            await self._set_message_reaction(
                message_id,
                emoji_name,
                add=False,
            )

    async def _set_status_reaction(
        self,
        chat_id: str,
        thread_id: str | None,
        message_id: str,
        emoji_name: str,
    ) -> None:
        await self._clear_status_reaction(chat_id, thread_id)
        if await self._set_message_reaction(
            message_id,
            emoji_name,
            add=True,
        ):
            key = self._conversation_key(chat_id, thread_id)
            self._status_reactions[key] = (message_id, emoji_name)

    async def _clear_all_status_reactions(self) -> None:
        statuses = list(self._status_reactions.values())
        self._status_reactions.clear()
        for message_id, emoji_name in statuses:
            await self._set_message_reaction(
                message_id,
                emoji_name,
                add=False,
            )

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """Send a normal Hermes response to its originating Zulip conversation."""
        if self._api_client is None:
            return SendResult(success=False, error="Zulip API client is not connected")
        metadata = metadata or {}
        message_ids: list[str] = []
        try:
            async with self._api_lock:
                for chunk in _split_message(content, max(self.max_message_length, 1)):
                    payload = self._send_payload(chat_id, chunk, metadata)
                    result = await asyncio.to_thread(
                        self._api_client.send_message, payload
                    )
                    if result.get("result") != "success":
                        return SendResult(
                            success=False,
                            error=str(result.get("msg") or "Zulip send failed"),
                        )
                    if result.get("id") is not None:
                        message_ids.append(str(result["id"]))
        except Exception as exc:  # noqa: BLE001 - report transport failures to Hermes
            return SendResult(success=False, error=str(exc))

        if message_ids:
            await self._set_status_reaction(
                chat_id,
                metadata.get("thread_id"),
                message_ids[-1],
                self.listen_reaction,
            )
        return SendResult(
            success=True,
            message_id=message_ids[-1] if message_ids else None,
            continuation_message_ids=tuple(message_ids[:-1]),
        )

    def _send_payload(
        self,
        chat_id: str,
        content: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if chat_id.startswith("stream:"):
            thread_id = metadata.get("thread_id")
            return {
                "type": "stream",
                "to": _stream_from_chat_id(chat_id),
                "topic": _topic_from_thread_id(thread_id),
                "content": content,
            }
        if chat_id.startswith("dm:"):
            return {
                "type": "direct",
                "to": _dm_recipients(chat_id),
                "content": content,
            }
        raise ValueError(f"Unsupported Zulip chat_id: {chat_id}")

    async def send_typing(
        self,
        chat_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self._send_typing(chat_id, "start", metadata or {})

    async def stop_typing(
        self,
        chat_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self._send_typing(chat_id, "stop", metadata or {})

    async def _send_typing(
        self,
        chat_id: str,
        op: str,
        metadata: dict[str, Any],
    ) -> None:
        if self._api_client is None:
            return
        try:
            async with self._api_lock:
                if chat_id.startswith("stream:"):
                    stream = _stream_from_chat_id(chat_id)
                    stream_id = self._stream_ids_by_name.get(stream.lower())
                    if stream_id is None:
                        stream_result = await asyncio.to_thread(
                            self._api_client.get_stream_id, stream
                        )
                        if stream_result.get("result") != "success":
                            return
                        stream_id = int(stream_result["stream_id"])
                        self._stream_ids_by_name[stream.lower()] = stream_id
                    payload = {
                        "type": "stream",
                        "op": op,
                        "stream_id": stream_id,
                        "topic": _topic_from_thread_id(metadata.get("thread_id")),
                    }
                else:
                    recipients = _dm_recipients(chat_id)
                    user_ids = [
                        self._user_ids_by_email[email.lower()]
                        for email in recipients
                        if email.lower() in self._user_ids_by_email
                    ]
                    if not user_ids:
                        return
                    payload = {"type": "direct", "op": op, "to": user_ids}
                result = await asyncio.to_thread(
                    self._api_client.set_typing_status,
                    payload,
                )
                if result.get("result") != "success":
                    logger.warning(
                        "Zulip typing indicator rejected: %s",
                        result.get("msg") or "unknown error",
                    )
        except Exception:
            logger.debug("Zulip typing indicator failed", exc_info=True)

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        preview = command if len(command) <= 300 else command[:297] + "..."
        result = await self.send(
            chat_id,
            "⚠️ **Command approval required**\n\n"
            f"```text\n{preview}\n```\n"
            f"{description}\n\n"
            "React 👍 to approve once, ♾️ to approve for this session, or 👎 to reject.",
            metadata=metadata,
        )
        if result.success and result.message_id:
            self._approval_messages[str(result.message_id)] = session_key
        return result

    async def _handle_reaction_event(self, event: dict[str, Any]) -> None:
        if str(event.get("op") or "add").lower() not in {"add", "add_reaction"}:
            return
        user_id = str(event.get("user_id") or event.get("user", {}).get("user_id") or "")
        email = str(
            event.get("user", {}).get("email")
            or self._user_emails_by_id.get(user_id)
            or ""
        ).lower()
        if self._bot_user_id and user_id == self._bot_user_id:
            return
        if (
            not self.allow_all_users
            and user_id.lower() not in self.allowed_users
            and email not in self.allowed_users
        ):
            return

        message_id = str(event.get("message_id") or "")
        reaction = _reaction_name(event)
        if reaction and reaction == self.dismiss_reaction:
            await self._dismiss_topic_for_reaction(message_id)

        approval = APPROVAL_REACTIONS.get(reaction)
        session_key = self._approval_messages.get(message_id)
        if approval and session_key:
            try:
                from tools.approval import resolve_gateway_approval

                count = resolve_gateway_approval(session_key, approval)
                if count:
                    self._approval_messages.pop(message_id, None)
            except Exception:
                logger.warning("Failed to resolve Zulip approval reaction", exc_info=True)

        handler = getattr(self, "_reaction_handler", None)
        if handler is not None:
            await handler(
                {
                    "platform": "zulip",
                    "event_name": "reaction:added",
                    "reaction": reaction,
                    "user_id": user_id,
                    "channel_id": str(event.get("stream_id") or ""),
                    "message_ts": message_id,
                    "event_ts": str(event.get("id") or ""),
                    "raw_event": event,
                }
            )

    async def _dismiss_topic_for_reaction(self, message_id: str) -> None:
        """Interrupt a bot-authored topic session until it is mentioned again."""
        if not message_id or self._api_client is None:
            return
        try:
            result = await self._api_call(
                self._api_client.get_messages,
                {
                    "anchor": message_id,
                    "num_before": 0,
                    "num_after": 0,
                    "include_anchor": True,
                    "apply_markdown": False,
                },
            )
            messages = result.get("messages") if result.get("result") == "success" else None
            message = messages[0] if messages else None
            if not isinstance(message, dict) or not self._is_self_message(message):
                return
            if str(message.get("type") or "").lower() != "stream":
                return
            source = self._source_for_message(message, is_dm=False)
            topic_key = (source.chat_id, source.thread_id or "")
            await self._set_status_reaction(
                source.chat_id,
                source.thread_id,
                message_id,
                self.stopped_reaction,
            )
            self._activated_topics.discard(topic_key)
            self._deactivated_topics.add(topic_key)

            from gateway.session import build_session_key

            session_key = build_session_key(
                source,
                group_sessions_per_user=_extra(self.config).get(
                    "group_sessions_per_user", True
                ),
                thread_sessions_per_user=_extra(self.config).get(
                    "thread_sessions_per_user", False
                ),
            )
            cancel = getattr(self, "cancel_session_processing", None)
            if callable(cancel):
                await cancel(session_key)
        except Exception:
            logger.debug("Could not dismiss Zulip topic session", exc_info=True)

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        if chat_id.startswith("stream:"):
            return {"name": _stream_from_chat_id(chat_id), "type": "channel"}
        if chat_id.startswith("dm:"):
            return {"name": ", ".join(_dm_recipients(chat_id)), "type": "dm"}
        return {"name": chat_id, "type": "unknown"}


def check_requirements() -> bool:
    path = os.environ.get("ZULIP_RC_PATH", "").strip()
    return bool(path and Path(path).expanduser().is_file())


def validate_config(config: PlatformConfig) -> bool:
    value = _setting(config, "ZULIP_RC_PATH", "zuliprc", "")
    return bool(value and Path(str(value)).expanduser().is_file())


def _env_enablement() -> dict[str, Any] | None:
    path = os.environ.get("ZULIP_RC_PATH", "").strip()
    if not path or not Path(path).expanduser().is_file():
        return None
    extra: dict[str, Any] = {
        "zuliprc": str(Path(path).expanduser()),
        "allowed_users": sorted(
            _values(os.environ.get("ZULIP_ALLOWED_USERS"), lowercase=True)
        ),
        "allow_all_users": _bool_setting(
            object(), "ZULIP_ALLOW_ALL_USERS", "allow_all_users", False
        ),
    }
    return extra


def register(ctx: Any) -> None:
    """Register Zulip as a first-class Hermes gateway platform."""
    ctx.register_platform(
        name="zulip",
        label="Zulip",
        adapter_factory=lambda config: ZulipAdapter(config),
        check_fn=check_requirements,
        validate_config=validate_config,
        required_env=["ZULIP_RC_PATH"],
        install_hint="Install zulipmcp into the Hermes environment",
        env_enablement_fn=_env_enablement,
        allowed_users_env="ZULIP_ALLOWED_USERS",
        allow_all_env="ZULIP_ALLOW_ALL_USERS",
        max_message_length=DEFAULT_MAX_MESSAGE_LENGTH,
        platform_hint=(
            "You are chatting through Zulip. The current stream/topic is a "
            "persistent Hermes session. Reply normally: the gateway delivers "
            "your final response, so do not use ZulipMCP reply/listen/session "
            "tools. Use explicit ZulipMCP tools only for deliberate actions "
            "such as reading history, reacting, uploading, or messaging a "
            "different conversation."
        ),
        emoji="💬",
    )


__all__ = [
    "ZulipAdapter",
    "check_requirements",
    "register",
    "validate_config",
]
