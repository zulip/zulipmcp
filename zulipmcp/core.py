import os
import re
import time
import tempfile
import json
import urllib.parse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Callable, Optional

import diskcache
import requests
import zulip

_DEFAULT_TIMEZONE = "America/Los_Angeles"


def _load_timezone() -> ZoneInfo:
    """Load the configured IANA timezone."""
    name = os.environ.get("ZULIPMCP_TIMEZONE", _DEFAULT_TIMEZONE)
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(
            f"Invalid ZULIPMCP_TIMEZONE {name!r}; use an IANA timezone name"
        ) from exc


TIMEZONE = _load_timezone()

_client: Optional[zulip.Client] = None
_cache = diskcache.Cache(os.environ.get("ZULIPMCP_CACHE_DIR",
    Path(tempfile.gettempdir()) / "zulipmcp_cache"))
_ignored_streams: set[str] = set()
_ALL_PRIVATE_STREAMS = "__ALL__"
_dismiss_emoji: set[str] = {"stop_sign"}

# Zulip silently truncates messages longer than this server-side, so a sender
# never learns its message was clipped. Matches Zulip's default; override via
# env for realms configured with a different cap. Bad values fall back rather
# than crashing the server at import.
try:
    MAX_MESSAGE_LENGTH = int(os.environ.get("ZULIP_MAX_MESSAGE_LENGTH", "10000"))
except ValueError:
    MAX_MESSAGE_LENGTH = 10000

_FENCE_RE = re.compile(r'^(`{3,}|~{3,})')
# Inline code spans with variable-length delimiters (`` `x` ``, ``` ``x`` ```),
# mirroring Python-Markdown's backtick pairing.
_INLINE_CODE_RE = re.compile(r'(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)')
# Link URLs allow one level of balanced parens (GFM-style) so targets like
# .../wiki/Foo_(bar) terminate at the real closing paren.
_LINK_RE = re.compile(r'(?<!!)\[([^\]]*)\]\((<?(?:\([^()\s]*\)|[^()\s])+>?)\)')
_URL_ABUTS_BOLD_RE = re.compile(r'(https?://[^\s<>\[\]()*]+)(?=\*\*)')


def _autofix_enabled() -> bool:
    """Whether outgoing markdown normalization is enabled.

    Controlled by ``ZULIPMCP_MARKDOWN_AUTOFIX``: set to ``0`` or ``false``
    (case-insensitive) to disable all normalization.  Read at call time so
    it can be toggled without a restart.
    """
    return os.environ.get('ZULIPMCP_MARKDOWN_AUTOFIX', '').lower() not in ('0', 'false')


def _is_table_separator(line: str) -> bool:
    """Whether *line* is a markdown table separator (e.g. ``| --- | --- |``)."""
    stripped = line.strip()
    if not stripped or '-' not in stripped:
        return False
    inner = stripped.lstrip('|').rstrip('|')
    return bool(inner) and set(inner) <= set('|:- ')


def _restyle_bold_in_link(m: 're.Match[str]') -> str:
    """Rewrite bold inside a markdown link's text so Zulip renders it.

    Whole-text bold moves outside the link (``[**a**](url)`` becomes
    ``**[a](url)**``); partial bold is stripped from the text.
    """
    text, url = m.group(1), m.group(2)
    if '**' not in text:
        return m.group(0)
    if text.startswith('**') and text.endswith('**') and text.count('**') == 2:
        return f'**[{text[2:-2]}]({url})**'
    return f'[{text.replace("**", "")}]({url})'


def _sub_outside_code(pattern: 're.Pattern[str]',
                      repl: Callable[['re.Match[str]'], str], line: str) -> str:
    """Apply ``pattern.sub(repl, line)``, leaving matches inside inline code alone.

    A match is exempt only when wholly enclosed in an inline code span; a
    match that merely *contains* a code span is still rewritten.
    """
    code_spans = [span.span() for span in _INLINE_CODE_RE.finditer(line)]

    def guarded(m: 're.Match[str]') -> str:
        if any(s <= m.start() and m.end() <= e for s, e in code_spans):
            return m.group(0)
        return repl(m)

    return pattern.sub(guarded, line)


def _fix_bold_links(line: str) -> str:
    """Rewrite bold/link combinations that Zulip's renderer breaks on."""
    if '**' not in line:
        return line
    line = _sub_outside_code(_LINK_RE, _restyle_bold_in_link, line)
    line = _sub_outside_code(_URL_ABUTS_BOLD_RE,
                             lambda m: f'[{m.group(1)}]({m.group(1)})', line)
    return line


def normalize_zulip_markdown(content: str) -> str:
    """Normalize outgoing markdown for Zulip's renderer.

    Two deterministic fixes for patterns LLMs trained on GFM emit constantly:

    1. Blank lines before tables.  Zulip's Python-Markdown parser requires a
       blank line before table header rows; injects the missing one.
    2. Bold/link combos Zulip breaks on.  Bold inside link text
       (``[**a**](url)``) renders as literal escaped asterisks with an
       unclickable link (zulip/zulip#36087, fix PR stalled upstream) —
       whole-text bold moves outside the link, partial bold is stripped.
       A bare URL immediately followed by ``**`` is wrapped as
       ``[url](url)``: a trailing ``**`` otherwise gets swallowed into the
       autolinked URL, and a URL *preceded* by ``**`` never autolinks at
       all, so the rewrite also upgrades bold URLs to clickable links.

    Both fixes skip fenced code blocks and indented code.  The bold/link
    fix also skips inline code spans; the table fix also skips blockquotes.
    Accepted edge case: a URL whose path literally contains ``**`` is
    truncated at the asterisks (literal ``*`` in URLs should be
    percent-encoded).

    Set ``ZULIPMCP_MARKDOWN_AUTOFIX=0`` (or ``false``) to disable all
    normalization and send content verbatim.
    """
    if not _autofix_enabled():
        return content

    lines = content.split('\n')
    result: list[str] = []
    in_fence = False
    fence_char: str | None = None
    fence_len = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        m = _FENCE_RE.match(stripped)
        if m:
            fence_chars = m.group(1)
            after_fence = stripped[len(fence_chars):]
            if not in_fence:
                in_fence, fence_char, fence_len = True, fence_chars[0], len(fence_chars)
            elif fence_chars[0] == fence_char and len(fence_chars) >= fence_len and not after_fence:
                in_fence = False
                fence_char = None
            result.append(line)
            continue

        if in_fence:
            result.append(line)
            continue

        if (
            stripped
            and '|' in stripped
            and result
            and result[-1].strip()
            and not stripped.startswith('>')
            and not line.startswith('    ')
            and not line.startswith('\t')
            and i + 1 < len(lines)
            and _is_table_separator(lines[i + 1])
        ):
            result.append('')

        if not line.startswith('    ') and not line.startswith('\t'):
            line = _fix_bold_links(line)

        result.append(line)

    return '\n'.join(result)


def _is_allow_all_token(value: object) -> bool:
    """Whether a value is the special allow-all sentinel."""
    return isinstance(value, str) and value.strip().upper() == _ALL_PRIVATE_STREAMS


def _parse_stream_allowlist(env_var: str, default_all: bool) -> set[str]:
    """Parse a stream allowlist env var into a lowercase set."""
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return {_ALL_PRIVATE_STREAMS} if default_all else set()
    if _is_allow_all_token(raw):
        return {_ALL_PRIVATE_STREAMS}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            if any(_is_allow_all_token(s) for s in parsed):
                return {_ALL_PRIVATE_STREAMS}
            return {str(s).lower() for s in parsed if str(s).strip()}
        if isinstance(parsed, str):
            if _is_allow_all_token(parsed):
                return {_ALL_PRIVATE_STREAMS}
            return {parsed.lower()}
    except json.JSONDecodeError:
        return {s.strip().lower() for s in raw.split(",") if s.strip()}
    return set()


def _allowed_private_streams() -> set[str]:
    """Return allowed private streams from env.

    Security model:
    - Unset/empty env => no private stream access (default-deny)
    - "__ALL__" => explicit full private stream access
    - list/string => explicit allowlist (normalized to lowercase)
    """
    return _parse_stream_allowlist("BOT_ALLOWED_PRIVATE_STREAMS", default_all=False)


def _allowed_write_streams() -> set[str]:
    """Return allowed write streams from env.

    Security model:
    - Unset/empty env => all stream writes allowed (backwards-compatible)
    - "__ALL__" => explicit full write access
    - list/string => explicit allowlist (normalized to lowercase)
    """
    return _parse_stream_allowlist("BOT_ALLOWED_WRITE_STREAMS", default_all=True)


def is_private_stream_allowed(stream_name: str) -> bool:
    """Whether the current process is allowed to access this private stream."""
    if not is_stream_private(stream_name):
        return True
    allowed = _allowed_private_streams()
    if _ALL_PRIVATE_STREAMS in allowed:
        return True
    return stream_name.lower() in allowed


def is_stream_write_allowed(stream_name: str) -> bool:
    """Whether the current process is allowed to send to this stream."""
    allowed = _allowed_write_streams()
    if _ALL_PRIVATE_STREAMS in allowed:
        return True
    return stream_name.lower() in allowed


def _stream_write_error(stream: str) -> Optional[dict]:
    """Return an API-style error dict if writes to this stream are forbidden."""
    if is_stream_private(stream) and not is_private_stream_allowed(stream):
        return {"result": "error", "msg": f"Private stream access denied: {stream}"}
    if not is_stream_write_allowed(stream):
        return {"result": "error", "msg": f"Stream write access denied: {stream}"}
    return None


def _get_stream_message_for_write(message_id: int) -> tuple[Optional[dict], Optional[dict]]:
    """Return a stream message or an API-style error dict for write operations."""
    msg = get_message_by_id(message_id)
    if not msg:
        return None, {"result": "error", "msg": f"Message {message_id} not found or inaccessible"}
    if msg.get("type") != "stream":
        return None, {"result": "error", "msg": "This operation only supports stream messages"}
    stream = msg.get("display_recipient", "")
    if not stream:
        return None, {"result": "error", "msg": f"Message {message_id} stream not found"}
    err = _stream_write_error(stream)
    if err:
        return None, err
    return msg, None


def set_ignored_streams(streams: set[str]) -> None:
    """Set streams to exclude from all message fetches."""
    global _ignored_streams
    _ignored_streams = {s.lower() for s in streams}


def get_ignored_streams() -> set[str]:
    """Return the current ignored streams set."""
    return _ignored_streams


def set_dismiss_emoji(emoji: set[str] | str) -> None:
    """Set the emoji names that trigger session dismissal when reacted.

    Pass an empty set to disable dismiss-by-reaction.
    """
    global _dismiss_emoji
    if isinstance(emoji, str):
        emoji = {emoji}
    _dismiss_emoji = {e.strip(":") for e in emoji if e.strip(":")}


def get_dismiss_emoji() -> set[str]:
    """Return the current dismiss emoji set."""
    return _dismiss_emoji


def is_dismiss_reaction(event: dict, bot_user_id: int,
                        stream: Optional[str] = None,
                        topic: Optional[str] = None) -> bool:
    """Check if a reaction event is a dismiss signal on a bot-authored message.

    Returns True if:
    - The reaction is an "add" operation
    - The emoji matches a dismiss emoji
    - The reactor is not the bot itself
    - The reacted-on message was sent by the bot
    - The reacted-on message is in the given stream/topic (if specified)

    The last checks require one API call to fetch the message.

    Note: Zulip's event queue narrow does NOT filter reaction events, so
    the caller receives reactions from all streams.  Pass stream/topic to
    ensure only reactions in the current conversation trigger dismissal.
    """
    if event.get("op") != "add":
        return False
    if event.get("emoji_name") not in _dismiss_emoji:
        return False
    if event.get("user_id") == bot_user_id:
        return False

    # Verify the reacted-on message was sent by this bot (and optionally
    # that it belongs to the expected stream/topic).
    msg_id = event.get("message_id")
    if not msg_id:
        return False
    try:
        result = get_client().call_endpoint(
            url=f"/messages/{msg_id}",
            method="GET",
        )
        if result.get("result") != "success":
            return False
        msg = result.get("message", {})
        if msg.get("sender_id") != bot_user_id:
            return False
        # Verify stream/topic if provided.
        # display_recipient is a string for stream messages but a list for DMs.
        display_recipient = msg.get("display_recipient", "")
        if stream:
            if not isinstance(display_recipient, str) or display_recipient.lower() != stream.lower():
                return False
        if topic and msg.get("subject", "").lower() != topic.lower():
            return False
        return True
    except Exception:
        return False


def check_dismissed(message_id: int, bot_user_id: int) -> Optional[str]:
    """Check if a bot message already has a dismiss reaction.

    Fetches the message via REST API and inspects its reactions list.
    Returns the emoji name if a dismiss reaction from a non-bot user exists,
    or None if no dismiss reaction is found.

    This catches the race condition where a user reacts while the bot is
    busy (not in listen()), so the reaction event was never consumed.
    """
    if not _dismiss_emoji:
        return None
    try:
        result = get_client().call_endpoint(
            url=f"/messages/{message_id}",
            method="GET",
        )
        if result.get("result") != "success":
            return None
        msg = result.get("message", {})
        for reaction in msg.get("reactions", []):
            if reaction.get("emoji_name") in _dismiss_emoji and reaction.get("user_id") != bot_user_id:
                return reaction["emoji_name"]
        return None
    except Exception:
        return None


def is_stream_private(stream_name: str) -> bool:
    """Check if a stream is private (invite_only). Cached for 1 hour."""
    cache_key = ("is_stream_private", stream_name.lower())
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    # Fetch all streams and cache every one — avoids repeated API calls
    result = get_client().get_streams(include_public=True, include_subscribed=True)
    if result["result"] != "success":
        return True  # can't determine — assume private (safer)

    for s in result.get("streams", []):
        is_private = s.get("invite_only", False)
        _cache.set(("is_stream_private", s["name"].lower()), is_private, expire=3600)

    # Re-check cache after populating
    cached = _cache.get(cache_key)
    return cached if cached is not None else True  # not in results — must be private


def get_client() -> zulip.Client:
    """Get or create the Zulip client singleton.

    Config resolution order:
        1. ZULIP_RC_PATH environment variable (absolute path to zuliprc)
        2. .zuliprc in the current working directory
    """
    global _client
    if _client is None:
        import os
        import sys
        rc_env = os.environ.get("ZULIP_RC_PATH")
        if rc_env:
            config_path = Path(rc_env)
        else:
            config_path = Path.cwd() / ".zuliprc"
        if not config_path.exists():
            raise FileNotFoundError(
                f"zuliprc not found at: {config_path}\n"
                f"Set ZULIP_RC_PATH env var or place .zuliprc in the working directory."
            )
        _client = zulip.Client(config_file=str(config_path))
        # Log identity so it's visible in MCP startup output
        email = _client.email or "unknown"
        print(f"[zulipmcp] Zulip client initialized as: {email} (from {config_path})", file=sys.stderr)
    return _client


def get_user_email(full_name: str) -> Optional[str]:
    """Look up a user's email by their full name."""
    client = get_client()
    result = client.get_users()
    if result["result"] != "success":
        return None
    for user in result["members"]:
        if user["full_name"] == full_name:
            return user["email"]
    return None


def resolve_name(query: str) -> list[dict]:
    """Find active users whose display name contains the query (case-insensitive).

    Returns list of dicts with 'full_name' and 'email' for each match.
    """
    client = get_client()
    result = client.get_users()
    if result["result"] != "success":
        return []
    q = query.lower()
    matches = []
    for user in result["members"]:
        if user.get("is_bot", False) or not user.get("is_active", True):
            continue
        if q in user["full_name"].lower():
            matches.append({"full_name": user["full_name"], "email": user["email"]})
    matches.sort(key=lambda u: u["full_name"])
    return matches


# ============================================================================
# Bot visibility filtering — /nobots and /nb support
# ============================================================================

_NOBOTS_KEYWORDS = ("/nobots", "/nb")


def _should_hide_from_bot(msg: dict) -> bool:
    """Check if a message should be hidden from bots.

    Returns True if:
    - The topic contains '/nobots' or '/nb' (case-insensitive)
    - The message content starts with '/nobots' or '/nb' (case-insensitive, after stripping)
    """
    topic = msg.get("subject", "").lower()
    if any(kw in topic for kw in _NOBOTS_KEYWORDS):
        return True
    content = msg.get("content", "").strip().lower()
    if any(content.startswith(kw) for kw in _NOBOTS_KEYWORDS):
        return True
    return False


def filter_for_bot(messages: list[dict]) -> list[dict]:
    """Filter out messages that should be hidden from bots.

    Use this before displaying messages to the bot or checking if new messages exist.
    """
    return [m for m in messages if not _should_hide_from_bot(m)]


def _format_timestamp(timestamp: int, prev_timestamp: Optional[int] = None) -> str:
    """Smart timestamp formatting.

    - Always shows date on the first message
    - Shows date only when it changes from the previous message
    - Shows time only when there's a 5+ minute gap from the previous message
    - Uses the configured timezone
    """
    dt = datetime.fromtimestamp(timestamp, tz=TIMEZONE)
    show_date = True
    show_time = True

    if prev_timestamp is not None:
        prev_dt = datetime.fromtimestamp(prev_timestamp, tz=TIMEZONE)
        show_date = dt.date() != prev_dt.date()
        gap_minutes = (timestamp - prev_timestamp) / 60
        show_time = gap_minutes >= 5 or show_date

    if show_date and show_time:
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    elif show_time:
        return dt.strftime("%H:%M:%S %Z")
    else:
        return ""


def _time_attr(msg: dict, prev_timestamp: Optional[int]) -> str:
    """Build the time attribute string for a <msg> tag."""
    timestamp = msg.get("timestamp", 0)
    if "time_range_end" in msg:
        start = datetime.fromtimestamp(timestamp, tz=TIMEZONE)
        end = datetime.fromtimestamp(msg["time_range_end"], tz=TIMEZONE)
        if start.date() != end.date():
            return f'time="{start.strftime("%Y-%m-%d %H:%M:%S %Z")}-{end.strftime("%Y-%m-%d %H:%M:%S %Z")}"'
        return f'time="{start.strftime("%Y-%m-%d %H:%M:%S %Z")}-{end.strftime("%H:%M:%S %Z")}"'
    ts_str = _format_timestamp(timestamp, prev_timestamp)
    if ts_str:
        return f'time="{ts_str}"'
    return ""


def get_messages(hours_back: int = 24, channels: Optional[list[str]] = None,
                 sender: Optional[str] = None,
                 as_of: Optional[datetime] = None) -> list[dict]:
    """General-purpose paginated fetch returning raw message dicts sorted by ID.

    channels=None → all public streams
    channels=["eng", "ops"] → specific streams
    sender="John Dean" → auto-resolves name to email, filters by sender
    as_of=datetime(2025,6,1) → treat this as "now" (fetch hours_back before it)
    """
    cache_key = ("get_messages", hours_back, tuple(channels) if channels else None,
                 sender, as_of.isoformat() if as_of else None)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    client = get_client()
    anchor_time = int(as_of.timestamp()) if as_of else int(time.time())
    cutoff = anchor_time - (hours_back * 3600)

    narrow = []
    if channels is None:
        narrow.append({"operator": "channels", "operand": "public"})
    # If specific channels given, we fetch per-channel and merge
    if sender:
        email = get_user_email(sender)
        if not email:
            return []
        narrow.append({"operator": "sender", "operand": email})

    if channels is not None:
        # Fetch from each channel separately and merge
        all_messages = []
        for channel in channels:
            channel_narrow = narrow + [{"operator": "channel", "operand": channel}]
            msgs = _paginated_fetch(client, channel_narrow, cutoff, anchor_time)
            all_messages.extend(msgs)
        # Deduplicate by id and sort
        seen = set()
        unique = []
        for m in all_messages:
            if m["id"] not in seen:
                seen.add(m["id"])
                unique.append(m)
        unique.sort(key=lambda m: m["id"])
        _cache.set(cache_key, unique, expire=600)
        return unique
    else:
        result = _paginated_fetch(client, narrow, cutoff, anchor_time)
        _cache.set(cache_key, result, expire=600)
        return result


def _paginated_fetch(client: zulip.Client, narrow: list[dict],
                     cutoff: int, upper_bound: Optional[int] = None) -> list[dict]:
    """Paginate backward from newest in batches of 500 until time cutoff."""
    cache_key = ("_paginated_fetch", tuple(tuple(d.items()) for d in narrow), cutoff, upper_bound)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    all_messages = []
    anchor = "newest"
    while True:
        result = client.get_messages({
            "anchor": anchor,
            "num_before": 500,
            "num_after": 0,
            "narrow": narrow,
            "apply_markdown": False,
        })
        if result["result"] != "success":
            break
        batch = result["messages"]
        if not batch:
            break

        found_old = False
        for msg in batch:
            if upper_bound and msg["timestamp"] > upper_bound:
                continue
            if msg["timestamp"] >= cutoff:
                all_messages.append(msg)
            else:
                found_old = True

        if found_old:
            break

        # Move anchor to oldest message in this batch
        anchor = batch[0]["id"]
        # If we got fewer than requested, we've hit the beginning
        if len(batch) < 500:
            break

    all_messages.sort(key=lambda m: m["id"])
    if _ignored_streams:
        all_messages = [
            m for m in all_messages
            if m.get("display_recipient", "").lower() not in _ignored_streams
        ]
    _cache.set(cache_key, all_messages, expire=600)
    return all_messages


def format_messages(messages: list[dict], include_topic: bool = False,
                    combine: bool = True) -> str:
    """Format messages as XML <msg> tags.

    combine=True merges consecutive messages from same user within 2 min.
    Smart timestamps: date shown only when it changes, time shown on 5+ min gaps.
    include_topic=True adds stream and topic attributes to each <msg> tag.

    Note: Messages in /nobots or /nb topics or starting with /nobots or /nb are automatically
    filtered out and will not be shown to the bot.
    """
    if not messages:
        return ""

    # Filter out messages that should be hidden from bots
    messages = filter_for_bot(messages)
    if not messages:
        return ""

    if combine:
        messages = _combine_messages(messages)

    lines = []
    prev_timestamp = None
    for msg in messages:
        sender = msg.get("sender_full_name", "Unknown")
        msg_id = msg.get("id", "")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", 0)

        # Build the time attribute
        time_attr = _time_attr(msg, prev_timestamp)

        attrs = [f'user="{sender}"']
        if time_attr:
            attrs.append(time_attr)
        if include_topic and msg.get("type") == "stream":
            stream_name = msg.get("display_recipient", "")
            visibility = "private" if is_stream_private(stream_name) else "public"
            attrs.append(f'stream="{stream_name}"')
            attrs.append(f'visibility="{visibility}"')
            attrs.append(f'topic="{msg.get("subject", "")}"')
        attrs.append(f'id="{msg_id}"')

        # Add reaction summary as attribute if present
        reaction_str = summarize_reactions(msg.get("reactions", []))
        if reaction_str:
            attrs.append(f'reactions="{reaction_str}"')

        tag_open = "<msg " + " ".join(attrs) + ">"
        lines.append(f"{tag_open}\n{content}\n</msg>")

        prev_timestamp = msg.get("time_range_end", timestamp)

    return "\n".join(lines)


def _combine_messages(messages: list[dict]) -> list[dict]:
    """Merge consecutive messages from same user within 2 minutes."""
    if not messages:
        return []

    combined = []
    current = dict(messages[0])
    current["reactions"] = list(current.get("reactions", []))

    for msg in messages[1:]:
        same_user = msg.get("sender_full_name") == current.get("sender_full_name")
        time_gap = msg.get("timestamp", 0) - current.get("time_range_end", current.get("timestamp", 0))

        if same_user and time_gap <= 120:
            # Merge: append content, update end time, id, and reactions
            current["content"] = current["content"] + "\n\n" + msg.get("content", "")
            current["time_range_end"] = msg.get("timestamp", 0)
            current["id"] = msg.get("id", current["id"])
            current["reactions"].extend(msg.get("reactions", []))
        else:
            combined.append(current)
            current = dict(msg)
            current["reactions"] = list(current.get("reactions", []))

    combined.append(current)
    return combined


def _context_window(messages: list[dict], sender_email: str) -> list[dict]:
    """Keep messages around where sender participated.

    Before each user message: 3 messages OR 10 min (whichever captures more).
    After each user message: 3 messages OR (2 hours AND <10 messages).
    """
    if not messages:
        return []

    include = set()
    sender_indices = [i for i, m in enumerate(messages)
                      if m.get("sender_email") == sender_email]

    for idx in sender_indices:
        include.add(idx)
        ts = messages[idx]["timestamp"]

        # Before: 3 messages or 10 min
        for i in range(idx - 1, -1, -1):
            if (idx - i) <= 3 or (ts - messages[i]["timestamp"]) <= 600:
                include.add(i)
            else:
                break

        # After: 3 messages or (2hr and <10 messages)
        for i in range(idx + 1, len(messages)):
            dist = i - idx
            if dist <= 3 or (messages[i]["timestamp"] - ts <= 7200 and dist < 10):
                include.add(i)
            else:
                break

    return [messages[i] for i in sorted(include)]


def format_topics(messages: list[dict], combine: bool = True) -> str:
    """Group messages by stream>topic, format each group.

    Topics sorted by last message ID. Groups separated by ---.
    """
    if not messages:
        return ""

    # Group by (stream, topic)
    groups: dict[tuple[str, str], list[dict]] = {}
    for msg in messages:
        stream = msg.get("display_recipient", "unknown")
        topic = msg.get("subject", "unknown")
        key = (stream, topic)
        if key not in groups:
            groups[key] = []
        groups[key].append(msg)

    # Sort groups by last message ID
    sorted_groups = sorted(groups.items(), key=lambda item: item[1][-1]["id"])

    sections = []
    for (stream, topic), group_msgs in sorted_groups:
        visibility = "[private]" if is_stream_private(stream) else "[public]"
        header = f"Stream: {visibility} {stream}, Topic: {topic}"
        body = format_messages(group_msgs, include_topic=False, combine=combine)
        sections.append(f"{header}\n{body}")

    return "\n---\n".join(sections)


def get_messages_formatted(hours_back: int = 24, channels: Optional[list[str]] = None,
                           sender: Optional[str] = None,
                           group_by_topic: bool = True,
                           as_of: Optional[datetime] = None) -> str:
    """Convenience: fetch + format in one call.

    When sender is set with group_by_topic=True, fetches full conversation
    context around the sender's messages in each topic they participated in.
    as_of=datetime(2025,6,1) → treat this as "now" (fetch hours_back before it)
    """
    if sender and group_by_topic:
        return _get_sender_conversations(hours_back, channels, sender, as_of)

    messages = get_messages(hours_back=hours_back, channels=channels, sender=sender, as_of=as_of)
    if not messages:
        return "No messages found."
    if group_by_topic:
        return format_topics(messages)
    else:
        return format_messages(messages, include_topic=True)


def _get_sender_conversations(hours_back: int, channels: Optional[list[str]],
                              sender: str, as_of: Optional[datetime] = None) -> str:
    """Two-pass fetch: find sender's topics, then get full context."""
    client = get_client()
    email = get_user_email(sender)
    if not email:
        return "No messages found."

    # Pass 1: get sender's messages to discover topics
    sender_messages = get_messages(hours_back=hours_back, channels=channels, sender=sender, as_of=as_of)
    if not sender_messages:
        return "No messages found."

    # Extract unique (stream, topic) pairs
    topics = set()
    for msg in sender_messages:
        if msg.get("type") == "stream":
            topics.add((msg["display_recipient"], msg["subject"]))

    # Pass 2: fetch all messages from each topic, apply context window
    anchor_time = int(as_of.timestamp()) if as_of else int(time.time())
    cutoff = anchor_time - (hours_back * 3600)
    all_context_messages = []
    for stream, topic in topics:
        narrow = [
            {"operator": "channel", "operand": stream},
            {"operator": "topic", "operand": topic},
        ]
        topic_messages = _paginated_fetch(client, narrow, cutoff, anchor_time)
        windowed = _context_window(topic_messages, email)
        all_context_messages.extend(windowed)

    if not all_context_messages:
        return "No messages found."

    return format_topics(all_context_messages)


# ============================================================================
# Direct API wrappers — return raw Python objects for composability
# ============================================================================

def get_profile() -> dict:
    """Get the bot's own profile. Returns the profile dict from the API."""
    return get_client().get_profile()


def get_own_status() -> dict:
    """Get the status of the currently authenticated Zulip user."""
    client = get_client()
    profile = client.get_profile()
    if profile.get("result") != "success":
        return profile

    user_id = profile.get("user_id")
    if user_id is None:
        return {
            "result": "error",
            "msg": "Authenticated Zulip profile did not include user_id",
        }

    return client.call_endpoint(
        url=f"/users/{user_id}/status",
        method="GET",
    )


def update_own_status(
    status_text: str,
    emoji_name: Optional[str] = None,
    emoji_code: Optional[str] = None,
    reaction_type: Optional[str] = None,
) -> dict:
    """Update the status of the currently authenticated Zulip user.

    Optional emoji fields are omitted when None, preserving the existing
    status emoji. Passing empty strings explicitly clears those fields.
    """
    request: dict[str, object] = {
        "status_text": status_text,
    }

    if emoji_name is not None:
        request["emoji_name"] = emoji_name
    if emoji_code is not None:
        request["emoji_code"] = emoji_code
    if reaction_type is not None:
        request["reaction_type"] = reaction_type

    return get_client().call_endpoint(
        url="/users/me/status",
        method="POST",
        request=request,
    )


def clear_own_status() -> dict:
    """Clear both status text and status emoji for the authenticated user."""
    return update_own_status(
        status_text="",
        emoji_name="",
        emoji_code="",
        reaction_type="unicode_emoji",
    )


def list_streams(include_private: bool = False) -> list[dict]:
    """List available streams. Returns list of stream dicts."""
    result = get_client().get_streams(include_public=True, include_subscribed=True)
    if result["result"] != "success":
        return []
    streams = result.get("streams", [])
    allowed = _allowed_private_streams()
    if _ALL_PRIVATE_STREAMS not in allowed:
        streams = [
            s for s in streams
            if not s.get("invite_only", False) or s["name"].lower() in allowed
        ]
    if not include_private:
        streams = [s for s in streams if not s.get("invite_only", False)]
    return sorted(streams, key=lambda s: s["name"])


def get_stream_topics(stream: str, limit: int = 20) -> list[dict]:
    """Get recent topics in a stream. Returns list of topic dicts.

    Note: Topics containing '/nobots' or '/nb' are filtered out and will not be shown to bots.
    """
    if not is_private_stream_allowed(stream):
        raise ValueError(f"Private stream access denied: {stream}")
    client = get_client()
    result = client.get_stream_id(stream)
    if result["result"] != "success":
        raise ValueError(f"Stream '{stream}' not found: {result.get('msg', '')}")
    result = client.get_stream_topics(result["stream_id"])
    if result["result"] != "success":
        raise ValueError(f"Error fetching topics: {result.get('msg', '')}")
    topics = result.get("topics", [])
    # Filter out /nobots and /nb topics
    topics = [t for t in topics
              if not any(kw in t.get("name", "").lower() for kw in _NOBOTS_KEYWORDS)]
    return topics[:limit]


def get_topic_messages(stream: str, topic: str, num_messages: int = 20,
                       before_message_id: Optional[int] = None) -> list[dict]:
    """Get messages from a specific stream/topic. Returns raw message dicts sorted by ID."""
    if not is_private_stream_allowed(stream):
        return []
    result = get_client().get_messages({
        "narrow": [
            {"operator": "stream", "operand": stream},
            {"operator": "topic", "operand": topic},
        ],
        "anchor": before_message_id or "newest",
        "num_before": min(num_messages, 100),
        "num_after": 0,
        "apply_markdown": False,
    })
    if result["result"] != "success":
        return []
    msgs = result.get("messages", [])
    msgs.sort(key=lambda m: m["id"])
    return msgs


def fetch_new_messages(stream: str, topic: str, after_id: int,
                       exclude_user_id: Optional[int] = None) -> list[dict]:
    """Fetch messages after a given ID, optionally excluding a user. Sorted by ID."""
    if not is_private_stream_allowed(stream):
        return []
    result = get_client().get_messages({
        "narrow": [
            {"operator": "stream", "operand": stream},
            {"operator": "topic", "operand": topic},
        ],
        "anchor": after_id,
        "num_before": 0,
        "num_after": 100,
        "include_anchor": False,
        "apply_markdown": False,
    })
    if result["result"] != "success":
        return []
    msgs = result.get("messages", [])
    if exclude_user_id:
        msgs = [m for m in msgs if m.get("sender_id") != exclude_user_id]
    msgs.sort(key=lambda m: m["id"])
    return msgs


def send_message(stream: str, topic: str, content: str) -> dict:
    """Send a message. Returns API result dict with 'id' on success."""
    err = _stream_write_error(stream)
    if err:
        return err
    return get_client().send_message({
        "type": "stream",
        "to": stream,
        "subject": topic,
        "content": normalize_zulip_markdown(content),
    })


def send_direct_message(recipients: list[str], content: str) -> dict:
    """Send a direct message (DM) to one or more users.

    Args:
        recipients: List of email addresses to send to.
        content: Message content (supports Zulip markdown).

    Returns:
        API result dict with 'id' on success.
    """
    return get_client().send_message({
        "type": "direct",
        "to": recipients,
        "content": normalize_zulip_markdown(content),
    })


def add_reaction(message_id: int, emoji_name: str) -> dict:
    """Add emoji reaction to a message. Returns API result dict."""
    return get_client().add_reaction({
        "message_id": message_id,
        "emoji_name": emoji_name,
    })


def remove_reaction(message_id: int, emoji_name: str,
                     reaction_type: Optional[str] = None) -> dict:
    """Remove emoji reaction from a message. Returns API result dict.

    Pass *reaction_type* (``"unicode_emoji"``, ``"realm_emoji"``, or
    ``"zulip_extra_emoji"``) for non-unicode emoji — the server defaults
    to ``"unicode_emoji"`` when omitted, so custom-emoji removals fail
    without it.
    """
    params: dict = {
        "message_id": message_id,
        "emoji_name": emoji_name,
    }
    if reaction_type is not None:
        params["reaction_type"] = reaction_type
    return get_client().remove_reaction(params)


def edit_message(message_id: int, content: str) -> dict:
    """Edit a message's content. Returns API result dict."""
    _, err = _get_stream_message_for_write(message_id)
    if err:
        return err
    return get_client().update_message({
        "message_id": message_id,
        "content": normalize_zulip_markdown(content),
    })


def move_messages(message_id: int, topic: str, stream: Optional[str] = None,
                  propagate_mode: str = "change_one", notify: bool = True) -> dict:
    """Move message(s) to a different topic and/or stream.

    Uses the Zulip update_message API with topic/stream_id params.

    Args:
        message_id: The anchor message to move.
        topic: Destination topic name.
        stream: Destination stream name (optional, only for cross-channel moves).
        propagate_mode: "change_one", "change_later", or "change_all".
        notify: Send notifications to old and new threads (default True).
            Set to False for silent renames like topic resolution.

    Returns:
        API result dict.
    """
    _, err = _get_stream_message_for_write(message_id)
    if err:
        return err

    request: dict = {
        "message_id": message_id,
        "topic": topic,
        "propagate_mode": propagate_mode,
        "send_notification_to_old_thread": notify,
        "send_notification_to_new_thread": notify,
    }
    if stream:
        err = _stream_write_error(stream)
        if err:
            return err
        client = get_client()
        result = client.get_stream_id(stream)
        if result["result"] != "success":
            return result
        request["stream_id"] = result["stream_id"]
    return get_client().update_message(request)


_stream_id_cache: dict[str, int] = {}


def send_typing(stream: str, topic: str, op: str = "start") -> dict:
    """Send a typing indicator start/stop to a stream/topic.

    Args:
        stream: Stream name.
        topic: Topic name.
        op: "start" or "stop".

    Returns API result dict.
    """
    client = get_client()
    stream_id = _stream_id_cache.get(stream)
    if stream_id is None:
        stream_result = client.get_stream_id(stream)
        if stream_result["result"] != "success":
            return stream_result
        stream_id = stream_result["stream_id"]
        _stream_id_cache[stream] = stream_id
    return client.call_endpoint(
        url="/typing",
        method="POST",
        request={
            "op": op,
            "type": "stream",
            "stream_id": stream_id,
            "topic": topic,
        },
    )


def list_emoji(query: str = "") -> tuple[list[str], int]:
    """List custom emoji, optionally filtered by substring.

    Returns (matching_names, total_count).
    """
    client = get_client()
    result = client.get_realm_emoji()
    if result.get("result") != "success":
        return [], 0
    all_emoji = sorted(
        info["name"]
        for info in result.get("emoji", {}).values()
        if not info.get("deactivated", False)
    )
    total = len(all_emoji)
    if query:
        q = query.lower()
        return [name for name in all_emoji if q in name.lower()], total
    return all_emoji, total


def get_emoji_info() -> tuple[int, set[str]]:
    """Return (count, names) of active custom emoji.

    Single API call — use this instead of separate count/has checks.
    Returns ``(0, set())`` on error.
    """
    try:
        result = get_client().get_realm_emoji()
        if result.get("result") != "success":
            return 0, set()
        names = {
            info["name"]
            for info in result.get("emoji", {}).values()
            if not info.get("deactivated", False)
        }
        return len(names), names
    except Exception:
        return 0, set()


def get_emoji_count() -> int:
    """Return count of active custom emoji. Returns 0 on error."""
    count, _ = get_emoji_info()
    return count


def summarize_reactions(reactions: list) -> str:
    """Summarize a list of Zulip reaction dicts into a compact string.

    Returns e.g. ":thumbs_up: x2  :check:" or empty string if none.
    """
    counts: dict[str, int] = {}
    for r in reactions:
        name = r.get("emoji_name", "")
        if name:
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        return ""
    return "  ".join(
        f":{name}: x{n}" if n > 1 else f":{name}:"
        for name, n in counts.items()
    )


def check_reactions_on(message_id: int) -> str:
    """Fetch a single message by ID and return its reaction summary, or empty string."""
    try:
        result = get_client().call_endpoint(
            url=f"/messages/{message_id}",
            method="GET",
        )
        if result.get("result") != "success":
            return ""
        msg = result.get("message", {})
        reaction_str = summarize_reactions(msg.get("reactions", []))
        if not reaction_str:
            return ""
        return f"Reactions on message {message_id}: {reaction_str}"
    except Exception:
        return ""


def discover_message_context(message_id: int) -> Optional[tuple[str, str]]:
    """Look up a message by ID and return its (stream, topic), or None for DMs/errors."""
    result = get_client().get_messages({
        "anchor": message_id,
        "num_before": 0,
        "num_after": 0,
        "include_anchor": True,
        "apply_markdown": False,
    })
    if result.get("result") != "success" or not result.get("messages"):
        return None
    target = result["messages"][0]
    if target.get("type") != "stream":
        return None
    stream = target.get("display_recipient", "")
    topic = target.get("subject", "")
    if not stream or not topic:
        return None
    return stream, topic


def get_custom_profile_fields() -> list[dict]:
    """Get the org's custom profile field definitions. Cached aggressively."""
    cache_key = ("custom_profile_fields",)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    result = get_client().call_endpoint(url="/realm/profile_fields", method="GET")
    if result.get("result") != "success":
        return []
    fields = result.get("custom_fields", [])
    _cache.set(cache_key, fields, expire=3600)
    return fields


def get_user_info(email: str) -> Optional[dict]:
    """Get user info by email, including custom profile fields. Returns user dict or None."""
    result = get_client().call_endpoint(
        url=f"/users/{email}",
        method="GET",
        request={"include_custom_profile_fields": True},
    )
    if result.get("result") != "success":
        return None
    return result["user"]


def get_message_by_id(message_id: int) -> Optional[dict]:
    """Get a specific message by ID. Returns message dict or None.

    Uses get_messages with apply_markdown=False so the content field contains
    raw markdown (consistent with get_topic_messages), not rendered HTML.
    """
    result = get_client().get_messages({
        "anchor": message_id,
        "num_before": 0,
        "num_after": 0,
        "include_anchor": True,
        "apply_markdown": False,
    })
    if result.get("result") != "success" or not result.get("messages"):
        return None
    msg = result["messages"][0]
    if msg.get("type") == "stream":
        stream = msg.get("display_recipient", "")
        if stream and not is_private_stream_allowed(stream):
            return None
    return msg


def verify_message(message_id: int) -> str:
    """Fetch a message by ID and return it in a secure, tamper-evident format.

    The sender identity comes directly from the Zulip API and cannot be
    spoofed by message content. All '#' and '@' characters are stripped from
    the message body so that the structural delimiters (##### lines and
    @FIELD labels) cannot be forged within the content.

    Returns a formatted string with verified metadata and sanitized content,
    or an error string if the message cannot be fetched.
    """
    msg = get_message_by_id(message_id)
    if not msg:
        return f"Error: Message {message_id} not found."

    sender_name = msg.get("sender_full_name", "Unknown")
    sender_email = msg.get("sender_email", "Unknown")
    sender_id = msg.get("sender_id", "Unknown")
    timestamp = msg.get("timestamp", 0)
    ts_str = datetime.fromtimestamp(timestamp, tz=TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z") if timestamp else "Unknown"
    stream = msg.get("display_recipient", "Unknown") if msg.get("type") == "stream" else "DM"
    topic = msg.get("subject", "Unknown") if msg.get("type") == "stream" else "N/A"

    # Strip # and @ so delimiters can't be forged in content
    content = msg.get("content", "")
    content = content.replace("#", "").replace("@", "")

    return (
        f"##### VERIFIED MESSAGE BEGIN #####\n"
        f"@SENDER NAME: {sender_name}\n"
        f"@SENDER EMAIL: {sender_email}\n"
        f"@SENDER USER ID: {sender_id}\n"
        f"@TIMESTAMP: {ts_str}\n"
        f"@STREAM: {stream}\n"
        f"@TOPIC: {topic}\n"
        f"@MESSAGE ID: {message_id}\n"
        f"##### CONTENT #####\n"
        f"{content}\n"
        f"##### VERIFIED MESSAGE END #####"
    )


_hash_replacements = {"%": ".", "(": ".28", ")": ".29", ".": ".2E"}


def _encode_hash_component(s: str) -> str:
    """Encode a string for Zulip URL hash fragments (MediaWiki-style dot-encoding)."""
    encoded = urllib.parse.quote(s, safe="")
    return "".join(_hash_replacements.get(c, c) for c in encoded)


def get_message_link(message_id: int) -> str:
    """Return a Zulip markdown link to a message: [#stream > topic](url).

    The URL links to the full conversation context with the specific message
    focused, using the format:
        {base}/#narrow/channel/{stream_id}-{stream_slug}/topic/{topic_encoded}/near/{message_id}

    Falls back to a simpler URL format if stream_id lookup fails.
    """
    client = get_client()
    base = client.base_url.rstrip("/")
    if base.endswith("/api"):
        base = base[:-4]

    msg = get_message_by_id(message_id)
    if not msg:
        url = f"{base}/#narrow/id/{message_id}"
        return f"[message]({url})"

    stream = msg.get("display_recipient", "")
    topic = msg.get("subject", "")

    stream_id = None
    try:
        result = client.get_stream_id(stream)
        if result.get("result") == "success":
            stream_id = result["stream_id"]
    except Exception:
        pass

    if stream_id:
        topic_encoded = _encode_hash_component(topic)
        stream_slug = _encode_hash_component(stream.replace(" ", "-"))
        url = f"{base}/#narrow/channel/{stream_id}-{stream_slug}/topic/{topic_encoded}/near/{message_id}"
    else:
        url = f"{base}/#narrow/id/{message_id}"

    return f"[#{stream} > {topic}]({url})"


def get_subscribed_streams() -> list[dict]:
    """Get streams the bot is subscribed to. Returns list of subscription dicts."""
    result = get_client().get_subscriptions()
    if result["result"] != "success":
        return []
    subs = result.get("subscriptions", [])
    allowed = _allowed_private_streams()
    if allowed is not None:
        subs = [
            s for s in subs
            if not s.get("invite_only", False) or s["name"].lower() in allowed
        ]
    return sorted(subs, key=lambda s: s["name"])


def get_stream_members(stream: str) -> list[dict]:
    """Get members of a stream. Returns list of user dicts with 'full_name' and 'email'."""
    if not is_private_stream_allowed(stream):
        raise ValueError(f"Private stream access denied: {stream}")
    client = get_client()
    result = client.get_stream_id(stream)
    if result["result"] != "success":
        raise ValueError(f"Stream '{stream}' not found: {result.get('msg', '')}")
    stream_id = result["stream_id"]
    result = client.call_endpoint(url=f"/streams/{stream_id}/members", method="GET")
    if result.get("result") != "success":
        raise ValueError(f"Error fetching members: {result.get('msg', '')}")
    # Resolve user IDs to names/emails
    user_ids = set(result.get("subscribers", []))
    users_result = client.get_users()
    if users_result["result"] != "success":
        return []
    members = []
    for user in users_result["members"]:
        if user["user_id"] in user_ids:
            members.append({"full_name": user["full_name"], "email": user["email"]})
    return sorted(members, key=lambda u: u["full_name"])


def download_file(path: str) -> tuple[bytes, str]:
    """Download a file from Zulip. Returns (content_bytes, content_type).

    Raises ValueError on any download error.
    """
    client = get_client()
    client.ensure_session()
    if client.session is None:
        raise ValueError("Failed to create Zulip session")

    base = client.base_url.rstrip("/")
    if base.endswith("/api"):
        base = base[:-4]
    if not path.startswith("/"):
        path = "/" + path

    response = client.session.get(base + path, timeout=30)
    if response.status_code == 403:
        raise ValueError(f"Access denied (403). The bot may not have permission to access: {path}")
    elif response.status_code == 404:
        raise ValueError(f"File not found (404): {path}")
    elif response.status_code != 200:
        raise ValueError(f"HTTP {response.status_code}: {response.text[:200]}")
    content_type = response.headers.get("Content-Type", "application/octet-stream")
    return response.content, content_type


def save_file(path: str, save_dir: Optional[str] = None) -> tuple[str, int, str]:
    """Download a Zulip file and save it locally.

    Returns (saved_path, size_bytes, content_type).
    """
    content, content_type = download_file(path)
    filename = Path(path).name

    if save_dir:
        save_path = Path(save_dir) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        save_path = Path(tempfile.mkdtemp()) / filename

    save_path.write_bytes(content)
    return str(save_path), len(content), content_type


def save_image(path: str) -> str:
    """Download a Zulip image and save to a temp file. Returns the temp file path."""
    content, content_type = download_file(path)

    ext_map = {
        "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
        "image/webp": ".webp", "image/svg+xml": ".svg",
    }
    ext = ext_map.get((content_type or "").split(";")[0].strip(), "")
    if not ext:
        ext = Path(path).suffix.lower() if Path(path).suffix.lower() in ext_map.values() else ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
        f.write(content)
        return f.name


def upload_file(file_path: str) -> tuple[str, str]:
    """Upload a local file to Zulip.

    Args:
        file_path: Absolute path to the file to upload.

    Returns:
        Tuple of (uri, filename) where uri is the Zulip upload path like
        "/user_uploads/2/ab/cdef/image.png".

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the upload fails.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    client = get_client()
    with open(path, "rb") as f:
        result = client.upload_file(f)

    if result.get("result") != "success":
        raise ValueError(f"Upload failed: {result.get('msg', 'Unknown error')}")

    return result["uri"], path.name


# ============================================================================
# Event queue — long-polling for real-time message delivery
# ============================================================================

def is_bot_subscribed(stream: str) -> bool:
    """Check if the bot is currently subscribed to a stream."""
    result = get_client().get_subscriptions()
    if result.get("result") != "success":
        return False
    return any(sub["name"].lower() == stream.lower()
               for sub in result.get("subscriptions", []))


def ensure_subscribed(stream: str) -> bool:
    """Subscribe the bot to a stream. No-op if already subscribed.

    For public streams, auto-subscribes. For private streams (or streams
    the bot can't see), returns True only if already subscribed.
    """
    # Fast path: already subscribed
    if is_bot_subscribed(stream):
        return True
    # Private streams can't be joined — must be invited
    if is_stream_private(stream):
        return False
    # Try to subscribe (works for public streams)
    result = get_client().add_subscriptions(
        streams=[{"name": stream}],
    )
    return result.get("result") == "success"


def register_event_queue(stream: str, topic: str) -> tuple[str, int, int]:
    """Register an event queue narrowed to a stream/topic.

    Returns (queue_id, last_event_id, longpoll_timeout_seconds).
    Raises ValueError on failure.
    """
    result = get_client().call_endpoint(
        url="/register",
        method="POST",
        request={
            "event_types": json.dumps(["message", "reaction"]),
            "narrow": json.dumps([
                ["channel", stream],
                ["topic", topic],
            ]),
            "apply_markdown": False,
        },
    )
    if result.get("result") != "success":
        raise ValueError(f"Failed to register event queue: {result.get('msg', '')}")
    timeout = result.get("event_queue_longpoll_timeout_seconds", 90)
    return result["queue_id"], result["last_event_id"], timeout


def get_events(queue_id: str, last_event_id: int,
               longpoll_timeout: int = 90) -> tuple[list[dict], list[dict], int]:
    """Long-poll for events. Blocks until events arrive or server heartbeat.

    Returns (messages, reactions, new_last_event_id). Lists may be empty
    (heartbeat only). Raises ValueError on BAD_EVENT_QUEUE_ID (caller
    should re-register).
    """
    try:
        result = get_client().call_endpoint(
            url="/events",
            method="GET",
            request={
                "queue_id": queue_id,
                "last_event_id": last_event_id,
            },
            timeout=longpoll_timeout + 30,  # HTTP timeout > server timeout
        )
    except requests.exceptions.ReadTimeout:
        return [], [], last_event_id
    if result.get("result") != "success":
        code = result.get("code", "")
        if code == "BAD_EVENT_QUEUE_ID":
            raise ValueError("BAD_EVENT_QUEUE_ID")
        raise ValueError(f"get_events failed: {result.get('msg', '')}")
    events = result.get("events", [])
    messages = []
    reactions = []
    new_last = last_event_id
    for ev in events:
        new_last = max(new_last, ev.get("id", new_last))
        if ev.get("type") == "message":
            messages.append(ev["message"])
        elif ev.get("type") == "reaction":
            reactions.append(ev)
    return messages, reactions, new_last


def delete_event_queue(queue_id: str) -> None:
    """Delete an event queue. Ignores errors (best-effort cleanup)."""
    try:
        get_client().call_endpoint(
            url="/events",
            method="DELETE",
            request={"queue_id": queue_id},
        )
    except Exception:
        pass
