"""Custom Subgroups Feature.

Lets users create named subgroups (e.g. ``blogmaintainers``) and populate
them with WhatsApp users.  When someone writes ``@blogmaintainers`` in any
group the bot is in, every member of that subgroup receives a silent
notification (ghost mention) — their names do not appear in the message.

Subgroups are global (not per-group) and persisted in PostgreSQL.

Commands:
  !add-subgroup <name> | @user1 @user2 …
  !remove-from-subgroup <name> | @user1 @user2 …
  !delete-subgroup <name>
  !list-subgroups
  !subgroup-info <name>
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from neonize.proto.waE2E.WAWebProtobufsE2E_pb2 import (
    ContextInfo,
    ExtendedTextMessage,
    Message,
)

from db.subgroup_store import SubgroupStore
from db.auth import gate, get_active_admin_jids, normalize_jid

if TYPE_CHECKING:
    from neonize.client import NewClient
    from neonize.events import MessageEv

log = logging.getLogger(__name__)

# Subgroup names: alphanumeric, hyphens, underscores (2-32 chars).
# Natural-language compilation normalizes spaces/punctuation to hyphens before
# reaching this validator, so names such as "2nd year" become "2nd-year".
_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,31}$")


def normalize_collection_name(raw: str) -> str | None:
    """Convert a natural-language collection name to the stored name grammar."""
    if not isinstance(raw, str):
        return None
    value = raw.strip().lstrip("@").casefold()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    if not _NAME_RE.fullmatch(value):
        return None
    return value[:32].rstrip("-_") or None

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _read_subgroups(store: SubgroupStore) -> dict[str, list[str]]:
    return store.read()


def _write_subgroups(store: SubgroupStore, data: dict[str, list[str]]) -> None:
    store.write(data)


# ---------------------------------------------------------------------------
# Message text extraction
# ---------------------------------------------------------------------------

def _get_text(message: "MessageEv") -> str:
    """Extract plain text body from a message."""
    msg = message.Message
    if msg.conversation:
        return msg.conversation.strip()
    if msg.extendedTextMessage and msg.extendedTextMessage.text:
        return msg.extendedTextMessage.text.strip()
    if msg.imageMessage and msg.imageMessage.caption:
        return msg.imageMessage.caption.strip()
    return ""


def _get_mentioned_jids(message: "MessageEv") -> list[str]:
    """Extract mentionedJID strings from the message's contextInfo."""
    msg = message.Message
    mentions: list[str] = []

    # Walk through wrapper layers (ephemeral, viewOnce, etc.)
    for _ in range(5):
        found_wrapper = False
        for field_desc, value in msg.ListFields():
            if field_desc.name in (
                "ephemeralMessage",
                "viewOnceMessage",
                "viewOnceMessageV2",
                "documentWithCaptionMessage",
                "groupMentionedMessage",
            ):
                inner = getattr(value, "message", None)
                if inner and inner.ListFields():
                    msg = inner
                    found_wrapper = True
                    break
        if not found_wrapper:
            break

    # Look for contextInfo on the actual message type
    for field_desc, value in msg.ListFields():
        if field_desc.name.endswith("Message"):
            ctx = getattr(value, "contextInfo", None)
            if ctx is not None and ctx.ListFields():
                mentions.extend(ctx.mentionedJID)
                break

    # Natural-language test messages may use the literal ``@me`` alias even
    # when WhatsApp has not produced mention metadata for it. The translator
    # attaches the sender JID to the cloned message so normal command handlers
    # can resolve the alias exactly like a native WhatsApp mention.
    sender_alias = getattr(message, "_pbbot_me_jid", "")
    if sender_alias and sender_alias not in mentions:
        mentions.append(sender_alias)
    return mentions


# ---------------------------------------------------------------------------
# Reply helper
# ---------------------------------------------------------------------------

def _reply(client: "NewClient", chat_jid, text: str) -> None:
    client.send_message(chat_jid, text)


# ---------------------------------------------------------------------------
# JID Resolution Helpers
# ---------------------------------------------------------------------------

def _resolve_lid_to_pn(client: "NewClient", jid: str) -> str:
    """Convert a WhatsApp LID to a Phone JID."""
    normalized = normalize_jid(jid)
    if not normalized or not normalized.endswith("@lid"):
        return normalized
    try:
        from db.work_store import _JID_ALIASES
        from db.auth import jid_user
        user_part = jid_user(normalized)
        if user_part in _JID_ALIASES:
            return f"{_JID_ALIASES[user_part]}@s.whatsapp.net"
    except Exception:
        pass
    return normalized

def _resolve_pn_to_lid(client: "NewClient", jid: str) -> str:
    """Convert a WhatsApp Phone JID to an LID."""
    normalized = normalize_jid(jid)
    if not normalized or not normalized.endswith("@s.whatsapp.net"):
        return normalized
    try:
        from db.work_store import _JID_ALIASES
        from db.auth import jid_user
        user_part = jid_user(normalized)
        for lid_u, phone_u in _JID_ALIASES.items():
            if phone_u == user_part:
                return f"{lid_u}@lid"
    except Exception:
        pass
    return normalized

# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def add_subgroup_members(
    store: SubgroupStore,
    name: str,
    member_jids: list[str],
) -> tuple[int, int]:
    """Create/update a subgroup from concrete runtime member JIDs.

    This is the domain operation shared by the legacy command handler and the
    natural-language executor. It deliberately does not depend on a WhatsApp
    message or synthetic mention metadata.
    """
    name = name.strip().lstrip("@").lower()
    if not _NAME_RE.fullmatch(name):
        raise ValueError(
            "Subgroup name must be 2-32 characters: letters, digits, hyphens, "
            "or underscores, starting with a letter."
        )
    members = list(dict.fromkeys(
        normalize_jid(jid) for jid in member_jids if normalize_jid(jid)
    ))
    if not members:
        raise ValueError("Mention at least one user after the pipe.")

    result = store.add_members(name, members)
    if isinstance(result, tuple) and len(result) == 2:
        added, total = result
        return len(added), total
    # Compatibility for lightweight store doubles and old integrations.
    subgroups = _read_subgroups(store)
    existing = list(subgroups.get(name, []))
    existing_keys = {normalize_jid(jid) for jid in existing}
    added = [jid for jid in members if jid not in existing_keys]
    subgroups[name] = existing + added
    _write_subgroups(store, subgroups)
    return len(added), len(subgroups[name])


def remove_subgroup_members(
    store: SubgroupStore,
    name: str,
    member_jids: list[str],
) -> tuple[int, int, bool]:
    """Remove concrete members and report (removed, remaining, deleted)."""
    name = name.strip().lstrip("@").lower()
    if not _NAME_RE.fullmatch(name):
        raise ValueError("invalid subgroup name")
    members = {
        normalize_jid(jid) for jid in member_jids if normalize_jid(jid)
    }
    if not members:
        raise ValueError("Mention at least one user after the pipe.")
    try:
        result = store.remove_members(name, members)
        if isinstance(result, tuple) and len(result) == 3:
            return result
    except ValueError as exc:
        raise ValueError(f"Subgroup {name} does not exist.") from exc
    # Compatibility for lightweight store doubles and old integrations.
    subgroups = _read_subgroups(store)
    if name not in subgroups:
        raise ValueError(f"Subgroup {name} does not exist.")
    original = subgroups[name]
    remaining = [jid for jid in original if normalize_jid(jid) not in members]
    removed = len(original) - len(remaining)
    deleted = not remaining
    if deleted:
        del subgroups[name]
    else:
        subgroups[name] = remaining
    _write_subgroups(store, subgroups)
    return removed, len(remaining), deleted


def _cmd_add_subgroup(
    client, chat_jid, body: str, mentioned_jids: list[str], store: SubgroupStore
) -> None:
    """!add-subgroup <name> | @user1 @user2 … or !add-subgroup <name> | @everyone"""
    parts = body.split("|", 1)
    raw_name = parts[0].strip().lstrip("@")

    if not raw_name:
        _reply(client, chat_jid, "⚠️ Usage: `!add-subgroup <name> | @user1 @user2 …` or `!add-subgroup <name> | @everyone`")
        return

    name = raw_name.lower()

    if not _NAME_RE.match(name):
        _reply(
            client, chat_jid,
            "⚠️ Subgroup name must be 2-32 characters: letters, digits, hyphens, "
            "or underscores, starting with a letter.",
        )
        return

    after_pipe = parts[1].strip() if len(parts) > 1 else ""
    if re.search(r"@everyone\b|@all\b", after_pipe, re.IGNORECASE):
        from features.community_tag import get_group_member_jids
        everyone_jids = get_group_member_jids(client, chat_jid)
        mentioned_jids = list(dict.fromkeys(mentioned_jids + everyone_jids))

    try:
        added_count, total = add_subgroup_members(store, name, mentioned_jids)
    except ValueError as exc:
        _reply(client, chat_jid, f"⚠️ {exc}")
        return

    if added_count:
        _reply(
            client, chat_jid,
            f"✅ Added {added_count} member(s) to *@{name}* (total: {total}).",
        )
    else:
        _reply(client, chat_jid, f"ℹ️ All mentioned users are already in *@{name}* ({total} members).")


def _cmd_remove_from_subgroup(
    client, chat_jid, body: str, mentioned_jids: list[str], store: SubgroupStore
) -> None:
    """!remove-from-subgroup <name> | @user1 @user2 … or !remove-from-subgroup <name> | @everyone"""
    parts = body.split("|", 1)
    raw_name = parts[0].strip().lstrip("@")
    name = raw_name.lower()

    if not name:
        _reply(client, chat_jid, "⚠️ Usage: `!remove-from-subgroup <name> | @user1 @user2 …`")
        return

    after_pipe = parts[1].strip() if len(parts) > 1 else ""
    if re.search(r"@everyone\b|@all\b", after_pipe, re.IGNORECASE):
        from features.community_tag import get_group_member_jids
        everyone_jids = get_group_member_jids(client, chat_jid)
        mentioned_jids = list(dict.fromkeys(mentioned_jids + everyone_jids))

    try:
        removed_count, remaining_count, deleted = remove_subgroup_members(
            store, name, mentioned_jids
        )
    except ValueError as exc:
        _reply(client, chat_jid, f"⚠️ {exc}")
        return

    if not deleted:
        _reply(
            client, chat_jid,
            f"✅ Removed {removed_count} member(s) from *@{name}* ({remaining_count} remaining).",
        )
    else:
        _reply(
            client, chat_jid,
            f"🗑️ Subgroup *@{name}* deleted (no members remaining).",
        )


def _cmd_delete_subgroup(client, chat_jid, name_raw: str, store: SubgroupStore) -> None:
    """!delete-subgroup <name>"""
    name = name_raw.strip().lstrip("@").lower()

    if not name:
        _reply(client, chat_jid, "⚠️ Usage: `!delete-subgroup <name>`")
        return

    subgroups = _read_subgroups(store)
    if name not in subgroups:
        _reply(client, chat_jid, f"⚠️ Subgroup *@{name}* does not exist.")
        return

    del subgroups[name]
    _write_subgroups(store, subgroups)
    _reply(client, chat_jid, f"🗑️ Subgroup *@{name}* deleted.")


def _cmd_list_subgroups(client, chat_jid, store: SubgroupStore) -> None:
    """!list-subgroups"""
    subgroups = _read_subgroups(store)
    if not subgroups:
        _reply(client, chat_jid, "📭 No subgroups defined yet.")
        return

    lines = [f"• *@{name}* — {len(members)} member(s)" for name, members in sorted(subgroups.items())]
    _reply(client, chat_jid, f"*📋 Subgroups ({len(subgroups)})*\n\n" + "\n".join(lines))


def _cmd_subgroup_info(client, chat_jid, name_raw: str, store: SubgroupStore) -> None:
    """!subgroup-info <name>"""
    name = name_raw.strip().lstrip("@").lower()

    if not name:
        _reply(client, chat_jid, "⚠️ Usage: `!subgroup-info <name>`")
        return

    subgroups = _read_subgroups(store)
    if name not in subgroups:
        _reply(client, chat_jid, f"⚠️ Subgroup *@{name}* does not exist.")
        return

    members = subgroups[name]
    # Build @mentions so members render as tagged contacts (with names)
    mention_parts = [f"@{jid.split('@')[0]}" for jid in members]
    text = f"*@{name}* — {len(members)} member(s)\n\n" + "\n".join(f"  • {m}" for m in mention_parts)

    # Send as protobuf so mentionedJID makes WhatsApp resolve names
    proto_msg = Message(
        extendedTextMessage=ExtendedTextMessage(
            text=text,
            contextInfo=ContextInfo(
                mentionedJID=list(members),
            ),
        ),
    )
    try:
        client.send_message(chat_jid, proto_msg)
    except Exception as exc:
        log.error("Failed to send subgroup info: %s", exc)


# ---------------------------------------------------------------------------
# Subgroup @mention detection & tagging
# ---------------------------------------------------------------------------

def _detect_subgroup_mentions(text: str, subgroups: dict[str, list[str]]) -> list[str]:
    """Return subgroup names that are @mentioned in the text."""
    if not subgroups or "@" not in text:
        return []

    # Build a regex that matches any known subgroup name preceded by @
    # Sort by length (longest first) so "blog-team" matches before "blog"
    names = sorted(subgroups.keys(), key=len, reverse=True)
    pattern = r"@(" + "|".join(re.escape(n) for n in names) + r")(?:\b|$)"
    return [m.group(1) for m in re.finditer(pattern, text, re.IGNORECASE)]


def _is_subgroup_command(text: str) -> bool:
    """Return whether text is an explicit subgroup command.

    TEST-ONLY FEATURE: own-account command handling exists only for testing.
    Own-account commands are allowed for testing, but own-account replies and
    generated mention messages must not be fed back into the feature.
    """
    lower = text.lower()
    return any(
        lower == command or lower.startswith(f"{command} ")
        for command in (
            "!add-subgroup",
            "!remove-from-subgroup",
            "!delete-subgroup",
            "!list-subgroups",
            "!subgroup-info",
        )
    )


# ---------------------------------------------------------------------------
# Feature registration
# ---------------------------------------------------------------------------

def register(client: "NewClient", config: dict) -> callable:
    # Clean blocked users (remove spaces, +, -, etc) so they match sender.User
    blocked_users: set[str] = {
        "".join(c for c in u if c.isdigit())
        for u in config.get("subgroup_blocked_users", set())
    }

    session_factory = config.get("db_session_factory")
    if session_factory is None:
        raise RuntimeError("Subgroups feature requires db_session_factory")
    store = SubgroupStore(session_factory)

    def on_message(client: "NewClient", message: "MessageEv"):
        if not message.Info or not message.Info.MessageSource:
            return

        chat = message.Info.MessageSource.Chat

        # Only process group messages
        if getattr(chat, "Server", "") != "g.us":
            return

        # Block listed users from using the subgroup feature
        sender = message.Info.MessageSource.Sender
        sender_user = getattr(sender, "User", "")
        if sender_user in blocked_users:
            return

        body = _get_text(message)
        if not body:
            return

        # TEST-ONLY FEATURE: review/remove the own-account command exception
        # for production. Production should ignore every IsFromMe message.
        # Allow explicit commands sent from the bot's own account for testing,
        # but ignore all other own-account messages. This prevents generated
        # subgroup replies/tags from triggering recursive loops.
        if message.Info.MessageSource.IsFromMe and not _is_subgroup_command(body):
            return

        # ----- Command handling -----
        lower = body.lower()

        actor = normalize_jid(sender)

        if lower == "!add-subgroup" or lower.startswith("!add-subgroup "):
            if not gate(session_factory, sender, client, chat, "admin", "subgroup.add"): return
            args = body[len("!add-subgroup"):].strip()
            if not args:
                _reply(client, chat, "⚠️ Usage: `!add-subgroup <name> | @user1 @user2 …`")
            else:
                _cmd_add_subgroup(client, chat, args, _get_mentioned_jids(message), store)
            return

        if lower == "!remove-from-subgroup" or lower.startswith("!remove-from-subgroup "):
            if not gate(session_factory, sender, client, chat, "admin", "subgroup.remove"): return
            args = body[len("!remove-from-subgroup"):].strip()
            if not args:
                _reply(client, chat, "⚠️ Usage: `!remove-from-subgroup <name> | @user1 @user2 …`")
            else:
                _cmd_remove_from_subgroup(client, chat, args, _get_mentioned_jids(message), store)
            return

        if lower == "!delete-subgroup" or lower.startswith("!delete-subgroup "):
            if not gate(session_factory, sender, client, chat, "admin", "subgroup.delete"): return
            args = body[len("!delete-subgroup"):].strip()
            if not args:
                _reply(client, chat, "⚠️ Usage: `!delete-subgroup <name>`")
            else:
                _cmd_delete_subgroup(client, chat, args, store)
            return

        if lower == "!list-subgroups":
            if not gate(session_factory, sender, client, chat, "member", "subgroup.list"): return
            _cmd_list_subgroups(client, chat, store)
            return

        if lower == "!subgroup-info" or lower.startswith("!subgroup-info "):
            if not gate(session_factory, sender, client, chat, "member", "subgroup.info"): return
            args = body[len("!subgroup-info"):].strip()
            if not args:
                _reply(client, chat, "⚠️ Usage: `!subgroup-info <name>`")
            else:
                _cmd_subgroup_info(client, chat, args, store)
            return

        # ----- @subgroup and @admins tag detection -----
        if not gate(session_factory, sender, client, chat, "member", "subgroup.tag"):
            return
        subgroups = _read_subgroups(store)
        matched = _detect_subgroup_mentions(body, subgroups)
        has_admin_mention = bool(re.search(r"@admins?\b", body, re.IGNORECASE))
        if not matched and not has_admin_mention:
            return

        # Collect unique JIDs across all matched subgroups and admins
        all_jids: list[str] = []
        seen: set[str] = set()
        tag_names: list[str] = []

        if has_admin_mention:
            admin_jids = get_active_admin_jids(session_factory)
            if admin_jids:
                tag_names.append("admins")
                for jid in admin_jids:
                    if jid not in seen:
                        seen.add(jid)
                        all_jids.append(jid)

        for name in matched:
            members = subgroups.get(name, [])
            if not members:
                continue
            tag_names.append(name)
            for jid in members:
                if jid not in seen:
                    seen.add(jid)
                    all_jids.append(jid)

        if not all_jids:
            return

        label = ", ".join(f"@{n}" for n in tag_names)
        text = f"📢 Tagging: {label}"

        proto_msg = Message(
            extendedTextMessage=ExtendedTextMessage(
                text=text,
                contextInfo=ContextInfo(
                    mentionedJID=all_jids,
                ),
            ),
        )

        try:
            client.send_message(chat, proto_msg)
            log.info(
                "Sent subgroup tag for %s — %d members mentioned.",
                label, len(all_jids),
            )
        except Exception as exc:
            log.error("Failed to send subgroup tag message: %s", exc)

    log.info("✅ Subgroups feature registered")
    return on_message
