"""Direct operation adapters for semantic natural-language execution.

These functions are the mutation boundary shared by the NL executor and the
legacy handlers' domain services. They accept resolved entities/JIDs directly;
they never manufacture a WhatsApp command or mention context.
"""

from __future__ import annotations

from typing import Callable
import re

from db.auth import jid_user, normalize_jid
from features.work import _send

# Mutating capabilities use this module as their single semantic execution
# boundary. Legacy command handlers call the same feature-domain services.
from features.agent_runtime import TOOL_SPECS
from features.agent_runtime import tool_spec


def _jid_text(value) -> str:
    """Serialize Neonize JIDs to stable ``user@server`` planner values."""
    if value is None:
        return ""
    user = getattr(value, "User", None)
    server = getattr(value, "Server", None)
    if user and server:
        return f"{user}@{server}"
    raw = str(value or "")
    return raw if "@" in raw else ""


def _object_name(obj) -> str:
    """Extract a transient WhatsApp/contact name from a Neonize object."""
    if obj is None:
        return ""
    for field in (
        "DisplayName", "PushName", "Pushname", "FullName", "Name",
        "Notify", "VerifiedName", "BusinessName", "ShortName",
    ):
        value = getattr(obj, field, None)
        if value is None:
            continue
        if getattr(value, "User", None) and getattr(value, "Server", None):
            continue
        text = str(value).strip()
        if text:
            return text
    for field in ("Contact", "User", "Info"):
        nested = getattr(obj, field, None)
        if nested is not None and nested is not obj:
            name = _object_name(nested)
            if name:
                return name
    return ""


def _phone_for_jid(client, jid) -> str:
    normalized = normalize_jid(_jid_text(jid))
    if normalized.endswith("@s.whatsapp.net"):
        return normalized
    if normalized.endswith("@lid"):
        from features.subgroups import _resolve_lid_to_pn
        pn = _resolve_lid_to_pn(client, normalized)
        if pn != normalized and pn.endswith("@s.whatsapp.net"):
            return pn
    return ""


def _get_display_name_map(client, chat, jids):
    """Resolve current WhatsApp/contact names only for outgoing formatting.

    Names are never persisted in the application database. The group
    Participant object currently returns an empty DisplayName in this bot's
    session, so contact/user-info APIs are also attempted.
    """
    wanted = []
    for jid in jids:
        value = normalize_jid(_jid_text(jid))
        if value and value not in wanted:
            wanted.append(value)
    if not wanted:
        return {}

    names = {}
    phone_by_jid = {jid: _phone_for_jid(client, jid) for jid in wanted}

    try:
        info = client.get_group_info(chat)
        participants = getattr(info, "Participants", []) or []
        for participant in participants:
            raw_jid = getattr(participant, "JID", None) or getattr(participant, "LID", None)
            participant_jid = normalize_jid(_jid_text(raw_jid))
            if not participant_jid:
                continue
            name = _object_name(participant)
            phone_raw = re.sub(r"[^0-9]", "", str(getattr(participant, "PhoneNumber", "") or ""))
            phone_jid = normalize_jid(f"{phone_raw}@s.whatsapp.net") if phone_raw else ""
            if name:
                if participant_jid in wanted:
                    names[participant_jid] = name
                if phone_jid in wanted:
                    names[phone_jid] = name
            if phone_jid:
                for jid in wanted:
                    if jid.endswith("@lid") and phone_by_jid.get(jid) == phone_jid:
                        if name:
                            names[jid] = name
    except Exception as exc:
        # Name lookup must never break the actual operation.
        pass

    unresolved = [jid for jid in wanted if jid not in names]
    for method_name in ("get_contact", "get_contact_info"):
        method = getattr(client, method_name, None)
        if not callable(method):
            continue
        for jid in unresolved:
            candidates = [jid]
            phone = phone_by_jid.get(jid)
            if phone and phone not in candidates:
                candidates.append(phone)
            for candidate in candidates:
                try:
                    name = _object_name(method(candidate))
                    if name:
                        names[jid] = name
                        break
                except Exception:
                    pass
        unresolved = [jid for jid in wanted if jid not in names]
        if not unresolved:
            break

    get_user_info = getattr(client, "get_user_info", None)
    if callable(get_user_info) and unresolved:
        query = []
        for jid in unresolved:
            query.append(jid)
            phone = phone_by_jid.get(jid)
            if phone and phone not in query:
                query.append(phone)
        try:
            for obj in list(get_user_info(*query) or []):
                returned = normalize_jid(_jid_text(getattr(obj, "JID", None)))
                name = _object_name(obj)
                if not name:
                    continue
                for jid in wanted:
                    if jid == returned or phone_by_jid.get(jid) == returned:
                        names[jid] = name
        except Exception:
            pass

    return names


def _mention_text(client, chat, jids):
    """Return display-name mention text while preserving real JIDs for metadata."""
    jids = list(dict.fromkeys(_jid_text(jid) for jid in jids if _jid_text(jid)))
    names = _get_display_name_map(client, chat, jids)
    return ", ".join(
        f"@{names.get(jid, jid_user(jid))}"
        for jid in jids
    )


def _authorize_tool(factory, sender, client, chat, capability):
    """Apply the permission declared by the canonical tool registry."""
    from db.auth import gate

    return gate(
        factory,
        sender,
        client,
        chat,
        tool_spec(capability).permission,
        capability,
    )


class _OperationMessageProxy:
    """Expose a plan-resolved chat while preserving the original message."""

    def __init__(self, message, chat):
        self._message = message
        self._chat = chat

    def __getattr__(self, name):
        if name != "Info":
            return getattr(self._message, name)
        return _OperationInfoProxy(self._message.Info, self._chat)


class _OperationInfoProxy:
    def __init__(self, info, chat):
        self._info = info
        self._chat = chat

    def __getattr__(self, name):
        if name != "MessageSource":
            return getattr(self._info, name)
        return _OperationSourceProxy(self._info.MessageSource, self._chat)


class _OperationSourceProxy:
    def __init__(self, source, chat):
        self._source = source
        self._chat = chat

    def __getattr__(self, name):
        if name == "Chat":
            return self._chat
        return getattr(self._source, name)


def _resolve_operation_chat(message, intent):
    """Resolve only a current-chat alias or a plan-produced WhatsApp JID."""
    raw = intent.get("arguments", {}).get("target_chat")
    if raw is None:
        return None
    if isinstance(raw, dict):
        resolver = raw.get("resolver") or raw.get("kind")
        if resolver == "current_chat":
            return message.Info.MessageSource.Chat
        return None
    value = str(raw).strip()
    if "@" not in value:
        return None
    user, server = value.split("@", 1)
    if not user or server not in {"g.us", "s.whatsapp.net"}:
        return None
    from neonize.utils import build_jid

    return build_jid(user, server)


DIRECT_CAPABILITIES = frozenset(
    capability for capability, spec in TOOL_SPECS.items()
    if spec.executor == "direct"
)

# Every direct capability must belong to exactly one of these routing groups.
# Keeping this audit next to the domain dispatcher makes a newly registered
# direct tool fail verification instead of silently returning no result.
_DIRECT_SPECIAL_CAPABILITIES = frozenset({
    "collections.add", "collections.remove", "collections.delete", "collections.list", "collections.info",
    "labels.add", "labels.remove",
    "work.assign", "work.unassign", "work.create_event", "work.create_task",
    "work.my", "work.overview", "work.list_event_tasks",
    "whatsapp.add_group_members", "whatsapp.remove_group_members",
    "whatsapp.set_group_announce", "whatsapp.set_group_locked",
    "whatsapp.set_group_topic", "whatsapp.set_disappearing_timer",
    "whatsapp.send_contact", "whatsapp.send_poll",
})
_DIRECT_HANDLER_CAPABILITIES = frozenset({
    "whatsapp.send", "whatsapp.reply", "whatsapp.react", "whatsapp.group_info",
    "whatsapp.group_members", "whatsapp.user_info", "whatsapp.send_attachment",
    "whatsapp.rename_group", "whatsapp.group_invite", "whatsapp.joined_groups",
    "whatsapp.community_subgroups", "whatsapp.profile_pictures",
    "whatsapp.group_join_requests", "whatsapp.linked_group_members",
    "whatsapp.create_group", "whatsapp.join_group", "whatsapp.leave_group",
    "whatsapp.is_on_whatsapp", "whatsapp.block_contacts", "whatsapp.unblock_contacts",
    "whatsapp.pin_message", "whatsapp.revoke_message", "whatsapp.set_group_photo",
    "whatsapp.contact_devices", "whatsapp.blocklist", "whatsapp.resolve_contact",
    "whatsapp.group_info_from_link", "whatsapp.link_group", "whatsapp.unlink_group",
    "whatsapp.contact_qr", "whatsapp.set_profile_name", "whatsapp.set_status",
    "whatsapp.set_profile_photo",
    "whatsapp.account_info",
})


def validate_direct_registry() -> dict[str, list[str]]:
    """Report direct tools missing from or duplicated in the dispatcher map."""
    routed = _DIRECT_SPECIAL_CAPABILITIES | _DIRECT_HANDLER_CAPABILITIES
    overlap = _DIRECT_SPECIAL_CAPABILITIES & _DIRECT_HANDLER_CAPABILITIES
    return {
        "missing": sorted(DIRECT_CAPABILITIES - routed),
        "unknown": sorted(routed - DIRECT_CAPABILITIES),
        "overlap": sorted(overlap),
    }


def is_direct_capability(capability: str) -> bool:
    return capability in DIRECT_CAPABILITIES


def execute_whatsapp_send(client, message, intent: dict, factory) -> dict | None:
    """Send an explicitly requested announcement to the current chat only."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.send"):
        return None
    text = str(intent.get("arguments", {}).get("text") or "").strip()
    if not text:
        return None
    client.send_message(chat, text)
    return {"sent": True, "chat": _jid_text(chat), "text": text}


def execute_whatsapp_reply(client, message, intent: dict, factory) -> dict | None:
    """Reply to the triggering message in its current group only."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.reply"):
        return None
    text = str(intent.get("arguments", {}).get("text") or "").strip()
    if not text:
        return None
    try:
        client.reply_message(text, message)
    except Exception:
        client.send_message(chat, text)
    return {"replied": True, "chat": _jid_text(chat), "text": text}


def execute_whatsapp_react(client, message, intent: dict, factory) -> dict | None:
    """React to the triggering message; never accept an arbitrary message ID."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.react"):
        return None
    reaction = str(intent.get("arguments", {}).get("reaction") or "").strip()
    if len(reaction) > 8:
        raise ValueError("reaction must be a short emoji or symbol")
    message_id = getattr(message.Info, "ID", "")
    if not message_id or not reaction:
        return None
    payload = client.build_reaction(chat, source.Sender, message_id, reaction)
    client.send_message(chat, payload)
    return {"reacted": True, "message_id": message_id, "reaction": reaction}


def _serialize_group_info(info) -> dict:
    """Convert Neonize protobuf group metadata into planner-safe JSON."""
    group_name = getattr(info, "GroupName", "")
    group_topic = getattr(info, "GroupTopic", "")
    group_name = getattr(group_name, "Name", group_name) or ""
    group_topic = getattr(group_topic, "Topic", group_topic) or ""
    participants = []
    for participant in getattr(info, "Participants", []) or []:
        jid = getattr(participant, "JID", None) or getattr(participant, "LID", None)
        value = _jid_text(jid)
        if not value:
            continue
        participants.append({
            "jid": value,
            "phone_number": str(getattr(participant, "PhoneNumber", "") or ""),
            "display_name": str(getattr(participant, "DisplayName", "") or ""),
            "is_admin": bool(getattr(participant, "IsAdmin", False)),
            "is_super_admin": bool(getattr(participant, "IsSuperAdmin", False)),
        })
    return {
        "group_jid": _jid_text(getattr(info, "JID", None)),
        "name": str(group_name),
        "topic": str(group_topic),
        "member_count": len(participants),
        "members": participants,
        "member_jids": [item["jid"] for item in participants],
    }


def _current_group_info(client, message):
    source = message.Info.MessageSource
    chat = source.Chat
    if getattr(chat, "Server", "") != "g.us":
        raise ValueError("group information is available only in group chats")
    return client.get_group_info(chat)


def execute_whatsapp_group_info(client, message, intent: dict, factory) -> dict | None:
    """Read metadata for the group containing the triggering message."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.group_info"):
        return None
    data = _serialize_group_info(_current_group_info(client, message))
    client.send_message(
        chat,
        f"👥 *{data['name'] or 'Current group'}* — {data['member_count']} member(s)\n"
        f"Topic: {data['topic'] or '_none_'}",
    )
    return data


def execute_whatsapp_group_members(client, message, intent: dict, factory) -> dict | None:
    """Read the current group's concrete members for later plan reasoning."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.group_members"):
        return None
    data = _serialize_group_info(_current_group_info(client, message))
    lines = [f"👥 *Group members* — {data['member_count']}"]
    for member in data["members"]:
        label = member["display_name"] or member["phone_number"] or member["jid"].split("@", 1)[0]
        suffix = " (admin)" if member["is_admin"] or member["is_super_admin"] else ""
        lines.append(f"• {label}{suffix}")
    client.send_message(chat, "\n".join(lines))
    return {
        "group_jid": data["group_jid"],
        "member_count": data["member_count"],
        "members": data["members"],
        "member_jids": data["member_jids"],
    }


def execute_whatsapp_user_info(client, message, intent: dict, members: list[str], factory) -> dict | None:
    """Read profile metadata only for locally resolved audience members."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.user_info"):
        return None
    if not members:
        return {"users": [], "user_count": 0}
    from neonize.utils import build_jid

    jids = []
    for member in members:
        normalized = str(member)
        user, server = normalized.split("@", 1) if "@" in normalized else (normalized, "s.whatsapp.net")
        jids.append(build_jid(user, server))
    rows = []
    for item in client.get_user_info(*jids):
        rows.append({
            "jid": _jid_text(getattr(item, "JID", None)),
            "status": str(getattr(item, "Status", "") or ""),
            "business_name": str(getattr(item, "BusinessName", "") or ""),
        })
    client.send_message(
        chat,
        "👤 User info\n" + "\n".join(
            f"• {row['jid'] or 'unknown'}"
            + (f" — {row['business_name']}" if row["business_name"] else "")
            for row in rows
        ) if rows else "👤 No user information found.",
    )
    return {"users": rows, "user_count": len(rows)}


def execute_whatsapp_send_attachment(client, message, intent: dict, factory) -> dict | None:
    """Forward only the triggering message's attached media to its group."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.send_attachment"):
        return None
    payload = getattr(message, "Message", None)
    if payload is None or not hasattr(client, "download_any"):
        return None
    fields = {field.name for field, _ in payload.ListFields()} if hasattr(payload, "ListFields") else set()
    media_kind = next(
        (kind for kind in ("image", "video", "audio", "document", "sticker")
         if f"{kind}Message" in fields),
        None,
    )
    if media_kind is None:
        return None
    data = client.download_any(payload)
    if not data:
        return None
    arguments = intent.get("arguments", {})
    caption = str(arguments.get("caption") or "").strip() or None
    filename = str(arguments.get("filename") or "").strip() or None
    if media_kind == "image":
        client.send_image(chat, data, caption=caption)
    elif media_kind == "video":
        client.send_video(chat, data, caption=caption)
    elif media_kind == "audio":
        client.send_audio(chat, data)
    elif media_kind == "sticker":
        client.send_sticker(chat, data)
    else:
        client.send_document(chat, data, caption=caption, filename=filename)
    return {"sent": True, "chat": _jid_text(chat), "media_type": media_kind, "filename": filename}


def execute_whatsapp_group_membership(
    client, message, intent: dict, members: list[str], factory
) -> dict | None:
    """Apply an explicit add/remove participant operation to the current group."""
    source = message.Info.MessageSource
    chat = source.Chat
    if getattr(chat, "Server", "") != "g.us":
        return None
    if not _authorize_tool(factory, source.Sender, client, chat, intent["capability"]):
        return None
    if not members:
        return None
    from neonize.utils import ParticipantChange, build_jid

    jids = []
    for member in dict.fromkeys(members):
        user, server = str(member).split("@", 1) if "@" in str(member) else (str(member), "s.whatsapp.net")
        jids.append(build_jid(user, server))
    action = (
        ParticipantChange.ADD
        if intent["capability"] == "whatsapp.add_group_members"
        else ParticipantChange.REMOVE
    )
    client.update_group_participants(chat, jids, action)
    verb = "Added" if action == ParticipantChange.ADD else "Removed"
    client.send_message(chat, f"✅ {verb} {len(jids)} group member(s).")
    return {"updated": True, "action": action.value, "count": len(jids), "members": list(members)}


def execute_whatsapp_rename_group(client, message, intent: dict, factory) -> dict | None:
    """Rename only the current group, with admin authorization."""
    source = message.Info.MessageSource
    chat = source.Chat
    if getattr(chat, "Server", "") != "g.us":
        return None
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.rename_group"):
        return None
    name = str(intent.get("arguments", {}).get("name") or "").strip()
    if not name or len(name) > 100:
        raise ValueError("group name must be between 1 and 100 characters")
    client.set_group_name(chat, name)
    client.send_message(chat, f"✅ Group renamed to *{name}*.")
    return {"renamed": True, "name": name, "group_jid": _jid_text(chat)}


def execute_whatsapp_group_invite(client, message, intent: dict, factory) -> dict | None:
    """Retrieve or explicitly revoke the current group's invite link."""
    source = message.Info.MessageSource
    chat = source.Chat
    if getattr(chat, "Server", "") != "g.us":
        return None
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.group_invite"):
        return None
    raw_revoke = intent.get("arguments", {}).get("revoke", False)
    revoke = raw_revoke is True or str(raw_revoke).strip().casefold() in {"true", "yes", "1", "revoke"}
    link = client.get_group_invite_link(chat, revoke=revoke)
    client.send_message(chat, f"🔗 Group invite link: {link}")
    return {"link": str(link), "revoked": revoke, "group_jid": _jid_text(chat)}


def _nested_text(value, field: str) -> str:
    nested = getattr(value, field, value)
    return str(nested or "")


def execute_whatsapp_joined_groups(client, message, intent: dict, factory) -> dict | None:
    """List joined groups using only Neonize's current session state."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.joined_groups"):
        return None
    groups = []
    for info in list(client.get_joined_groups())[:100]:
        groups.append({
            "group_jid": _jid_text(getattr(info, "JID", None)),
            "name": _nested_text(getattr(info, "GroupName", ""), "Name"),
            "member_count": len(getattr(info, "Participants", []) or []),
        })
    client.send_message(
        chat,
        "👥 Joined groups\n" + "\n".join(
            f"• {item['name'] or item['group_jid']} ({item['member_count']})"
            for item in groups
        ) if groups else "📭 No joined groups found.",
    )
    return {"groups": groups, "group_count": len(groups)}


def execute_whatsapp_community_subgroups(client, message, intent: dict, factory) -> dict | None:
    """List linked subgroups for the current community/group JID."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.community_subgroups"):
        return None
    groups = []
    for info in list(client.get_sub_groups(chat))[:100]:
        jid = getattr(info, "JID", "")
        groups.append({
            "group_jid": _jid_text(jid),
            "name": _nested_text(getattr(info, "GroupName", ""), "Name"),
            "default": bool(getattr(getattr(info, "GroupIsDefaultSub", None), "IsDefaultSub", False)),
        })
    client.send_message(
        chat,
        "👥 Community subgroups\n" + "\n".join(
            f"• {item['name'] or item['group_jid']}" for item in groups
        ) if groups else "📭 No linked subgroups found.",
    )
    return {"groups": groups, "group_count": len(groups)}


def _coerce_bool(value) -> bool:
    return value is True or str(value).strip().casefold() in {"true", "yes", "1", "on", "enabled"}


def execute_whatsapp_group_setting(client, message, intent: dict, factory) -> dict | None:
    """Apply one explicit admin-only setting to the current group."""
    source = message.Info.MessageSource
    chat = source.Chat
    capability = intent["capability"]
    if getattr(chat, "Server", "") != "g.us":
        return None
    if not _authorize_tool(factory, source.Sender, client, chat, capability):
        return None
    arguments = intent.get("arguments", {})
    try:
        if capability == "whatsapp.set_group_announce":
            enabled = _coerce_bool(arguments.get("enabled"))
            client.set_group_announce(chat, enabled)
            value = {"enabled": enabled}
        elif capability == "whatsapp.set_group_locked":
            locked = _coerce_bool(arguments.get("locked"))
            client.set_group_locked(chat, locked)
            value = {"locked": locked}
        elif capability == "whatsapp.set_group_topic":
            topic = str(arguments.get("topic") or "").strip()
            if not topic or len(topic) > 512:
                raise ValueError("group topic must be between 1 and 512 characters")
            info = client.get_group_info(chat)
            previous = getattr(getattr(info, "GroupTopic", None), "TopicID", "") or ""
            import uuid
            client.set_group_topic(chat, str(previous), uuid.uuid4().hex, topic)
            value = {"topic": topic}
        else:
            seconds = int(arguments.get("seconds"))
            if seconds < 0 or seconds > 2_147_483_647:
                raise ValueError("disappearing timer seconds must be between 0 and 2147483647")
            client.set_disappearing_timer(chat, seconds * 1_000_000_000)
            value = {"seconds": seconds}
        client.send_message(chat, "✅ Group setting updated.")
        return {"updated": True, "capability": capability, **value, "group_jid": _jid_text(chat)}
    except (TypeError, ValueError) as exc:
        client.send_message(chat, f"⚠️ {exc}")
        return None


def execute_whatsapp_message_primitive(client, message, intent: dict, factory) -> dict | None:
    """Send a bounded contact or poll using Neonize's typed builders."""
    source = message.Info.MessageSource
    chat = source.Chat
    capability = intent["capability"]
    if not _authorize_tool(factory, source.Sender, client, chat, capability):
        return None
    arguments = intent.get("arguments", {})
    try:
        if capability == "whatsapp.send_contact":
            name = str(arguments.get("name") or "").strip()
            number = str(arguments.get("number") or "").strip()
            import re
            if not name or len(name) > 100 or not re.fullmatch(r"\+?[0-9][0-9 ()-]{3,30}", number):
                raise ValueError("contact name or number is invalid")
            client.send_contact(chat, name, number)
            client.send_message(chat, f"✅ Sent contact card for *{name}*.")
            return {"sent": True, "type": "contact", "name": name, "number": number}

        question = str(arguments.get("question") or "").strip()
        options = arguments.get("options", [])
        if isinstance(options, str):
            options = [item.strip() for item in options.split(",") if item.strip()]
        if not question or len(question) > 256 or not isinstance(options, list) or not 2 <= len(options) <= 10:
            raise ValueError("polls need a question and 2-10 options")
        options = [str(item).strip() for item in options]
        if any(not item or len(item) > 100 for item in options):
            raise ValueError("poll options must be 1-100 characters")
        selectable = int(arguments.get("selectable_count") or 1)
        if selectable < 1 or selectable > len(options):
            raise ValueError("selectable_count must fit the option list")
        from neonize.utils import VoteType
        poll = client.build_poll_vote_creation(
            question,
            options,
            VoteType.SINGLE if selectable == 1 else VoteType.MULTIPLE,
        )
        client.send_message(chat, poll)
        client.send_message(chat, "✅ Poll sent.")
        return {"sent": True, "type": "poll", "question": question, "options": options, "selectable_count": selectable}
    except (TypeError, ValueError) as exc:
        client.send_message(chat, f"⚠️ {exc}")
        return None


def execute_whatsapp_profile_pictures(
    client, message, intent: dict, members: list[str], factory
) -> dict | None:
    """Read profile-picture metadata for locally resolved users only."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.profile_pictures"):
        return None
    if not members:
        return None
    from neonize.utils import build_jid

    profiles = []
    for member in dict.fromkeys(members):
        user, server = str(member).split("@", 1) if "@" in str(member) else (str(member), "s.whatsapp.net")
        info = client.get_profile_picture(build_jid(user, server))
        profiles.append({
            "jid": str(member),
            "url": str(getattr(info, "URL", "") or ""),
            "picture_id": str(getattr(info, "ID", "") or ""),
        })
    client.send_message(
        chat,
        "🖼️ Profile pictures\n" + "\n".join(
            f"• {row['jid']}: {row['url'] or 'no public picture'}" for row in profiles
        ),
    )
    return {"profiles": profiles, "profile_count": len(profiles)}


def _participant_request_rows(items) -> list[dict]:
    rows = []
    for item in list(items or []):
        participant = getattr(item, "Participant", None) or getattr(item, "JID", None)
        jid = getattr(participant, "JID", participant)
        if not jid:
            continue
        rows.append({"jid": _jid_text(jid), "requested_at": str(getattr(item, "TimeAt", "") or "")})
    return rows


def execute_whatsapp_group_join_requests(client, message, intent: dict, factory) -> dict | None:
    """Read pending join requests for the current group."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.group_join_requests"):
        return None
    rows = _participant_request_rows(client.get_group_request_participants(chat))
    client.send_message(
        chat,
        "📥 Group join requests\n" + "\n".join(f"• {row['jid']}" for row in rows)
        if rows else "📭 No pending group join requests.",
    )
    return {"requests": rows, "request_count": len(rows)}


def execute_whatsapp_linked_group_members(client, message, intent: dict, factory) -> dict | None:
    """Read participants linked to the current community/group."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.linked_group_members"):
        return None
    rows = _participant_request_rows(client.get_linked_group_participants(chat))
    client.send_message(
        chat,
        "👥 Linked group participants\n" + "\n".join(f"• {row['jid']}" for row in rows)
        if rows else "📭 No linked group participants found.",
    )
    return {"members": rows, "member_count": len(rows)}


def execute_whatsapp_create_group(
    client, message, intent: dict, members: list[str], factory
) -> dict | None:
    """Create a group from a resolved audience; never accept raw JIDs from the model."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.create_group"):
        return None
    arguments = intent.get("arguments", {})
    name = str(arguments.get("name") or "").strip()
    if not name or len(name) > 100:
        raise ValueError("group name must be between 1 and 100 characters")
    from neonize.utils import build_jid

    participants = []
    for member in dict.fromkeys(members):
        user, server = str(member).split("@", 1) if "@" in str(member) else (str(member), "s.whatsapp.net")
        if server != "s.whatsapp.net":
            continue
        participants.append(build_jid(user, server))
    info = client.create_group(name, participants)
    group_jid = _jid_text(getattr(info, "JID", None))
    if not group_jid:
        return None
    client.send_message(chat, f"✅ Group created: *{name}* ({len(participants)} member(s)).")
    return {"group_jid": group_jid, "name": name, "member_jids": list(dict.fromkeys(members))}


def execute_whatsapp_join_group(client, message, intent: dict, factory) -> dict | None:
    """Join a group using a user-supplied invite link/code."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.join_group"):
        return None
    invite = str(intent.get("arguments", {}).get("invite") or "").strip()
    if not invite or len(invite) > 512:
        raise ValueError("a WhatsApp group invite link or code is required")
    if "chat.whatsapp.com/" in invite:
        invite = invite.split("chat.whatsapp.com/", 1)[1].split("?", 1)[0].strip("/")
    group_jid = client.join_group_with_link(invite)
    value = _jid_text(group_jid) or (group_jid if "@" in group_jid else "")
    if not value:
        return None
    client.send_message(chat, f"✅ Joined group {value}.")
    return {"group_jid": value}


def execute_whatsapp_leave_group(client, message, intent: dict, factory) -> dict | None:
    """Leave only the group containing the triggering message."""
    source = message.Info.MessageSource
    chat = source.Chat
    if getattr(chat, "Server", "") != "g.us":
        return None
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.leave_group"):
        return None
    client.leave_group(chat)
    client.send_message(chat, "✅ Bot left this group.")
    return {"left": True, "group_jid": _jid_text(chat)}


def execute_whatsapp_is_on_whatsapp(client, message, intent: dict, factory) -> dict | None:
    """Check explicit phone numbers through Neonize without inventing contacts."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.is_on_whatsapp"):
        return None
    raw = intent.get("arguments", {}).get("numbers", [])
    values = [raw] if isinstance(raw, str) else list(raw or [])
    import re
    numbers = []
    for value in values[:50]:
        number = re.sub(r"[^0-9+]", "", str(value))
        if re.fullmatch(r"\+?[0-9]{5,20}", number):
            numbers.append(number.lstrip("+"))
    if not numbers:
        raise ValueError("at least one valid phone number is required")
    rows = []
    for result in client.is_on_whatsapp(*numbers):
        rows.append({
            "number": str(getattr(result, "Query", "") or ""),
            "jid": _jid_text(getattr(result, "JID", None)),
            "exists": bool(getattr(result, "IsIn", False)),
        })
    client.send_message(
        chat,
        "📱 WhatsApp availability\n" + "\n".join(
            f"• {row['number'] or row['jid']}: {'yes' if row['exists'] else 'no'}" for row in rows
        ),
    )
    return {"numbers": rows, "number_count": len(rows)}


def execute_whatsapp_blocklist(
    client, message, intent: dict, members: list[str], factory
) -> dict | None:
    """Block or unblock only locally resolved contacts."""
    source = message.Info.MessageSource
    chat = source.Chat
    capability = intent["capability"]
    if not _authorize_tool(factory, source.Sender, client, chat, capability):
        return None
    if not members:
        return None
    from neonize.utils import BlocklistAction, build_jid

    jids = []
    for member in dict.fromkeys(members):
        user, server = str(member).split("@", 1) if "@" in str(member) else (str(member), "s.whatsapp.net")
        if server == "s.whatsapp.net":
            jids.append(build_jid(user, server))
    if not jids:
        return None
    action = BlocklistAction.BLOCK if capability == "whatsapp.block_contacts" else BlocklistAction.UNBLOCK
    for jid in jids:
        client.update_blocklist(jid, action)
    verb = "Blocked" if action == BlocklistAction.BLOCK else "Unblocked"
    client.send_message(chat, f"✅ {verb} {len(jids)} contact(s).")
    return {"updated": True, "action": action.value, "count": len(jids), "members": list(members)}


def execute_whatsapp_message_moderation(client, message, intent: dict, factory) -> dict | None:
    """Pin or revoke the triggering message, never an arbitrary user-supplied ID."""
    source = message.Info.MessageSource
    chat = source.Chat
    capability = intent["capability"]
    if getattr(chat, "Server", "") != "g.us":
        return None
    if not _authorize_tool(factory, source.Sender, client, chat, capability):
        return None
    message_id = str(getattr(message.Info, "ID", "") or "")
    if not message_id:
        return None
    if capability == "whatsapp.pin_message":
        raw_seconds = intent.get("arguments", {}).get("seconds", 86400)
        seconds = int(raw_seconds or 0)
        if seconds < 0 or seconds > 2_592_000:
            raise ValueError("pin duration must be between 0 and 2592000 seconds")
        client.pin_message(chat, source.Sender, message_id, seconds)
        client.send_message(chat, "✅ Message pinned.")
        return {"pinned": True, "message_id": message_id, "seconds": seconds}
    client.revoke_message(chat, source.Sender, message_id)
    client.send_message(chat, "✅ Message revoked.")
    return {"revoked": True, "message_id": message_id}


def execute_whatsapp_set_group_photo(client, message, intent: dict, factory) -> dict | None:
    """Set the current group's photo from the triggering attached image only."""
    source = message.Info.MessageSource
    chat = source.Chat
    if getattr(chat, "Server", "") != "g.us":
        return None
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.set_group_photo"):
        return None
    payload = getattr(message, "Message", None)
    if payload is None or not hasattr(client, "download_any"):
        return None
    fields = {field.name for field, _ in payload.ListFields()} if hasattr(payload, "ListFields") else set()
    if "imageMessage" not in fields:
        return None
    data = client.download_any(payload)
    if not data:
        return None
    client.set_group_photo(chat, data)
    client.send_message(chat, "✅ Group photo updated.")
    return {"updated": True, "group_jid": _jid_text(chat)}


def execute_whatsapp_contact_devices(
    client, message, intent: dict, members: list[str], factory
) -> dict | None:
    """Read device JIDs for a locally resolved audience."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.contact_devices"):
        return None
    if not members:
        return {"devices": [], "device_count": 0}
    from neonize.utils import build_jid

    jids = []
    for member in dict.fromkeys(members):
        user, server = str(member).split("@", 1) if "@" in str(member) else (str(member), "s.whatsapp.net")
        if server == "s.whatsapp.net":
            jids.append(build_jid(user, server))
    devices = [_jid_text(item) for item in client.get_user_devices(*jids)]
    client.send_message(
        chat,
        "📱 Contact devices\n" + "\n".join(f"• {device}" for device in devices)
        if devices else "📭 No additional contact devices found.",
    )
    return {"devices": devices, "device_count": len(devices)}


def execute_whatsapp_blocklist_read(client, message, intent: dict, factory) -> dict | None:
    """Read the account blocklist as structured context for later planning."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.blocklist"):
        return None
    blocklist = client.get_blocklist()
    contacts = []
    for item in getattr(blocklist, "JID", blocklist if isinstance(blocklist, (list, tuple)) else []) or []:
        contacts.append(_jid_text(getattr(item, "JID", item)))
    contacts = [value for value in contacts if value]
    client.send_message(
        chat,
        "🚫 Blocked contacts\n" + "\n".join(f"• {value}" for value in contacts)
        if contacts else "📭 No blocked contacts.",
    )
    return {"contacts": contacts, "contact_count": len(contacts)}


def execute_whatsapp_resolve_contact(client, message, intent: dict, factory) -> dict | None:
    """Resolve phone/LID forms through Neonize, fixing identifier ambiguity at the boundary."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.resolve_contact"):
        return None
    identifier = str(intent.get("arguments", {}).get("identifier") or "").strip()
    if not identifier or len(identifier) > 80:
        raise ValueError("a phone number or WhatsApp LID is required")
    from neonize.utils import build_jid

    compact = identifier.replace("+", "").replace(" ", "").replace("-", "")
    if "@" in compact:
        user, server = compact.split("@", 1)
        jid = build_jid(user, server)
        if server == "lid":
            lid_jid = jid
            phone_jid = client.get_pn_from_lid(jid)
        else:
            phone_jid = jid
            lid_jid = client.get_lid_from_pn(jid)
    else:
        phone_jid = build_jid(compact, "s.whatsapp.net")
        lid_jid = client.get_lid_from_pn(phone_jid)
    result = {"phone_jid": str(phone_jid), "lid_jid": str(lid_jid)}
    client.send_message(chat, f"🔎 Contact IDs\n• phone: {result['phone_jid']}\n• LID: {result['lid_jid']}")
    return result


def execute_whatsapp_group_info_from_link(client, message, intent: dict, factory) -> dict | None:
    """Inspect an invite link without joining or mutating group state."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.group_info_from_link"):
        return None
    link = str(intent.get("arguments", {}).get("link") or "").strip()
    if not link or len(link) > 512:
        raise ValueError("a WhatsApp group invite link is required")
    if "chat.whatsapp.com/" in link:
        code = link.split("chat.whatsapp.com/", 1)[1].split("?", 1)[0].strip("/")
    else:
        code = link
    info = client.get_group_info_from_link(code)
    result = _serialize_group_info(info)
    client.send_message(
        chat,
        f"👥 *{result['name'] or result['group_jid']}* — {result['member_count']} member(s)",
    )
    return result


def _resolve_group_endpoint(message, value):
    if isinstance(value, dict):
        resolver = value.get("resolver") or value.get("kind")
        if resolver == "current_chat":
            return message.Info.MessageSource.Chat
        return None
    raw = str(value or "").strip()
    if "@" not in raw:
        return None
    user, server = raw.split("@", 1)
    if not user or server != "g.us":
        return None
    from neonize.utils import build_jid

    return build_jid(user, server)


def execute_whatsapp_group_link(client, message, intent: dict, factory) -> dict | None:
    """Link or unlink two plan-resolved group endpoints."""
    source = message.Info.MessageSource
    chat = source.Chat
    capability = intent["capability"]
    if not _authorize_tool(factory, source.Sender, client, chat, capability):
        return None
    arguments = intent.get("arguments", {})
    parent = _resolve_group_endpoint(message, arguments.get("parent_chat"))
    child = _resolve_group_endpoint(message, arguments.get("child_chat"))
    if parent is None or child is None:
        return None
    if capability == "whatsapp.link_group":
        client.link_group(parent, child)
        verb = "linked"
    else:
        client.unlink_group(parent, child)
        verb = "unlinked"
    client.send_message(chat, f"✅ Group {verb}.")
    return {"updated": True, "parent_chat": _jid_text(parent), "child_chat": _jid_text(child), "action": verb}


def execute_whatsapp_profile_operation(client, message, intent: dict, factory) -> dict | None:
    """Manage bot identity through bounded profile operations."""
    source = message.Info.MessageSource
    chat = source.Chat
    capability = intent["capability"]
    if not _authorize_tool(factory, source.Sender, client, chat, capability):
        return None
    arguments = intent.get("arguments", {})
    if capability == "whatsapp.contact_qr":
        raw_revoke = arguments.get("revoke", False)
        revoke = _coerce_bool(raw_revoke) or str(raw_revoke).casefold() == "revoke"
        link = client.get_contact_qr_link(revoke=revoke)
        client.send_message(chat, f"🔗 Bot contact QR link: {link}")
        return {"link": str(link), "revoked": revoke}
    if capability == "whatsapp.set_profile_name":
        name = str(arguments.get("name") or "").strip()
        if not name or len(name) > 100:
            raise ValueError("profile name must be between 1 and 100 characters")
        client.set_profile_name(name)
        client.send_message(chat, "✅ Bot profile name updated.")
        return {"updated": True, "name": name}
    if capability == "whatsapp.set_status":
        status = str(arguments.get("status") or "").strip()
        if len(status) > 139:
            raise ValueError("status must be at most 139 characters")
        client.set_status_message(status)
        client.send_message(chat, "✅ Bot status updated.")
        return {"updated": True, "status": status}
    payload = getattr(message, "Message", None)
    if payload is None or not hasattr(client, "download_any"):
        return None
    fields = {field.name for field, _ in payload.ListFields()} if hasattr(payload, "ListFields") else set()
    if "imageMessage" not in fields:
        return None
    data = client.download_any(payload)
    if not data:
        return None
    client.set_profile_photo(data)
    client.send_message(chat, "✅ Bot profile photo updated.")
    return {"updated": True, "photo": True}


def execute_whatsapp_account_info(client, message, intent: dict, factory) -> dict | None:
    """Expose non-secret account identity metadata for agent context."""
    source = message.Info.MessageSource
    chat = source.Chat
    if not _authorize_tool(factory, source.Sender, client, chat, "whatsapp.account_info"):
        return None
    device = client.get_me()
    result = {
        "jid": _jid_text(getattr(device, "JID", None)),
        "lid": _jid_text(getattr(device, "LID", None)),
        "name": str(getattr(device, "PushName", "") or getattr(device, "BussinessName", "") or ""),
        "platform": str(getattr(device, "Platform", "") or ""),
    }
    client.send_message(
        chat,
        f"🤖 Bot account\n• JID: {result['jid']}\n• LID: {result['lid']}\n• Name: {result['name']}\n• Platform: {result['platform']}",
    )
    return result


def execute_direct_tool(
    client,
    message,
    intent: dict,
    members: list[str],
    factory,
    text: str = "",
    *,
    resolve_collection: Callable[[object], str | None] | None = None,
    resolve_or_create_collection: Callable[[object], str | None] | None = None,
    normalize_target_arguments: Callable[[dict, str], dict] | None = None,
    resolve_target: Callable[[dict], str | None] | None = None,
) -> dict | None:
    """Single capability-to-domain boundary for all direct agent tools.

    Entity-name and work-target callbacks remain injected from the application
    layer, while the operation selection and execution policy live here with
    the adapters. This keeps the planner independent of dispatch details.
    """
    capability = str(intent.get("capability") or "")
    target_chat = _resolve_operation_chat(message, intent)
    if intent.get("arguments", {}).get("target_chat") is not None and target_chat is None:
        return None
    if target_chat is not None:
        message = _OperationMessageProxy(message, target_chat)
    if capability.startswith(("collections.", "labels.")):
        action = capability.split(".", 1)[1]

        def resolve_collection_name(current_factory, requested):
            if action == "add" and resolve_or_create_collection is not None:
                return resolve_or_create_collection(requested)
            return resolve_collection(requested) if resolve_collection is not None else None

        if capability.startswith("collections."):
            return execute_collection_mutation(
                client, message, intent, members, factory, resolve_collection_name
            )
        return execute_label_mutation(
            client, message, intent, members, factory, resolve_collection_name
        )
    if capability in {"work.assign", "work.unassign"}:
        arguments = (
            normalize_target_arguments(intent.get("arguments", {}), text)
            if normalize_target_arguments is not None
            else intent.get("arguments", {})
        )
        scoped_intent = {**intent, "arguments": arguments}
        return execute_work_assignment(
            client,
            message,
            scoped_intent,
            members,
            factory,
            resolve_target or (lambda _arguments: None),
        )
    if capability in {"work.create_event", "work.create_task"}:
        return execute_work_creation(client, message, intent, factory)
    handlers = {
        "whatsapp.send": lambda: execute_whatsapp_send(client, message, intent, factory),
        "whatsapp.reply": lambda: execute_whatsapp_reply(client, message, intent, factory),
        "whatsapp.react": lambda: execute_whatsapp_react(client, message, intent, factory),
        "whatsapp.group_info": lambda: execute_whatsapp_group_info(client, message, intent, factory),
        "whatsapp.group_members": lambda: execute_whatsapp_group_members(client, message, intent, factory),
        "whatsapp.user_info": lambda: execute_whatsapp_user_info(client, message, intent, members, factory),
        "whatsapp.send_attachment": lambda: execute_whatsapp_send_attachment(client, message, intent, factory),
        "whatsapp.rename_group": lambda: execute_whatsapp_rename_group(client, message, intent, factory),
        "whatsapp.group_invite": lambda: execute_whatsapp_group_invite(client, message, intent, factory),
        "whatsapp.joined_groups": lambda: execute_whatsapp_joined_groups(client, message, intent, factory),
        "whatsapp.community_subgroups": lambda: execute_whatsapp_community_subgroups(client, message, intent, factory),
        "whatsapp.profile_pictures": lambda: execute_whatsapp_profile_pictures(client, message, intent, members, factory),
        "whatsapp.group_join_requests": lambda: execute_whatsapp_group_join_requests(client, message, intent, factory),
        "whatsapp.linked_group_members": lambda: execute_whatsapp_linked_group_members(client, message, intent, factory),
        "whatsapp.create_group": lambda: execute_whatsapp_create_group(client, message, intent, members, factory),
        "whatsapp.join_group": lambda: execute_whatsapp_join_group(client, message, intent, factory),
        "whatsapp.leave_group": lambda: execute_whatsapp_leave_group(client, message, intent, factory),
        "whatsapp.is_on_whatsapp": lambda: execute_whatsapp_is_on_whatsapp(client, message, intent, factory),
        "whatsapp.block_contacts": lambda: execute_whatsapp_blocklist(client, message, intent, members, factory),
        "whatsapp.unblock_contacts": lambda: execute_whatsapp_blocklist(client, message, intent, members, factory),
        "whatsapp.pin_message": lambda: execute_whatsapp_message_moderation(client, message, intent, factory),
        "whatsapp.revoke_message": lambda: execute_whatsapp_message_moderation(client, message, intent, factory),
        "whatsapp.set_group_photo": lambda: execute_whatsapp_set_group_photo(client, message, intent, factory),
        "whatsapp.contact_devices": lambda: execute_whatsapp_contact_devices(client, message, intent, members, factory),
        "whatsapp.blocklist": lambda: execute_whatsapp_blocklist_read(client, message, intent, factory),
        "whatsapp.resolve_contact": lambda: execute_whatsapp_resolve_contact(client, message, intent, factory),
        "whatsapp.group_info_from_link": lambda: execute_whatsapp_group_info_from_link(client, message, intent, factory),
        "whatsapp.link_group": lambda: execute_whatsapp_group_link(client, message, intent, factory),
        "whatsapp.unlink_group": lambda: execute_whatsapp_group_link(client, message, intent, factory),
        "whatsapp.contact_qr": lambda: execute_whatsapp_profile_operation(client, message, intent, factory),
        "whatsapp.set_profile_name": lambda: execute_whatsapp_profile_operation(client, message, intent, factory),
        "whatsapp.set_status": lambda: execute_whatsapp_profile_operation(client, message, intent, factory),
        "whatsapp.set_profile_photo": lambda: execute_whatsapp_profile_operation(client, message, intent, factory),
        "whatsapp.account_info": lambda: execute_whatsapp_account_info(client, message, intent, factory),
    }
    if capability in {
        "whatsapp.set_group_announce", "whatsapp.set_group_locked",
        "whatsapp.set_group_topic", "whatsapp.set_disappearing_timer",
    }:
        return execute_whatsapp_group_setting(client, message, intent, factory)
    if capability in {"whatsapp.add_group_members", "whatsapp.remove_group_members"}:
        return execute_whatsapp_group_membership(client, message, intent, members, factory)
    if capability in {"whatsapp.send_contact", "whatsapp.send_poll"}:
        return execute_whatsapp_message_primitive(client, message, intent, factory)
    if capability in {"work.my", "work.overview", "work.list_event_tasks"}:
        return execute_work_read(client, message, intent, factory, resolve_target=resolve_target)
    handler = handlers.get(capability)
    if handler is None or capability not in TOOL_SPECS:
        return None
    return handler()


def _structured_work_row(row: dict) -> dict:
    """Keep work read results JSON-friendly for later plan references."""
    result = {}
    for key, value in row.items():
        if hasattr(value, "isoformat"):
            result[key] = value.isoformat()
        else:
            result[key] = value
    return result


def execute_work_read(client, message, intent: dict, factory, resolve_target=None) -> dict | None:
    """Read work through domain stores and expose structured rows to later steps."""
    source = message.Info.MessageSource
    chat = source.Chat
    capability = intent["capability"]
    actor = _authorize_tool(factory, source.Sender, client, chat, capability)
    if not actor:
        return None
    from db.task_store import TaskStore
    from db.work_store import WorkStore
    from features.work import _format

    sender = source.Sender
    arguments = intent.get("arguments", {})
    _raw_status = str(arguments.get("status") or "").strip().lower()
    # Only accept statuses that actually exist in the DB schema.
    # The NL model sometimes sets status="assigned" when the user says
    # "assigned to @X" — discard those so the query isn't filtered to nothing.
    from db.work_store import PROGRESS_STATUSES
    status = _raw_status if _raw_status in PROGRESS_STATUSES else None
    store = WorkStore(factory)

    # Build visible_mentions from the message protobuf (same logic as the NL compiler).
    from features.subgroups import _get_mentioned_jids, _resolve_lid_to_pn
    from db.auth import normalize_jid, jid_user
    _self_jids = {normalize_jid(sender)}
    try:
        from neonize.utils import Jid2String
        bot_jid = Jid2String(client.get_me().JID)
        if bot_jid:
            _self_jids.add(normalize_jid(bot_jid))
    except Exception:
        pass
        
    visible_mentions = []
    for _jid in _get_mentioned_jids(message):
        _resolved = _resolve_lid_to_pn(client, normalize_jid(_jid))
        if _resolved and _resolved not in _self_jids and _resolved not in visible_mentions:
            visible_mentions.append(_resolved)

    # Resolve LID/phone aliases for the target user using the persistent cache.
    def _alias_jids(jid: str) -> list[str]:
        """Return [jid] plus its LID or phone counterpart if known."""
        from db.auth import normalize_jid as _nj, jid_user as _ju
        jid = _nj(jid)
        aliases = [jid]
        try:
            from db.work_store import _JID_ALIASES
            user_part = _ju(jid)
            
            # Forward: LID -> Phone
            if jid.endswith("@lid"):
                if user_part in _JID_ALIASES:
                    aliases.append(f"{_JID_ALIASES[user_part]}@s.whatsapp.net")
            
            # Reverse: Phone -> LID
            elif jid.endswith("@s.whatsapp.net"):
                for lid_u, phone_u in _JID_ALIASES.items():
                    if phone_u == user_part:
                        aliases.append(f"{lid_u}@lid")
        except Exception:
            pass
        return aliases

    try:
        if capability == "work.list_event_tasks":
            event_id = arguments.get("event_id", arguments.get("target_id"))
            if not str(event_id or "").isdigit() and resolve_target:
                ref = resolve_target({"target_type": "event", **arguments})
                if ref and ref.startswith("event "):
                    event_id = ref.split(" ", 1)[1]
            if not str(event_id or "").isdigit() and factory:
                from db.event_store import EventStore
                name_query = arguments.get("event_name") or arguments.get("target_name") or arguments.get("name") or arguments.get("event")
                if name_query:
                    events = EventStore(factory).list_events(status="active")
                    for e in events:
                        if e["name"].lower() == str(name_query).lower() or str(name_query).lower() in e["name"].lower():
                            event_id = e["id"]
                            break
            if not str(event_id or "").isdigit():
                # If the NL model misrouted a user-workload query to this
                # capability (e.g. "list all tasks of @Shuvam"), fall back
                # gracefully to a workload query for the mentioned users.
                if visible_mentions:
                    # Re-route to user workload
                    target_jid = visible_mentions[0]
                    all_aliases = _alias_jids(target_jid)
                    rows = store.overview(
                        user_jid=all_aliases[0],
                        also_jids=all_aliases[1:],
                        status=status,
                    )
                    heading = "📌 *Workload*"
                    all_work_jids = [row["user_jid"] for row in rows if row.get("user_jid")]
                    display_names = _get_display_name_map(client, chat, all_work_jids)
                    from features.work import _format as _fmt
                    lines = [heading]
                    lines.extend(_fmt(row, display_names) for row in rows)
                    _send(client, chat, "\n".join(lines) if rows else heading + "\n\n📭 No matching work.", mention_jids=all_work_jids)
                    return {"rows": [_structured_work_row(row) for row in rows], "row_count": len(rows)}
                raise ValueError("event reference is required")
            tasks = TaskStore(factory).list_for_event(int(event_id), status=status)
            rows = []
            for task in tasks:
                assignments = store.overview(
                    target_type="task", target_id=task.id, admin=True
                )
                rows.append({
                    "task_id": task.id,
                    "title": task.title,
                    "status": task.status,
                    "priority": task.priority,
                    "event_id": task.event_id,
                    "assignees": [row["user_jid"] for row in assignments],
                })
            lines = [f"🧩 *Tasks under event {event_id}*"]
            all_assignee_jids = [jid for row in rows for jid in row["assignees"]]
            display_names = _get_display_name_map(client, chat, all_assignee_jids)
            lines.extend(
                f"• `task {row['task_id']}` *{row['title']}* — `{row['status']}` "
                f"({row['priority']}) | "
                + (", ".join(
                    f"@{display_names.get(jid, jid_user(jid))}"
                    for jid in row["assignees"]
                ) or "unassigned")
                for row in rows
            )
            _send(client, chat, "\n".join(lines) if rows else f"📭 No tasks found under event {event_id}.", mention_jids=all_assignee_jids)
            return {"tasks": rows, "task_count": len(rows), "event_id": int(event_id)}

        if capability == "work.my":
            # "work.my" shows the sender's own workload.  The audience argument
            # may contain a mentioned user when the sender asks about someone
            # else (e.g. "@me show tasks for @Shuvam").
            audience_jids = arguments.get("audience") or []
            if isinstance(audience_jids, str):
                audience_jids = [audience_jids]
            if not audience_jids and visible_mentions:
                audience_jids = list(visible_mentions)
            if audience_jids:
                # Admin asking about someone else's workload.
                target_jid = audience_jids[0]
                all_aliases = _alias_jids(target_jid)
                rows = store.overview(
                    user_jid=all_aliases[0],
                    also_jids=all_aliases[1:],
                    status=status,
                )
                heading = f"📌 *Workload*"
            else:
                norm_sender = normalize_jid(sender)
                all_aliases = _alias_jids(norm_sender)
                import logging
                logging.getLogger("features.natural_language").info(
                    "work.my self-query debugging: sender=%s norm=%s aliases=%s",
                    sender, norm_sender, all_aliases
                )
                rows = store.overview(
                    user_jid=all_aliases[0],
                    also_jids=all_aliases[1:],
                    status=status,
                )
                heading = "📌 *My Workload*"
        else:
            target_type = arguments.get("target_type")
            target_id = arguments.get("target_id")
            rows = store.overview(
                user_jid=None if actor.role == "admin" else sender,
                admin=actor.role == "admin",
                status=status,
                target_type=target_type,
                target_id=int(target_id) if str(target_id or "").isdigit() else None,
            )
            if actor.role == "admin":
                rows += store.unassigned(target_type=target_type)
            heading = "📋 *Work Overview*"
        all_work_jids = [row["user_jid"] for row in rows if row.get("user_jid")]
        display_names = _get_display_name_map(client, chat, all_work_jids)
        lines = [heading]
        lines.extend(_format(row, display_names) for row in rows)
        _send(client, chat, "\n".join(lines) if rows else heading + "\n\n📭 No matching work.", mention_jids=all_work_jids)
        return {"rows": [_structured_work_row(row) for row in rows], "row_count": len(rows)}
    except Exception as exc:
        client.send_message(chat, f"⚠️ {exc}")
        return None


def _parse_date(value):
    if not value:
        return None
    from datetime import datetime
    try:
        return datetime.strptime(str(value), "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("dates must use YYYY-MM-DD") from exc


def execute_work_creation(client, message, intent: dict, factory) -> dict | None:
    """Create an event/task and return its durable identifiers to the plan."""
    source = message.Info.MessageSource
    chat = source.Chat
    capability = intent["capability"]
    arguments = intent.get("arguments", {})
    from db.auth import audit

    actor = _authorize_tool(factory, source.Sender, client, chat, capability)
    if not actor:
        return None

    try:
        if capability == "work.create_event":
            from db.event_store import EventStore
            labels = arguments.get("labels") or []
            if isinstance(labels, str):
                labels = [item.strip() for item in labels.split(",") if item.strip()]
            name = str(arguments.get("name") or "").strip()
            if not name:
                client.send_message(
                    chat,
                    "⚠️ To create an event I need at least a *name*.\n"
                    "Example: `@me create a hackathon event called PBCTF 5.0`\n"
                    "Optional extras: type (participation/organization), category, description, start/end dates.",
                )
                return None
            event = EventStore(factory).create_event(
                name=name,
                type=str(arguments.get("type") or "organization").strip(),
                category=str(arguments.get("category") or "other").strip(),
                description=arguments.get("description"),
                labels=labels,
                start_date=_parse_date(arguments.get("start")),
                end_date=_parse_date(arguments.get("end")),
                status="active",
            )
            audit(factory, actor, "event.create", "natural_language", {
                "event_id": event["id"], "name": event["name"],
            })
            client.send_message(chat, f"✅ Event `{event['id']}` created: *{event['name']}*")
            return {"event_id": event["id"], "event": event}

        from db.task_store import TaskStore
        title = str(arguments.get("title") or "").strip()
        if not title:
            client.send_message(
                chat,
                "⚠️ To create a task I need at least a *title*.\n"
                "Example: `@me create a task called Design poster due 2026-08-20 under event LFX`\n"
                "Optional extras: description, due date (YYYY-MM-DD), priority (low/medium/high), event name.",
            )
            return None
        raw_event = arguments.get("event_id") if arguments.get("event_id") is not None else (arguments.get("event_name") or arguments.get("event"))
        resolved_event_id = None
        if raw_event is not None:
            if isinstance(raw_event, int) or (isinstance(raw_event, str) and raw_event.strip().isdigit()):
                resolved_event_id = int(str(raw_event).strip())
            elif isinstance(raw_event, str) and raw_event.strip() and factory:
                try:
                    from db.event_store import EventStore
                    from features.natural_language import _entity_match_score
                    events = EventStore(factory).list_events(status="active")
                    match = next((e for e in events if e["name"].casefold() == raw_event.strip().casefold()), None)
                    if not match:
                        ranked = sorted(
                            ((e, _entity_match_score(raw_event, e["name"], e.get("category", ""))) for e in events),
                            key=lambda item: -item[1],
                        )
                        if ranked and ranked[0][1] >= 0.4:
                            match = ranked[0][0]
                    if match:
                        resolved_event_id = match["id"]
                except Exception:
                    pass

        task = TaskStore(factory).create(
            title=title,
            created_by_jid=source.Sender,
            description=arguments.get("description"),
            event_id=resolved_event_id,
            due_date=_parse_date(arguments.get("due")),
            priority=str(arguments.get("priority") or "medium").lower(),
        )
        audit(factory, actor, "task.create", "natural_language", {
            "task_id": task.id, "title": task.title, "event_id": task.event_id,
        })
        parent = f" under event {task.event_id}" if task.event_id else ""
        client.send_message(chat, f"✅ Task `{task.id}` created{parent}: *{task.title}*")
        return {"task_id": task.id, "event_id": task.event_id, "task": {
            "id": task.id, "title": task.title, "event_id": task.event_id,
        }}
    except (TypeError, ValueError) as exc:
        client.send_message(chat, f"⚠️ {exc}")
        return None

def execute_collection_mutation(
    client,
    message,
    intent: dict,
    members: list[str],
    factory,
    resolve_collection: Callable[[object, str], str | None],
) -> dict | None:
    """Execute a subgroup add/remove/delete/list/info operation."""
    source = message.Info.MessageSource
    chat = source.Chat
    action = intent["capability"].split(".", 1)[1]

    actor = _authorize_tool(factory, source.Sender, client, chat, intent["capability"])
    if not actor:
        return None

    from db.subgroup_store import SubgroupStore
    from features.subgroups import add_subgroup_members, remove_subgroup_members
    store = SubgroupStore(factory)

    try:
        if action == "list":
            subgroups = store.read()
            if not subgroups:
                client.send_message(chat, "📭 No subgroups defined yet.")
                return {"subgroups": [], "collection_count": 0}
            lines = [f"• *@{name}* — {len(m)} member(s)" for name, m in sorted(subgroups.items())]
            client.send_message(chat, f"*📋 Subgroups ({len(subgroups)})*\n\n" + "\n".join(lines))
            return {"subgroups": list(subgroups.keys()), "collection_count": len(subgroups)}

        if action == "info":
            raw_coll = intent.get("arguments", {}).get("collection")
            collection = resolve_collection(factory, raw_coll) if raw_coll else None
            if not collection:
                client.send_message(chat, "⚠️ I couldn't resolve the subgroup name.")
                return None
            subgroups = store.read()
            if collection not in subgroups:
                client.send_message(chat, f"⚠️ Subgroup *@{collection}* does not exist.")
                return None
            members_list = subgroups[collection]
            display_names = _get_display_name_map(client, chat, members_list)
            mention_parts = [
                f"@{display_names.get(jid, jid_user(jid))}"
                for jid in members_list
            ]
            text = f"*@{collection}* — {len(members_list)} member(s)\n\n" + "\n".join(f"  • {m}" for m in mention_parts)
            client.send_message(chat, text)
            return {"collection": collection, "members": members_list, "member_count": len(members_list)}

        if action == "delete":
            raw_coll = intent.get("arguments", {}).get("collection")
            collection = resolve_collection(factory, raw_coll) if raw_coll else None
            if collection:
                deleted = store.delete(collection)
                if deleted:
                    client.send_message(chat, f"🗑️ Subgroup *@{collection}* deleted.")
                    return {"collection": collection, "action": action, "deleted": True}
                else:
                    client.send_message(chat, f"⚠️ Subgroup *@{collection}* does not exist.")
                    return None
            else:
                # Delete ALL subgroups
                subgroups = store.read()
                if not subgroups:
                    client.send_message(chat, "📭 No subgroups defined.")
                    return {"action": action, "deleted_count": 0}
                count = len(subgroups)
                store.write({})
                client.send_message(chat, f"🗑️ Deleted all {count} subgroup(s).")
                return {"action": action, "deleted_count": count}

        # For add and remove, require a resolved collection name
        raw_coll = intent.get("arguments", {}).get("collection")
        collection = resolve_collection(factory, raw_coll) if raw_coll else None
        if not collection:
            client.send_message(chat, "⚠️ I couldn't resolve the subgroup name.")
            return None

        if action == "add":
            added, total = add_subgroup_members(store, collection, members)
            if added:
                client.send_message(
                    chat,
                    f"✅ Added {added} member(s) to @{collection} (total: {total}).",
                )
            else:
                client.send_message(
                    chat,
                    f"ℹ️ All mentioned users are already in @{collection} ({total} members).",
                )
        else:
            removed, remaining, deleted = remove_subgroup_members(
                store, collection, members
            )
            if deleted:
                client.send_message(
                    chat,
                    f"🗑️ Subgroup @{collection} deleted (no members remaining).",
                )
            else:
                client.send_message(
                    chat,
                    f"✅ Removed {removed} member(s) from @{collection} "
                    f"({remaining} remaining).",
                )
    except ValueError as exc:
        client.send_message(chat, f"⚠️ {exc}")
        return None
    return {"collection": collection, "action": action, "members": members}


def execute_label_mutation(
    client,
    message,
    intent: dict,
    members: list[str],
    factory,
    resolve_collection: Callable[[object, str], str | None],
) -> dict | None:
    """Execute label add/remove using direct resolved members."""
    source = message.Info.MessageSource
    chat = source.Chat
    action = intent["capability"].split(".", 1)[1]
    collection = resolve_collection(
        factory, intent.get("arguments", {}).get("collection")
    )
    if not collection:
        client.send_message(chat, "⚠️ I couldn't resolve the label name.")
        return None
    from db.auth import audit, jid_user, normalize_jid
    from db.subgroup_store import SubgroupStore
    from features.labels import add_label_members, remove_label_members

    actor = _authorize_tool(factory, source.Sender, client, chat, intent["capability"])
    if not actor:
        return None

    targets = [normalize_jid(member) for member in members if normalize_jid(member)]
    if actor.role != "admin":
        sender_user = jid_user(source.Sender)
        if any(jid_user(target) != sender_user for target in targets):
            client.send_message(
                chat,
                "⛔ You can only add or remove yourself. "
                "Ask an admin to change someone else's labels.",
            )
            return None
    try:
        store = SubgroupStore(factory)
        if action == "add":
            added, total = add_label_members(store, collection, targets)
            audit(
                factory,
                actor,
                "label.assign",
                "natural_language",
                {"label": collection, "added": added},
            )
            client.send_message(
                chat,
                f"✅ Label {collection} now has {total} member(s)."
                + (f" Added {len(added)}." if added else " No new members."),
            )
        else:
            removed, deleted = remove_label_members(store, collection, targets)
            audit(
                factory,
                actor,
                "label.remove",
                "natural_language",
                {"label": collection, "removed": removed},
            )
            client.send_message(
                chat,
                f"✅ Removed {removed} member(s) from {collection}."
                + ("" if not deleted else " Label deleted (now empty)."),
            )
    except ValueError as exc:
        client.send_message(chat, f"⚠️ {exc}")
        return None
    return {"collection": collection, "action": action, "members": targets}


def execute_work_assignment(
    client,
    message,
    intent: dict,
    members: list[str],
    factory,
    resolve_work_target: Callable[[dict], str | None],
) -> dict | None:
    """Assign/unassign a work item to concrete resolved members."""
    source = message.Info.MessageSource
    chat = source.Chat
    from db.auth import audit, normalize_jid
    from db.work_store import WorkStore

    actor = _authorize_tool(factory, source.Sender, client, chat, intent["capability"])
    if not actor:
        return None
    action = intent["capability"].split(".", 1)[1]
    reference = resolve_work_target(intent.get("arguments", {}))
    if not reference:
        if action == "unassign":
            # No specific target — unassign from ALL currently assigned work items.
            store = WorkStore(factory)
            targets = list(dict.fromkeys(
                normalize_jid(member) for member in members if normalize_jid(member)
            ))
            try:
                all_rows = store.overview(admin=True)
                # Group by (target_type, target_id)
                seen: set[tuple] = set()
                work_items: list[tuple[str, int]] = []
                for row in all_rows:
                    key = (row.get("target_type", ""), row.get("target_id"))
                    if key[0] and key[1] is not None and key not in seen:
                        seen.add(key)
                        work_items.append(key)
                if not work_items:
                    client.send_message(chat, "📭 No assignments found to remove.")
                    return {"action": action, "members": []}
                total_removed: list[str] = []
                for t_type, t_id in work_items:
                    t_targets = list(targets)
                    if not t_targets:
                        # Remove all assignees for this specific item
                        item_rows = store.overview(target_type=t_type, target_id=t_id, admin=True)
                        t_targets = list(dict.fromkeys(
                            normalize_jid(r["user_jid"]) for r in item_rows if normalize_jid(r["user_jid"])
                        ))
                    if t_targets:
                        removed = store.unassign_many(t_type, t_id, t_targets)
                        audit(factory, actor, f"{t_type}.unassign", "natural_language",
                              {"target_id": t_id, "users": removed})
                        total_removed.extend(removed)
                if total_removed:
                    client.send_message(chat, f"✅ Removed {len(total_removed)} assignment(s) across all work items.")
                else:
                    client.send_message(chat, "📭 No matching assignments found.")
                return {"action": action, "members": total_removed}
            except Exception as exc:
                client.send_message(chat, f"⚠️ {exc}")
                return None
        target_name = intent.get("arguments", {}).get("target_name") or intent.get("arguments", {}).get("target_id") or ""
        if target_name:
            client.send_message(
                chat,
                f"⚠️ I couldn't find an event or task named *{target_name}*.\n"
                "Use `!work` to see current events and their IDs, then try again with the exact ID.\n"
                "Example: `@me assign event 1 to subgroup abc`",
            )
        else:
            client.send_message(chat, "⚠️ Please specify which event or task to assign. Example: `@me assign event LFX to subgroup abc`")
        return None

    target_type, target_id = reference.split()
    store = WorkStore(factory)
    targets = list(dict.fromkeys(normalize_jid(member) for member in members if normalize_jid(member)))

    try:
        if action == "unassign" and not targets:
            # No explicit audience — remove ALL current assignees for this work item.
            current_rows = store.overview(
                target_type=target_type, target_id=int(target_id), admin=True
            )
            targets = list(dict.fromkeys(
                normalize_jid(row["user_jid"])
                for row in current_rows
                if normalize_jid(row["user_jid"])
            ))
            if not targets:
                client.send_message(chat, f"📭 No assignments found on {target_type} {target_id}.")
                return {"target": f"{target_type} {target_id}", "action": action, "members": []}
        elif not targets:
            client.send_message(chat, "⚠️ I couldn't resolve any assignees.")
            return None


        if action == "assign":
            rows = store.assign_many(target_type, int(target_id), targets)
            assigned_jids = [row["user_jid"] for row in rows]
            audit(
                factory,
                actor,
                f"{target_type}.assign",
                "natural_language",
                {"target_id": int(target_id), "users": targets},
            )
            _send(
                client,
                chat,
                f"✅ Assigned {target_type} {target_id} to "
                + _mention_text(client, chat, assigned_jids)
                + ".",
                mention_jids=assigned_jids,
            )
            return {"target": f"{target_type} {target_id}", "action": action, "members": assigned_jids}
        else:
            removed = store.unassign_many(target_type, int(target_id), targets)
            audit(
                factory,
                actor,
                f"{target_type}.unassign",
                "natural_language",
                {"target_id": int(target_id), "users": removed},
            )
            client.send_message(
                chat,
                f"✅ Removed {len(removed)} assignment(s) from "
                f"{target_type} {target_id}."
                if removed
                else "📭 No matching assignments found.",
            )
            return {"target": f"{target_type} {target_id}", "action": action, "members": removed}
    except Exception as exc:
        client.send_message(chat, f"⚠️ {exc}")
        return None
