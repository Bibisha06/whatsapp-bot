"""Unified user and admin workflow for events, tasks, assignments and progress."""
from __future__ import annotations

from datetime import datetime
import logging
import re
from typing import TYPE_CHECKING

from db.auth import audit, gate, jid_user, normalize_jid
from db.event_store import EventStore, validate_event_type_category
from db.schema_store import FIELD_TYPES, SchemaStore
from db.subgroup_store import SubgroupStore
from db.task_store import TaskStore
from db.work_store import PROGRESS_STATUSES, WorkStore
from db.reminder_store import ReminderStore
from features.subgroups import _get_mentioned_jids, _get_text

if TYPE_CHECKING:
    from neonize.client import NewClient

log = logging.getLogger(__name__)
WORK_COMMANDS = ("!my", "!work", "!events", "!tasks", "!task",
                 "!update", "!update-edit", "!history", "!status", "!set-status",
                 "!complete-task", "!assign", "!unassign", "!add-task",
                 "!update-task", "!delete-task", "!create-event", "!delete-event",
                 "!update-event", "!schema")
WORK_SUBCOMMANDS = {"assign", "unassign", "update", "edit", "history", "status", "set-status", "complete", "start", "create", "tasks", "reminders", "reminder", "schema"}


def _format(row: dict, display_names: dict[str, str] | None = None) -> str:
    typ = row["target_type"]
    ident = row.get("event_id") if typ == "event" else row.get("task_id")
    raw_jid = normalize_jid(row.get("user_jid")) if row.get("user_jid") else ""
    # Keep a JID-backed token in the text.  _send resolves it to the current
    # WhatsApp contact name and attaches a real mention to the message.
    who = f" @+{jid_user(raw_jid)}" if raw_jid else " unassigned"
    due = f" | due {row['due_date'].strftime('%Y-%m-%d')}" if row.get("due_date") else ""
    progress = row.get("status") or "unassigned"
    event_kind = f" | {row['event_type']}/{row['event_category']}" if typ == "event" and row.get("event_type") else ""
    lifecycle = f" | lifecycle `{row['lifecycle_status']}`" if row.get("lifecycle_status") else ""
    return f"• `{typ} {ident}` *{row.get('title', row.get('name', ''))}* — `{progress}`{event_kind}{who}{due}{lifecycle}"


def _text_value(value) -> str:
    """Convert Neonize/protobuf scalar or nested values to clean text."""
    if value is None:
        return ""
    # protobuf JID-like objects are not useful as names.
    if getattr(value, "User", None) and getattr(value, "Server", None):
        return ""
    text = str(value).strip()
    return text if text else ""


def _object_name(obj) -> str:
    """Extract a contact/profile name from whatever Neonize object is returned."""
    if obj is None:
        return ""

    # These are the common WhatsApp/contact name fields across Neonize/WA
    # protobuf versions. We deliberately do not write any of these to the DB.
    for field in (
        "DisplayName",
        "PushName",
        "Pushname",
        "FullName",
        "Name",
        "Notify",
        "VerifiedName",
        "BusinessName",
        "ShortName",
    ):
        value = getattr(obj, field, None)
        text = _text_value(value)
        if text:
            return text

    # Some protobuf/contact wrappers expose the useful data one level down.
    for field in ("Contact", "User", "Info"):
        nested = getattr(obj, field, None)
        if nested is not None and nested is not obj:
            name = _object_name(nested)
            if name:
                return name

    return ""


def _phone_for_jid(client, jid: str) -> str:
    """Return a real phone JID for either a phone JID or a WhatsApp LID."""
    normalized = normalize_jid(jid)
    if not normalized:
        return ""
    if normalized.endswith("@s.whatsapp.net"):
        return normalized
    if normalized.endswith("@lid"):
        try:
            phone = normalize_jid(client.get_pn_from_lid(normalized))
            if phone.endswith("@s.whatsapp.net"):
                return phone
        except Exception:
            pass
    return ""


def _get_display_name_map(client, chat, jids) -> dict[str, str]:
    """Resolve current WhatsApp/contact names without storing them in our DB.

    The group Participant protobuf on this Neonize/WhatsApp session currently
    returns DisplayName='', so relying only on Participants cannot work. We
    therefore try, in order:
      1. group participant DisplayName;
      2. Neonize contact APIs, when available;
      3. get_user_info() fields such as PushName/FullName;
      4. the real phone number for LID users.

    The returned mapping is temporary and exists only for formatting the
    outgoing message. No name is persisted.
    """
    wanted = []
    for jid in jids or []:
        normalized = normalize_jid(jid)
        if normalized and normalized not in wanted:
            wanted.append(normalized)
    if not wanted:
        return {}

    names: dict[str, str] = {}
    phone_by_jid: dict[str, str] = {}
    for jid in wanted:
        phone = _phone_for_jid(client, jid)
        if phone:
            phone_by_jid[jid] = phone

    # ------------------------------------------------------------
    # 1. Group participant metadata.
    # ------------------------------------------------------------
    try:
        info = client.get_group_info(chat)
        participants = getattr(info, "Participants", []) or []
        for participant in participants:
            raw = getattr(participant, "JID", None) or getattr(participant, "LID", None)
            participant_jid = normalize_jid(raw)
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
                for wanted_jid in wanted:
                    if wanted_jid == phone_jid:
                        phone_by_jid[wanted_jid] = phone_jid
                    elif wanted_jid.endswith("@lid"):
                        try:
                            if normalize_jid(client.get_pn_from_lid(wanted_jid)) == phone_jid:
                                phone_by_jid[wanted_jid] = phone_jid
                        except Exception:
                            pass
    except Exception as exc:
        log.info("Group participant name lookup failed: %s", exc)

    # ------------------------------------------------------------
    # 2. Try contact APIs exposed by the installed Neonize version.
    # ------------------------------------------------------------
    unresolved = [jid for jid in wanted if jid not in names]
    contact_methods = ("get_contact", "get_contact_info")
    for method_name in contact_methods:
        method = getattr(client, method_name, None)
        if not callable(method):
            continue
        for jid in list(unresolved):
            candidates = [jid]
            phone = phone_by_jid.get(jid) or _phone_for_jid(client, jid)
            if phone and phone not in candidates:
                candidates.append(phone)
            for candidate in candidates:
                try:
                    obj = method(candidate)
                    name = _object_name(obj)
                    if name:
                        names[jid] = name
                        break
                except Exception:
                    continue
        unresolved = [jid for jid in wanted if jid not in names]
        if not unresolved:
            break

    # ------------------------------------------------------------
    # 3. Neonize's known user-info API. It may expose PushName in some
    # versions even though the current version's common fields are sparse.
    # ------------------------------------------------------------
    unresolved = [jid for jid in wanted if jid not in names]
    get_user_info = getattr(client, "get_user_info", None)
    if callable(get_user_info) and unresolved:
        query_jids = []
        for jid in unresolved:
            query_jids.append(jid)
            phone = phone_by_jid.get(jid) or _phone_for_jid(client, jid)
            if phone and phone not in query_jids:
                query_jids.append(phone)
        try:
            results = get_user_info(*query_jids)
            for obj in list(results or []):
                returned = normalize_jid(getattr(obj, "JID", None))
                name = _object_name(obj)
                if not name:
                    continue
                keys = [returned]
                if returned.endswith("@lid"):
                    phone = _phone_for_jid(client, returned)
                    if phone:
                        keys.append(phone)
                for key in keys:
                    if key in wanted:
                        names[key] = name
        except Exception as exc:
            log.info("User-info name lookup failed: %s", exc)

    return names


def _parse(args: str):
    """Parse overview filters while accepting the old colon form silently."""
    status = None
    typ = None
    ident = None
    jid = None
    tokens = args.split()
    index = 0
    while index < len(tokens):
        token = tokens[index]
        low = token.lower()
        if low in PROGRESS_STATUSES:
            status = low
        elif low in ("event", "task") and index + 1 < len(tokens):
            typ = low
            match = re.fullmatch(r"(\d+)(?:@(.+))?", tokens[index + 1])
            if match:
                ident, jid = int(match.group(1)), match.group(2)
                index += 1
                if index + 1 < len(tokens) and tokens[index + 1].startswith("@") and jid is None:
                    jid = tokens[index + 1][1:]
                    index += 1
        elif low.startswith("event:") or low.startswith("task:"):
            match = re.fullmatch(r"(event|task):(\d+)(?:@(.+))?", token, re.I)
            if match:
                typ, ident, jid = match.group(1).lower(), int(match.group(2)), match.group(3)
        elif low.isdigit() and ident is None:
            ident, typ = int(low), "event"
        index += 1
    return status, typ, ident, jid.lstrip("@") if jid else None


def _target(tokens: list[str], start: int = 0):
    """Return (type, id, optional jid, next index) from space-based syntax."""
    if start >= len(tokens):
        raise ValueError("target must start with `event` or `task`")
    legacy = re.fullmatch(r"(event|task):(\d+)(?:@(.+))?", tokens[start], re.I)
    if legacy:
        return legacy.group(1).lower(), int(legacy.group(2)), legacy.group(3), start + 1
    if tokens[start].lower() not in ("event", "task"):
        raise ValueError("target must start with `event` or `task`")
    typ = tokens[start].lower()
    if start + 1 >= len(tokens):
        raise ValueError(f"usage: {typ} <id>")
    match = re.fullmatch(r"(\d+)(?:@(.+))?", tokens[start + 1])
    if not match:
        raise ValueError(f"usage: {typ} <id>")
    ident, jid = int(match.group(1)), match.group(2)
    next_index = start + 2
    if jid is None and next_index < len(tokens) and tokens[next_index].startswith("@"):
        jid, next_index = tokens[next_index][1:], next_index + 1
    return typ, ident, jid, next_index


def _phone_jid_for_mention(client, chat, jid: str) -> str:
    """Resolve a WhatsApp LID mention to its real phone JID when available."""
    normalized = normalize_jid(jid)
    if not normalized or normalized.endswith("@s.whatsapp.net"):
        return normalized
    if not normalized.endswith("@lid"):
        return normalized
    from features.subgroups import _resolve_lid_to_pn
    pn = _resolve_lid_to_pn(client, normalized)
    if pn != normalized and pn.endswith("@s.whatsapp.net"):
        return pn
    try:
        for participant in getattr(client.get_group_info(chat), "Participants", []) or []:
            participant_jid = normalize_jid(
                getattr(participant, "JID", None) or getattr(participant, "LID", None)
            )
            if participant_jid != normalized:
                continue
            phone = re.sub(r"[^0-9]", "", str(getattr(participant, "PhoneNumber", "") or ""))
            if phone:
                return f"{phone}@s.whatsapp.net"
    except Exception:
        pass
    return normalized


def _assign_targets(client, chat, message, remainder: str, inline_jid: str | None, factory) -> tuple[list[str], dict[str, str]]:
    """Collect assignees and map temporary WhatsApp LIDs to phone JIDs."""
    candidates = []
    if inline_jid:
        candidates.append(inline_jid)
    candidates.extend(_get_mentioned_jids(message))
    subgroups = SubgroupStore(factory).read()
    for name in re.findall(r"@([A-Za-z0-9_-]{2,32})", remainder or ""):
        candidates.extend(subgroups.get(name.lower(), []))
    unique = {}
    aliases: dict[str, str] = {}
    for candidate in candidates:
        normalized = normalize_jid(candidate)
        if normalized:
            canonical = _phone_jid_for_mention(client, chat, normalized)
            if canonical != normalized:
                aliases[normalized] = canonical
            unique.setdefault(jid_user(canonical), canonical)
    return list(unique.values()), aliases


def _reference(typ: str, ident: int, jid: str | None, sender: str, *, use_sender: bool = True) -> str:
    target_jid = jid or (sender if use_sender else None)
    return f"{typ} {ident}" + (f" @{target_jid}" if target_jid else "")


def _resolve_admin_target(store: WorkStore, typ: str, ident: int, jid: str | None) -> str | None:
    """Resolve an admin's target without silently choosing a user."""
    if jid:
        return jid
    rows = store.overview(admin=True, target_type=typ, target_id=ident)
    if not rows:
        return None
    if len(rows) > 1:
        raise ValueError(f"mention the target user for {typ} {ident}; multiple users are assigned")
    return rows[0]["user_jid"]


def _send(client, chat, text: str, mention_jids: list[str] | None = None) -> None:
    """Send text with real WhatsApp mention JIDs and transient display labels.

    The database JID is never changed. Visible names are resolved at send time.
    If WhatsApp does not expose a name, the real phone number is shown instead
    of the internal LID.
    """
    seen: set[str] = set()
    all_jids: list[str] = []

    def _add(jid: str) -> None:
        normalized = normalize_jid(jid)
        if normalized and normalized not in seen:
            seen.add(normalized)
            all_jids.append(normalized)

    for jid in mention_jids or []:
        _add(jid)

    # Pick up literal @phone/@LID tokens already present in the text.
    for match in re.finditer(
        r"@(?:\+)?(\d{7,16})(?:@(s\.whatsapp\.net|lid))?\b",
        text,
    ):
        number = match.group(1)
        server = match.group(2) or "s.whatsapp.net"
        _add(f"{number}@{server}")

    if not all_jids:
        client.send_message(chat, text)
        return

    # Resolve visible labels independently from the mention metadata.
    display_names = _get_display_name_map(client, chat, all_jids)
    resolved_mentions: list[str] = []

    for original_jid in all_jids:
        normalized = normalize_jid(original_jid)
        resolved_jid = normalized
        if normalized.endswith("@s.whatsapp.net"):
            try:
                lid = normalize_jid(client.get_lid_from_pn(normalized))
                if lid.endswith("@lid"):
                    resolved_jid = lid
            except Exception:
                pass

        if resolved_jid not in resolved_mentions:
            resolved_mentions.append(resolved_jid)

        name = display_names.get(normalized) or display_names.get(resolved_jid)
        if name:
            label = name
        else:
            phone = _phone_for_jid(client, resolved_jid) or _phone_for_jid(client, normalized)
            label = jid_user(phone or normalized)

        # Replace both +LID/+phone and plain @LID/@phone forms.
        original_user = jid_user(normalized)
        resolved_user = jid_user(resolved_jid)
        for user in dict.fromkeys((original_user, resolved_user)):
            if not user:
                continue
            text = re.sub(
                rf"@\+?{re.escape(user)}\b",
                f"@{label}",
                text,
            )

    try:
        from neonize.proto.waE2E.WAWebProtobufsE2E_pb2 import (
            ContextInfo,
            ExtendedTextMessage,
            Message,
        )

        proto_msg = Message(
            extendedTextMessage=ExtendedTextMessage(
                text=text,
                contextInfo=ContextInfo(
                    mentionedJID=resolved_mentions,
                ),
            ),
        )
        client.send_message(chat, proto_msg)
        return
    except Exception as exc:
        log.info("Failed to send protobuf mention message: %s", exc)

    client.send_message(chat, text)


def _parse_date(label: str, value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"{label} must use YYYY-MM-DD")


def _legacy_field_args(raw: str) -> dict:
    result = {}
    for part in raw.split("|")[1:]:
        key, _, value = part.strip().partition(" ")
        if not value and ":" in part:
            key, value = (item.strip() for item in part.split(":", 1))
        key, value = key.lower(), value.strip()
        if key in ("description", "desc"):
            result["description"] = value
        elif key == "title":
            result["title"] = value
        elif key == "name":
            result["name"] = value
        elif key in ("priority", "p"):
            result["priority"] = value.lower()
        elif key in ("due", "due_date", "date"):
            result["due_date"] = _parse_date("due date", value)
        elif key in ("start", "start_date"):
            result["start_date"] = _parse_date("start date", value)
        elif key in ("end", "end_date"):
            result["end_date"] = _parse_date("end date", value)
        elif key in ("labels", "label"):
            result["labels"] = [item.strip().lower() for item in value.split(",") if item.strip()]
        elif key == "category":
            result["category"] = value.lower()
        elif key == "type":
            result["type"] = value.lower()
        elif key == "event":
            if not value.isdigit():
                raise ValueError("event must be a numeric event id")
            result["event_id"] = int(value)
        elif key in ("status", "s"):
            result["status"] = value.lower().replace(" ", "_")
    return result


def _create_task(store: TaskStore, raw: str, sender: str):
    parts = [part.strip() for part in raw.split("|")]
    title = parts[0].strip()
    if not title:
        raise ValueError("task title is required")
    tail = parts[1:]
    description = None
    if tail and tail[0] and not re.match(
        r"^(description|desc|due|due_date|date|priority|p|event|status|s)\b",
        tail[0],
        re.IGNORECASE,
    ):
        description = tail.pop(0)
    fields = _legacy_field_args("|" + "|".join(tail))
    return store.create(title, sender, description=fields.get("description") or description,
                        due_date=fields.get("due_date"), priority=fields.get("priority", "medium"),
                        event_id=fields.get("event_id"),
                        )


def _overview(client, chat, store: WorkStore, actor, sender: str, command: str, args: str) -> None:
    status, typ, ident, mentioned_jid = _parse(args)
    is_admin = actor.role == "admin"
    if command in ("!my", "!task"):
        rows = store.overview(user_jid=sender, status=status,
                              target_type="task" if command == "!task" else None,
                              target_id=ident if command == "!task" else None)
        heading = "📌 *My Workload*"
    else:
        alias_type = "event" if command == "!events" else "task" if command == "!tasks" else None
        rows = store.overview(user_jid=None if is_admin else sender, admin=is_admin,
                              status=status, target_type=typ or alias_type, target_id=ident,
                              assignee_jid=mentioned_jid)
        if is_admin:
            rows += store.unassigned(target_type=typ or alias_type)
        heading = "📋 *Work Overview*" if command == "!work" else ("📅 *Events*" if command == "!events" else "✅ *Tasks*")
    if status:
        heading += f" — `{status}`"
    if not rows:
        _send(client, chat, heading + "\n\n📭 No matching work.")
        return
    event_rows = [r for r in rows if r["target_type"] == "event"]
    task_rows = [r for r in rows if r["target_type"] == "task"]
    events_by_id = {
        row.get("event_id"): row
        for row in event_rows
        if row.get("event_id") is not None
    }
    tasks_by_event: dict[int, list[dict]] = {}
    standalone_tasks: list[dict] = []
    for row in task_rows:
        parent_id = row.get("parent_event_id")
        if parent_id is None:
            standalone_tasks.append(row)
            continue
        tasks_by_event.setdefault(parent_id, []).append(row)
        if parent_id not in events_by_id:
            events_by_id[parent_id] = {
                "target_type": "event",
                "event_id": parent_id,
                "title": row.get("parent_event_name", f"Event {parent_id}"),
                "name": row.get("parent_event_name", f"Event {parent_id}"),
                "status": None,
                "user_jid": None,
                "lifecycle_status": row.get("parent_event_status"),
            }
    display_names = _get_display_name_map(
        client, chat, [row.get("user_jid") for row in rows if row.get("user_jid")]
    )
    lines = [heading]
    if events_by_id:
        lines += ["", "*Events*"]
        for event_id, event_row in events_by_id.items():
            lines.append(_format(event_row, display_names))
            for task_row in tasks_by_event.get(event_id, []):
                lines.append("  └─ " + _format(task_row, display_names).lstrip())
    if standalone_tasks:
        lines += ["", "*Tasks*"] + [_format(r, display_names) for r in standalone_tasks]
    totals = {s: sum(1 for r in rows if r.get("status") == s) for s in PROGRESS_STATUSES}
    lines += ["", "Totals: " + ", ".join(f"{s}={n}" for s, n in sorted(totals.items()))]
    _send(client, chat, "\n".join(lines))


def _handle_work_subcommand(client, chat, message, actor, sender: str, args: str, factory) -> bool:
    tokens = args.split()
    if not tokens or tokens[0].lower() not in WORK_SUBCOMMANDS:
        if tokens:
            _send(client, chat, "ℹ️ Use `!work` for the overview. Try `!work event <id>`, `!work update event <id> note <text>`, or `!work history event <id>`." )
        return True
    action = tokens[0].lower()
    store = WorkStore(factory)
    is_admin = actor.role == "admin"

    try:
        if action == "tasks":
            remainder = args[len(tokens[0]):].strip()
            parts = remainder.split()
            if len(parts) < 2 or parts[0].lower() != "event" or not parts[1].isdigit():
                raise ValueError("usage: !work tasks event <id> [todo|in_progress|done|cancelled]")
            event_id = int(parts[1])
            status = parts[2].lower() if len(parts) > 2 else None
            if status is not None and status not in ("todo", "in_progress", "done", "cancelled"):
                raise ValueError("status must be todo, in_progress, done, or cancelled")
            tasks = TaskStore(factory).list_for_event(event_id, status=status)
            if not tasks:
                _send(client, chat, f"📭 No tasks found under event {event_id}.")
                return True
            lines = [f"🧩 *Tasks under event {event_id}*"]
            all_assignee_jids: list[str] = []
            for task in tasks:
                assignments = WorkStore(factory).overview(
                    target_type="task", target_id=task.id, admin=True
                )
                assignee_jids = [row["user_jid"] for row in assignments if row.get("user_jid")]
                all_assignee_jids.extend(assignee_jids)
                assignees = ", ".join(
                    f"@+{jid_user(j)}" for j in assignee_jids
                ) or "unassigned"
                due = f" | due {task.due_date.strftime('%Y-%m-%d')}" if task.due_date else ""
                lines.append(
                    f"• `task {task.id}` *{task.title}* — `{task.status}` "
                    f"({task.priority}){due} | {assignees}"
                )
            _send(client, chat, "\n".join(lines), mention_jids=all_assignee_jids)
            return True

        if action in ("reminders", "reminder"):
            # Reminder controls are part of the unified work workflow. The
            # old !reminders commands remain aliases in features/reminders.
            from features.reminders import (
                _cmd_config, _cmd_history, _cmd_reminders_summary, _cmd_run,
            )
            remainder = args[len(tokens[0]):].strip()
            subcommand, _, sub_args = remainder.partition(" ")
            subcommand = subcommand.lower()
            reminder_store = ReminderStore(factory)
            if not subcommand or subcommand in ("status", "summary"):
                _cmd_reminders_summary(client, chat, reminder_store, actor=actor)
                return True
            if subcommand == "config":
                if not is_admin:
                    _send(client, chat, "⛔ Only administrators can configure reminders.")
                    return True
                mentions = _get_mentioned_jids(message)
                _cmd_config(client, chat, sub_args, actor, mentions, reminder_store)
                return True
            if subcommand == "run":
                if not is_admin:
                    _send(client, chat, "⛔ Only administrators can run reminders.")
                    return True
                _cmd_run(client, chat, actor, reminder_store)
                return True
            if subcommand == "history":
                _cmd_history(client, chat, sub_args, reminder_store, actor=actor)
                return True
            raise ValueError("usage: !work reminders [status|config|run|history [assignment_id]]")

        if action == "create":
            if not is_admin:
                _send(client, chat, "⛔ Only administrators can create work.")
                return True
            parts = [p.strip() for p in args[len(tokens[0]):].strip().split("|")]
            if len(parts) < 2 or parts[0].lower() not in ("event", "task"):
                _send(client, chat, "Usage: `!work create event | <participation|organization> | <category> | <name> | [description]` or `!work create task | <title> | [description] | [due YYYY-MM-DD] | [priority low|medium|high]`")
                return True
            if parts[0].lower() == "event":
                if len(parts) < 4:
                    raise ValueError("event creation needs a type, category, and name")
                event_type, category = validate_event_type_category(parts[1], parts[2])
                # Everything after the description may carry `start`, `end`, and
                # `labels` fields, so parse the tail the same way tasks do.
                extras = _legacy_field_args("|" + "|".join(parts[5:])) if len(parts) > 5 else {}
                event = EventStore(factory).create_event(
                    name=parts[3], type=event_type, category=category,
                    description=parts[4] if len(parts) > 4 else "", status="active",
                    labels=extras.get("labels"), start_date=extras.get("start_date"),
                    end_date=extras.get("end_date"))
                audit(factory, actor, "event.create", "whatsapp", {"event_id": event["id"], "name": event["name"]})
                _send(client, chat, f"✅ Event `{event['id']}` created: *{event['name']}*")
            else:
                task = _create_task(TaskStore(factory), "|".join(parts[1:]), sender)
                audit(factory, actor, "task.create", "whatsapp", {"task_id": task.id, "title": task.title})
                _send(client, chat, f"✅ Task `{task.id}` created: *{task.title}*")
            return True

        if action == "schema":
            remainder = args[len(tokens[0]):].strip()
            verb, _, rest = remainder.partition(" ")
            # PRS names these schema.create/update/delete; keep those spellings
            # working alongside the shorter set/add/remove forms.
            verb = {"create": "set", "update": "add"}.get(verb.lower(), verb.lower())
            store_schema = SchemaStore(factory)
            if verb in ("event", "task", "fields", "show", "view", "info"):
                head = rest if verb in ("fields", "show", "view", "info") else remainder
                typ, ident, _, _ = _target(head.split(), 0)
                fields = store_schema.list_fields(ident)
                if not fields:
                    _send(client, chat, f"📭 No schema defined for {typ} {ident}. Admins can add one with "
                                        f"`!schema set {typ} {ident} | org text | prs number`.")
                    return True
                lines = [f"📐 *Schema — {typ} {ident}*"]
                lines += [f"• `{item['name']}` {item['field_type']}"
                          + (f" — {', '.join(item['options'])}" if item["options"] else "")
                          for item in fields]
                _send(client, chat, "\n".join(lines))
                return True
            if not is_admin:
                _send(client, chat, "⛔ Only administrators can change event schemas.")
                return True
            head, _, spec_text = rest.partition("|")
            typ, ident, _, _ = _target(head.split(), 0)
            if verb in ("set", "add"):
                specs = [part for part in spec_text.split("|") if part.strip()]
                if not specs:
                    raise ValueError(f"usage: !schema {verb} {typ} {ident} | <name> <type>  (types: "
                                     + ", ".join(FIELD_TYPES) + ")")
                fields = (store_schema.set_fields(ident, specs) if verb == "set"
                          else store_schema.add_field(ident, specs[0]))
                audit(factory, actor, f"schema.{verb}", "whatsapp", {"event_id": ident, "specs": specs})
                _send(client, chat, f"✅ Schema for `{typ} {ident}` now has "
                                    + ", ".join(f"`{item['name']}`" for item in fields) + ".")
            elif verb in ("remove", "delete") and spec_text.strip():
                removed = store_schema.remove_field(ident, spec_text)
                audit(factory, actor, "schema.delete", "whatsapp", {"event_id": ident, "field": spec_text.strip().lower()})
                _send(client, chat, f"🗑️ Field `{spec_text.strip().lower()}` removed."
                      if removed else "📭 No such field on this event.")
            elif verb in ("clear", "delete", "remove"):
                count = store_schema.clear(ident)
                audit(factory, actor, "schema.delete", "whatsapp", {"event_id": ident, "cleared": count})
                _send(client, chat, f"🗑️ Cleared {count} field(s) from `{typ} {ident}`.")
            else:
                raise ValueError("usage: !schema <set|add|remove|clear|fields> event <id> | <name> <type>")
            return True

        if action == "edit":
            if len(tokens) < 3 or not tokens[1].isdigit():
                raise ValueError("usage: !work edit <revision_id> <new value>")
            revision = store.edit_update(int(tokens[1]), " ".join(tokens[2:]), sender)
            audit(factory, actor, "update.edit", "whatsapp", {"revision_id": revision["id"]})
            _send(client, chat, f"✅ Update `{revision['id']}` edited successfully.")
            return True

        typ, ident, jid, next_index = _target(tokens, 1)
        target_jid = jid or (sender if not is_admin else None)
        if action in ("assign", "unassign"):
            if not is_admin:
                _send(client, chat, "⛔ Only administrators can change assignments.")
                return True
            targets, aliases = _assign_targets(client, chat, message, " ".join(tokens[next_index:]), jid, factory)
            if not targets:
                raise ValueError("mention at least one user or subgroup to assign or unassign")
            for temporary_jid, phone_jid in aliases.items():
                store.reconcile_user_identity(temporary_jid, phone_jid)
            if action == "assign":
                rows = store.assign_many(typ, ident, targets)
                assigned_jids = [row["user_jid"] for row in rows if row.get("user_jid")]
                audit(factory, actor, f"{typ}.assign", "whatsapp", {"target_id": ident, "users": targets})
                visible = ", ".join(f"@+{jid_user(j)}" for j in assigned_jids)
                _send(client, chat, f"✅ Assigned `{typ} {ident}` to {visible}.", mention_jids=assigned_jids)
            else:
                removed = store.unassign_many(typ, ident, targets)
                audit(factory, actor, f"{typ}.unassign", "whatsapp", {"target_id": ident, "users": removed})
                _send(client, chat, f"✅ Removed {len(removed)} assignment(s) from `{typ} {ident}`."
                      if removed else "📭 No matching assignments found.")
            return True

        if action == "history":
            if is_admin:
                target_jid = _resolve_admin_target(store, typ, ident, target_jid)
            history = store.history(_reference(typ, ident, target_jid, sender, use_sender=not is_admin))
            if not history:
                _send(client, chat, "📭 No progress history yet.")
            else:
                lines = [f"🕘 *History — {typ} {ident}*"]
                lines.extend(f"• `{item['field']}`: {item['value']} _(update {item['id']})_" for item in history)
                _send(client, chat, "\n".join(lines))
            return True

        if action == "update":
            if is_admin:
                target_jid = _resolve_admin_target(store, typ, ident, target_jid)
            if next_index >= len(tokens) or not tokens[next_index].strip():
                raise ValueError("usage: !work update event <id> <field> <value>")
            field = tokens[next_index]
            value = " ".join(tokens[next_index + 1:]).strip()
            if not value:
                raise ValueError("update value is required")
            result = store.submit_update(_reference(typ, ident, target_jid, sender, use_sender=not is_admin), field, value, sender)
            audit(factory, actor, "update.submit", "whatsapp", {"target": f"{typ} {ident}", "field": field, "revision_id": result["id"]})
            _send(client, chat, f"✅ Update `{result['id']}` recorded for `{typ} {ident}`.")
            return True

        status = "completed" if action == "complete" else "in_progress" if action == "start" else None
        if action in ("status", "set-status", "start"):
            if action == "set-status" and next_index >= len(tokens):
                raise ValueError("usage: !work set-status event <id> <status>")
            status = tokens[next_index].lower() if action == "set-status" else status
            if action == "status":
                if is_admin:
                    target_jid = _resolve_admin_target(store, typ, ident, target_jid)
                rows = store.overview(user_jid=None if is_admin else sender, admin=is_admin,
                                      target_type=typ, target_id=ident, assignee_jid=target_jid)
                _send(client, chat, "\n".join([f"📌 *{typ.title()} {ident}*"] + ([_format(row, _get_display_name_map(client, chat, [r.get("user_jid") for r in rows if r.get("user_jid")])) for row in rows] if rows else ["📭 No assignment found."])))
                return True
            if not is_admin and action == "set-status":
                _send(client, chat, "⛔ Use `!work complete` or update your own work; administrators set explicit statuses.")
                return True
        if action in ("complete", "start") and status:
            if action == "start" and is_admin:
                target_jid = _resolve_admin_target(store, typ, ident, target_jid)
            result = store.set_status(_reference(typ, ident, target_jid, sender, use_sender=not is_admin), status, sender)
            audit(factory, actor, f"{typ}.status", "whatsapp", {"target_id": ident, "status": result["status"]})
            if action == "complete" and typ == "task":
                # Keep the task lifecycle in sync with the assignee's
                # completed progress while retaining separate status fields.
                TaskStore(factory).update(ident, status="done", force_status=True)
            _send(client, chat, f"✅ `{typ} {ident}` marked `{result['status']}`.")
            return True
        if action == "set-status" and status:
            if is_admin:
                target_jid = _resolve_admin_target(store, typ, ident, target_jid)
            result = store.set_status(_reference(typ, ident, target_jid, sender, use_sender=not is_admin), status, sender)
            audit(factory, actor, f"{typ}.status", "whatsapp", {"target_id": ident, "status": result["status"]})
            _send(client, chat, f"✅ `{typ} {ident}` set to `{result['status']}`.")
            return True
    except Exception as exc:
        log.info("work command failed: %s", exc)
        _send(client, chat, f"⚠️ {exc}")
    return True


def handle(client, message, session_factory) -> bool:
    if not message.Info or not message.Info.MessageSource:
        return False
    source = message.Info.MessageSource
    chat = source.Chat
    if getattr(chat, "Server", "") != "g.us":
        return False
    body = _get_text(message)
    command, _, args = body.partition(" ")
    command = command.lower()
    if command not in WORK_COMMANDS:
        return False
    push_name = str(getattr(message.Info, "Pushname", "") or "")
    actor = gate(session_factory, source.Sender, client, chat, "member", f"work.{command[1:]}", push_name=push_name)
    if not actor:
        return True
    sender = normalize_jid(source.Sender)
    if command == "!work" and args.strip().split()[:1] and args.strip().split()[0].lower() in WORK_SUBCOMMANDS:
        return _handle_work_subcommand(client, chat, message, actor, sender, args.strip(), session_factory)
    if command in ("!assign", "!unassign"):
        if actor.role != "admin":
            _send(client, chat, "⛔ Only administrators can change assignments.")
            return True
        parts = [part.strip() for part in args.split("|", 1)]
        head = parts[0].split()
        typ = head[0].lower() if head and head[0].lower() in ("event", "task") else "event"
        ident_token = head[1] if typ in ("event", "task") and len(head) > 1 else (head[0] if head else "")
        targets, aliases = _assign_targets(client, chat, message, parts[1] if len(parts) > 1 else "", None, session_factory)
        if not ident_token.isdigit() or not targets:
            _send(client, chat, f"Usage: `{command} {typ} <id> | @user`")
            return True
        try:
            store = WorkStore(session_factory)
            for temporary_jid, phone_jid in aliases.items():
                store.reconcile_user_identity(temporary_jid, phone_jid)
            if command == "!assign":
                rows = store.assign_many(typ, int(ident_token), targets)
                assigned_jids = [row["user_jid"] for row in rows if row.get("user_jid")]
                visible = ", ".join(f"@+{jid_user(j)}" for j in assigned_jids)
                _send(client, chat, f"✅ Assigned `{typ} {ident_token}` to {visible}.", mention_jids=assigned_jids)
            else:
                removed = store.unassign_many(typ, int(ident_token), targets)
                _send(client, chat, f"✅ Removed {len(removed)} assignment(s)." if removed else "📭 Assignment not found.")
        except Exception as exc:
            _send(client, chat, f"⚠️ {exc}")
        return True
    if command in ("!add-task", "!update-task", "!delete-task"):
        if actor.role != "admin":
            _send(client, chat, "⛔ Only administrators can manage tasks.")
            return True
        try:
            tasks = TaskStore(session_factory)
            if command == "!add-task":
                task = _create_task(tasks, args, sender)
                _send(client, chat, f"✅ Task `{task.id}` created: *{task.title}*")
            elif command == "!delete-task":
                if not args.strip().isdigit():
                    raise ValueError("usage: !delete-task <id>")
                tasks.delete(int(args.strip()))
                audit(session_factory, actor, "task.delete", "whatsapp", {"task_id": int(args.strip())})
                _send(client, chat, f"🗑️ Task `{args.strip()}` deleted.")
            else:
                parts = [part.strip() for part in args.split("|", 1)]
                if len(parts) != 2 or not parts[0].isdigit():
                    raise ValueError("usage: !update-task <id> | field value")
                fields = _legacy_field_args("|" + parts[1])
                task = tasks.update(int(parts[0]), title=fields.get("title"), description=fields.get("description"),
                                    due_date=fields.get("due_date"), priority=fields.get("priority"), status=fields.get("status"), force_status=True)
                audit(session_factory, actor, "task.update", "whatsapp", {"task_id": task.id, "fields": sorted(fields)})
                _send(client, chat, f"✅ Task `{task.id}` updated.")
        except Exception as exc:
            _send(client, chat, f"⚠️ {exc}")
        return True
    if command in ("!create-event", "!delete-event", "!update-event"):
        if actor.role != "admin":
            _send(client, chat, "⛔ Only administrators can manage events.")
            return True
        try:
            events = EventStore(session_factory)
            if command == "!update-event":
                parts = [part.strip() for part in args.split("|", 1)]
                if len(parts) != 2 or not parts[0].isdigit():
                    raise ValueError("usage: !update-event <id> | name <value> [| desc <value>] [| start YYYY-MM-DD] [| end YYYY-MM-DD] [| labels a,b]")
                fields = _legacy_field_args("|" + parts[1])
                if not fields:
                    raise ValueError("nothing to update; use name, desc, type, category, start, end, or labels")
                event = events.update_event(int(parts[0]), name=fields.get("name"), type=fields.get("type"),
                                            description=fields.get("description"), category=fields.get("category"),
                                            labels=fields.get("labels"), start_date=fields.get("start_date"),
                                            end_date=fields.get("end_date"))
                audit(session_factory, actor, "event.update", "whatsapp", {"event_id": event["id"], "fields": sorted(fields)})
                _send(client, chat, f"✅ Event `{event['id']}` updated: *{event['name']}*")
            elif command == "!create-event":
                parts = [part.strip() for part in args.split("|")]
                if len(parts) < 2:
                    raise ValueError("usage: !create-event <type> | <name> | [description]")
                event_type, category = validate_event_type_category(parts[0], "other")
                event = events.create_event(type=event_type, category=category, name=parts[1], description=parts[2] if len(parts) > 2 else "", status="active")
                _send(client, chat, f"✅ Event `{event['id']}` created: *{event['name']}*")
            else:
                if not args.strip().isdigit():
                    raise ValueError("usage: !delete-event <id>")
                if not events.delete_event(int(args.strip())):
                    raise ValueError("event not found")
                audit(session_factory, actor, "event.delete", "whatsapp", {"event_id": int(args.strip())})
                _send(client, chat, f"🗑️ Event `{args.strip()}` deleted.")
        except Exception as exc:
            _send(client, chat, f"⚠️ {exc}")
        return True
    legacy_actions = {
        "!update": "update", "!update-edit": "edit", "!history": "history",
        "!status": "status", "!set-status": "set-status",
    }
    if command in legacy_actions:
        if command == "!set-status" and "|" in args:
            if actor.role != "admin":
                _send(client, chat, "⛔ Only administrators can change lifecycle status.")
                return True
            parts = [part.strip() for part in args.split("|", 1)]
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1]:
                _send(client, chat, "Usage: `!set-status <event_id> | <draft|active|completed|cancelled>`")
                return True
            try:
                EventStore(session_factory).set_status(int(parts[0]), parts[1].lower())
                _send(client, chat, f"✅ Event `{parts[0]}` lifecycle set to `{parts[1].lower()}`.")
            except Exception as exc:
                _send(client, chat, f"⚠️ {exc}")
            return True
        return _handle_work_subcommand(client, chat, message, actor, sender,
                                       f"{legacy_actions[command]} {args}".strip(), session_factory)
    if command == "!complete-task":
        return _handle_work_subcommand(client, chat, message, actor, sender,
                                       f"complete task {args}".strip(), session_factory)
    if command == "!schema":
        return _handle_work_subcommand(client, chat, message, actor, sender,
                                       f"schema {args}".strip(), session_factory)
    _overview(client, chat, WorkStore(session_factory), actor, sender, command, args)
    return True


def register(client: "NewClient", config: dict) -> callable:
    factory = config.get("db_session_factory")
    if factory is None:
        raise RuntimeError("Work feature requires db_session_factory")
    return lambda client, message: handle(client, message, factory)
