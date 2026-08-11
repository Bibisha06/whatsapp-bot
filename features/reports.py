"""Reporting and audit surfaces for admins (PRS 7.8 and 7.9)."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from db.auth import gate, jid_user, normalize_jid
from db.report_store import ReportStore
from features.subgroups import _get_mentioned_jids, _get_text
from features.work import _send

if TYPE_CHECKING:
    from neonize.client import NewClient

log = logging.getLogger(__name__)
REPORT_CMDS = ("!reports", "!report", "!audit")


_MAX_CELL = 24


def _cell(value: str) -> str:
    """Keep columns phone-readable: long lists collapse to a count, and the full
    values stay available through `!work history`."""
    text = str(value)
    if len(text) <= _MAX_CELL:
        return text
    items = [item for item in text.split(",") if item.strip()]
    if len(items) > 1:
        return f"{len(items)} items"
    return text[:_MAX_CELL - 1] + "…"


def _table(fields: list[str], rows: list[dict]) -> str:
    """Render a fixed-width table so WhatsApp's monospace block keeps columns aligned."""
    headers = ["member"] + fields + ["status"]
    body = [[row["name"]] + [_cell(row["values"].get(field, "-")) for field in fields] + [row["status"]]
            for row in rows]
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *body)] if body \
        else [len(header) for header in headers]
    lines = ["  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)).rstrip()
             for row in [headers] + body]
    return "```\n" + "\n".join(lines) + "\n```"


def _owner_label(jid: str) -> str:
    """Leave a token for the mention-aware sender to resolve at send time."""
    return f"@+{jid_user(jid)}" if jid else "unassigned"


def _cmd_progress(client, chat, store: ReportStore, args: str) -> None:
    tokens = args.split()
    if len(tokens) < 2 or tokens[0].lower() != "event" or not tokens[1].isdigit():
        client.send_message(chat, "Usage: `!reports progress event <id>`")
        return
    data = store.cohort(int(tokens[1]))
    if not data["rows"]:
        client.send_message(chat, f"📭 Nobody is assigned to *{data['event_name']}* yet.")
        return
    # Build a simple readable list instead of (or alongside) the table
    lines = [f"📊 *{data['event_name']}* — {len(data['rows'])} assignment(s)", ""]
    for row in data["rows"]:
        row["name"] = _owner_label(row.get("user_jid", ""))
        scope = row.get("scope", "")
        scope_label = f" _(via {scope})_" if scope else ""
        status_label = f"`{row['status']}`"
        task_title = row.get("values", {}).get("task", "")
        task_label = f" — _{task_title}_" if task_title else ""
        lines.append(f"• *{row['name']}*{scope_label}{task_label} — {status_label}")
    if data["fields"]:
        lines.append("")
        lines.append(_table([f for f in data["fields"] if f != "task"], data["rows"]))
    _send(client, chat, "\n".join(lines), [row["user_jid"] for row in data["rows"] if row.get("user_jid")])


def _cmd_status_list(client, chat, store: ReportStore, status: str) -> None:
    rows = store.by_status(status)
    if not rows:
        client.send_message(chat, f"📭 Nothing is `{status}`.")
        return
    lines = [f"📋 *{status.replace('_', ' ').title()}* — {len(rows)}"]
    for row in rows:
        row["name"] = _owner_label(row.get("user_jid", ""))
        missed = f" | missed {row['missed_count']}" if row["missed_count"] else ""
        lines.append(f"• `{row['target_type']} {row['target_id']}` *{row['title']}* — {row['name']}{missed}")
    _send(client, chat, "\n".join(lines), [row["user_jid"] for row in rows if row.get("user_jid")])


def _cmd_summary(client, chat, store: ReportStore) -> None:
    data = store.summary()
    counts = ", ".join(f"{status}={count}" for status, count in sorted(data["counts"].items())) or "none"
    client.send_message(chat, "\n".join([
        "📈 *Work Report*",
        f"• Events: `{data['events']}` (unassigned: `{data['unassigned_events']}`)",
        f"• Tasks: `{data['tasks']}`",
        f"• Assignments: `{data['assignments']}`",
        f"• By status: {counts}",
        "",
        "Try `!reports progress event <id>`, `!reports pending`, or `!reports completed`.",
    ]))


def _cmd_audit(client, chat, store: ReportStore, args: str, mentions: list[str]) -> None:
    tokens = args.split()
    actor_jid = normalize_jid(mentions[0]) if mentions else None
    operation = None
    if tokens and tokens[0].lower() in ("op", "operation") and len(tokens) > 1:
        operation = tokens[1].lower()
    elif tokens and tokens[0].lower() not in ("user", "actor"):
        operation = tokens[0].lower()
    entries = store.audit_entries(actor_jid=actor_jid, operation=operation)
    if not entries:
        client.send_message(chat, "📭 No audit entries match.")
        return
    lines = ["🧾 *Audit Log*"]
    for entry in entries:
        entry["actor"] = _owner_label(entry.get("actor_jid", ""))
        stamp = entry["timestamp"].strftime("%Y-%m-%d %H:%M")
        lines.append(f"• `{entry['operation']}` by {entry['actor']} ({entry['actor_role']}) "
                     f"via {entry['source']} — {entry['result']} _{stamp}_")
    _send(client, chat, "\n".join(lines), [entry["actor_jid"] for entry in entries if entry.get("actor_jid")])


def register(client: "NewClient", config: dict) -> callable:
    factory = config.get("db_session_factory")
    if factory is None:
        raise RuntimeError("Reports feature requires db_session_factory")

    def on_message(client: "NewClient", message) -> None:
        if not message.Info or not message.Info.MessageSource:
            return
        source = message.Info.MessageSource
        chat = source.Chat
        if getattr(chat, "Server", "") != "g.us":
            return
        body = _get_text(message)
        command, _, args = body.partition(" ")
        command = command.lower()
        if command not in REPORT_CMDS:
            return
        actor = gate(factory, source.Sender, client, chat, "admin", f"report.{command[1:]}")
        if not actor:
            return
        store = ReportStore(factory)
        args = args.strip()
        try:
            if command == "!audit":
                _cmd_audit(client, chat, store, args, _get_mentioned_jids(message))
                return
            action, _, rest = args.partition(" ")
            action = action.lower()
            if action == "progress":
                _cmd_progress(client, chat, store, rest.strip())
            elif action in ("pending", "in_progress", "completed", "cancelled"):
                _cmd_status_list(client, chat, store, action)
            elif not action or action in ("generate", "summary"):
                _cmd_summary(client, chat, store)
            else:
                client.send_message(chat, "Usage: `!reports [progress event <id>|pending|completed]`")
        except Exception as exc:
            log.info("report command failed: %s", exc)
            client.send_message(chat, f"⚠️ {exc}")

    return on_message
