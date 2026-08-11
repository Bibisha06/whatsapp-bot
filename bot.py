#!/usr/bin/env python3
"""WhatsApp bot entrypoint.

The process owns configuration, database setup, legacy migration, Neonize
initialisation, feature registration, dispatch, and the connection lifecycle.
Feature business rules remain in the feature modules.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from neonize.client import NewClient
from neonize.events import ConnectedEv, MessageEv, PairStatusEv

from db.database import create_database
from db.auth import normalize_jid
from db.migration import migrate_legacy_json, migrate_unified_work, upgrade_unified_schema
from features.registry import register_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bot")

load_dotenv()


def _parse_group_ids(*env_keys: str) -> set[str]:
    """Parse comma-separated group IDs from one or more environment values."""
    ids: set[str] = set()
    for key in env_keys:
        for group_id in os.getenv(key, "").split(","):
            group_id = group_id.strip()
            if group_id:
                ids.add(group_id)
    return ids


def _build_config() -> dict:
    return {
        "group_ids": _parse_group_ids("GROUP_ID", "GROUP_IDS"),
        # The primary PBBot group is the only group allowed to trigger bot
        # handlers. Keep GROUP_ID as the backwards-compatible default.
        "pbbot_group_id": (
            os.getenv("PBBOT_GROUP_ID", "").strip()
            or os.getenv("GROUP_ID", "").strip()
            or None
        ),
        "media_group_id": os.getenv("MEDIA_GROUP_ID", "").strip() or None,
        "incident_group_id": os.getenv("INCIDENT_GROUP_ID", "").strip() or None,
        "incident_port": int(os.getenv("INCIDENT_PORT", "8081")),
        "subgroup_blocked_users": _parse_group_ids("SUBGROUP_BLOCKED_USERS"),
        "database_url": os.getenv("DATABASE_URL", "").strip(),
        "mistral_api_key": os.getenv("MISTRAL_API_KEY", "").strip(),
        "mistral_model": os.getenv("MISTRAL_MODEL", "mistral-small-latest").strip(),
        "mistral_card_model": os.getenv("MISTRAL_CARD_MODEL", "mistral-medium-3-5").strip(),
        "natural_language_knowledge_urls": os.getenv("NATURAL_LANGUAGE_KNOWLEDGE_URLS", "").strip(),
        "bot_jid": os.getenv("BOT_JID", "").strip(),
    }


def _jid_string(jid) -> str:
    """Return a canonical WhatsApp JID, matching list_groups.py output."""
    user = getattr(jid, "User", "")
    server = getattr(jid, "Server", "")
    if user and server:
        return f"{user}@{server}"
    return str(jid)


# Kept available for callers that import bot configuration, without creating a
# database connection or a Neonize session as an import side effect.
config = _build_config()


def _excepthook(exc_type, exc_value, exc_tb):
    log.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))


sys.excepthook = _excepthook


def main() -> None:
    """Initialise all runtime dependencies and connect Neonize."""
    database = create_database(config["database_url"])
    database.initialize()
    upgrade_unified_schema(database)
    migrate_legacy_json(database.session_factory, Path.cwd())
    migrate_unified_work(database.session_factory)

    # Load persisted LID<->phone alias mappings from DB so that the alias
    # registry is available immediately, before the WhatsApp reconciliation
    # background thread runs.
    from db.work_store import load_persistent_aliases
    load_persistent_aliases(database.session_factory)

    runtime_config = {
        **config,
        "db_session_factory": database.session_factory,
    }

    # Keep neonize.db independent from PostgreSQL: it is the WhatsApp session
    # database and must remain persistent across container restarts.
    session_db = Path(os.getenv("NEONIZE_SESSION_DB", "neonize.db"))
    if not session_db.is_absolute():
        session_db = Path.cwd() / session_db
    client = NewClient(str(session_db))
    allowed_group = runtime_config.get("pbbot_group_id")

    # TEST-ONLY GUARD: remove this block when production routing is designed.
    # Enforce the destination policy at the last possible point. This also
    # covers outbound messages originating from independent features such as
    # the incident webhook, not just replies to inbound messages.
    original_send_message = client.send_message

    def guarded_send_message(chat_jid, *args, **kwargs):
        if isinstance(chat_jid, str):
            chat_str = normalize_jid(chat_jid)
            if "@" in chat_str:
                u, s = chat_str.split("@", 1)
            else:
                u, s = chat_str, "s.whatsapp.net"
            try:
                from neonize.utils import build_jid
                chat_jid = build_jid(u, s)
            except Exception as e:
                log.warning("Failed to build JID for %s: %s", chat_jid, e)

        chat_id = _jid_string(chat_jid) if chat_jid else ""
        if (
            chat_id.endswith("@s.whatsapp.net")
            or chat_id.endswith("@lid")
            or (allowed_group and chat_id.endswith("@g.us") and chat_id == allowed_group)
        ):
            return original_send_message(chat_jid, *args, **kwargs)
        log.warning("Blocked outbound message to non-PBBot chat: %s", chat_id or "(unknown)")
        return None

    client.send_message = guarded_send_message

    # WhatsApp may replay pending/history messages immediately after a
    # reconnect. Fail closed during startup and ignore anything timestamped
    # before this process started, so old commands cannot run.
    startup_timestamp = int(time.time())
    accept_messages_after = time.monotonic() + 10

    @client.event(PairStatusEv)
    def on_pair_status(_client: NewClient, event: PairStatusEv):
        log.info("📱 Pair status: %s", event)

    def _start_reminder_scheduler(client: NewClient, session_factory) -> None:
        import threading
        from db.auth import upsert_user, current_user
        from db.reminder_store import ReminderStore

        # ConnectedEv can fire again after a reconnect. Keep exactly one
        # scheduler thread attached to the bot process.
        if getattr(client, "_pbbot_reminder_scheduler_started", False):
            log.info("⏰ Reminder scheduler already running; skipping duplicate start.")
            return
        client._pbbot_reminder_scheduler_started = True

        def run_scheduler_loop():
            # Wait a bit for the system to settle down after connection
            time.sleep(30)
            try:
                upsert_user(session_factory, "system@s.whatsapp.net", role="admin")
            except Exception as e:
                log.warning("Could not upsert system user: %s", e)

            store = ReminderStore(session_factory)
            while True:
                try:
                    system_user = current_user(session_factory, "system@s.whatsapp.net")
                    if system_user:
                        log.info("⏰ Running background reminders check...")
                        res = store.run_reminders(client, system_user, force_ignore_window=False, source="system")
                        log.info("⏰ Background reminders run result: %s", res)
                except Exception as exc:
                    log.exception("Error in background reminder scheduler loop: %s", exc)
                time.sleep(900)

        thread = threading.Thread(target=run_scheduler_loop, name="ReminderScheduler", daemon=True)
        thread.start()
        log.info("⏰ Background reminder scheduler thread started.")

    @client.event(ConnectedEv)
    def on_connected(_client: NewClient, _event: ConnectedEv):
        log.info("✅ Bot connected to WhatsApp — all features active")
        try:
            from neonize.utils import Jid2String
            runtime_config["bot_jid"] = Jid2String(client.get_me().JID)
        except Exception as e:
            log.warning("Could not determine bot JID: %s", e)
        _start_reminder_scheduler(client, runtime_config["db_session_factory"])
        _reconcile_lid_assignments(client, runtime_config["db_session_factory"])

    def _reconcile_lid_assignments(client: NewClient, session_factory) -> None:
        """Resolve LID-based assignments to phone JIDs using the live client.

        Always resolves known LIDs and populates the in-memory alias registry
        (_JID_ALIASES) so that overview() can match assignments regardless of
        whether the sender arrives as a LID or phone JID.
        """
        import threading

        def _run():
            import time as _time
            _time.sleep(5)  # let connection fully settle
            try:
                from db.work_store import WorkStore, _JID_ALIASES
                from neonize.utils import build_jid, Jid2String
                from db.auth import normalize_jid, jid_user
                from db.models import Assignment, User
                from sqlalchemy import select

                store = WorkStore(session_factory)

                # Collect all unique user JIDs currently in assignments (any form).
                with session_factory() as session:
                    all_assignment_jids = list(dict.fromkeys(
                        row.user_jid
                        for row in session.scalars(select(Assignment)).all()
                        if row.user_jid
                    ))
                    # Also collect any user JIDs from the users table that look like LIDs.
                    all_user_jids = list(dict.fromkeys(
                        u.jid
                        for u in session.scalars(select(User)).all()
                        if u.jid
                    ))

                # Build a candidate set of LIDs to resolve.
                # Include both LID-form JIDs from assignments AND try to get LIDs
                # for phone-form JIDs (reverse lookup) so we cover both directions.
                lid_to_phone: dict[str, str] = {}

                # Forward: LID -> phone
                candidate_lids = [j for j in (all_assignment_jids + all_user_jids)
                                  if j.endswith("@lid")]
                for lid in dict.fromkeys(candidate_lids):
                    try:
                        u, s = lid.split("@", 1)
                        lid_jid = build_jid(u, s)
                        pn_jid = client.get_pn_from_lid(lid_jid)
                        if pn_jid:
                            phone = normalize_jid(Jid2String(pn_jid))
                            if phone and phone.endswith("@s.whatsapp.net"):
                                lid_to_phone[lid] = phone
                                log.info("📋 LID resolved: %s -> %s", lid, phone)
                    except Exception as exc:
                        log.warning("Could not resolve LID %s: %s", lid, exc)

                # Reverse: phone -> LID  (covers the case where DB is already
                # fully migrated and has zero @lid rows — we still need the
                # alias so that incoming LID mentions resolve correctly).
                candidate_phones = [j for j in (all_assignment_jids + all_user_jids)
                                    if j.endswith("@s.whatsapp.net")]
                for phone in dict.fromkeys(candidate_phones):
                    if phone in lid_to_phone.values():
                        continue  # already covered by forward lookup
                    try:
                        u, s = phone.split("@", 1)
                        pn_jid = build_jid(u, s)
                        lid_jid = client.get_lid_from_pn(pn_jid)
                        if lid_jid:
                            lid = normalize_jid(Jid2String(lid_jid))
                            if lid and lid.endswith("@lid"):
                                lid_to_phone[lid] = phone
                                log.info("📋 Reverse LID resolved: %s -> %s", lid, phone)
                    except Exception as exc:
                        log.warning("Could not reverse-resolve phone %s: %s", phone, exc)

                if lid_to_phone:
                    migrated = store.reconcile_all_lids(lid_to_phone)
                    log.info("📋 LID reconciliation complete: %d row(s) migrated, "
                             "%d alias(es) registered.", migrated, len(lid_to_phone))

                else:
                    log.info("📋 No LID->phone mappings resolved; alias registry empty.")
            except Exception as exc:
                log.exception("Error during LID assignment reconciliation: %s", exc)

        thread = threading.Thread(target=_run, name="LIDReconciler", daemon=True)
        thread.start()


    dispatch = register_features(client, runtime_config)

    @client.event(MessageEv)
    def on_message(_client: NewClient, message: MessageEv):
        info = getattr(message, "Info", None)
        if not info:
            return
        
        # Dynamically register incoming LID senders to the alias registry
        # so overview() can find their phone-based assignments.
        sender_str = getattr(info.MessageSource, "Sender", None)
        if sender_str:
            from db.auth import normalize_jid as _nj, jid_user as _ju
            sender = _nj(sender_str)
            if sender.endswith("@lid"):
                from db.work_store import _JID_ALIASES
                if _ju(sender) not in _JID_ALIASES:
                    def resolve_in_bg():
                        try:
                            import time
                            time.sleep(1)
                            from neonize.utils import build_jid, Jid2String
                            user, server = sender.split("@")
                            lid_jid = build_jid(user, server)
                            pn_jid = _client.get_pn_from_lid(lid_jid)
                            if pn_jid:
                                phone = _nj(Jid2String(pn_jid))
                                if phone and phone.endswith("@s.whatsapp.net"):
                                    _JID_ALIASES[_ju(sender)] = _ju(phone)
                                    log.info("📋 Dynamically registered LID alias in bg: %s -> %s", sender, phone)
                                    from db.work_store import save_persistent_alias
                                    save_persistent_alias(runtime_config["db_session_factory"], _ju(sender), _ju(phone))
                        except Exception as e:
                            log.warning("Failed background LID resolve for %s: %s", sender, e)
                    
                    import threading
                    threading.Thread(target=resolve_in_bg, daemon=True).start()

        source = getattr(info, "MessageSource", None)
        chat = getattr(source, "Chat", None)
        chat_id = _jid_string(chat) if chat else ""
        is_from_me = getattr(info, "IsFromMe", False)

        if time.monotonic() < accept_messages_after:
            return

        message_timestamp = int(getattr(info, "Timestamp", 0) or 0)
        if message_timestamp <= startup_timestamp:
            return

        # TEST-ONLY GUARD: remove this exact-group check when production
        # routing is designed.
        if (
            not allowed_group
            or not chat_id.endswith("@g.us")
            or chat_id != allowed_group
        ):
            return
        dispatch(message)

    log.info("Starting WhatsApp bot...")
    log.info("Groups: %s", config["group_ids"] or "(none)")
    log.info("Media group: %s", config["media_group_id"] or "(not set)")
    log.info("Incident group: %s", config["incident_group_id"] or "(not set)")
    client.connect()


if __name__ == "__main__":
    main()
