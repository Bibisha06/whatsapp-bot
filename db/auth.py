"""JID normalization, authorization, user management, and audit logging."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import func, select

from .models import AuditLog, User

log = logging.getLogger(__name__)

ROLES = {"admin", "member"}


def normalize_jid(value) -> str:
    """Normalize a WhatsApp JID into a stable string representation."""
    if value is None:
        return ""

    if getattr(value, "User", "") and getattr(value, "Server", ""):
        user_str = str(value.User).split(":")[0].split(".")[0]
        value = f"{user_str}@{value.Server}"

    value = str(value).strip().lower().replace("whatsapp:", "")

    value = re.sub(r"[:.][0-9]+@", "@", value)
    value = re.sub(r"[^0-9a-z._@-]", "", value)

    if "@" not in value and value:
        value += "@s.whatsapp.net"

    return value


def jid_user(value) -> str:
    """Return the user portion of a JID."""
    return normalize_jid(value).split("@", 1)[0]


def current_user(session_factory: Callable, jid) -> User | None:
    """
    Find a user using either the exact JID or the stable user portion.

    WhatsApp may represent the same person as:
        123456789@s.whatsapp.net
    or:
        123456789@lid
    """
    with session_factory() as session:
        normalized = normalize_jid(jid)

        if not normalized:
            return None

        user = session.get(User, normalized)

        if user is not None:
            return user

        wanted = jid_user(normalized)

        return next(
            (
                candidate
                for candidate in session.scalars(select(User)).all()
                if jid_user(candidate.jid) == wanted
            ),
            None,
        )


def authorize(
    session_factory,
    actor_jid,
    operation: str,
    required_role: str = "member",
    push_name: str = "",
) -> User | None:
    """
    Authorize a WhatsApp user.

    IMPORTANT:
    - push_name is deliberately NOT stored in the database.
    - Contact/profile names are handled elsewhere at display time.
    - Existing users are never renamed from WhatsApp PushName.
    """
    actor = normalize_jid(actor_jid)

    if not actor:
        return None

    user = current_user(session_factory, actor)

    # Automatically create normal members on first interaction.
    # Do not store WhatsApp PushName/contact names.
    if user is None and required_role == "member":
        user = upsert_user(
            session_factory,
            actor,
            role="member",
            display_name="",
            operation="user.auto_create",
        )

    # IMPORTANT:
    # This must be calculated regardless of whether the user
    # already existed or was newly created.
    allowed = bool(
        user
        and user.active
        and (
            required_role == "member"
            or user.role == "admin"
        )
    )

    log.info(
        "authorization actor=%s operation=%s required_role=%s allowed=%s",
        actor,
        operation,
        required_role,
        allowed,
    )

    return user if allowed else None


def require_admin(
    session_factory,
    actor_jid,
    operation: str = "unknown",
) -> User | None:
    """Require an active administrator."""
    return authorize(
        session_factory,
        actor_jid,
        operation,
        "admin",
    )


def require_member(
    session_factory,
    actor_jid,
    operation: str = "unknown",
) -> User | None:
    """Require an active member."""
    return authorize(
        session_factory,
        actor_jid,
        operation,
        "member",
    )


# Default denial messages.

_DENY_MSG = {
    "admin": "⛔ You need to be an active administrator to use this command.",
    "member": "⛔ An active user account is required.",
}


def gate(
    session_factory,
    sender,
    client,
    chat,
    role: str = "member",
    operation: str = "unknown",
    push_name: str = "",
):
    """Single-call authorization gate for feature handlers."""
    jid = normalize_jid(sender)

    actor = authorize(
        session_factory,
        jid,
        operation,
        role,
        push_name=push_name,
    )

    if not actor:
        try:
            client.send_message(
                chat,
                _DENY_MSG.get(role, "⛔ Access denied."),
            )
        except Exception as exc:
            log.warning(
                "gate: failed to send denial: %s",
                exc,
            )

    return actor


def audit(
    session_factory,
    actor: User,
    operation: str,
    source: str,
    payload: dict,
    result: str = "success",
) -> None:
    """Write an audit record."""
    with session_factory() as session:
        session.add(
            AuditLog(
                actor_jid=actor.jid,
                actor_role=actor.role,
                operation=operation,
                source=source,
                payload=payload,
                result=result,
                timestamp=datetime.now(timezone.utc),
            )
        )

        session.commit()


def upsert_user(
    session_factory,
    jid,
    role="member",
    display_name="",
    *,
    deactivate=False,
    actor=None,
    operation="user.create",
) -> User:
    """
    Create or update a user.

    display_name is intentionally NOT populated from WhatsApp PushName.
    The database stores the stable JID and account information only.
    """
    jid = normalize_jid(jid)

    if not jid:
        raise ValueError("jid is required")

    if role not in ROLES:
        raise ValueError("role must be admin or member")

    now = datetime.now(timezone.utc)

    with session_factory() as session:
        user = session.get(User, jid)

        if user is None:
            # Do not store a WhatsApp contact/profile name.
            # If the model requires a non-empty value, use the JID user
            # portion only as a neutral identifier.
            user = User(
                jid=jid,
                display_name=jid.split("@")[0],
                role=role,
                active=not deactivate,
                created_at=now,
                updated_at=now,
            )

            user.deactivated_at = now if deactivate else None

            session.add(user)

        else:
            # Never overwrite display_name with PushName/contact name.
            if (
                user.active
                and user.role == "admin"
                and role != "admin"
                and active_admin_count(session_factory) <= 1
            ):
                raise ValueError(
                    "cannot demote the last active admin"
                )

            user.role = role
            user.active = not deactivate

            if not deactivate:
                user.deactivated_at = None

            user.updated_at = now

        session.commit()
        session.refresh(user)

    if actor:
        audit(
            session_factory,
            actor,
            operation,
            "whatsapp",
            {
                "jid": jid,
                "role": role,
            },
        )

    return user


def active_admin_count(session_factory) -> int:
    """Return the number of active administrators."""
    with session_factory() as session:
        return (
            session.scalar(
                select(func.count())
                .select_from(User)
                .where(
                    User.active.is_(True),
                    User.role == "admin",
                )
            )
            or 0
        )


def get_active_admin_jids(session_factory) -> list[str]:
    """Return JIDs of all active administrators."""
    with session_factory() as session:
        users = (
            session.query(User)
            .filter(
                User.role == "admin",
                User.active.is_(True),
            )
            .all()
        )

        return [user.jid for user in users]


def set_role(
    session_factory,
    jid,
    role,
    actor=None,
) -> User:
    """Change a user's role."""
    if role not in ROLES:
        raise ValueError("role must be admin or member")

    jid = normalize_jid(jid)

    with session_factory() as session:
        user = session.get(User, jid)

        if not user:
            raise ValueError("user not found")

        if (
            user.active
            and user.role == "admin"
            and role != "admin"
            and active_admin_count(session_factory) <= 1
        ):
            raise ValueError(
                "cannot demote the last active admin"
            )

        user.role = role
        user.updated_at = datetime.now(timezone.utc)

        session.commit()
        session.refresh(user)

    if actor:
        audit(
            session_factory,
            actor,
            "user.set_role",
            "whatsapp",
            {
                "jid": jid,
                "role": role,
            },
        )

    return user


def deactivate_user(
    session_factory,
    jid,
    actor=None,
) -> User:
    """Deactivate a user."""
    jid = normalize_jid(jid)

    with session_factory() as session:
        user = session.get(User, jid)

        if not user:
            raise ValueError("user not found")

        if (
            user.active
            and user.role == "admin"
            and active_admin_count(session_factory) <= 1
        ):
            raise ValueError(
                "cannot deactivate the last active admin"
            )

        now = datetime.now(timezone.utc)

        user.active = False
        user.deactivated_at = now
        user.updated_at = now

        session.commit()
        session.refresh(user)

    if actor:
        audit(
            session_factory,
            actor,
            "user.remove",
            "whatsapp",
            {
                "jid": jid,
            },
        )

    return user


def remove_or_demote_user(
    session_factory,
    jid,
    actor=None,
) -> tuple[User, str]:
    """
    If user is an admin, demote to member.
    If user is a member, deactivate.
    """
    jid = normalize_jid(jid)

    with session_factory() as session:
        user = session.get(User, jid)

        if not user:
            raise ValueError(
                f"user {jid} not found"
            )

        if user.active and user.role == "admin":
            if active_admin_count(session_factory) <= 1:
                raise ValueError(
                    "cannot demote the last active admin"
                )

            user.role = "member"
            user.updated_at = datetime.now(timezone.utc)

            action = "demoted to member"
            op = "user.demote"

        else:
            now = datetime.now(timezone.utc)

            user.active = False
            user.deactivated_at = now
            user.updated_at = now

            action = "deactivated"
            op = "user.remove"

        session.commit()
        session.refresh(user)

    if actor:
        audit(
            session_factory,
            actor,
            op,
            "whatsapp",
            {
                "jid": jid,
                "action": action,
            },
        )

    return user, action