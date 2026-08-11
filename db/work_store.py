"""Shared assignment, progress, and workload service for events and tasks."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from .auth import jid_user, normalize_jid
from .models import Assignment, Event, ProgressRevision, Task, User
from .schema_store import validate_submission

PROGRESS_STATUSES = frozenset({"pending", "in_progress", "completed", "cancelled"})


def normalize_reference(reference: str) -> str:
    """Normalize canonical work references without colon syntax.

    Canonical forms are ``event <id>`` / ``task <id>`` with an optional
    assignee suffix (``event 4@user@s.whatsapp.net``).  The former
    ``event:4`` and ``task:4`` forms remain accepted for compatibility.
    """
    value = " ".join(reference.strip().split())
    match = re.fullmatch(r"(event|task)(?::\s*|\s+)(\d+)(?:\s*@(.+))?", value, re.I)
    if match:
        target_type, target_id, jid = match.groups()
        return f"{target_type.lower()}:{target_id}" + (f"@{jid}" if jid else "")

    return value


class WorkStore:
    """The single source of truth for assignment and progress operations."""

    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _ensure_user(session: Session, jid: str) -> str:
        wanted = normalize_jid(jid)
        if not wanted:
            raise ValueError("assignee must be a valid user")
        user = session.get(User, wanted)
        if user:
            return user.jid
        equivalent = next((u for u in session.scalars(select(User)).all()
                           if jid_user(u.jid) == jid_user(wanted)), None)
        if equivalent:
            return equivalent.jid
        now = WorkStore._now()
        session.add(User(jid=wanted, display_name=wanted.split("@", 1)[0],
                         role="member", active=True, created_at=now, updated_at=now))
        session.flush()
        return wanted

    @staticmethod
    def _target(session: Session, target_type: str, target_id: int):
        target_type = target_type.lower()
        model = Event if target_type == "event" else Task if target_type == "task" else None
        if model is None:
            raise ValueError("target type must be event or task")
        target = session.get(model, target_id)
        if target is None or getattr(target, "deleted_at", None) is not None:
            raise ValueError(f"{target_type} #{target_id} not found")
        return target

    @staticmethod
    def _assignment(session: Session, target_type: str, target_id: int, jid: str | None = None):
        column = Assignment.event_id if target_type == "event" else Assignment.task_id
        rows = session.scalars(select(Assignment).where(column == target_id)).all()
        if jid is None:
            return rows
        wanted = jid_user(jid)
        return next((row for row in rows if jid_user(row.user_jid) == wanted), None)

    @staticmethod
    def _row(row: Assignment, target=None, parent=None, session: Session | None = None) -> dict:
        result = {
            "id": row.id, "assignment_id": row.id, "target_type": row.target_type,
            "event_id": row.event_id, "task_id": row.task_id, "user_jid": row.user_jid,
            "user_id": row.user_jid, "status": row.status,
            "reminder_state": row.reminder_state, "missed_count": row.missed_count,
            "last_update_at": row.last_update_at, "created_at": row.created_at,
        }
        result["target_id"] = row.event_id if row.target_type == "event" else row.task_id
        if session is not None:
            user = session.get(User, row.user_jid)
            result["display_name"] = (user.display_name if user and user.display_name else row.user_jid.split("@", 1)[0])
        if isinstance(target, Event):
            result.update(name=target.name, title=target.name, lifecycle_status=target.status,
                          event_type=target.type, event_category=target.category, due_date=target.end_date)
        elif isinstance(target, Task):
            result.update(name=target.title, title=target.title, lifecycle_status=target.status,
                          priority=target.priority, due_date=target.due_date,
                          description=target.description)
            if parent is not None and parent.deleted_at is None:
                result.update(
                    parent_event_id=parent.id,
                    parent_event_name=parent.name,
                    parent_event_status=parent.status,
                )
        return result

    def assign(self, target_type: str, target_id: int, user_jid: str) -> dict:
        with self.session_factory.begin() as session:
            return self._assign_in(session, target_type, target_id, user_jid)

    def _assign_in(self, session: Session, target_type: str, target_id: int, user_jid: str) -> dict:
        target_type = target_type.lower()
        target = self._target(session, target_type, target_id)
        jid = self._ensure_user(session, user_jid)
        existing = self._assignment(session, target_type, target_id, jid)
        if existing:
            parent = session.get(Event, target.event_id) if isinstance(target, Task) and target.event_id else None
            return self._row(existing, target, parent)
        row = Assignment(
            target_type=target_type,
            event_id=target_id if target_type == "event" else None,
            task_id=target_id if target_type == "task" else None,
            user_jid=jid,
            status="pending",
            created_at=self._now(),
        )
        session.add(row)
        session.flush()
        parent = session.get(Event, target.event_id) if isinstance(target, Task) and target.event_id else None
        return self._row(row, target, parent)

    def assign_many(self, target_type: str, target_id: int, user_jids: list[str]) -> list[dict]:
        """Assign all users in one transaction or persist none of them."""
        with self.session_factory.begin() as session:
            return [
                self._assign_in(session, target_type, target_id, user_jid)
                for user_jid in dict.fromkeys(user_jids)
            ]

    def unassign(self, target_type: str, target_id: int, user_jid: str) -> bool:
        with self.session_factory.begin() as session:
            return self._unassign_in(session, target_type, target_id, user_jid)

    def _unassign_in(self, session: Session, target_type: str, target_id: int, user_jid: str) -> bool:
        row = self._assignment(session, target_type, target_id, user_jid)
        if not row:
            return False
        session.delete(row)
        return True

    def unassign_many(self, target_type: str, target_id: int, user_jids: list[str]) -> list[str]:
        """Remove all requested assignments in one transaction."""
        with self.session_factory.begin() as session:
            removed: list[str] = []
            for user_jid in dict.fromkeys(user_jids):
                if self._unassign_in(session, target_type, target_id, user_jid):
                    removed.append(normalize_jid(user_jid))
            return removed

    def resolve(self, reference: str) -> Assignment:
        with self.session_factory() as session:
            return self._resolve_in(session, reference)

    def set_status(self, reference: str, status: str, author_jid: str = "system@s.whatsapp.net") -> dict:
        status = status.lower().strip()
        if status not in PROGRESS_STATUSES:
            raise ValueError("status must be one of: " + ", ".join(sorted(PROGRESS_STATUSES)))
        with self.session_factory.begin() as session:
            row = self._resolve_in(session, reference)
            now = self._now()
            old = session.scalar(select(ProgressRevision).where(ProgressRevision.assignment_id == row.id,
                                                                  ProgressRevision.field == "status").order_by(ProgressRevision.id.desc()))
            row.status, row.last_update_at = status, now
            row.missed_count = 0
            row.reminder_state = None
            session.add(ProgressRevision(assignment_id=row.id, field="status", value=status,
                                         author_jid=normalize_jid(author_jid), timestamp=now,
                                         superseded_revision_id=old.id if old else None))
            return self._row(row)

    @staticmethod
    def _resolve_in(session: Session, reference: str) -> Assignment:
        reference = normalize_reference(reference)
        if reference.isdigit():
            row = session.get(Assignment, int(reference))
        else:
            import re
            match = re.fullmatch(r"(event|task):(\d+)(?:@(.+))?", reference, re.I)
            if not match:
                raise ValueError(f"Assignment '{reference}' not found.")
            typ, ident, jid = match.groups()
            rows = WorkStore._assignment(session, typ.lower(), int(ident), jid)
            row = rows if jid else (rows or [None])[0]
        if row is None:
            raise ValueError(f"Assignment '{reference}' not found.")
        return row

    def submit_update(self, reference: str, field: str, value: str, author_jid: str) -> dict:
        # Field names are case-insensitive so a cohort typing "prs" and "PRs"
        # does not split one metric across two revision chains.
        field, value = field.strip().lower(), value.strip()
        if not field or not value:
            raise ValueError("Both update field and value are required.")
        with self.session_factory.begin() as session:
            row = self._resolve_in(session, reference)
            if row.target_type == "event" and row.event_id:
                field, value = validate_submission(session, row.event_id, field, value)
            previous = session.scalar(select(ProgressRevision).where(ProgressRevision.assignment_id == row.id,
                                                                      ProgressRevision.field == field).order_by(ProgressRevision.id.desc()))
            now = self._now(); row.last_update_at = now
            row.missed_count = 0
            row.reminder_state = None
            revision = ProgressRevision(assignment_id=row.id, field=field, value=value,
                                        author_jid=normalize_jid(author_jid), timestamp=now,
                                        superseded_revision_id=previous.id if previous else None)
            session.add(revision); session.flush()
            return {"id": revision.id, "assignment_id": row.id, "field": field, "value": value,
                    "author_jid": revision.author_jid, "timestamp": now,
                    "superseded_revision_id": revision.superseded_revision_id}

    def edit_update(self, revision_id: int, value: str, author_jid: str) -> dict:
        with self.session_factory.begin() as session:
            old = session.get(ProgressRevision, revision_id)
            if old is None: raise ValueError(f"Revision '{revision_id}' not found.")
            value = value.strip()
            if not value: raise ValueError("The new update value is required.")
            now = self._now()
            assignment = session.get(Assignment, old.assignment_id)
            if assignment is not None and assignment.target_type == "event" and assignment.event_id:
                _, value = validate_submission(session, assignment.event_id, old.field, value)
            revision = ProgressRevision(assignment_id=old.assignment_id, field=old.field, value=value,
                                        author_jid=normalize_jid(author_jid), timestamp=now,
                                        superseded_revision_id=old.id)
            session.add(revision); session.flush()
            if assignment is not None:
                assignment.last_update_at = now
                assignment.missed_count = 0
                assignment.reminder_state = None
            return {"id": revision.id, "assignment_id": revision.assignment_id, "field": revision.field,
                    "value": revision.value, "timestamp": now, "superseded_revision_id": old.id}

    def history(self, reference: str) -> list[dict]:
        with self.session_factory() as session:
            row = self._resolve_in(session, reference)
            revisions = session.scalars(select(ProgressRevision).where(ProgressRevision.assignment_id == row.id)
                                        .order_by(ProgressRevision.timestamp, ProgressRevision.id)).all()
            return [{"id": r.id, "assignment_id": r.assignment_id, "field": r.field, "value": r.value,
                     "author_jid": r.author_jid, "timestamp": r.timestamp,
                     "superseded_revision_id": r.superseded_revision_id} for r in revisions]

    def overview(self, *, user_jid: str | None = None, admin: bool = False,
                 status: str | None = None, target_type: str | None = None,
                 target_id: int | None = None, assignee_jid: str | None = None) -> list[dict]:
        with self.session_factory() as session:
            rows = session.scalars(select(Assignment).order_by(Assignment.id)).all()
            result = []
            for row in rows:
                if user_jid and jid_user(row.user_jid) != jid_user(user_jid): continue
                if not admin and user_jid is None: continue
                if status and row.status != status: continue
                if target_type and row.target_type != target_type: continue
                if target_id is not None and (row.event_id or row.task_id) != target_id: continue
                if assignee_jid and jid_user(row.user_jid) != jid_user(assignee_jid): continue
                target = session.get(Event if row.target_type == "event" else Task,
                                     row.event_id if row.target_type == "event" else row.task_id)
                if target is None or getattr(target, "deleted_at", None) is not None: continue
                parent = session.get(Event, target.event_id) if isinstance(target, Task) and target.event_id else None
                result.append(self._row(row, target, parent, session=session))
            return result

    def unassigned(self, *, target_type: str | None = None) -> list[dict]:
        with self.session_factory() as session:
            output=[]
            if target_type in (None, "event"):
                for t in session.scalars(select(Event).where(Event.deleted_at.is_(None))).all():
                    if not self._assignment(session, "event", t.id): output.append({"target_type":"event","event_id":t.id,"task_id":None,"title":t.name,"status":None,"user_jid":None,"lifecycle_status":t.status})
            if target_type in (None, "task"):
                for t in session.scalars(select(Task).where(Task.deleted_at.is_(None))).all():
                    if not self._assignment(session, "task", t.id):
                        row = {
                            "target_type": "task", "task_id": t.id, "event_id": None,
                            "title": t.title, "status": None, "user_jid": None,
                            "due_date": t.due_date, "lifecycle_status": t.status,
                        }
                        if t.event_id:
                            parent = session.get(Event, t.event_id)
                            if parent is not None and parent.deleted_at is None:
                                row.update(
                                    parent_event_id=parent.id,
                                    parent_event_name=parent.name,
                                    parent_event_status=parent.status,
                                )
                        output.append(row)
            return output


# Public names used by integrations that call this domain service directly.
AssignmentService = WorkStore
WorkService = WorkStore
