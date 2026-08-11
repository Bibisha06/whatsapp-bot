"""Read-only reporting over assignments, progress revisions, and the audit log (PRS 7.8, 7.9)."""

from __future__ import annotations

from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Assignment, AuditLog, Event, ProgressRevision, Task, User
from .schema_store import SchemaStore


class ReportStore:
    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    @staticmethod
    def _display(session: Session, jid: str) -> str:
        user = session.get(User, jid)
        return (user.display_name if user and user.display_name else jid.split("@", 1)[0])

    @staticmethod
    def _latest_values(session: Session, assignment_id: int) -> dict[str, str]:
        """Collapse the append-only revision log to the current value per field."""
        rows = session.scalars(
            select(ProgressRevision)
            .where(ProgressRevision.assignment_id == assignment_id)
            .order_by(ProgressRevision.id)
        ).all()
        return {row.field: row.value for row in rows}

    def cohort(self, event_id: int) -> dict:
        """Latest value per schema field for every member assigned to an event.

        Includes both direct event assignments AND assignments on any task
        that belongs to this event (task.event_id == event_id).
        """
        with self.session_factory() as session:
            event = session.get(Event, event_id)
            if event is None or event.deleted_at is not None:
                raise ValueError(f"event #{event_id} not found")

            # Direct event-level assignments
            direct = session.scalars(
                select(Assignment)
                .where(Assignment.event_id == event_id)
                .order_by(Assignment.id)
            ).all()

            # Task-level assignments for tasks belonging to this event
            tasks_under = session.scalars(
                select(Task).where(
                    Task.event_id == event_id,
                    Task.deleted_at.is_(None),
                )
            ).all()
            task_ids = {t.id: t.title for t in tasks_under}
            task_assignments = []
            if task_ids:
                task_assignments = session.scalars(
                    select(Assignment)
                    .where(Assignment.task_id.in_(task_ids.keys()))
                    .order_by(Assignment.id)
                ).all()

            all_assignments = list(direct) + list(task_assignments)
            rows = []
            seen_fields = []
            for assignment in all_assignments:
                # set_status records a "status" revision; the assignment status
                # column already reports it, so keep it out of the field pivot.
                values = {field: value for field, value
                          in self._latest_values(session, assignment.id).items()
                          if field != "status"}
                # For task assignments, annotate which task this is
                if assignment.task_id and assignment.task_id in task_ids:
                    values["task"] = task_ids[assignment.task_id]
                for field in values:
                    if field not in seen_fields:
                        seen_fields.append(field)
                rows.append({
                    "name": self._display(session, assignment.user_jid),
                    "user_jid": assignment.user_jid,
                    "status": assignment.status,
                    "values": values,
                    "scope": (
                        f"task {assignment.task_id}" if assignment.task_id
                        else f"event {event_id}"
                    ),
                })
        declared = [item["name"] for item in SchemaStore(self.session_factory).list_fields(event_id)]
        fields = declared + [field for field in seen_fields if field not in declared]
        return {"event_name": event.name, "fields": fields, "rows": rows}

    def summary(self) -> dict:
        with self.session_factory() as session:
            assignments = session.scalars(select(Assignment)).all()
            counts: dict[str, int] = {}
            for assignment in assignments:
                counts[assignment.status] = counts.get(assignment.status, 0) + 1
            events = session.scalar(select(Event).where(Event.deleted_at.is_(None)).order_by(Event.id))
            return {
                "assignments": len(assignments),
                "counts": counts,
                "events": len(session.scalars(select(Event).where(Event.deleted_at.is_(None))).all()),
                "tasks": len(session.scalars(select(Task).where(Task.deleted_at.is_(None))).all()),
                "unassigned_events": len([
                    event for event in session.scalars(select(Event).where(Event.deleted_at.is_(None))).all()
                    if not session.scalar(select(Assignment).where(Assignment.event_id == event.id))
                ]),
                "first_event": events.name if events else None,
            }

    def by_status(self, status: str) -> list[dict]:
        with self.session_factory() as session:
            assignments = session.scalars(
                select(Assignment).where(Assignment.status == status).order_by(Assignment.id)
            ).all()
            result = []
            for assignment in assignments:
                target = session.get(Event if assignment.target_type == "event" else Task,
                                     assignment.event_id or assignment.task_id)
                if target is None or target.deleted_at is not None:
                    continue
                result.append({
                    "target_type": assignment.target_type,
                    "target_id": assignment.event_id or assignment.task_id,
                    "title": getattr(target, "name", None) or getattr(target, "title", ""),
                    "name": self._display(session, assignment.user_jid),
                    "user_jid": assignment.user_jid,
                    "missed_count": assignment.missed_count,
                    "last_update_at": assignment.last_update_at,
                })
            return result

    def audit_entries(self, *, actor_jid: str | None = None, operation: str | None = None,
                      limit: int = 20) -> list[dict]:
        with self.session_factory() as session:
            query = select(AuditLog).order_by(AuditLog.id.desc())
            if actor_jid:
                query = query.where(AuditLog.actor_jid == actor_jid)
            if operation:
                query = query.where(AuditLog.operation.startswith(operation))
            return [{
                "id": row.id, "actor": self._display(session, row.actor_jid),
                "actor_jid": row.actor_jid,
                "actor_role": row.actor_role, "operation": row.operation,
                "source": row.source, "result": row.result, "timestamp": row.timestamp,
            } for row in session.scalars(query.limit(limit)).all()]
