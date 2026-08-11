"""Minimal tables used by the bot's persisted features."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, CheckConstraint, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# JSONB in PostgreSQL, portable JSON when the stores are exercised with SQLite.
JsonDocument = JSON().with_variant(JSONB, "postgresql")


class User(Base):
    __tablename__ = "users"
    jid: Mapped[str] = mapped_column(String(128), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="member")
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deactivated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    __table_args__ = (CheckConstraint("role IN ('admin', 'member')", name="ck_users_role"),)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor_jid: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    actor_role: Mapped[str] = mapped_column(String(16), nullable=False)
    operation: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JsonDocument, nullable=False)
    result: Mapped[str] = mapped_column(String(32), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class JidAlias(Base):
    """Maps linked device LIDs to primary phone JIDs."""
    __tablename__ = "jid_aliases"
    lid: Mapped[str] = mapped_column(String(128), primary_key=True)
    phone: Mapped[str] = mapped_column(String(128), nullable=False)


class Event(Base):
    """Future PRD event record; included now so assignments have a stable target."""
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(String(16), nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False, default="other")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft", index=True)
    start_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    end_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EventLabel(Base):
    __tablename__ = "event_labels"
    __table_args__ = (UniqueConstraint("event_id", "label", name="uq_event_label"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(64), nullable=False, index=True)


class EventFieldSchema(Base):
    """PRS 7.4: per-event dynamic form definition for participation events."""
    __tablename__ = "event_field_schemas"
    __table_args__ = (UniqueConstraint("event_id", "name", name="uq_event_field_name"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    field_type: Mapped[str] = mapped_column(String(16), nullable=False)
    options: Mapped[list | None] = mapped_column(JsonDocument, nullable=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class Assignment(Base):
    __tablename__ = "assignments"
    __table_args__ = (
        UniqueConstraint("event_id", "user_jid", name="uq_event_user"),
        UniqueConstraint("task_id", "user_jid", name="uq_task_user"),
        CheckConstraint(
            "(event_id IS NOT NULL AND task_id IS NULL) OR "
            "(event_id IS NULL AND task_id IS NOT NULL)",
            name="ck_assignment_one_target",
        ),
        CheckConstraint("target_type IN ('event', 'task')", name="ck_assignment_target_type"),
        CheckConstraint("status IN ('pending', 'in_progress', 'completed', 'cancelled')", name="ck_assignment_status"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    target_type: Mapped[str] = mapped_column(String(8), nullable=False, default="event", index=True)
    event_id: Mapped[int | None] = mapped_column(ForeignKey("events.id"), nullable=True, index=True)
    task_id: Mapped[int | None] = mapped_column(ForeignKey("tasks.id"), nullable=True, index=True)
    user_jid: Mapped[str] = mapped_column(ForeignKey("users.jid"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    reminder_state: Mapped[str | None] = mapped_column(String(16), nullable=True)
    missed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_update_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class MediaPost(Base):
    __tablename__ = "media_posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    posted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    platform_status: Mapped[dict] = mapped_column(JsonDocument, nullable=False)
    # The field is present in the existing JSON entries and is retained for
    # compatibility, although command output does not currently display it.
    created_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    # JSON arrays have ordering semantics in the compatibility shape.
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class MediaPostCounter(Base):
    """High-water mark for the JSON ``nextId`` value."""

    __tablename__ = "media_post_counters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    next_id: Mapped[int] = mapped_column(Integer, nullable=False)


class Subgroup(Base):
    __tablename__ = "subgroups"

    name: Mapped[str] = mapped_column(String(32), primary_key=True)
    member_jids: Mapped[list] = mapped_column(JsonDocument, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class IncidentState(Base):
    __tablename__ = "incident_states"

    incident_url: Mapped[str] = mapped_column(Text, primary_key=True)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    last_updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class Task(Base):
    """PRD FR-5: trackable work item, optionally linked to an organization event."""

    __tablename__ = "tasks"
    __table_args__ = (
        CheckConstraint(
            "status IN ('todo', 'in_progress', 'done', 'cancelled')",
            name="ck_tasks_status",
        ),
        CheckConstraint(
            "priority IN ('low', 'medium', 'high')",
            name="ck_tasks_priority",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    event_id: Mapped[int | None] = mapped_column(
        ForeignKey("events.id"), nullable=True, index=True
    )
    # Deprecated compatibility column. New code uses assignments.task_id.
    assignee_jid: Mapped[str | None] = mapped_column(ForeignKey("users.jid"), nullable=True, index=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="todo", index=True
    )
    priority: Mapped[str] = mapped_column(
        String(8), nullable=False, default="medium"
    )
    due_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_by_jid: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class ProgressRevision(Base):
    """Append-only history for every assignment progress field."""
    __tablename__ = "progress_revisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    assignment_id: Mapped[int] = mapped_column(ForeignKey("assignments.id"), nullable=False, index=True)
    field: Mapped[str] = mapped_column(String(64), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    author_jid: Mapped[str] = mapped_column(String(128), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    superseded_revision_id: Mapped[int | None] = mapped_column(
        ForeignKey("progress_revisions.id"), nullable=True, index=True
    )


class ReminderConfig(Base):
    """PRD FR-7: System-wide configuration for scheduled reminders."""

    __tablename__ = "reminder_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    frequency_hours: Mapped[int] = mapped_column(Integer, nullable=False, default=24)
    active_window_start: Mapped[str] = mapped_column(String(5), nullable=False, default="09:00")
    active_window_end: Mapped[str] = mapped_column(String(5), nullable=False, default="18:00")
    escalation_threshold: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    escalation_channel: Mapped[str | None] = mapped_column(String(128), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ReminderLog(Base):
    """PRD FR-7: Immutable log of reminder attempts, outcomes, and escalations."""

    __tablename__ = "reminder_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    assignment_id: Mapped[int] = mapped_column(
        ForeignKey("assignments.id"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    channel: Mapped[str] = mapped_column(String(32), nullable=False, default="whatsapp")
    result: Mapped[str] = mapped_column(String(32), nullable=False)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)
