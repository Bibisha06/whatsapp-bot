"""Semantic intent contracts and deterministic runtime target resolution.

The language model may describe an operation and an audience expression, but
it never supplies executable JIDs. This module is the boundary between that
semantic plan and the legacy command handlers:

    intent -> contract validation -> target resolution -> ready-to-compile

Keeping the contract here prevents individual compiler branches from silently
accepting incomplete intents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from db.auth import get_active_admin_jids, normalize_jid
from db.subgroup_store import SubgroupStore


@dataclass(frozen=True)
class CapabilityContract:
    """Execution requirements for one semantic capability."""

    target: str = "none"  # none, optional, or required
    entity_scope: bool = False  # operation should honor a named entity scope
    repeatable: bool = True  # may appear multiple times in one plan
    produces: frozenset[str] = frozenset()  # durable fields exposed to later steps


CAPABILITY_CONTRACTS: dict[str, CapabilityContract] = {
    # These handlers cannot perform the operation without concrete members.
    "collections.add": CapabilityContract("required"),
    "collections.remove": CapabilityContract("required"),
    "collections.delete": CapabilityContract("optional"),
    "collections.list": CapabilityContract("optional"),
    "collections.info": CapabilityContract("optional"),
    "labels.remove": CapabilityContract("required"),
    "work.assign": CapabilityContract("required"),
    # For unassign, audience is optional: omitting it means "remove all current assignees".
    "work.unassign": CapabilityContract("optional"),
    "whatsapp.user_info": CapabilityContract("required"),
    "whatsapp.add_group_members": CapabilityContract("required"),
    "whatsapp.remove_group_members": CapabilityContract("required"),
    "whatsapp.profile_pictures": CapabilityContract("required"),
    "whatsapp.block_contacts": CapabilityContract("required"),
    "whatsapp.unblock_contacts": CapabilityContract("required"),
    "whatsapp.create_group": CapabilityContract("optional", produces=frozenset({"group_jid", "name", "member_jids"})),
    # A bare label add intentionally preserves legacy "add myself" semantics.
    "labels.add": CapabilityContract("optional"),
    # Scoped reads must not silently degrade to a global overview when the
    # user's wording names an event, task, label, or other stored entity.
    "work.overview": CapabilityContract(entity_scope=True),
    "work.history": CapabilityContract(entity_scope=True),
    "work.status": CapabilityContract(entity_scope=True),
    "work.start": CapabilityContract(entity_scope=True),
    "work.complete": CapabilityContract(entity_scope=True),
    "work.update": CapabilityContract(entity_scope=True),
    "work.list_event_tasks": CapabilityContract(entity_scope=True),
    "reports.progress": CapabilityContract(entity_scope=True),
    "schema.show": CapabilityContract(entity_scope=True),
    "work.create_event": CapabilityContract(produces=frozenset({"event_id"})),
    "work.create_task": CapabilityContract(produces=frozenset({"task_id"})),
}

TARGET_RESOLVERS = frozenset(
    {
        "current_chat_members",
        "collection_members",
        "active_admins",
        "sender",
        "explicit_mentions",
        "plan_output",
    }
)


@dataclass(frozen=True)
class TargetResolution:
    members: tuple[str, ...] = ()
    resolver: str = ""
    error: str = ""

    @property
    def ready(self) -> bool:
        return bool(self.members) and not self.error


def contract_for(capability: str) -> CapabilityContract:
    return CAPABILITY_CONTRACTS.get(capability, CapabilityContract())


def verify_operation_result(intent: dict, result: object) -> str | None:
    """Check deterministic postconditions before exposing a tool result."""
    capability = str(intent.get("capability") or "")
    if not isinstance(result, dict):
        from features.agent_runtime import tool_spec

        if tool_spec(capability).produces:
            return "the operation returned no structured result"
        return None
    from features.agent_runtime import tool_spec
    missing_outputs = [
        field for field in tool_spec(capability).produces
        if field not in result
    ]
    if missing_outputs:
        return "the operation omitted declared outputs: " + ", ".join(sorted(missing_outputs))
    if capability not in {"work.create_event", "work.create_task"}:
        return None
    arguments = intent.get("arguments", {})
    if capability == "work.create_event" and not result.get("event_id"):
        return "event creation returned no event ID"
    if capability == "work.create_task":
        if not result.get("task_id"):
            return "task creation returned no task ID"
        expected_event = arguments.get("event_id")
        if (
            expected_event is not None
            and str(result.get("event_id")) != str(expected_event)
        ):
            return "task was not linked to the requested event"
    return None


def _normalize_collection_value(value: object) -> object:
    """Flatten a single-element list to a plain string.

    The model occasionally wraps a collection name in a list (e.g.
    ``['abc']``).  Downstream resolvers require a plain string, so unwrap
    here when the list contains exactly one item.
    """
    if isinstance(value, list):
        # Strip any leading @ before comparing
        strings = [v.strip().lstrip("@") if isinstance(v, str) else v for v in value if v]
        if len(strings) == 1 and isinstance(strings[0], str):
            return strings[0]
    if isinstance(value, str):
        return value.strip().lstrip("@") if value.strip().startswith("@") else value
    return value


def target_expression(arguments: dict) -> tuple[str, object]:
    """Return the canonical (resolver, value) target expression."""
    audience = arguments.get("audience")
    if isinstance(audience, dict):
        resolver = audience.get("resolver") or audience.get("kind") or ""
        value = audience.get("value") or audience.get("name") or audience.get("collection") or ""
        return str(resolver), _normalize_collection_value(value)

    # Transitional compatibility for early structured responses. Do not
    # interpret a work item string in arguments["target"] as an audience.
    target = arguments.get("target")
    if isinstance(target, dict):
        resolver = target.get("resolver") or target.get("kind") or ""
        value = target.get("value") or target.get("name") or target.get("collection") or ""
        return str(resolver), _normalize_collection_value(value)

    resolver = arguments.get("target_scope") or ""
    value = arguments.get("target_collection") or ""
    if resolver:
        return str(resolver), _normalize_collection_value(value)

    for key in ("collections", "labels"):
        val = arguments.get(key)
        if val:
            return "collection_members", _normalize_collection_value(val)

    return "", ""


def target_is_declared(arguments: dict, visible_mentions: list[str]) -> bool:
    resolver, _ = target_expression(arguments)
    if resolver == "explicit_mentions":
        return bool(visible_mentions)
    if resolver:
        return True
    return bool(arguments.get("mention_indices") and visible_mentions)


def target_is_required_and_missing(
    intent: dict,
    visible_mentions: list[str],
) -> bool:
    capability = str(intent.get("capability") or "")
    return (
        contract_for(capability).target == "required"
        and not target_is_declared(intent.get("arguments", {}), visible_mentions)
    )


def entity_scope_is_missing(intent: dict, entity_candidates: list[dict]) -> bool:
    """Detect a scoped operation that discarded a named runtime entity."""
    if not entity_candidates:
        return False
    contract = contract_for(str(intent.get("capability") or ""))
    if not contract.entity_scope:
        return False
    arguments = intent.get("arguments", {})
    return not any(
        arguments.get(key)
        for key in (
            "target_id", "target_name", "target", "event_id", "task_id", "collection",
            # Named entity keys: model supplies these when user names an entity in text
            "event_name", "task_name", "name", "event",
        )
    )


def _dedupe_members(members, self_jids: set[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for member in members:
        jid = normalize_jid(member)
        if not jid or jid in self_jids or jid in seen:
            continue
        seen.add(jid)
        result.append(jid)
    return tuple(result)


def resolve_target(
    client,
    message,
    intent: dict,
    self_jids: set[str],
    factory,
    resolve_collection: Callable[[object], str | list[str] | None],
    visible_mentions: list[str] | None = None,
) -> TargetResolution:
    """Resolve a declared semantic audience to concrete JIDs."""
    arguments = intent.get("arguments", {})
    resolver, value = target_expression(arguments)

    if not resolver and arguments.get("mention_indices") and visible_mentions:
        resolver = "explicit_mentions"
    if not resolver:
        return TargetResolution()
    if resolver not in TARGET_RESOLVERS:
        return TargetResolution(error="The requested audience resolver is not available.")

    try:
        if resolver == "explicit_mentions":
            mention_indices = arguments.get("mention_indices")
            if mention_indices is None:
                members = visible_mentions or []
            else:
                try:
                    members = [visible_mentions[index] for index in mention_indices]
                except (IndexError, TypeError):
                    return TargetResolution(
                        error="One or more requested mentions are unavailable."
                    )
        elif resolver == "sender":
            members = [message.Info.MessageSource.Sender]
        elif resolver == "active_admins":
            if factory is None:
                return TargetResolution(error="Active administrators are unavailable.")
            members = get_active_admin_jids(factory)
        elif resolver == "collection_members":
            if factory is None or not value:
                return TargetResolution(error="The referenced member collection is missing.")
            # Guard: if value is still a list after target_expression normalization
            # (e.g. multi-element list from model), unwrap or join into a single string.
            lookup_value: object = value
            if isinstance(lookup_value, list):
                lookup_value = " ".join(
                    str(v).strip().lstrip("@") for v in lookup_value if v
                ) or value
            resolved_names = resolve_collection(lookup_value)
            if isinstance(resolved_names, str):
                resolved_names = [resolved_names]
            if not resolved_names:
                return TargetResolution(error=f"Subgroup '{lookup_value}' not found. Use `!list-subgroups` to see available subgroups.")
            collections = SubgroupStore(factory).read()
            members = [
                member
                for name in resolved_names
                for member in collections.get(name, [])
            ]
            # For collection_members, never filter out self_jids — a human admin
            # who runs the bot from their own number is a legitimate group member.
            deduped = _dedupe_members(members, set())
            if not deduped:
                return TargetResolution(
                    resolver=resolver,
                    error=f"Subgroup '{resolved_names[0]}' exists but has no members yet.",
                )
            return TargetResolution(members=deduped, resolver=resolver)
        elif resolver == "current_chat_members":
            chat = message.Info.MessageSource.Chat
            if getattr(chat, "Server", "") != "g.us":
                return TargetResolution(error="Current group members are unavailable here.")
            from features.community_tag import get_group_member_jids

            members = get_group_member_jids(client, chat)
        elif resolver == "plan_output":
            if not isinstance(value, list):
                return TargetResolution(error="The referenced plan output is not a member list.")
            members = value
        else:
            members = []
    except Exception:
        return TargetResolution(error="I couldn't resolve that audience from runtime data.")

    active_self_jids = self_jids
    deduped = _dedupe_members(members, active_self_jids)
    if not deduped:
        return TargetResolution(
            resolver=resolver,
            error="The audience resolved to no eligible members.",
        )
    return TargetResolution(members=deduped, resolver=resolver)


def validate_execution_ready(
    intent: dict,
    resolution: TargetResolution,
    visible_mentions: list[str],
) -> str | None:
    """Return a user-safe error when a semantic step cannot execute."""
    capability = str(intent.get("capability") or "")
    contract = contract_for(capability)
    if contract.target == "required":
        if not target_is_declared(intent.get("arguments", {}), visible_mentions):
            return "I couldn't identify who this operation should affect."
        if not resolution.ready:
            return resolution.error or "I couldn't resolve the requested audience."
    if resolution.error and target_is_declared(intent.get("arguments", {}), visible_mentions):
        return resolution.error
    return None
