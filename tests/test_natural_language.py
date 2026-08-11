from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from features.natural_language import (
    MISTRAL_CHAT_URL,
    MistralCardDesigner,
    MistralCommandTranslator,
    build_knowledge_context,
    compile_card_design,
    compile_intent,
    fallback_command,
    register,
    _resolve_runtime_target_scope,
    resolve_named_collection_command,
    resolve_named_entity_command,
    validate_command,
    validate_plan,
    _needs_target_repair,
    _inherit_plan_context,
    _target_arguments,
    _plan_completeness_issue,
)
from features.subgroups import _get_mentioned_jids, normalize_collection_name


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload

    def raise_for_status(self):
        return None


class FakeHttpClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


def make_message(text, sender="member@s.whatsapp.net"):
    message = MagicMock()
    message.Info.MessageSource.Chat.Server = "g.us"
    message.Info.MessageSource.Sender = sender
    message.Message.conversation = text
    message.Message.extendedTextMessage = None
    message.Message.imageMessage = None
    return message


def test_mistral_returns_a_single_existing_command_in_json_mode():
    http = FakeHttpClient(
        FakeResponse({"choices": [{"message": {"content": '{"command":"!my","clarification":""}'}}]})
    )
    translator = MistralCommandTranslator("secret", client=http)

    command, clarification = translator.translate(
        "can you show me what I need to do?", ["123@s.whatsapp.net"]
    )

    assert command == "!my"
    assert clarification == ""
    assert http.calls[0][0][0] == MISTRAL_CHAT_URL
    assert http.calls[0][1]["json"]["response_format"] == {"type": "json_object"}
    assert http.calls[0][1]["json"]["temperature"] == 0


def test_mistral_returns_a_structured_capability_envelope():
    http = FakeHttpClient(
        FakeResponse({
            "choices": [{"message": {"content":
                '{"intent":{"capability":"labels.add","arguments":{"collection":"team","mention_indices":[0]}},"clarification":""}'
            }}]
        })
    )
    translator = MistralCommandTranslator("secret", client=http)

    intent, clarification = translator.translate("put the person in team", ["person@lid"])

    assert intent == {
        "capability": "labels.add",
        "arguments": {"collection": "team", "mention_indices": [0]},
    }
    assert clarification == ""
    assert "capability" in http.calls[0][1]["json"]["messages"][0]["content"]


def test_structured_intent_compiles_through_existing_command_syntax():
    intent = {
        "capability": "labels.add",
        "arguments": {"collection": "team", "mention_indices": [0]},
    }
    with patch("features.natural_language._resolve_collection_name", return_value="team"):
        command = compile_intent(intent, "add @Person to team", object(), ["person@lid"])
    assert command == "!labels add team"


def test_work_lifecycle_and_crud_capabilities_compile_to_legacy_handlers():
    with patch("features.natural_language._resolve_target_reference", side_effect=["event 4", "event 4", "task 7"]):
        assert compile_intent(
            {"capability": "work.set_lifecycle", "arguments": {"target_type": "event", "target_id": 4, "status": "completed"}},
            "complete event Zenith", object(), [],
        ) == "!set-status 4 | completed"
        assert compile_intent(
            {"capability": "work.update_event", "arguments": {"target_type": "event", "target_id": 4, "fields": {"name": "Renamed", "category": "hackathon"}}},
            "rename event", object(), [],
        ) == "!update-event 4 | name Renamed | category hackathon"
        assert compile_intent(
            {"capability": "work.delete_task", "arguments": {"target_type": "task", "target_id": 7}},
            "delete task 7", object(), [],
        ) == "!delete-task 7"


def test_natural_collection_names_are_normalized_to_stored_names():
    assert normalize_collection_name("2nd year") == "2nd-year"
    assert normalize_collection_name("  Backend / Maintainers! ") == "backend-maintainers"


def test_new_subgroup_is_created_when_name_does_not_exist_and_me_is_preserved():
    intent = {
        "capability": "collections.add",
        "arguments": {"collection": "2nd year", "mention_indices": [0, 1, 2]},
    }

    command = compile_intent(
        intent,
        "@me create a subgroup called 2nd year and add me and @Deval @Bibisha to it",
        object(),
        ["deval@lid", "bibisha@lid", "@me"],
    )

    assert command == "!add-subgroup 2nd-year | @me"


def test_existing_fuzzy_subgroup_wins_without_explicit_new_keyword():
    intent = {
        "capability": "collections.add",
        "arguments": {"collection": "lfx aplicants"},
    }
    with patch(
        "features.natural_language._resolve_collection_name",
        return_value="lfx-applicants",
    ):
        command = compile_intent(intent, "@me add me to lfx aplicants", object(), ["@me"])
    assert command == "!add-subgroup lfx-applicants | @me"


def test_explicit_new_keyword_overrides_similar_existing_subgroup():
    intent = {
        "capability": "collections.add",
        "arguments": {"collection": "lfx applicants"},
    }
    with patch(
        "features.natural_language._resolve_collection_name",
        return_value="lfx-applicants",
    ):
        command = compile_intent(
            intent,
            "@me create a new subgroup called lfx applicants",
            object(),
            ["@me"],
        )
    assert command == "!add-subgroup lfx-applicants"


def test_missing_label_is_created_with_normalized_name():
    intent = {
        "capability": "labels.add",
        "arguments": {"collection": "2nd year"},
    }
    command = compile_intent(intent, "@me add me to the 2nd year label", object(), ["@me"])
    assert command == "!labels add 2nd-year | @me"


def test_work_assignment_preserves_label_and_explicit_task_type():
    intent = {
        "capability": "work.assign",
        "arguments": {
            "target_type": "event",
            "target_id": 7,
            "collections": ["media"],
        },
    }
    with patch(
        "features.natural_language._resolve_collection_name",
        return_value="media",
    ):
        command = compile_intent(
            intent,
            "@me assign task 7 to label media",
            object(),
            [],
        )
    assert command == "!work assign event 7 | @media"


def test_leading_me_trigger_is_not_forwarded_as_a_target():
    client = MagicMock()
    client.get_me.return_value = SimpleNamespace(
        JID="bot@s.whatsapp.net", LID="999999@lid"
    )
    message = make_message("@me create a label called media")
    dispatch = MagicMock()

    with patch("features.natural_language._get_mentioned_jids", return_value=[]), patch.object(
        MistralCommandTranslator,
        "translate",
        return_value=("!labels add media", ""),
    ) as translate:
        handler = register(client, {"mistral_api_key": "secret"})
        assert handler(client, message, dispatch) is True

    translate.assert_called_once()
    assert translate.call_args.args[0] == "create a label called media"
    translated = dispatch.call_args.args[0]
    assert translated._pbbot_nl_no_target_mentions is True


def test_current_chat_member_scope_is_resolved_by_runtime_not_the_model():
    client = MagicMock()
    message = make_message("@me create a subgroup called everyone")
    intent = {
        "capability": "collections.add",
        "arguments": {
            "collection": "everyone",
            "target_scope": "current_chat_members",
        },
    }
    with patch(
        "features.community_tag.get_group_member_jids",
        return_value=["111@s.whatsapp.net", "222@s.whatsapp.net", "bot@s.whatsapp.net"],
    ):
        members, error = _resolve_runtime_target_scope(
            client, message, intent, {"bot@s.whatsapp.net"}
        )

    assert error is None
    assert members == ["111@s.whatsapp.net", "222@s.whatsapp.net"]


def test_canonical_audience_object_resolves_without_legacy_target_scope():
    client = MagicMock()
    message = make_message("@me add everyone here to backend")
    intent = {
        "capability": "collections.add",
        "arguments": {
            "collection": "backend",
            "audience": {"resolver": "current_chat_members"},
        },
    }
    with patch(
        "features.community_tag.get_group_member_jids",
        return_value=["111@s.whatsapp.net", "bot@s.whatsapp.net"],
    ):
        members, error = _resolve_runtime_target_scope(
            client,
            message,
            intent,
            {"bot@s.whatsapp.net"},
            visible_mentions=[],
        )

    assert error is None
    assert members == ["111@s.whatsapp.net"]


def test_required_target_cannot_reach_compilation_unresolved():
    from features.nl_runtime import TargetResolution, validate_execution_ready

    intent = {
        "capability": "collections.add",
        "arguments": {"collection": "backend"},
    }
    error = validate_execution_ready(intent, TargetResolution(), [])

    assert error == "I couldn't identify who this operation should affect."


def test_unresolved_required_target_is_stopped_before_legacy_dispatch():
    client = MagicMock()
    client.get_me.return_value = SimpleNamespace(
        JID="bot@s.whatsapp.net", LID="999999@lid"
    )
    message = make_message("@me create a subgroup for the people")
    dispatch = MagicMock()
    intent = {
        "capability": "collections.add",
        "arguments": {"collection": "people"},
    }

    with patch("features.natural_language._get_mentioned_jids", return_value=[]), \
         patch.object(MistralCommandTranslator, "translate", return_value=(intent, "")), \
         patch.object(MistralCommandTranslator, "repair_missing_target", return_value=None):
        handler = register(client, {"mistral_api_key": "secret"})
        assert handler(client, message, dispatch) is True

    dispatch.assert_not_called()
    assert any(
        "identify who this operation should affect" in str(call)
        for call in client.send_message.call_args_list
    )


def test_group_audience_repair_reaches_dispatch_with_resolved_members():
    client = MagicMock()
    client.get_me.return_value = SimpleNamespace(
        JID="bot@s.whatsapp.net", LID="999999@lid"
    )
    message = make_message(
        '@me I want to create a subgroup to assign everyone in this group '
        'in it and call it "everyone"'
    )
    dispatch = MagicMock()
    incomplete = {
        "capability": "collections.add",
        "arguments": {"collection": "everyone"},
    }
    repaired = {
        "capability": "collections.add",
        "arguments": {
            "collection": "everyone",
            "audience": {"resolver": "current_chat_members"},
        },
    }

    with patch("features.natural_language._get_mentioned_jids", return_value=[]), \
         patch.object(MistralCommandTranslator, "translate", return_value=(incomplete, "")), \
         patch.object(MistralCommandTranslator, "repair_missing_target", return_value=repaired), \
         patch(
             "features.community_tag.get_group_member_jids",
             return_value=["111@s.whatsapp.net", "222@s.whatsapp.net"],
         ), patch(
             "features.natural_language._execute_direct_operation",
             return_value=True,
         ) as execute:
        handler = register(client, {"mistral_api_key": "secret"})
        assert handler(client, message, dispatch) is True

    dispatch.assert_not_called()
    execute.assert_called_once()
    assert execute.call_args.args[2] == repaired
    assert execute.call_args.args[3] == ["111@s.whatsapp.net", "222@s.whatsapp.net"]


def test_exact_group_request_executes_direct_subgroup_operation():
    client = MagicMock()
    client.get_me.return_value = SimpleNamespace(
        JID="bot@s.whatsapp.net", LID="999999@lid"
    )
    message = make_message(
        '@me I want to create a subgroup to assign everyone in this group '
        'in it and call it "everyone"'
    )
    dispatch = MagicMock()
    incomplete = {
        "capability": "collections.add",
        "arguments": {"collection": "everyone"},
    }
    repaired = {
        "capability": "collections.add",
        "arguments": {
            "collection": "everyone",
            "audience": {"resolver": "current_chat_members"},
        },
    }
    factory = MagicMock()
    actor = SimpleNamespace(role="admin")

    with patch("features.natural_language._get_mentioned_jids", return_value=[]), \
         patch.object(MistralCommandTranslator, "translate", return_value=(incomplete, "")), \
         patch.object(MistralCommandTranslator, "repair_missing_target", return_value=repaired), \
         patch(
             "features.community_tag.get_group_member_jids",
             return_value=["111@s.whatsapp.net", "222@s.whatsapp.net"],
         ), patch(
             "db.auth.gate",
             return_value=actor,
         ), patch(
             "features.natural_language._resolve_or_create_collection_name",
             return_value="everyone",
         ), patch(
             "features.subgroups.add_subgroup_members",
             return_value=(2, 2),
         ) as add_members:
        handler = register(
            client,
            {"mistral_api_key": "secret", "db_session_factory": factory},
        )
        assert handler(client, message, dispatch) is True

    dispatch.assert_not_called()
    add_members.assert_called_once()
    assert add_members.call_args.args[1:] == (
        "everyone",
        ["111@s.whatsapp.net", "222@s.whatsapp.net"],
    )
    assert "Added 2 member(s)" in str(client.send_message.call_args)


def test_invalid_runtime_target_scope_is_rejected():
    from features.natural_language import validate_intent

    assert validate_intent({
        "capability": "collections.add",
        "arguments": {"collection": "everyone", "target_scope": "all_users"},
    }) is None
    assert validate_intent({
        "capability": "work.assign",
        "arguments": {"target": "task 7", "target_type": "task", "target_id": 7},
    }) is not None


def test_translator_accepts_a_bounded_multi_step_semantic_plan():
    http = FakeHttpClient(FakeResponse({
        "choices": [{"message": {"content": (
            '{"plan":[{"capability":"collections.add",'
            '"arguments":{"collection":"everyone",'
            '"target_scope":"current_chat_members"}}],'
            '"clarification":""}'
        )}}]
    }))
    translator = MistralCommandTranslator("secret", client=http)

    plan, clarification = translator.translate("put everyone here in a subgroup", [])

    assert plan == {"plan": [{
        "capability": "collections.add",
        "arguments": {
            "collection": "everyone",
            "target_scope": "current_chat_members",
        },
    }]}
    assert clarification == ""
    assert validate_plan(plan["plan"]) is not None


def test_collection_member_scope_resolves_from_persisted_store():
    client = MagicMock()
    message = make_message("@me assign it to backend")
    intent = {
        "capability": "work.assign",
        "arguments": {
            "target_scope": "collection_members",
            "target_collection": "backend",
        },
    }
    factory = MagicMock()
    with patch(
        "features.natural_language._resolve_collection_name",
        return_value="backend",
    ), patch(
        "db.subgroup_store.SubgroupStore.read",
        return_value={"backend": ["111@s.whatsapp.net"]},
    ):
        members, error = _resolve_runtime_target_scope(
            client, message, intent, set(), factory
        )

    assert error is None
    assert members == ["111@s.whatsapp.net"]


def test_missing_audience_is_sent_through_generic_repair_pass():
    intent = {
        "capability": "collections.add",
        "arguments": {"collection": "everyone"},
    }
    assert _needs_target_repair(
        intent,
        "create a subgroup to assign everyone in this group in it",
        [],
    ) is True
    assert _needs_target_repair(
        {
            **intent,
            "arguments": {
                **intent["arguments"],
                "target_scope": "explicit_mentions",
            },
        },
        "create a subgroup to assign everyone in this group in it",
        [],
    ) is True

    http = FakeHttpClient(FakeResponse({
        "choices": [{"message": {"content": (
            '{"intent":{"capability":"collections.add","arguments":'
            '{"collection":"everyone","target_scope":"current_chat_members"}},'
            '"clarification":""}'
        )}}]
    }))
    translator = MistralCommandTranslator("secret", client=http)
    repaired = translator.repair_missing_target(
        "create a subgroup for everyone here",
        intent,
        [],
    )

    assert repaired["arguments"]["target_scope"] == "current_chat_members"
    assert len(http.calls) == 1


def test_open_ended_card_design_keeps_original_template_without_style_request():
    intent = {
        "capability": "card.design",
        "arguments": {
            "base_template": "hackathon",
            "name": "Zodiak",
            "text": "For successfully failing to save your asses at PBCTF 5.0",
            "title": "Congratulations, Zodiak!",
            "accent": "#a855f7",
            "logo_urls": ["https://example.com/logo.svg"],
            "highlight_terms": ["PBCTF 5.0"],
            "tone": "sarcastic",
        },
    }

    command, design = compile_card_design(
        intent,
        "create a card for Zodiak using https://example.com/logo.svg as the logo",
    )

    assert command == (
        "!card hackathon | Zodiak | For successfully failing to save your asses at PBCTF 5.0 "
        "| https://example.com/logo.svg"
    )
    assert design is None


def test_open_ended_card_design_applies_spec_when_style_is_explicitly_requested():
    intent = {
        "capability": "card.design",
        "arguments": {
            "base_template": "hackathon",
            "name": "Zodiak",
            "text": "For successfully failing to save your asses at PBCTF 5.0",
            "title": "Congratulations, Zodiak!",
            "accent": "#a855f7",
            "highlight_terms": ["PBCTF 5.0"],
            "tone": "sarcastic",
        },
    }

    command, design = compile_card_design(
        intent,
        "create a sarcastic purple style hackathon card for Zodiak",
    )

    assert command.startswith("!card hackathon | Zodiak | ")
    assert design["base_template"] == "hackathon"
    assert design["accent"] == "#A855F7"
    assert design["highlight_terms"] == ["PBCTF 5.0"]
    assert design["tone"] == "sarcastic"


def test_dedicated_card_designer_preserves_semantic_brief_fields():
    http = FakeHttpClient(FakeResponse({
        "choices": [{"message": {"content": (
            '{"base_template":"custom","name":"Zodiak",'
            '"occasion":"PBCTF 5.0 failure","tone":"sarcastic",'
            '"headline":"Congratulations on the Failure",'
            '"body":"For failing to save the team at PBCTF 5.0",'
            '"accent":"#FF5C8A","pill":"PBCTF 5.0",'
            '"logo_urls":[],"highlight_terms":["failure"]}'
        )}}]
    }))
    from features.natural_language import MistralCardDesigner

    designer = MistralCardDesigner("secret", client=http)
    brief = designer.design("make a sarcastic card for Zodiak")

    assert brief["capability"] == "card.design"
    assert brief["arguments"]["tone"] == "sarcastic"
    assert brief["arguments"]["headline"] == "Congratulations on the Failure"
    assert http.calls[0][1]["json"]["model"] == "mistral-medium-3-5"


def test_explicit_logo_url_survives_when_design_model_omits_logo_urls():
    intent = {
        "capability": "card.design",
        "arguments": {
            "name": "Zodiak",
            "body": "For failing to save the team",
            "tone": "sarcastic",
        },
    }

    command, design = compile_card_design(
        intent,
        "make a sarcastic card for Zodiak; use https://example.com/logo.svg as the logo",
    )

    assert command.endswith(" | https://example.com/logo.svg")
    assert design["logo_url"] == "https://example.com/logo.svg"


def test_unknown_design_family_falls_back_to_custom_instead_of_help():
    intent = {
        "capability": "card.design",
        "arguments": {
            "base_template": "congratulations",
            "name": "Zodiak",
            "text": "PBCTF 5.0 legend",
        },
    }

    command, design = compile_card_design(intent, "make a congratulations card")

    assert command.startswith("!card custom | Zodiak | PBCTF 5.0 legend")
    assert design is None


def test_misclassified_unknown_card_type_is_promoted_to_design_mode():
    intent = {
        "capability": "card.create",
        "arguments": {
            "type": "congratulations",
            "name": "Zodiak",
            "text": "PBCTF 5.0 legend",
        },
    }

    command, design = compile_card_design(intent, "make a congratulations card")

    assert command == "!card custom | Zodiak | PBCTF 5.0 legend"
    assert design is None


def test_invalid_or_missing_model_command_is_rejected_without_guessing():
    http = FakeHttpClient(
        FakeResponse({"choices": [{"message": {"content": '{"command":null,"clarification":"Please clarify"}'}}]})
    )
    translator = MistralCommandTranslator("secret", client=http)

    command, clarification = translator.translate("make the thing", [])

    assert command is None
    assert clarification == ""


def test_translation_failure_does_not_reenter_dispatch_with_a_guessed_command():
    client = MagicMock()
    client.get_me.return_value = SimpleNamespace(
        JID="bot@s.whatsapp.net", LID="999999@lid"
    )
    message = make_message("@me please do something", "member@s.whatsapp.net")
    dispatch = MagicMock()

    with patch.object(
        MistralCommandTranslator,
        "translate",
        side_effect=RuntimeError("Mistral unavailable"),
    ):
        handler = register(client, {"mistral_api_key": "secret"})
        assert handler(client, message, dispatch) is True

    dispatch.assert_not_called()
    assert "safely resolve" in str(client.send_message.call_args)


def test_scoped_capability_is_reviewed_when_model_drops_named_entity():
    client = MagicMock()
    client.get_me.return_value = SimpleNamespace(
        JID="bot@s.whatsapp.net", LID="999999@lid"
    )
    message = make_message("@me what are the tasks left under Zenith 27")
    dispatch = MagicMock()
    candidate = {
        "capability": "work.overview",
        "arguments": {"status": "pending"},
    }
    repaired = {
        "capability": "work.list_event_tasks",
        "arguments": {"target_type": "event", "target_id": 10},
    }
    with patch("features.natural_language._get_mentioned_jids", return_value=[]), \
         patch("features.natural_language._named_entity_candidates", return_value=[{
             "type": "event", "id": 10, "name": "Zenith 27", "category": "other",
         }]), \
         patch.object(MistralCommandTranslator, "translate", return_value=(candidate, "")), \
         patch.object(MistralCommandTranslator, "repair_intent", return_value=repaired), \
         patch("features.natural_language._execute_direct_operation", return_value={"tasks": [], "task_count": 0}) as execute:
        handler = register(
            client,
            {"mistral_api_key": "secret", "db_session_factory": MagicMock()},
        )
        assert handler(client, message, dispatch) is True

    dispatch.assert_not_called()
    execute.assert_called_once()
    assert execute.call_args.args[2] == repaired


def test_task_step_inherits_unique_event_created_by_same_plan():
    intent = {
        "capability": "work.create_task",
        "arguments": {"title": "Raise funds"},
    }
    linked = _inherit_plan_context(intent, {"event": {"event_id": 10}})
    assert linked["arguments"]["event_id"] == 10


def test_full_message_target_extraction_does_not_override_explicit_step_target():
    arguments = _target_arguments(
        {"target_type": "task", "target_id": 13},
        "assign task 12 and 13 to @Deval PB",
    )
    assert arguments["target_id"] == 13


def test_plan_completeness_detects_multiple_collections_in_one_relation():
    plan = [{
        "step_id": "add",
        "capability": "labels.add",
        "arguments": {
            "collection": "congr",
            "audience": {
                "resolver": "collection_members",
                "value": "2nd-yearsand careers-page",
            },
        },
    }]
    with patch(
        "features.natural_language._named_collection_candidates",
        return_value=[
            {"name": "2nd-years", "score": 0.9},
            {"name": "careers-page", "score": 0.9},
        ],
    ):
        issue = _plan_completeness_issue(plan, "add both collections", object())
    assert "multiple existing collections" in issue


def test_fallback_selects_closest_existing_command():
    assert fallback_command("please create an LFX event") == "!work create event"
    assert fallback_command("make a task for the docs") == "!work create task"
    assert fallback_command("show me progress") == "!reports"


def test_knowledge_context_supplies_high_confidence_program_defaults():
    context = build_knowledge_context({}, "create an LFX event for this year")
    assert "type=participation, category=lfx" in context
    assert "Current date:" in context


def test_named_entity_resolution_corrects_typos_and_maps_update_intent():
    candidate = {"type": "event", "id": 42, "name": "Spring Term 2026", "score": 0.9}
    with patch("features.natural_language._named_entity_candidates", return_value=[candidate]):
        command = resolve_named_entity_command(
            "!help", "show changes on the Sprng Term event", object()
        )
    assert command == "!work history event 42"


def test_named_collection_membership_is_compiled_from_live_entity_context():
    candidate = {"name": "lfx-applicants", "score": 0.9}
    with patch("features.natural_language._named_collection_candidates", return_value=[candidate]):
        command = resolve_named_collection_command(
            "!help",
            "add @~Shaurya to lfx-applicants",
            object(),
            ["50990036295744@lid"],
        )
    assert command == "!labels add lfx-applicants"


def test_model_output_cannot_escape_the_command_allowlist():
    assert validate_command("!work update event 4 prs 3") == "!work update event 4 prs 3"
    assert validate_command("delete everything") is None
    assert validate_command("!update-event 4 | name old") is None
    assert validate_command("!reminders") is None
    assert validate_command("!work\n!remove 1") is None
    assert validate_command("```!my```") is None


def test_bot_mention_translates_and_reenters_dispatch_with_sender_context():
    client = MagicMock()
    client.get_me.return_value = SimpleNamespace(
        JID="bot@s.whatsapp.net", LID="999999@lid"
    )
    message = make_message("@Bot please show my workload")
    dispatch = MagicMock()

    with patch("features.natural_language._get_mentioned_jids", return_value=[
        "bot@s.whatsapp.net",
        "target@s.whatsapp.net",
    ]), patch.object(
        MistralCommandTranslator,
        "translate",
        return_value=("!my", ""),
    ):
        handler = register(client, {"mistral_api_key": "secret"})
        assert handler(client, message, dispatch) is True

    translated = dispatch.call_args.args[0]
    assert translated.Info is message.Info
    assert translated.Info.MessageSource.Sender == "member@s.whatsapp.net"
    assert translated.Message.conversation == "!my"
    assert translated._pbbot_nl_command is True


def test_me_alias_triggers_translation_and_resolves_to_sender_mention():
    client = MagicMock()
    client.get_me.return_value = SimpleNamespace(
        JID="bot@s.whatsapp.net", LID="999999@lid"
    )
    message = make_message("@me assign task 7 to me")
    dispatch = MagicMock()

    with patch("features.natural_language._get_mentioned_jids", return_value=[]), patch.object(
        MistralCommandTranslator,
        "translate",
        return_value=("!work assign task 7 | @me", ""),
    ):
        handler = register(client, {"mistral_api_key": "secret"})
        assert handler(client, message, dispatch) is True

    translated = dispatch.call_args.args[0]
    assert translated._pbbot_me_jid == "member@s.whatsapp.net"
    assert "member@s.whatsapp.net" in _get_mentioned_jids(translated)


def test_natural_language_card_design_reenters_dispatch_with_design_spec():
    client = MagicMock()
    client.get_me.return_value = SimpleNamespace(
        JID="bot@s.whatsapp.net", LID="999999@lid"
    )
    message = make_message(
        "@bot create a sarcastic congratulations card for Zodiak for PBCTF 5.0"
    )
    dispatch = MagicMock()
    intent = {
        "capability": "card.design",
        "arguments": {
            "base_template": "hackathon",
            "name": "Zodiak",
            "text": "For PBCTF 5.0",
            "accent": "#A855F7",
        },
    }

    with patch("features.natural_language._get_mentioned_jids", return_value=["bot@s.whatsapp.net"]), \
         patch.object(MistralCommandTranslator, "translate", return_value=(intent, "")), \
         patch.object(MistralCardDesigner, "design", return_value=intent):
        handler = register(client, {"mistral_api_key": "secret"})
        assert handler(client, message, dispatch) is True

    translated = dispatch.call_args.args[0]
    assert translated.Message.conversation == "!card custom | Zodiak | For PBCTF 5.0"
    assert translated._pbbot_card_design["base_template"] == "hackathon"
