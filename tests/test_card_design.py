from cards.render import _build_html, _build_original_html, validate_card_design
from features.cards import _LEADING_MENTION_COMMAND_RE


def test_card_design_schema_allows_text_but_rejects_unsafe_urls():
    assert validate_card_design({"base_template": "custom", "title": "<script>"}) is not None
    assert validate_card_design({"base_template": "custom", "logo_url": "javascript:alert(1)"}) is None
    assert validate_card_design({"base_template": "talk"}) is None
    assert validate_card_design({"base_template": "custom", "accent": "red"}) is None


def test_custom_design_overrides_copy_style_but_keeps_renderer_html_controlled():
    design = validate_card_design({
        "base_template": "hackathon",
        "title": "Congratulations, Zodiak!",
        "accent": "#a855f7",
        "pill": "PBCTF 5.0",
        "highlight_terms": ["PBCTF"],
        "tone": "sarcastic",
        "variation": 4,
    })

    html = _build_html(
        card_type="hackathon",
        name="Zodiak",
        text="For PBCTF",
        photo_data_url="data:image/png;base64,AA==",
        logo_data_url=None,
        design=design,
    )

    assert "Congratulations, Zodiak!" in html
    assert "#A855F7" in html
    assert 'class="highlight">PBCTF</span>' in html
    assert 'class="poster style-hackathon tone-sarcastic variation-4"' in html
    assert "text-transform: uppercase" in html
    assert "poster-wash" in html
    assert "poster-orbit" in html
    assert "<script" not in html


def test_default_renderer_uses_the_original_main_template():
    html = _build_original_html(
        card_type="hackathon",
        name="Zodiak",
        text="For [PBCTF]",
        photo_data_url="data:image/png;base64,AA==",
        logo_data_url=None,
    )

    assert 'class="grid-bg"' in html
    assert 'class="title">Congratulations</div>' in html
    assert 'class="avatar"' in html
    assert "poster-wash" not in html
    assert "poster-orbit" not in html


def test_leading_bot_mention_does_not_block_card_command_detection():
    caption = "@me !card-pdf talk | Akash Singh | Topic | Event"
    command = _LEADING_MENTION_COMMAND_RE.sub("", caption).strip()

    assert command.startswith("!card-pdf talk")
