"""Achievement card renderer.

Uses Playwright (headless Chromium) to render an HTML template into PNG and/or
PDF.  The template, CSS, and decorative SVG are identical to the original
Node.js/Puppeteer renderer so output is pixel-identical.

Card types control the accent colour and bottom pill.  The body text is
user-supplied; wrap any phrase in [brackets] to highlight it in the accent
colour.  ``internship`` and ``custom`` types also accept a logo URL. ``talk``
uses a dedicated speaker-thank-you layout with a required event name and up to
two optional event logos.
"""

from __future__ import annotations

import base64
import html as html_mod
import logging
import re
import secrets
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CARD_W = 1080
CARD_H = 1350
TALK_CARD_H = 1440

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_PB_LOGO_PATH = _ASSETS_DIR / "pb-logo.png"

# Embed the PB logo as a data-URL once at import time.
_PB_LOGO_DATA_URL: str | None = None
try:
    _pb_logo_bytes = _PB_LOGO_PATH.read_bytes()
    _PB_LOGO_DATA_URL = f"data:image/png;base64,{base64.b64encode(_pb_logo_bytes).decode()}"
except FileNotFoundError:
    log.warning("PB logo asset missing at %s", _PB_LOGO_PATH)

# ---------------------------------------------------------------------------
# Card type presets
# ---------------------------------------------------------------------------

TYPES: dict[str, dict[str, Any]] = {
    "gsoc": {
        "accent": "#FBBC04",
        "pill": "Google Summer of Code",
        "logoUrl": "https://developers.google.com/open-source/gsoc/resources/downloads/GSoC-Horizontal.png",
    },
    "lfx": {
        "accent": "#5C9BD6",
        "pill": "The Linux Foundation",
        "logoUrl": "https://lfx.linuxfoundation.org/wp-content/uploads/2023/01/logo_lfx_nopad.svg",
    },
    "hackathon": {"accent": "#A855F7", "pill": "Hackathon Winner"},
    "competitive": {"accent": "#2ED573", "pill": "Competitive Programming"},
    "acm": {
        "accent": "#F5A623",
        "pill": "ACM Summer / Winter School",
        "logoUrl": "https://www.acm.org/binaries/content/gallery/global/top-menu/acm_logo_tablet.svg",
    },
    "internship": {"accent": "#00BCD4", "pill": None},  # logo via logoUrl
    "talk": {
        "accent": "#48F80D",
        "pill": None,
    },
    "custom": {"accent": "#FFFFFF", "pill": None},
}

CARD_TYPES: list[str] = list(TYPES.keys())

CARD_TONES = frozenset({
    "sincere", "celebratory", "grateful", "professional",
    "playful", "sarcastic", "deadpan", "dramatic",
})

TONE_ACCENTS = {
    "sincere": "#5C9BD6",
    "celebratory": "#FBBC04",
    "grateful": "#48F80D",
    "professional": "#5C9BD6",
    "playful": "#A855F7",
    "sarcastic": "#FF5C8A",
    "deadpan": "#A0AEC0",
    "dramatic": "#F97316",
}

VARIATION_ACCENTS = (
    "#48F80D",
    "#FF5C8A",
    "#22D3EE",
    "#FACC15",
    "#A78BFA",
    "#FB923C",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _escape(s: str) -> str:
    return html_mod.escape(s)


_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")


def validate_card_design(value: object) -> dict[str, Any] | None:
    """Validate the bounded design vocabulary used by natural-language cards.

    The model is allowed to select an existing visual family and a small set
    of presentation values.  It is never allowed to provide HTML, CSS,
    arbitrary layout coordinates, or executable content.
    """
    if not isinstance(value, dict):
        return None

    base_template = value.get("base_template", value.get("template", "custom"))
    if not isinstance(base_template, str):
        return None
    base_template = base_template.strip().lower()
    # Talk is a separate, fixed layout with different required fields.
    if base_template not in TYPES or base_template == "talk":
        return None

    def bounded_text(key: str, limit: int) -> str | None:
        item = value.get(key)
        if item is None:
            return None
        if not isinstance(item, str):
            return None
        item = item.strip()
        return item if len(item) <= limit else None

    title = bounded_text("title", 64)
    pill = bounded_text("pill", 96)
    if title is None and value.get("title") is not None:
        return None
    if pill is None and value.get("pill") is not None:
        return None

    accent = value.get("accent")
    if accent is not None:
        if not isinstance(accent, str) or not _HEX_RE.match(accent.strip()):
            return None
        accent = "#" + _HEX_RE.match(accent.strip()).group(1).upper()

    logo_url = value.get("logo_url")
    if logo_url is not None:
        if not isinstance(logo_url, str) or len(logo_url.strip()) > 2000:
            return None
        logo_url = logo_url.strip()
        if not (logo_url.startswith("https://") or logo_url.startswith("http://")
                or logo_url.startswith("data:image/")):
            return None

    highlight_terms = value.get("highlight_terms", [])
    if highlight_terms is None:
        highlight_terms = []
    if not isinstance(highlight_terms, list) or len(highlight_terms) > 8:
        return None
    normalized_highlights: list[str] = []
    for term in highlight_terms:
        if not isinstance(term, str):
            return None
        term = term.strip()
        if not term or len(term) > 48:
            return None
        normalized_highlights.append(term)

    tone = value.get("tone", "celebratory")
    if not isinstance(tone, str) or tone.strip().lower() not in CARD_TONES:
        return None
    tone = tone.strip().lower()

    occasion = value.get("occasion")
    if occasion is not None:
        if not isinstance(occasion, str) or len(occasion.strip()) > 120:
            return None
        occasion = occasion.strip()

    variation = value.get("variation")
    if variation is not None and (
        not isinstance(variation, int) or isinstance(variation, bool) or not 0 <= variation < len(VARIATION_ACCENTS)
    ):
        return None

    return {
        "base_template": base_template,
        "title": title,
        "pill": pill,
        "accent": accent,
        "logo_url": logo_url,
        "highlight_terms": normalized_highlights,
        "tone": tone,
        "occasion": occasion,
        "variation": variation,
    }


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    m = _HEX_RE.match(hex_color)
    if not m:
        return f"rgba(255,255,255,{alpha})"
    v = m.group(1)
    r, g, b = int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _process_highlights(escaped_text: str, highlight_terms: list[str] | None = None) -> str:
    highlighted = re.sub(
        r"\[([^\[\]\n]+)\]",
        r'<span class="highlight">\1</span>',
        escaped_text,
    )
    for term in highlight_terms or []:
        escaped_term = _escape(term)
        highlighted = re.sub(
            re.escape(escaped_term),
            f'<span class="highlight">{escaped_term}</span>',
            highlighted,
            count=1,
            flags=re.IGNORECASE,
        )
    return highlighted


_EXT_TO_MIME: dict[str, str] = {
    "svg": "image/svg+xml",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}


async def _fetch_image_as_data_url(url: str) -> str:
    """Fetch an image URL and return a base64 ``data:`` URL."""
    async with httpx.AsyncClient(follow_redirects=True) as http:
        resp = await http.get(url, timeout=15)
        resp.raise_for_status()

    ct_mime = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    ext = url.split("?")[0].split("#")[0].rsplit(".", 1)[-1].lower()
    inferred = _EXT_TO_MIME.get(ext)
    mime = ct_mime if ct_mime.startswith("image/") else inferred
    if not mime:
        raise ValueError(f"Couldn't determine image type (content-type: {ct_mime or 'unknown'})")

    b64 = base64.b64encode(resp.content).decode()
    return f"data:{mime};base64,{b64}"


async def _resolve_image_as_data_url(source: str) -> str:
    """Return an image ``data:`` URL from either a data URL or a remote URL."""
    cleaned = source.strip()
    if cleaned.startswith("data:image/"):
        return cleaned
    return await _fetch_image_as_data_url(cleaned)


# ---------------------------------------------------------------------------
# Decorative SVG layer
# ---------------------------------------------------------------------------


def _build_deco_svg(accent_hex: str) -> str:
    stroke = _hex_to_rgba(accent_hex, 0.65)
    stroke_soft = _hex_to_rgba(accent_hex, 0.35)
    dot_fill = _hex_to_rgba(accent_hex, 0.75)
    dot_fill_soft = _hex_to_rgba(accent_hex, 0.45)

    dots = [
        (150, 240, 3.5), (930, 260, 3), (220, 180, 2.5), (880, 195, 2.5),
        (85, 365, 3), (1000, 395, 3.5), (120, 880, 3), (965, 905, 3.5),
        (70, 1090, 2.5), (1015, 1075, 3), (180, 1180, 3.5), (905, 1175, 2.5),
        (250, 765, 2), (840, 770, 2.5), (430, 175, 2), (665, 165, 2.5),
    ]
    plus_marks = [
        (110, 305), (980, 320), (60, 700), (1025, 720),
        (200, 1010), (890, 1010), (330, 220), (770, 230),
    ]

    dot_svg = "".join(
        f'<circle cx="{x}" cy="{y}" r="{r}" fill="{dot_fill if i % 3 == 0 else dot_fill_soft}"/>'
        for i, (x, y, r) in enumerate(dots)
    )
    plus_svg = "".join(
        f'<g stroke="{stroke_soft}" stroke-width="2" stroke-linecap="round">'
        f'<line x1="{x - 8}" y1="{y}" x2="{x + 8}" y2="{y}"/>'
        f'<line x1="{x}" y1="{y - 8}" x2="{x}" y2="{y + 8}"/>'
        f"</g>"
        for x, y in plus_marks
    )

    corners = (
        f'<path d="M 60 140 L 60 60 L 140 60" stroke="{stroke}" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="78" cy="78" r="3" fill="{dot_fill}"/>'
        f'<path d="M 940 60 L 1020 60 L 1020 140" stroke="{stroke}" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="1002" cy="78" r="3" fill="{dot_fill}"/>'
        f'<path d="M 60 1210 L 60 1290 L 140 1290" stroke="{stroke}" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="78" cy="1272" r="3" fill="{dot_fill}"/>'
        f'<path d="M 940 1290 L 1020 1290 L 1020 1210" stroke="{stroke}" stroke-width="3" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="1002" cy="1272" r="3" fill="{dot_fill}"/>'
    )

    avatar_rings = (
        f'<circle cx="540" cy="550" r="178" fill="none" stroke="{_hex_to_rgba(accent_hex, 0.4)}" stroke-width="2"/>'
        f'<circle cx="540" cy="550" r="194" fill="none" stroke="{_hex_to_rgba(accent_hex, 0.18)}" stroke-width="2" stroke-dasharray="6 10"/>'
    )

    return (
        f'<svg class="deco" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {CARD_W} {CARD_H}" '
        f'width="{CARD_W}" height="{CARD_H}" aria-hidden="true">'
        f"{corners}{dot_svg}{plus_svg}{avatar_rings}"
        f"</svg>"
    )


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------


def _build_original_html(
    *,
    card_type: str,
    name: str,
    text: str,
    photo_data_url: str,
    logo_data_url: str | None,
) -> str:
    """Build the unmodified main-branch achievement-card template.

    Natural-language processing may decide what a card says, but it must not
    change its visual template unless the sender explicitly asks for styling
    changes.  Keeping this builder separate makes the default output exactly
    match the original template.
    """
    cfg = TYPES.get(card_type, TYPES["custom"])
    accent = cfg["accent"]
    sentence_html = _process_highlights(_escape(text))
    accent_soft = _hex_to_rgba(accent, 0.08)
    title_glow = _hex_to_rgba(accent, 0.18)
    deco_svg = _build_deco_svg(accent)

    pill_html = ""
    if logo_data_url:
        pill_html = f'<div class="pill logo-pill"><img src="{logo_data_url}" alt="logo" /></div>'
    elif cfg.get("pill"):
        pill_html = f'<div class="pill">{_escape(cfg["pill"])}</div>'

    pb_logo_mark = (
        f'<img class="mark" src="{_PB_LOGO_DATA_URL}" alt="Point Blank mark" />'
        if _PB_LOGO_DATA_URL
        else '<div style="font-size:56px;color:#48F80D;line-height:1;">&lt;.&gt;</div>'
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&family=JetBrains+Mono:wght@600;700&display=swap">
<style>
  :root {{ --accent: {accent}; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  body {{
    width: {CARD_W}px; height: {CARD_H}px; background: #07070d;
    background-image: radial-gradient(ellipse 70% 50% at 50% 50%, {accent_soft} 0%, transparent 70%);
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif; color: #fff; position: relative; overflow: hidden;
  }}
  .grid-bg, .deco {{ position: absolute; top: 0; left: 0; width: {CARD_W}px; height: {CARD_H}px; pointer-events: none; }}
  .grid-bg {{ z-index: 0; }} .deco {{ z-index: 1; }}
  .pb-logo {{ position: absolute; top: 70px; left: 0; right: 0; text-align: center; z-index: 2; font-family: 'JetBrains Mono', monospace; font-weight: 700; letter-spacing: 0.05em; }}
  .pb-logo .mark {{ width: 220px; height: auto; display: block; margin: 0 auto; }}
  .pb-logo .row {{ font-size: 32px; margin-top: 20px; }} .pb-logo .point {{ color: #48F80D; }} .pb-logo .blank {{ color: #fff; }}
  .title {{ position: absolute; top: 230px; left: 0; right: 0; text-align: center; font-size: 86px; font-weight: 800; letter-spacing: -0.02em; text-shadow: 0 0 18px {title_glow}; z-index: 2; }}
  .avatar {{ position: absolute; left: 50%; top: 380px; transform: translateX(-50%); width: 340px; height: 340px; border-radius: 50%; overflow: hidden; background: #1a1a2e; box-shadow: 0 0 0 6px rgba(255,255,255,0.6); }}
  .avatar img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
  .person-name {{ position: absolute; top: 770px; left: 0; right: 0; text-align: center; font-size: 60px; font-weight: 700; }}
  .sentence {{ position: absolute; top: 905px; left: 90px; right: 90px; text-align: center; font-size: 36px; line-height: 1.5; font-weight: 500; }}
  .highlight {{ color: var(--accent); font-weight: 700; }}
  .pill {{ position: absolute; bottom: 110px; left: 50%; transform: translateX(-50%); background: #ffffff; color: #15192b; padding: 22px 44px; border-radius: 18px; font-weight: 700; font-size: 30px; white-space: nowrap; box-shadow: 0 10px 26px rgba(0,0,0,0.5); }}
  .pill.logo-pill {{ padding: 0; background: transparent; box-shadow: none; }}
  .pill.logo-pill img {{ max-height: 150px; max-width: 480px; object-fit: contain; display: block; margin: 0 auto; border-radius: 12px; }}
</style>
</head>
<body>
  <svg class="grid-bg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {CARD_W} {CARD_H}" width="{CARD_W}" height="{CARD_H}" aria-hidden="true">
    <defs><pattern id="grid-pattern" width="52" height="52" patternUnits="userSpaceOnUse" patternTransform="translate(20, 25)"><path d="M 52 0 L 0 0 0 52" fill="none" stroke="#969FBE" stroke-opacity="0.34" stroke-width="1"/></pattern></defs>
    <rect width="{CARD_W}" height="{CARD_H}" fill="url(#grid-pattern)"/>
  </svg>
  {deco_svg}
  <div class="pb-logo">{pb_logo_mark}<div class="row"><span class="point">Point</span> <span class="blank">Blank</span></div></div>
  <div class="title">Congratulations</div>
  <div class="avatar"><img src="{photo_data_url}" alt="profile" /></div>
  <div class="person-name">{_escape(name)}</div>
  <div class="sentence">{sentence_html}</div>
  {pill_html}
</body>
</html>"""


def _build_html(
    *,
    card_type: str,
    name: str,
    text: str,
    photo_data_url: str,
    logo_data_url: str | None,
    design: dict[str, Any] | None = None,
) -> str:
    cfg = TYPES.get(card_type, TYPES["custom"])
    design = design or {}
    tone = design.get("tone", "celebratory")
    accent = design.get("accent") or TONE_ACCENTS.get(tone) or cfg["accent"]
    variation = design.get("variation")
    if not isinstance(variation, int) or isinstance(variation, bool):
        variation = 0
    variation %= len(VARIATION_ACCENTS)
    secondary = VARIATION_ACCENTS[variation]
    sentence_html = _process_highlights(
        _escape(text), design.get("highlight_terms")
    )
    accent_soft = _hex_to_rgba(accent, 0.08)
    title_glow = _hex_to_rgba(accent, 0.18)
    deco_svg = _build_deco_svg(accent)

    pill_html = ""
    if logo_data_url:
        pill_html = f'<div class="pill logo-pill"><img src="{logo_data_url}" alt="logo" /></div>'
    else:
        pill = design["pill"] if "pill" in design else cfg.get("pill")
        if pill:
            pill_html = f'<div class="pill">{_escape(pill)}</div>'

    title = design.get("title") or "Congratulations"
    template_class = re.sub(r"[^a-z0-9-]", "", card_type.lower()) or "custom"
    tone_class = re.sub(r"[^a-z0-9-]", "", tone.lower()) or "celebratory"
    variation_class = f"variation-{variation}"
    title_font = _size_for_text(
        title, base=78, medium=62, small=48, medium_at=28, small_at=44
    )
    sentence_font = _size_for_text(
        text, base=36, medium=31, small=27, medium_at=90, small_at=140
    )
    avatar_top = 410 if len(title) >= 29 else 380
    name_top = avatar_top + 390
    sentence_top = name_top + 125

    pb_logo_mark = (
        f'<img class="mark" src="{_PB_LOGO_DATA_URL}" alt="Point Blank mark" />'
        if _PB_LOGO_DATA_URL
        else '<div style="font-size:56px;color:#48F80D;line-height:1;">&lt;.&gt;</div>'
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&family=JetBrains+Mono:wght@600;700&display=swap">
<style>
  :root {{ --accent: {accent}; }}
  :root {{ --secondary: {secondary}; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  html, body {{
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}
  body {{
    width: {CARD_W}px;
    height: {CARD_H}px;
    background: #07070d;
    background-image:
      radial-gradient(ellipse 70% 50% at 50% 50%, {accent_soft} 0%, transparent 70%);
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
    color: #fff;
    position: relative;
    overflow: hidden;
  }}

  .poster-wash, .poster-band {{
    position: absolute;
    pointer-events: none;
    z-index: 0;
  }}
  .poster-wash {{
    inset: 0;
    background:
      radial-gradient(ellipse 76% 47% at 50% 33%, {accent_soft} 0%, transparent 70%),
      linear-gradient(135deg, rgba(255,255,255,0.035), transparent 38%, rgba(0,0,0,0.34));
  }}
  .poster-band {{
    width: 145%;
    height: 250px;
    left: -22%;
    top: 575px;
    transform: rotate(-11deg);
    border-top: 2px solid {_hex_to_rgba(accent, 0.28)};
    border-bottom: 2px solid {_hex_to_rgba(accent, 0.12)};
    background: linear-gradient(90deg, transparent, {_hex_to_rgba(accent, 0.16)}, transparent);
    box-shadow: 0 0 90px {_hex_to_rgba(accent, 0.14)};
  }}
  .poster-orbit, .poster-burst {{
    position: absolute;
    pointer-events: none;
    z-index: 0;
  }}
  .poster-orbit {{
    width: 760px;
    height: 760px;
    left: 160px;
    top: 225px;
    border: 2px solid {_hex_to_rgba(secondary, 0.16)};
    border-radius: 50%;
    transform: rotate(-18deg) scaleX(0.78);
    box-shadow: 0 0 90px {_hex_to_rgba(secondary, 0.12)};
  }}
  .poster-orbit::before, .poster-orbit::after {{
    content: "";
    position: absolute;
    inset: 42px;
    border: 1px dashed {_hex_to_rgba(secondary, 0.18)};
    border-radius: 50%;
  }}
  .poster-orbit::after {{
    inset: 120px;
    border-style: solid;
    border-color: {_hex_to_rgba(accent, 0.12)};
  }}
  .poster-burst {{
    width: 420px;
    height: 420px;
    right: -175px;
    top: -120px;
    border: 2px solid {_hex_to_rgba(secondary, 0.22)};
    transform: rotate(26deg);
    background: linear-gradient(135deg, {_hex_to_rgba(secondary, 0.12)}, transparent 55%);
    clip-path: polygon(0 0, 100% 0, 100% 18%, 18% 100%, 0 100%);
  }}
  .variation-1 .poster-band {{
    height: 115px;
    top: 495px;
    transform: rotate(17deg);
  }}
  .variation-1 .poster-orbit {{
    left: 240px;
    top: 305px;
    transform: rotate(22deg) scaleX(0.58);
  }}
  .variation-2 .poster-band {{
    height: 390px;
    top: 460px;
    transform: rotate(-24deg);
    opacity: 0.72;
  }}
  .variation-2 .poster-orbit {{
    left: -90px;
    top: 340px;
    transform: rotate(-8deg) scaleX(0.62);
  }}
  .variation-3 .poster-band {{
    height: 72px;
    top: 760px;
    transform: rotate(-4deg);
    border-top-style: dashed;
    border-bottom-style: dashed;
  }}
  .variation-3 .poster-orbit {{
    width: 900px;
    height: 520px;
    left: 90px;
    top: 330px;
    transform: rotate(-12deg) scaleX(0.9);
  }}
  .variation-4 .poster-band {{
    height: 180px;
    top: 335px;
    transform: rotate(8deg);
    background: linear-gradient(90deg, transparent, {_hex_to_rgba(secondary, 0.24)}, transparent);
  }}
  .variation-4 .poster-orbit {{
    left: 180px;
    top: 390px;
    transform: rotate(4deg) scaleX(0.72);
  }}
  .variation-5 .poster-band {{
    height: 300px;
    top: 640px;
    transform: rotate(28deg);
  }}
  .variation-5 .poster-orbit {{
    left: 255px;
    top: 180px;
    transform: rotate(-30deg) scaleX(0.66);
  }}
  .style-gsoc .poster-band {{
    background: linear-gradient(90deg, transparent, rgba(251,188,4,0.22), transparent);
  }}
  .style-lfx .poster-band {{
    background: linear-gradient(90deg, transparent, rgba(92,155,214,0.22), transparent);
  }}
  .style-hackathon .poster-band, .style-custom .poster-band {{
    background: linear-gradient(90deg, transparent, rgba(168,85,247,0.24), rgba(249,115,22,0.12), transparent);
  }}
  .style-competitive .poster-band {{
    background: linear-gradient(90deg, transparent, rgba(46,213,115,0.22), transparent);
  }}
  .style-acm .poster-band {{
    background: linear-gradient(90deg, transparent, rgba(245,166,35,0.22), transparent);
  }}
  .style-internship .poster-band {{
    background: linear-gradient(90deg, transparent, rgba(0,188,212,0.22), transparent);
  }}

  .grid-bg, .deco {{
    position: absolute;
    top: 0; left: 0;
    width: {CARD_W}px;
    height: {CARD_H}px;
    pointer-events: none;
  }}
  .grid-bg {{ z-index: 0; }}
  .deco {{ z-index: 1; }}

  .pb-logo {{
    position: absolute;
    top: 70px; left: 0; right: 0;
    text-align: center;
    z-index: 2;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
  }}
  .pb-logo .mark {{
    width: 220px;
    height: auto;
    display: block;
    margin: 0 auto;
  }}
  .pb-logo .row {{ font-size: 32px; margin-top: 20px; }}
  .pb-logo .point {{ color: #48F80D; }}
  .pb-logo .blank {{ color: #fff; }}

  .title {{
    position: absolute;
    top: 215px; left: 70px; right: 70px;
    min-height: 155px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    font-size: {title_font}px;
    line-height: 1.08;
    font-weight: 800;
    letter-spacing: -0.02em;
    text-shadow: 0 0 18px {title_glow};
    z-index: 2;
    overflow-wrap: break-word;
  }}

  .tone-sarcastic .title {{
    letter-spacing: 0.015em;
    text-transform: uppercase;
  }}
  .tone-deadpan .title {{
    text-shadow: none;
  }}
  .tone-playful .pill {{
    border-radius: 999px;
  }}
  .tone-dramatic .title {{
    text-shadow: 0 0 18px {title_glow}, 0 0 36px {title_glow};
  }}

  .avatar {{
    position: absolute;
    left: 50%; top: {avatar_top}px;
    transform: translateX(-50%);
    width: 340px; height: 340px;
    border-radius: 50%;
    overflow: hidden;
    background: #1a1a2e;
    box-shadow: 0 0 0 6px rgba(255,255,255,0.6);
  }}
  .avatar img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}

  .person-name {{
    position: absolute;
    top: {name_top}px; left: 0; right: 0;
    text-align: center;
    font-size: {_size_for_text(name, base=60, medium=50, small=42, medium_at=20, small_at=30)}px;
    font-weight: 700;
  }}

  .sentence {{
    position: absolute;
    top: {sentence_top}px; left: 90px; right: 90px;
    text-align: center;
    font-size: {sentence_font}px;
    line-height: 1.5;
    font-weight: 500;
  }}
  .highlight {{ color: var(--accent); font-weight: 700; }}

  .pill {{
    position: absolute;
    bottom: 110px; left: 50%;
    transform: translateX(-50%);
    background: #ffffff;
    color: #15192b;
    padding: 22px 44px;
    border-radius: 18px;
    font-weight: 700;
    font-size: 30px;
    white-space: nowrap;
    box-shadow: 0 10px 26px rgba(0,0,0,0.5);
  }}
  .pill.logo-pill {{
    width: 480px;
    height: 150px;
    padding: 0;
    background: transparent;
    box-shadow: none;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .pill.logo-pill img {{
    width: 100%;
    height: 100%;
    max-height: 150px;
    max-width: 480px;
    object-fit: contain;
    display: block;
    margin: 0 auto;
    border-radius: 12px;
  }}
</style>
</head>
<body class="poster style-{_escape(template_class)} tone-{_escape(tone_class)} {variation_class}">
  <div class="poster-wash"></div>
  <div class="poster-band"></div>
  <div class="poster-orbit"></div>
  <div class="poster-burst"></div>
  <svg class="grid-bg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {CARD_W} {CARD_H}" width="{CARD_W}" height="{CARD_H}" aria-hidden="true">
    <defs>
      <pattern id="grid-pattern" width="52" height="52" patternUnits="userSpaceOnUse" patternTransform="translate(20, 25)">
        <path d="M 52 0 L 0 0 0 52" fill="none" stroke="#969FBE" stroke-opacity="0.34" stroke-width="1"/>
      </pattern>
    </defs>
    <rect width="{CARD_W}" height="{CARD_H}" fill="url(#grid-pattern)"/>
  </svg>

  {deco_svg}

  <div class="pb-logo">
    {pb_logo_mark}
    <div class="row"><span class="point">Point</span> <span class="blank">Blank</span></div>
  </div>

  <div class="title">{_escape(title)}</div>

  <div class="avatar"><img src="{photo_data_url}" alt="profile" /></div>

  <div class="person-name">{_escape(name)}</div>
  <div class="sentence">{sentence_html}</div>
  {pill_html}
</body>
</html>"""


def _size_for_text(
    text: str,
    *,
    base: int,
    medium: int,
    small: int,
    medium_at: int,
    small_at: int,
) -> int:
    length = len(text.strip())
    if length >= small_at:
        return small
    if length >= medium_at:
        return medium
    return base


def _build_talk_deco_svg(accent_hex: str) -> str:
    accent = _hex_to_rgba(accent_hex, 0.78)
    accent_soft = _hex_to_rgba(accent_hex, 0.28)
    accent_faint = _hex_to_rgba(accent_hex, 0.12)
    pb_blue = "#48F80D"

    stars = [
        (128, 222, 2.2), (913, 188, 1.8), (185, 588, 2.4), (862, 546, 2.1),
        (102, 862, 1.9), (982, 908, 2.4), (198, 1194, 2.1), (858, 1228, 1.8),
        (74, 1282, 2.3), (1006, 1310, 2.1), (376, 968, 1.6), (717, 1006, 1.7),
    ]
    star_svg = "".join(
        f'<circle cx="{x}" cy="{y}" r="{r}" fill="rgba(255,255,255,{0.74 if i % 3 == 0 else 0.42})"/>'
        for i, (x, y, r) in enumerate(stars)
    )

    return f"""
  <svg class="talk-deco" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {CARD_W} {TALK_CARD_H}" width="{CARD_W}" height="{TALK_CARD_H}" aria-hidden="true">
    <defs>
      <filter id="talk-soft-glow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="5" result="blur"/>
        <feMerge>
          <feMergeNode in="blur"/>
          <feMergeNode in="SourceGraphic"/>
        </feMerge>
      </filter>
      <linearGradient id="talk-blue-mark" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#48F80D"/>
        <stop offset="1" stop-color="#228B22"/>
      </linearGradient>
      <linearGradient id="talk-icon-fill" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#32CD32"/>
        <stop offset="1" stop-color="#006400"/>
      </linearGradient>
    </defs>
    <rect width="{CARD_W}" height="{TALK_CARD_H}" fill="#02080A"/>
    <path d="M-80 215 C 105 250 185 328 344 312 S 611 116 812 41" stroke="{accent_faint}" stroke-width="13" fill="none" filter="url(#talk-soft-glow)"/>
    <path d="M-36 1235 C 210 1325 404 1207 578 1117 C 763 1022 903 1050 1144 1130" stroke="{accent_faint}" stroke-width="12" fill="none" filter="url(#talk-soft-glow)"/>
    <path d="M744 -40 C 814 128 796 257 698 386 C 587 532 603 674 760 819 C 863 914 871 999 781 1092" stroke="{accent_soft}" stroke-width="4" fill="none"/>
    <path d="M-72 1040 C 230 914 342 751 317 548 C 295 374 363 241 558 157" stroke="{accent_soft}" stroke-width="3" fill="none"/>
    <g transform="translate(904 -18) rotate(29)" opacity="0.72">
      <path d="M8 42 L92 6 C109 -1 132 3 151 16 L151 143 C131 130 110 126 92 134 L8 170 Z" fill="url(#talk-blue-mark)"/>
      <path d="M151 16 C171 3 193 -1 211 6 L294 42 L294 170 L211 134 C193 126 171 130 151 143 Z" fill="#72ECF2" opacity="0.72"/>
      <path d="M151 24 L151 151" stroke="#001114" stroke-opacity="0.42" stroke-width="8" stroke-linecap="round"/>
      <path d="M42 63 L101 38 M42 98 L101 73 M199 38 L260 63 M199 73 L260 98" stroke="#001114" stroke-opacity="0.34" stroke-width="9" stroke-linecap="round"/>
    </g>
    <g transform="translate(-45 170) rotate(-18)" opacity="0.78">
      <rect x="63" y="4" width="86" height="142" rx="43" fill="url(#talk-icon-fill)"/>
      <path d="M35 86 C35 132 66 166 106 166 C146 166 177 132 177 86" fill="none" stroke="#001114" stroke-width="16" stroke-linecap="round"/>
      <path d="M106 166 L106 217 M64 217 L148 217" stroke="#001114" stroke-width="16" stroke-linecap="round"/>
      <path d="M80 43 L132 43 M80 75 L132 75 M80 107 L132 107" stroke="#B8FFFF" stroke-opacity="0.56" stroke-width="8" stroke-linecap="round"/>
    </g>
    <g transform="translate(946 884) rotate(24)" opacity="0.72">
      <rect x="54" y="0" width="76" height="125" rx="38" fill="url(#talk-icon-fill)"/>
      <path d="M27 75 C27 116 55 145 92 145 C129 145 157 116 157 75" fill="none" stroke="#001114" stroke-width="13" stroke-linecap="round"/>
      <path d="M92 145 L92 189 M57 189 L127 189" stroke="#001114" stroke-width="13" stroke-linecap="round"/>
      <path d="M72 38 L112 38 M72 67 L112 67 M72 96 L112 96" stroke="#B8FFFF" stroke-opacity="0.52" stroke-width="7" stroke-linecap="round"/>
    </g>
    <g transform="translate(-58 1280) rotate(15)" opacity="0.76">
      <path d="M8 28 L111 3 C135 -3 157 1 176 16 L176 154 C154 139 132 135 111 141 L8 166 Z" fill="url(#talk-blue-mark)"/>
      <path d="M176 16 C195 1 217 -3 241 3 L344 28 L344 166 L241 141 C220 135 198 139 176 154 Z" fill="#72ECF2" opacity="0.68"/>
      <path d="M176 26 L176 163" stroke="#001114" stroke-opacity="0.38" stroke-width="8" stroke-linecap="round"/>
      <g stroke="#001114" stroke-opacity="0.30" stroke-width="9" stroke-linecap="round">
        <line x1="44" y1="61" x2="120" y2="43"/>
        <line x1="44" y1="96" x2="120" y2="78"/>
        <line x1="232" y1="43" x2="308" y2="61"/>
        <line x1="232" y1="78" x2="308" y2="96"/>
      </g>
    </g>
    <g opacity="0.72">
      {star_svg}
    </g>
    <g stroke="{pb_blue}" stroke-width="3" stroke-linecap="round" opacity="0.32">
      <line x1="418" y1="40" x2="436" y2="40"/>
      <line x1="427" y1="31" x2="427" y2="49"/>
      <line x1="654" y1="44" x2="672" y2="44"/>
      <line x1="663" y1="35" x2="663" y2="53"/>
    </g>
  </svg>"""


def _build_event_logos_html(logo_data_urls: list[str]) -> str:
    if not logo_data_urls:
        return ""

    layout_class = "single-logo" if len(logo_data_urls) == 1 else "dual-logo"
    cards = "".join(
        f'<img class="talk-event-logo" src="{logo_url}" alt="event logo {idx}" />'
        for idx, logo_url in enumerate(logo_data_urls, start=1)
    )
    return f"""
  <div class="talk-logos {layout_class}">
    {cards}
  </div>"""


def _build_talk_html(
    *,
    card_type: str,
    name: str,
    text: str,
    photo_data_url: str,
    event_name: str,
    event_logo_data_urls: list[str],
) -> str:
    cfg = TYPES.get(card_type, TYPES["talk"])
    accent = cfg["accent"]
    event_text = event_name.strip()
    speaker_name = name.strip().upper()
    title_text = text.strip()
    accent_soft = _hex_to_rgba(accent, 0.10)
    accent_mid = _hex_to_rgba(accent, 0.45)
    title_font = _size_for_text(title_text, base=36, medium=32, small=27, medium_at=86, small_at=126)
    event_font = _size_for_text(event_text, base=36, medium=31, small=27, medium_at=42, small_at=60)
    name_font = _size_for_text(speaker_name, base=46, medium=40, small=34, medium_at=25, small_at=36)
    deco_svg = _build_talk_deco_svg(accent)
    event_logos = _build_event_logos_html(event_logo_data_urls)

    pb_logo_mark = (
        f'<img class="talk-pb-mark" src="{_PB_LOGO_DATA_URL}" alt="Point Blank mark" />'
        if _PB_LOGO_DATA_URL
        else '<span class="talk-mark-text">&lt;.&gt;</span>'
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@600;700&display=swap">
<style>
  :root {{ --accent: {accent}; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  html, body {{
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}
  body {{
    width: {CARD_W}px;
    height: {TALK_CARD_H}px;
    background:
      radial-gradient(circle at 50% 20%, rgba(47, 255, 235, 0.10), transparent 31%),
      radial-gradient(circle at 88% 90%, rgba(24, 247, 255, 0.14), transparent 24%),
      linear-gradient(145deg, #010608 0%, #050A0C 52%, #010305 100%);
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
    color: #fff;
    position: relative;
    overflow: hidden;
  }}

  .talk-deco {{
    position: absolute;
    top: 0; left: 0;
    width: {CARD_W}px;
    height: {TALK_CARD_H}px;
    pointer-events: none;
    z-index: 0;
  }}
  .talk-vignette {{
    position: absolute;
    inset: 0;
    background:
      radial-gradient(ellipse 70% 45% at 50% 45%, transparent 0%, rgba(0,0,0,0.18) 58%, rgba(0,0,0,0.72) 100%),
      linear-gradient(90deg, rgba(0,0,0,0.52), transparent 25%, transparent 75%, rgba(0,0,0,0.52));
    z-index: 1;
  }}
  .talk-content {{
    position: absolute;
    inset: 0;
    z-index: 2;
    text-align: center;
  }}
  .talk-brand {{
    position: absolute;
    top: 34px;
    left: 0; right: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 22px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 34px;
    font-weight: 700;
    line-height: 1;
  }}
  .talk-pb-mark {{
    width: 92px;
    height: auto;
    display: block;
  }}
  .talk-mark-text {{ color: #48F80D; }}
  .talk-brand .point {{ color: #48F80D; }}
  .talk-brand .blank {{ color: #FFFFFF; }}

  .talk-thanks {{
    position: absolute;
    top: 117px; left: 0; right: 0;
    font-size: 58px;
    font-weight: 500;
    line-height: 1.05;
    letter-spacing: 0;
    text-shadow: 0 0 12px rgba(255,255,255,0.96), 0 0 26px rgba(255,255,255,0.48);
  }}
  .talk-kicker {{
    position: absolute;
    top: 211px; left: 0; right: 0;
    font-size: 34px;
    font-weight: 400;
    line-height: 1.1;
  }}
  .talk-event {{
    position: absolute;
    top: 258px; left: 80px; right: 80px;
    font-size: {event_font}px;
    font-weight: 600;
    line-height: 1.16;
    letter-spacing: 3px;
    text-shadow: 0 0 9px rgba(255,255,255,0.86), 0 0 24px rgba(255,255,255,0.32);
  }}
  .talk-photo {{
    position: absolute;
    top: 318px;
    left: 50%;
    transform: translateX(-50%);
    width: 468px;
    height: 586px;
    border: 5px solid #FFFFFF;
    border-radius: 20px;
    overflow: hidden;
    background: #081317;
    box-shadow: 0 0 0 2px rgba(255,255,255,0.14), 0 18px 42px rgba(0,0,0,0.46);
  }}
  .talk-photo img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center;
    display: block;
  }}
  .talk-speaker {{
    position: absolute;
    top: 946px; left: 70px; right: 70px;
    font-size: {name_font}px;
    font-weight: 500;
    line-height: 1.08;
    letter-spacing: 2px;
    text-shadow: 0 0 10px rgba(255,255,255,0.95), 0 0 23px rgba(255,255,255,0.42);
  }}
  .talk-line {{
    position: absolute;
    top: 1032px; left: 90px; right: 90px;
    font-size: 32px;
    font-weight: 400;
    line-height: 1.22;
  }}
  .talk-title {{
    position: absolute;
    top: 1084px; left: 84px; right: 84px;
    min-height: 104px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--accent);
    font-size: {title_font}px;
    font-weight: 700;
    line-height: 1.35;
    letter-spacing: 3px;
    text-wrap: balance;
    overflow-wrap: break-word;
    text-shadow: 0 0 8px {accent}, 0 0 22px {accent_mid};
  }}
  .talk-logos {{
    position: absolute;
    top: 1210px; left: 0; right: 0;
    min-height: 190px;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 42px;
  }}
  .talk-event-logo {{
    display: block;
    width: auto;
    height: auto;
    max-width: 330px;
    max-height: 190px;
    object-fit: contain;
    border-radius: 12px;
    box-shadow: 0 14px 32px rgba(0,0,0,0.38);
  }}
  .talk-logos.single-logo .talk-event-logo {{
    width: auto;
    max-width: 500px;
  }}
  .talk-glow {{
    position: absolute;
    left: 50%;
    top: 766px;
    width: 640px;
    height: 410px;
    transform: translateX(-50%);
    background: radial-gradient(ellipse at center, {accent_soft}, transparent 69%);
    z-index: 1;
    pointer-events: none;
  }}
</style>
</head>
<body>
  {deco_svg}
  <div class="talk-vignette"></div>
  <div class="talk-glow"></div>
  <main class="talk-content">
    <div class="talk-brand">
      {pb_logo_mark}
      <div><span class="point">Point</span> <span class="blank">Blank</span></div>
    </div>
    <div class="talk-thanks">THANK YOU !!</div>
    <div class="talk-kicker">for representing us at</div>
    <div class="talk-event">{_escape(event_text)}</div>
    <div class="talk-photo"><img src="{photo_data_url}" alt="speaker" /></div>
    <div class="talk-speaker">{_escape(speaker_name)}</div>
    <div class="talk-line">and giving an insightful talk on</div>
    <div class="talk-title">{_escape(title_text)}</div>
    {event_logos}
  </main>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def render_card(
    *,
    card_type: str,
    name: str,
    text: str,
    photo_bytes: bytes,
    photo_mime: str = "image/jpeg",
    logo_url: str | None = None,
    event_name: str | None = None,
    event_logo_urls: list[str] | None = None,
    design: dict[str, Any] | None = None,
    formats: list[str] | None = None,
) -> dict[str, str]:
    """Render an achievement card and return ``{"png": base64, "pdf": base64}``.

    Only the requested *formats* (default ``["png"]``) are populated.
    """
    if formats is None:
        formats = ["png"]

    if not photo_bytes:
        raise ValueError("Missing profile photo")
    if card_type not in TYPES:
        raise ValueError(f'Unknown card type "{card_type}". Use one of: {", ".join(TYPES)}')

    design_spec = None
    if design is not None:
        design_spec = validate_card_design(design)
        if design_spec is None:
            raise ValueError("Invalid card design specification")
        if card_type == "talk":
            raise ValueError("Custom design overrides are not supported for talk cards")
        if design_spec["variation"] is None:
            # Generate one variation per render call. PNG and PDF from this
            # call share the same spec, while separate runs can look fresh.
            design_spec["variation"] = secrets.randbelow(len(VARIATION_ACCENTS))

    want_png = "png" in formats
    want_pdf = "pdf" in formats
    if not want_png and not want_pdf:
        raise ValueError("render_card: at least one format (png|pdf) required")

    # Resolve logo
    logo_data_url: str | None = None
    template_type = design_spec["base_template"] if design_spec else card_type
    cfg = TYPES[template_type]
    effective_logo_url = (
        (design_spec.get("logo_url") if design_spec else None)
        or logo_url
        or cfg.get("logoUrl")
    )
    if card_type != "talk" and effective_logo_url:
        try:
            logo_data_url = await _resolve_image_as_data_url(effective_logo_url)
        except Exception as exc:
            if logo_url:
                # User-supplied URL — fail loudly.
                raise ValueError(f"Couldn't fetch logo URL: {exc}") from exc
            # Preset default failed — fall back to text pill.
            log.warning(
                "Preset logo fetch failed for %s: %s (%s)",
                card_type, exc, effective_logo_url,
            )

    photo_data_url = f"data:{photo_mime};base64,{base64.b64encode(photo_bytes).decode()}"
    if card_type == "talk":
        if not event_name or not event_name.strip():
            raise ValueError("Missing event name for talk card")

        event_logo_urls = event_logo_urls or []
        if len(event_logo_urls) > 2:
            raise ValueError("Talk cards accept at most two event logo URLs")

        event_logo_data_urls: list[str] = []
        for idx, url in enumerate(event_logo_urls, start=1):
            try:
                event_logo_data_urls.append(await _resolve_image_as_data_url(url))
            except Exception as exc:
                raise ValueError(f"Couldn't fetch event logo URL #{idx}: {exc}") from exc

        page_html = _build_talk_html(
            card_type=card_type,
            name=name,
            text=text,
            photo_data_url=photo_data_url,
            event_name=event_name,
            event_logo_data_urls=event_logo_data_urls,
        )
        page_height = TALK_CARD_H
    else:
        if design_spec is None:
            page_html = _build_original_html(
                card_type=card_type,
                name=name,
                text=text,
                photo_data_url=photo_data_url,
                logo_data_url=logo_data_url,
            )
        else:
            page_html = _build_html(
                card_type=template_type,
                name=name,
                text=text,
                photo_data_url=photo_data_url,
                logo_data_url=logo_data_url,
                design=design_spec,
            )
        page_height = CARD_H

    from playwright.async_api import async_playwright  # lazy import

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )
        try:
            page = await browser.new_page(viewport={"width": CARD_W, "height": page_height})
            await page.set_content(page_html, wait_until="load", timeout=30_000)
            # Wait for Google Fonts
            await page.evaluate("() => document.fonts && document.fonts.ready")
            await page.wait_for_timeout(150)

            out: dict[str, str] = {}

            if want_png:
                png_bytes = await page.screenshot(type="png")
                out["png"] = base64.b64encode(png_bytes).decode()

            if want_pdf:
                pdf_bytes = await page.pdf(
                    width=f"{CARD_W}px",
                    height=f"{page_height}px",
                    print_background=True,
                    margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                    prefer_css_page_size=False,
                )
                out["pdf"] = base64.b64encode(pdf_bytes).decode()

            await page.close()
            return out
        finally:
            await browser.close()
