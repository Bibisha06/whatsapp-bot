"""Card generation feature.

Commands:
    !card <type> | <name> | <text>       — generate a PNG card (attach photo)
    !card-pdf <type> | <name> | <text>   — generate PNG + editable PDF

For the ``talk`` type, ``name`` is the speaker, ``text`` is the talk title,
and the fourth part is the event name. Optional fifth/sixth parts are event
logo URLs.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from typing import TYPE_CHECKING

from neonize.events import MessageEv

if TYPE_CHECKING:
    from neonize.client import NewClient

log = logging.getLogger(__name__)

# A native WhatsApp mention at the beginning of a caption is rendered as
# ``@me`` for the sender.  It is commonly used to wake the NLP layer, but a
# direct command must accept it too.
_LEADING_MENTION_COMMAND_RE = re.compile(
    r"^\s*@\S+\s+(?=!card(?:-pdf)?\b)", re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_text(message: MessageEv) -> str:
    """Extract text body from a message (handles both plain, extended, and image captions)."""
    text = message.Message.conversation or ""
    if message.Message.extendedTextMessage and message.Message.extendedTextMessage.text:
        text = message.Message.extendedTextMessage.text
    elif message.Message.imageMessage and message.Message.imageMessage.caption:
        text = message.Message.imageMessage.caption
    return text.strip()


def _get_command_text(message: MessageEv) -> str:
    """Return a caption/body with one leading bot mention removed."""
    return _LEADING_MENTION_COMMAND_RE.sub("", _get_text(message)).strip()


def _has_image(message: MessageEv) -> bool:
    """Check if the message has an attached image."""
    return bool(message.Message.imageMessage and message.Message.imageMessage.URL)


def _parse_talk_logo_urls(raw_parts: list[str]) -> list[str]:
    """Parse up to two talk logo URLs from pipe fields or comma-separated text."""
    urls: list[str] = []
    for raw_part in raw_parts:
        for item in raw_part.split(","):
            item = item.strip()
            if item:
                urls.append(item)
    return urls


async def _handle_card_command(
    client: "NewClient",
    message: MessageEv,
    cmd_prefix: str,
    *,
    with_pdf: bool,
) -> None:
    from cards import render_card, CARD_TYPES
    from features.help import MODULE_HELP

    body = _get_command_text(message)
    chat_jid = message.Info.MessageSource.Chat
    rest = body[len(cmd_prefix):].strip()
    design = getattr(message, "_pbbot_card_design", None)

    # Split on newlines or pipes
    if "\n" in rest:
        parts = [s.strip() for s in rest.split("\n")]
    else:
        parts = [s.strip() for s in rest.split("|")]

    raw_type = parts[0] if len(parts) > 0 else ""
    name = parts[1] if len(parts) > 1 else ""
    text = parts[2] if len(parts) > 2 else ""
    card_type = raw_type.lower()
    if card_type == "talk":
        logo_url = None
        event_name = parts[3] if len(parts) > 3 else ""
        event_logo_urls = _parse_talk_logo_urls(parts[4:])
    else:
        logo_url = parts[3] if len(parts) > 3 else None
        event_name = None
        event_logo_urls = None

    if not raw_type or not name or not text:
        client.send_message(chat_jid, MODULE_HELP["cards"])
        return

    if card_type not in CARD_TYPES:
        client.send_message(
            chat_jid,
            f'⚠️ Unknown card type "{raw_type}". Use one of: {", ".join(CARD_TYPES)}\n\n'
            "See `!help card` for details.",
        )
        return

    if card_type == "talk":
        if not event_name:
            client.send_message(
                chat_jid,
                "⚠️ Talk cards require an event name as the 4th field.\n\n"
                "Usage: `!card talk | <speaker> | <talk title> | <event name> | <logoUrl1> | <logoUrl2>`",
            )
            return

        if len(event_logo_urls) > 2:
            client.send_message(chat_jid, "⚠️ Talk cards support at most two event logo URLs.")
            return

    if not _has_image(message):
        client.send_message(
            chat_jid,
            f"⚠️ Attach a profile photo to the same message as the `{cmd_prefix}` command.\n\n"
            "See `!help card` for details.",
        )
        return

    try:
        fmt_label = "PNG + PDF" if with_pdf else "PNG"
        client.send_message(chat_jid, f"🎨 Rendering {card_type} card for {name} ({fmt_label})...")

        # Download attached image
        photo_bytes = client.download_any(message.Message)
        if not photo_bytes:
            client.send_message(chat_jid, "❌ Couldn't download the attached media.")
            return

        photo_mime = message.Message.imageMessage.mimetype or "image/jpeg"
        formats = ["png", "pdf"] if with_pdf else ["png"]

        out = await render_card(
            card_type=card_type,
            name=name,
            text=text,
            photo_bytes=photo_bytes,
            photo_mime=photo_mime,
            logo_url=logo_url,
            event_name=event_name,
            event_logo_urls=event_logo_urls,
            design=design,
            formats=formats,
        )

        safe_name = re.sub(r"[^a-zA-Z0-9\-_]+", "-", name)[:60] or "card"

        if out.get("png"):
            png_bytes = base64.b64decode(out["png"])
            png_msg = client.build_image_message(
                png_bytes,
                caption=f"🎉 {name}",
            )
            client.send_message(chat_jid, png_msg)

        if out.get("pdf"):
            pdf_bytes = base64.b64decode(out["pdf"])
            doc_msg = client.build_document_message(
                pdf_bytes,
                filename=f"{safe_name}-card.pdf",
                caption=f"📄 {name} — editable PDF",
                mimetype="application/pdf",
            )
            client.send_message(chat_jid, doc_msg)

        log.info(
            "Card rendered: type=%s template=%s name=%r formats=%s",
            card_type,
            (design or {}).get("base_template", card_type),
            name,
            "+".join(formats),
        )

    except Exception as exc:
        log.error("Card render error: %s", exc)
        client.send_message(chat_jid, f"❌ Card render failed: {exc}")


# ---------------------------------------------------------------------------
# Feature registration
# ---------------------------------------------------------------------------


def register(client: "NewClient", config: dict) -> callable:
    """Register the card generation feature on the neonize client."""

    def on_message(client: "NewClient", message: MessageEv):
        body = _get_command_text(message)
        lower = body.lower()
        chat_jid = str(message.Info.MessageSource.Chat)

        # Card commands can come from any registered group (media or CTF)
        if lower.startswith("!card-pdf") and (lower == "!card-pdf" or lower[9:10] in (" ", "\n")):
            try:
                asyncio.run(
                    _handle_card_command(client, message, "!card-pdf", with_pdf=True)
                )
            except Exception as exc:
                log.error("Card-pdf command error: %s", exc, exc_info=True)
            return

        if lower.startswith("!card") and (lower == "!card" or lower[5:6] in (" ", "\n")):
            try:
                asyncio.run(
                    _handle_card_command(client, message, "!card", with_pdf=False)
                )
            except Exception as exc:
                log.error("Card command error: %s", exc, exc_info=True)
            return

    log.info("✅ Card generation feature registered")
    return on_message
