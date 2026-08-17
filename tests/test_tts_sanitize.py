"""
Tests for sanitize_for_tts — the TTS input boundary.

Non-speech tokens derail Qwen3-TTS (verified by closed-loop probing:
a lone emoji synthesizes as CJK babble, "[risas]" is read aloud as
English "Recess", "*se ríe*" comes out garbled). Only speakable words
may reach the synthesizer; the console still shows the original text.
"""

from src.core.text_filters import sanitize_for_tts


async def test_plain_spanish_untouched():
    text = "¿Sabés qué? ¡Ñoquis! Bueno... está bien, mamá."
    assert sanitize_for_tts(text) == text


async def test_emoji_stripped_inline():
    assert sanitize_for_tts("Sí, obvio 😂 ¿vos qué pensás?") == "Sí, obvio ¿vos qué pensás?"


async def test_emoji_only_returns_empty():
    assert sanitize_for_tts("😂") == ""
    assert sanitize_for_tts("🎭🎂✨") == ""


async def test_bracket_tag_stripped():
    assert sanitize_for_tts("Hola mamá [risas] qué bueno verte.") == "Hola mamá qué bueno verte."
    assert sanitize_for_tts("[risas]") == ""


async def test_action_markup_stripped():
    assert sanitize_for_tts("*se ríe* No te creo nada.") == "No te creo nada."


async def test_variation_selector_and_zwj_stripped():
    # "woman shrugging" ZWJ sequence + heart with variation selector
    assert sanitize_for_tts("Bueno 🤷‍♀️ dale ❤️.") == "Bueno dale ."


async def test_whitespace_collapsed():
    assert sanitize_for_tts("Hola   [tag]   mamá.") == "Hola mamá."
