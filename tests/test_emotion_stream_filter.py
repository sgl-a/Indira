"""
Tests for EmotionTagFilter — the real-time counterpart of the provider's
end-of-stream _parse_emotion. Strips the leading [emoción] tag from a
chunk stream for the live display and streaming TTS; both parsers must
agree on the protocol (tag at the start = metadata, brackets later =
dialogue).
"""

from src.core.text_filters import EmotionTagFilter


def collect(chunks: list[str]) -> tuple[str, str | None]:
    """Run chunks through a filter; return (displayed_text, emotion)."""
    f = EmotionTagFilter()
    out = "".join(f.feed(c) for c in chunks)
    out += f.flush()
    return out, f.emotion


def test_tag_in_single_chunk():
    text, emotion = collect(["[contenta] Hola má, ¿sabés qué?"])
    assert text == "Hola má, ¿sabés qué?"
    assert emotion == "contenta"


def test_tag_split_across_chunks():
    text, emotion = collect(["[cá", "lida, nost", "álgica] Me ac", "uerdo de eso."])
    assert text == "Me acuerdo de eso."
    assert emotion == "cálida, nostálgica"


def test_chunks_pass_through_after_resolution():
    f = EmotionTagFilter()
    assert f.feed("[triste] ") == ""
    assert f.feed("No sé, má.") == "No sé, má."  # verbatim, no buffering
    assert f.emotion == "triste"


def test_untagged_reply_passes_through():
    text, emotion = collect(["Hola má, ", "¿cómo estás?"])
    assert text == "Hola má, ¿cómo estás?"
    assert emotion is None


def test_short_untagged_reply_released_by_flush():
    # "Sí." never trips the resolution heuristics — without flush() it
    # would be silently dropped from display and TTS (the bug B5 fixed)
    text, emotion = collect(["Sí."])
    assert text == "Sí."
    assert emotion is None


def test_unclosed_bracket_becomes_dialogue():
    long_bracket = "[" + "palabras que no cierran nunca " * 3
    text, emotion = collect([long_bracket])
    assert text == long_bracket
    assert emotion is None


def test_inline_brackets_are_dialogue_not_tags():
    text, emotion = collect(["Hola má, qué bueno ", "[risas] verte."])
    assert text == "Hola má, qué bueno [risas] verte."
    assert emotion is None


def test_leading_whitespace_before_tag():
    text, emotion = collect(["  [seria] Tenemos que hablar."])
    assert text == "Tenemos que hablar."
    assert emotion == "seria"
