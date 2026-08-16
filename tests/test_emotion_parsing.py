"""Tests for the emotion-tag protocol parsing in the Ollama provider."""

from src.providers.llm.ollama_provider import OllamaLLMProvider

parse = OllamaLLMProvider._parse_emotion


def test_leading_tag_is_parsed():
    emotion, text = parse("[cálida, nostálgica] Me acuerdo cuando me enseñaste esa palabra.")
    assert emotion == "cálida, nostálgica"
    assert text == "Me acuerdo cuando me enseñaste esa palabra."


def test_leading_whitespace_before_tag():
    emotion, text = parse("  \n[contenta] ¡Hola mamá!")
    assert emotion == "contenta"
    assert text == "¡Hola mamá!"


def test_mid_text_bracket_does_not_delete_dialogue():
    # Regression: the old regex matched the first bracket ANYWHERE and
    # silently dropped everything before it ("Hola mamá" disappeared).
    emotion, text = parse("Hola mamá [risas] qué bueno verte")
    assert emotion is None
    assert text == "Hola mamá [risas] qué bueno verte"


def test_leading_tag_with_later_bracket_keeps_later_bracket():
    emotion, text = parse("[divertida] Hola mamá [risas] qué bueno verte")
    assert emotion == "divertida"
    assert text == "Hola mamá [risas] qué bueno verte"


def test_no_tag_returns_full_text():
    emotion, text = parse("No sé qué decirte.")
    assert emotion is None
    assert text == "No sé qué decirte."


def test_multiline_text_after_tag():
    emotion, text = parse("[reflexiva] Primera línea.\nSegunda línea.")
    assert emotion == "reflexiva"
    assert text == "Primera línea.\nSegunda línea."


def test_tag_only_no_text():
    emotion, text = parse("[triste]")
    assert emotion == "triste"
    assert text == ""


def test_empty_input():
    emotion, text = parse("")
    assert emotion is None
    assert text == ""
