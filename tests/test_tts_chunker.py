"""
Tests for TTSChunker — where the streaming TTS cuts a reply — and its
wiring in Orchestrator._stream_and_speak.

Each piece is synthesized on its own: every sentence end cuts, a long
sentence can also cut at a comma once the piece is long enough (off by
default), and no cut may land inside a [tag] or *action*
(sanitize_for_tts strips whole spans only, so half of one would be
spoken aloud).
"""

import io

import pytest
from rich.console import Console

from src.core.interfaces.llm import LLMProvider, LLMResponse
from src.core.interfaces.tts import TTSResult
from src.core.orchestrator import Orchestrator
from src.core.text_filters import TTSChunker, sanitize_for_tts


def chunk(stream: list[str], **kwargs) -> list[str]:
    """Feed a chunk stream through a TTSChunker; return every piece."""
    c = TTSChunker(**kwargs)
    pieces = []
    for s in stream:
        pieces += c.feed(s)
    return pieces + c.flush()


def test_one_sentence_per_piece():
    assert chunk(["Hola má. ¿Cómo estás? ¡Bien!"]) == [
        "Hola má.", " ¿Cómo estás?", " ¡Bien!",
    ]


def test_boundary_waits_for_the_next_character():
    c = TTSChunker()
    assert c.feed("Hola má.") == []  # could still become "Hola má..."
    assert c.feed(" Bue") == ["Hola má."]


def test_empty_chunks_never_repeat_a_cut():
    # The old sentence counter re-counted a "." on every empty or
    # whitespace-only chunk that followed it
    assert chunk(["Primero.", "", " ", "", "\n", "Segundo."]) == [
        "Primero.", " \nSegundo.",
    ]


def test_streamed_ellipsis_is_seen_whole():
    stream = ["Bueno", ".", ".", ".", " está bien."]
    # An ellipsis is a pause, not a sentence end…
    assert chunk(stream) == ["Bueno... está bien."]
    # …and when it does cut, it cuts after all three dots
    assert chunk(stream, comma_min_chars=5) == ["Bueno...", " está bien."]


def test_question_ending_in_ellipsis_is_a_sentence_end():
    assert chunk(["¿Y entonces...? No sé."]) == ["¿Y entonces...?", " No sé."]


def test_long_sentence_cuts_at_first_comma_past_minimum():
    text = "Me acuerdo de lo que me habías dicho vos, cuando era chica, que nunca iba a estar sola."
    assert chunk([text], comma_min_chars=30) == [
        "Me acuerdo de lo que me habías dicho vos,",
        # " cuando era chica," is under 30 chars, so no cut there
        " cuando era chica, que nunca iba a estar sola.",
    ]


def test_comma_cuts_off_by_default():
    text = "Me acuerdo de lo que me habías dicho vos, cuando era chica, que nunca iba a estar sola."
    assert chunk([text]) == [text]


def test_numbers_are_not_cut():
    text = "Tenía 3,5 años y costaba 2.000 pesos, me acuerdo bien."
    assert chunk([text], comma_min_chars=1) == [
        "Tenía 3,5 años y costaba 2.000 pesos,", " me acuerdo bien.",
    ]


def test_never_cuts_inside_an_action():
    text = "Bueno *se ríe, nerviosa. Mira el piso.* Ya está, ya pasó, má."
    pieces = chunk([text], comma_min_chars=1)
    assert pieces[0] == "Bueno *se ríe, nerviosa. Mira el piso.* Ya está,"
    assert sanitize_for_tts(pieces[0]) == "Bueno Ya está,"


def test_never_cuts_inside_a_bracket_tag():
    pieces = chunk(["[risas, muchas. Fuertes] Sí, obvio."], comma_min_chars=1)
    assert pieces == ["[risas, muchas. Fuertes] Sí,", " obvio."]
    assert sanitize_for_tts(pieces[0]) == "Sí,"


def test_paragraph_break_cuts():
    assert chunk(["Hola má\n\nBueno, sí"]) == ["Hola má", "\n\nBueno, sí"]


def test_closing_quote_stays_with_its_sentence():
    assert chunk(['Me dijo "andate." Y me fui.']) == ['Me dijo "andate."', " Y me fui."]


def test_no_piece_without_a_letter():
    assert chunk(["1. Ir al parque. 2. Volver."]) == ["1. Ir al parque.", " 2. Volver."]


def test_short_reply_released_by_flush():
    # Same lesson as EmotionTagFilter's B5 bug: nothing may be held forever
    assert chunk(["Sí."]) == ["Sí."]


def test_long_run_without_punctuation_cuts_at_a_space():
    words = ["y", " entonces"] * 30
    pieces = chunk(words, max_chars=60)
    assert len(pieces) > 1
    assert all(len(p) <= 60 for p in pieces)
    assert all(p.startswith(" ") for p in pieces[1:])  # never mid-word
    assert "".join(pieces) == "".join(words)


TEXTS = [
    "Me acuerdo de lo que me habías dicho vos, cuando era chica, que nunca iba "
    "a estar sola. ¿Te acordás? Yo sí... todavía.",
    "Bueno *se ríe, nerviosa. Mira el piso.* Ya está, ya pasó todo eso, má.",
    "[risas, muchas. Fuertes] Sí, obvio. Tenía 3,5 años y costaba 2.000 pesos, me acuerdo bien.",
    'Hola má\n\nBueno, sí. Me dijo "andate." Y me fui. ',
    "y entonces " * 20 + "fin.",
]


def splits(text: str) -> list[list[str]]:
    """The same text streamed whole, char by char, and in odd-sized chunks."""
    return [
        [text],
        list(text),
        [text[i:i + 7] for i in range(0, len(text), 7)],
    ]


@pytest.mark.parametrize("comma_min_chars", [0, 30])
@pytest.mark.parametrize("text", TEXTS)
def test_stream_splitting_never_changes_the_cuts(text, comma_min_chars):
    results = [chunk(s, comma_min_chars=comma_min_chars, max_chars=60) for s in splits(text)]
    assert results[0] == results[1] == results[2]


@pytest.mark.parametrize("comma_min_chars", [0, 30])
@pytest.mark.parametrize("text", TEXTS)
def test_nothing_lost_and_no_markup_spoken(text, comma_min_chars):
    pieces = chunk(list(text), comma_min_chars=comma_min_chars, max_chars=60)
    assert "".join(pieces) == text.rstrip()
    for p in pieces:
        spoken = sanitize_for_tts(p)
        assert "*" not in spoken and "[" not in spoken and "]" not in spoken


# ─── Wiring: what reaches the TTS from _stream_and_speak ───


class StreamingLLM(LLMProvider):
    def __init__(self, chunks: list[str], text: str, emotion: str | None):
        self._chunks, self._text, self._emotion = chunks, text, emotion

    async def initialize(self, config: dict) -> None:
        pass

    async def generate(self, system_prompt, messages, temperature=0.7, max_tokens=512):
        raise AssertionError("the streaming path should be used")

    async def stream_generate_with_metadata(
        self, system_prompt, messages, temperature=0.7, max_tokens=512
    ):
        for c in self._chunks:
            yield c
        yield LLMResponse(text=self._text, emotion=self._emotion)

    async def get_model_info(self) -> dict:
        return {}


class RecordingTTS:
    def __init__(self):
        self.calls: list[tuple[str, str | None]] = []

    async def synthesize(self, text, voice_profile=None, emotion=None):
        self.calls.append((text, emotion))
        return TTSResult(audio_data=b"RIFF")


class SilentPlayback:
    @classmethod
    async def start(cls, audio_data: bytes):
        return cls()

    async def wait(self) -> None:
        pass

    def cleanup(self) -> None:
        pass


async def speak(tmp_path, monkeypatch, chunks, text, emotion=None, **tts_config) -> RecordingTTS:
    monkeypatch.setattr("src.core.orchestrator.WavPlayback", SilentPlayback)
    orch = Orchestrator({
        "system": {
            "state_file": str(tmp_path / "performance_state.json"),
            "transcript_file": str(tmp_path / "transcript.jsonl"),
        },
        "tts": {"streaming": True, **tts_config},
    })
    orch.state.start_performance()
    orch.llm = StreamingLLM(chunks, text, emotion)
    orch.tts = RecordingTTS()
    await orch._stream_and_speak("hola hija", Console(file=io.StringIO()))
    return orch.tts


# A sentence end lands mid-chunk (". ¿Te") — the old cutter only checked the
# end of the buffer and sent this whole reply as one chunk
REPLY_CHUNKS = [
    "[cont", "enta] Me acuerdo de lo que me habías dicho vos", ", cuando",
    " era chica *se ríe, ", "nerviosa*", ". ¿Te acordás", "?",
]
REPLY_TEXT = (
    "Me acuerdo de lo que me habías dicho vos, cuando era chica "
    "*se ríe, nerviosa*. ¿Te acordás?"
)


async def test_reply_reaches_tts_one_sentence_per_piece(tmp_path, monkeypatch):
    tts = await speak(tmp_path, monkeypatch, REPLY_CHUNKS, REPLY_TEXT, "contenta")
    assert tts.calls == [
        ("Me acuerdo de lo que me habías dicho vos, cuando era chica .", "contenta"),
        ("¿Te acordás?", "contenta"),
    ]


async def test_reply_reaches_tts_cut_at_comma_when_enabled(tmp_path, monkeypatch):
    tts = await speak(
        tmp_path, monkeypatch, REPLY_CHUNKS, REPLY_TEXT, "contenta",
        chunk_comma_min_chars=30,
    )
    assert tts.calls == [
        ("Me acuerdo de lo que me habías dicho vos,", "contenta"),
        ("cuando era chica .", "contenta"),
        ("¿Te acordás?", "contenta"),
    ]


async def test_short_untagged_reply_is_spoken(tmp_path, monkeypatch):
    tts = await speak(tmp_path, monkeypatch, chunks=["Sí."], text="Sí.")
    assert [t for t, _ in tts.calls] == ["Sí."]


async def test_action_only_reply_is_not_spoken(tmp_path, monkeypatch):
    tts = await speak(tmp_path, monkeypatch, chunks=["*se ríe*"], text="*se ríe*")
    assert tts.calls == []
