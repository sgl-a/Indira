from __future__ import annotations

"""
Text filters for the TTS boundary.

Qwen3-TTS (and TTS models generally) derail on non-speech tokens: probing
showed a lone emoji synthesizes as CJK babble, "[risas]" is read aloud as
English "Recess", and "*se rie*" comes out garbled. Anything the LLM emits
that isn't speakable words must be stripped before synthesis - the console
still displays the original text.
"""

import re

# Inline stage directions the LLM may emit despite the prompt rules
_BRACKET_TAG = re.compile(r"\[[^\]]*\]")
_ACTION_MARK = re.compile(r"\*[^*\n]*\*")

# Emoji / pictograph / symbol ranges (deliberately excludes Latin-1
# punctuation and accented letters)
_EMOJI = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # emoji, pictographs, extended symbols
    "\U0001FB00-\U0001FBFF"
    "\u2600-\u27BF"          # misc symbols & dingbats
    "\u2B00-\u2BFF"          # misc symbols & arrows
    "\u2190-\u21FF"          # arrows
    "\uFE0E\uFE0F\u200D"    # variation selectors & zero-width joiner
    "]+"
)


class EmotionTagFilter:
    """
    Incrementally strips the leading [emoción] tag from a token stream.

    The end-of-stream parse (ollama_provider._parse_emotion) is the
    authority for state/history; this is its real-time counterpart for the
    live display and streaming TTS, which need clean text and the emotion
    BEFORE the response is complete. Tags can arrive split across chunk
    boundaries, so early chunks are held back until the tag either closes
    or provably isn't coming.
    """

    # Give up waiting for "]" past this much text — it's dialogue, not a tag
    _MAX_TAG_LEN = 50
    # If the text doesn't start with "[" after this many chars, no tag is coming
    _MAX_PLAIN_HOLD = 5

    def __init__(self) -> None:
        self._buf = ""
        self._resolved = False
        # Set once a leading tag is parsed; stays None if no tag arrives
        self.emotion: str | None = None

    def feed(self, chunk: str) -> str:
        """Feed a raw chunk, get back the displayable text (may be "")."""
        if self._resolved:
            return chunk
        self._buf += chunk
        bracket_end = self._buf.find("]")
        if bracket_end >= 0 and self._buf.lstrip().startswith("["):
            bracket_start = self._buf.find("[")
            self.emotion = self._buf[bracket_start + 1:bracket_end].strip()
            self._resolved = True
            out = self._buf[bracket_end + 1:].lstrip()
            self._buf = ""
            return out
        if len(self._buf) > self._MAX_TAG_LEN or (
            not self._buf.lstrip().startswith("[") and len(self._buf) > self._MAX_PLAIN_HOLD
        ):
            self._resolved = True
            out, self._buf = self._buf, ""
            return out
        return ""

    def flush(self) -> str:
        """
        Release anything still held at end of stream.

        Without this, short untagged replies (e.g. "Sí.") that never trip
        the resolution heuristics would be silently dropped from the
        display and streaming TTS.
        """
        out, self._buf = self._buf, ""
        self._resolved = True
        return out


def sanitize_for_tts(text: str) -> str:
    """
    Return only the speakable words of `text` for TTS synthesis.

    Strips bracket tags, *action* markup, and emojis; collapses the
    leftover whitespace. May return "" - callers must skip synthesis then.
    """
    text = _BRACKET_TAG.sub(" ", text)
    text = _ACTION_MARK.sub(" ", text)
    text = _EMOJI.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip()
