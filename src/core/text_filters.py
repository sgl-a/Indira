from __future__ import annotations

"""
Text filters for the TTS boundary.

Qwen3-TTS (and TTS models generally) derail on non-speech tokens: probing
showed a lone emoji synthesizes as CJK babble, "[risas]" is read aloud as
English "Recess", and "*se rie*" comes out garbled. Anything the LLM emits
that isn't speakable words must be stripped before synthesis - the console
still displays the original text. TTSChunker decides where the streaming
TTS cuts the reply, and never cuts through a span the stripping removes.
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

# Where TTSChunker may cut. A punctuation boundary only counts once the
# character after it has arrived and is whitespace: that keeps "3,5" and
# "2.000" whole, and an ellipsis streamed as "." "." "." is seen complete
# before it's classified. Closing quotes/parens stay with the punctuation
# they follow. A newline (paragraph break) cuts before itself.
_CLOSERS = "\"'\u201D\u2019\u00BB)"
_CHUNK_BOUNDARY = re.compile(r"(?:[.!?\u2026]+|[,;:])[\"'\u201D\u2019\u00BB)]*(?=\s)|\n")
_LETTER = re.compile(r"[^\W\d_]")
_WHITESPACE = re.compile(r"\s")


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


class TTSChunker:
    """
    Cuts the streamed reply into the pieces the streaming TTS synthesizes.

    Each piece is synthesized on its own, so the cut points trade latency
    for seams: nothing plays until the first piece is complete, and every
    cut resets the voice's intonation. The rules:

    - every sentence end (. ? ! or a paragraph break) cuts: one sentence
      per piece at most;
    - a pause (, ; : or an ellipsis) cuts only once the piece has reached
      `comma_min_chars` (0 = never, the default). This gets a long first
      sentence speaking early, but Qwen3-TTS sometimes keeps "speaking"
      past a chunk that ends mid-sentence (~3% in probing, 2026-10-15,
      vs 0% at sentence ends) — off until that's solved;
    - a piece that reaches `max_chars` (0 = no limit) with neither is cut
      at the last space within the limit;
    - never inside a [tag] or *action*: sanitize_for_tts strips whole
      spans only, so half of one would be spoken aloud;
    - never a piece without a letter (a list's "1.", a stray "...").

    Pieces come back verbatim, whitespace and markup included, and nothing
    is dropped but trailing whitespace: callers sanitize each piece and
    skip it when that returns "".
    """

    def __init__(self, comma_min_chars: int = 0, max_chars: int = 200) -> None:
        self._comma_min_chars = comma_min_chars or 0
        self._max_chars = max_chars or 0
        self._buf = ""

    def feed(self, text: str) -> list[str]:
        """Feed streamed text, get back the pieces it completed (may be [])."""
        self._buf += text
        pieces = []
        while (cut := self._next_cut()) is not None:
            pieces.append(self._buf[:cut])
            self._buf = self._buf[cut:]
        return pieces

    def flush(self) -> list[str]:
        """End of stream: release whatever is left as the last piece."""
        rest, self._buf = self._buf, ""
        return [rest] if rest.strip() else []

    def _next_cut(self) -> int | None:
        """Position of the first allowed cut in the buffer, or None."""
        buf = self._buf
        boundary = None
        for m in _CHUNK_BOUNDARY.finditer(buf):
            if m.group() == "\n":
                cut, pause = m.start(), False
            else:
                cut, pause = m.end(), self._is_pause(m.group().rstrip(_CLOSERS))
            if pause and (not self._comma_min_chars or cut < self._comma_min_chars):
                continue
            if self._can_cut(buf[:cut]):
                boundary = cut
                break

        if boundary is not None and (not self._max_chars or boundary <= self._max_chars):
            return boundary
        if self._max_chars and len(buf) > self._max_chars:
            # No boundary in time: cut at the last space within the limit
            spaces = [m.start() for m in _WHITESPACE.finditer(buf, 0, self._max_chars + 1)]
            for pos in reversed(spaces):
                if self._can_cut(buf[:pos]):
                    return pos
        # Either nothing to cut yet, or nothing cuttable within the limit
        # (a run of markup): then the next boundary, however far
        return boundary

    @staticmethod
    def _is_pause(punct: str) -> bool:
        """, ; : and ellipses are pauses; . ? ! end the sentence."""
        if punct[-1] in "?!":
            return False
        return punct[-1] in ",;:…" or ".." in punct

    @staticmethod
    def _can_cut(piece: str) -> bool:
        """A piece may end here: it has a word, and no span is left open."""
        if not _LETTER.search(piece):
            return False
        # Inside an unclosed [tag]?
        if piece.rfind("[") > piece.rfind("]"):
            return False
        # Inside an *action*? (sanitize's spans never cross a newline)
        return piece[piece.rfind("\n") + 1:].count("*") % 2 == 0


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
