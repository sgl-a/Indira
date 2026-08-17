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
