from __future__ import annotations

"""
Console logging helpers.

The conversation loop streams a response to stdout one chunk at a time,
leaving the line open. Any log record emitted in that window (TTS chunks
synthesized by the background consumer, age transitions, provider warnings)
lands in the middle of Indira's sentence and makes the console unreadable.

`defer_console_logs()` holds records back for the duration of a turn;
`flush_console_logs()` releases them once the line is closed.
"""

import logging


class DeferrableHandler(logging.Handler):
    """Wraps a handler so its output can be held back and flushed later."""

    def __init__(self, inner: logging.Handler):
        super().__init__()
        self._inner = inner
        self._deferred = False
        self._buffer: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        if self._deferred:
            self._buffer.append(record)
        else:
            self._inner.handle(record)

    def pause(self) -> None:
        self._deferred = True

    def flush_deferred(self) -> None:
        self._deferred = False
        buffered, self._buffer = self._buffer, []
        for record in buffered:
            self._inner.handle(record)


_console_handler: DeferrableHandler | None = None


def set_console_handler(handler: DeferrableHandler) -> None:
    """Register the handler that the defer/flush helpers control."""
    global _console_handler
    _console_handler = handler


def defer_console_logs() -> None:
    """Hold console log output back (while streaming text to stdout)."""
    if _console_handler is not None:
        _console_handler.pause()


def flush_console_logs() -> None:
    """Release any log output held back by `defer_console_logs()`."""
    if _console_handler is not None:
        _console_handler.flush_deferred()
