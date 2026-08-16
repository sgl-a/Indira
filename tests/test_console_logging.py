"""Tests for deferred console logging.

Regression: with streaming TTS on, the background consumer logs while the
response line is still open, so log records landed mid-sentence:

    Indira (age 10-15): ¡Bien también! Estoy acá.    INFO  Qwen3-TTS generate...
     ¿Hiciste algo divertido hoy?
"""

import logging

from src.core.logging_utils import (
    DeferrableHandler,
    defer_console_logs,
    flush_console_logs,
    set_console_handler,
)


class RecordingHandler(logging.Handler):
    """Stand-in for RichHandler that just records what it was asked to write."""

    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def make_logger(name: str) -> tuple[logging.Logger, RecordingHandler]:
    inner = RecordingHandler()
    handler = DeferrableHandler(inner)
    set_console_handler(handler)

    logger = logging.getLogger(name)
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger, inner


def test_logs_pass_through_when_not_deferred():
    logger, inner = make_logger("test.passthrough")
    logger.info("hola")
    assert inner.messages == ["hola"]


def test_logs_are_held_back_while_deferred():
    logger, inner = make_logger("test.deferred")

    defer_console_logs()
    logger.info("TTS chunk 1")
    logger.info("TTS chunk 2")
    assert inner.messages == []  # nothing cut into the streamed line

    flush_console_logs()
    assert inner.messages == ["TTS chunk 1", "TTS chunk 2"]


def test_flush_restores_live_logging():
    logger, inner = make_logger("test.restore")

    defer_console_logs()
    flush_console_logs()
    logger.info("after")
    assert inner.messages == ["after"]


def test_defer_is_safe_without_a_registered_handler():
    set_console_handler(None)  # type: ignore[arg-type]
    defer_console_logs()
    flush_console_logs()
