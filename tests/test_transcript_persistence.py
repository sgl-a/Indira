"""Tests for conversation-transcript persistence (crash recovery of short-term memory)."""

import json
import time

from src.core.orchestrator import Orchestrator


def make_config(tmp_path, **system_overrides) -> dict:
    system = {
        "state_file": str(tmp_path / "performance_state.json"),
        "transcript_file": str(tmp_path / "transcript.jsonl"),
        "performance_duration_hours": 72,
    }
    system.update(system_overrides)
    return {"system": system}


def test_turns_are_appended_to_disk(tmp_path):
    orch = Orchestrator(make_config(tmp_path))
    orch._append_transcript(orch.state.add_turn("user", "hola hija"))
    orch._append_transcript(orch.state.add_turn("assistant", "¡Hola má!", emotion="contenta"))

    lines = (tmp_path / "transcript.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["content"] == "hola hija"
    assert json.loads(lines[1])["emotion"] == "contenta"


def test_resume_restores_transcript(tmp_path):
    config = make_config(tmp_path)

    # A run that crashed at hour 50, with two turns on disk
    (tmp_path / "performance_state.json").write_text(
        json.dumps({"performance_start_time": time.time() - 50 * 3600})
    )
    (tmp_path / "transcript.jsonl").write_text(
        json.dumps({"role": "user", "content": "hola hija"}) + "\n"
        + json.dumps({"role": "assistant", "content": "¡Hola má!", "emotion": "contenta"}) + "\n"
    )

    orch = Orchestrator(config)
    assert orch._start_or_resume_performance() is True
    assert len(orch.state.conversation_history) == 2
    # Emotion survives the round-trip → tag replay in get_recent_messages works
    messages = orch.state.get_recent_messages()
    assert messages[1]["content"] == "[contenta] ¡Hola má!"


def test_fresh_start_clears_transcript(tmp_path):
    config = make_config(tmp_path, fresh_start=True)
    (tmp_path / "performance_state.json").write_text(
        json.dumps({"performance_start_time": time.time() - 50 * 3600})
    )
    (tmp_path / "transcript.jsonl").write_text(
        json.dumps({"role": "user", "content": "hola hija"}) + "\n"
    )

    orch = Orchestrator(config)
    assert orch._start_or_resume_performance() is False
    assert not (tmp_path / "transcript.jsonl").exists()
    assert orch.state.conversation_history == []


def test_torn_last_line_is_skipped(tmp_path):
    config = make_config(tmp_path)
    (tmp_path / "performance_state.json").write_text(
        json.dumps({"performance_start_time": time.time() - 10 * 3600})
    )
    (tmp_path / "transcript.jsonl").write_text(
        json.dumps({"role": "user", "content": "hola hija"}) + "\n"
        + '{"role": "assistant", "content": "truncated by cra'  # torn write
    )

    orch = Orchestrator(config)
    assert orch._start_or_resume_performance() is True
    assert len(orch.state.conversation_history) == 1
    assert orch.state.conversation_history[0].content == "hola hija"


def test_missing_transcript_resumes_empty(tmp_path):
    config = make_config(tmp_path)
    (tmp_path / "performance_state.json").write_text(
        json.dumps({"performance_start_time": time.time() - 10 * 3600})
    )

    orch = Orchestrator(config)
    assert orch._start_or_resume_performance() is True
    assert orch.state.conversation_history == []
