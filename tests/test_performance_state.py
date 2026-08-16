"""Tests for performance-state persistence (crash recovery of the 72h run)."""

import json
import time

from src.core.orchestrator import Orchestrator


def make_config(tmp_path, **system_overrides) -> dict:
    system = {
        "state_file": str(tmp_path / "performance_state.json"),
        "performance_duration_hours": 72,
    }
    system.update(system_overrides)
    return {"system": system}


def test_fresh_start_writes_state_file(tmp_path):
    orch = Orchestrator(make_config(tmp_path))
    resumed = orch._start_or_resume_performance()

    assert resumed is False
    data = json.loads((tmp_path / "performance_state.json").read_text())
    assert data["performance_start_time"] == orch.state.performance_start_time


def test_restart_resumes_saved_hour(tmp_path):
    config = make_config(tmp_path)

    # Simulate a run that started 50 hours ago, then crashed
    ten_to_seventy_hours_ago = time.time() - 50 * 3600
    (tmp_path / "performance_state.json").write_text(
        json.dumps({"performance_start_time": ten_to_seventy_hours_ago})
    )

    orch = Orchestrator(config)
    resumed = orch._start_or_resume_performance()

    assert resumed is True
    assert abs(orch.state.hours_elapsed - 50) < 0.01
    # Age stage must reflect hour 50, not hour 0 (stage 40-50 spans hours 45-54)
    assert orch.state.current_age_stage == "40-50"


def test_expired_state_starts_fresh(tmp_path):
    config = make_config(tmp_path)
    (tmp_path / "performance_state.json").write_text(
        json.dumps({"performance_start_time": time.time() - 100 * 3600})
    )

    orch = Orchestrator(config)
    resumed = orch._start_or_resume_performance()

    assert resumed is False
    assert orch.state.hours_elapsed < 0.01


def test_fresh_flag_ignores_saved_state(tmp_path):
    config = make_config(tmp_path, fresh_start=True)
    (tmp_path / "performance_state.json").write_text(
        json.dumps({"performance_start_time": time.time() - 50 * 3600})
    )

    orch = Orchestrator(config)
    resumed = orch._start_or_resume_performance()

    assert resumed is False
    assert orch.state.hours_elapsed < 0.01


def test_corrupt_state_file_starts_fresh(tmp_path):
    config = make_config(tmp_path)
    (tmp_path / "performance_state.json").write_text("not json{{{")

    orch = Orchestrator(config)
    resumed = orch._start_or_resume_performance()

    assert resumed is False
    assert orch.state.performance_start_time is not None
