"""Research-only tool profile: server-side binding for console sessions.

A session created with ``tool_profile="research_only"`` must be bound to
that profile at creation, in a server-reserved field (not the
client-writable config), and every registry built for it must exclude all
non-read-only tools. Sessions created without the marker keep the full
historical registry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.session.events import EventBus
from src.session.models import Session
from src.session.service import SessionService
from src.session.store import SessionStore
from src.tools import (
    READ_ONLY_PROFILE_HARD_EXCLUSIONS,
    build_registry,
)

#: Tools a research-only session must never be able to call, by category.
FORBIDDEN_IN_RESEARCH_ONLY = {
    # file writes / edits
    "write_file", "edit_file",
    # shell / code execution, incl. backtest subprocess
    "bash", "background_run", "cancel_background", "backtest",
    "generate_backtest_config", "scaffold_signal_engine",
    # order placement / cancellation
    "trading_place_order", "trading_cancel_order",
    "etoro_close_position", "etoro_cancel_close_order",
    # swarm + skill mutation
    "run_swarm", "save_skill", "patch_skill", "delete_skill", "skill_file",
    # persistent-state mutation
    "remember", "run_research_autopilot",
    "start_research_goal", "add_goal_evidence", "update_research_goal_status",
    # previously mis-classified (inherited is_readonly=True): these persist
    # profiles, execute backtests, or write report/analysis files to disk.
    "extract_shadow_strategy", "run_shadow_backtest", "render_shadow_report",
    "factor_analysis",
}


class _DummyIndex:
    def index_session(self, session_id: str, title: str) -> None:
        del session_id, title

    def index_message(self, session_id: str, role: str, content: str) -> None:
        del session_id, role, content


def _service(tmp_path: Path, monkeypatch) -> SessionService:
    monkeypatch.setattr("src.session.service.get_shared_index", lambda: _DummyIndex())
    return SessionService(
        store=SessionStore(tmp_path / "sessions"),
        event_bus=EventBus(),
        runs_dir=tmp_path / "runs",
    )


# ---------------------------------------------------------------------------
# Registry filtering
# ---------------------------------------------------------------------------


def test_research_only_registry_has_no_write_exec_order_swarm_or_skill_tools():
    registry = build_registry(tool_profile="research_only")
    names = set(registry.tool_names)

    assert names, "research-only registry should still offer read tools"
    assert not (names & FORBIDDEN_IN_RESEARCH_ONLY)
    assert not (names & READ_ONLY_PROFILE_HARD_EXCLUSIONS)
    for name in names:
        tool = registry.get(name)
        assert tool is not None and tool.is_readonly is True, name
    # Representative read tools survive.
    assert "get_market_data" in names
    assert "read_file" in names
    assert "load_skill" in names


#: Shell tools are excluded from EVERY default registry by the pre-existing
#: shell opt-in gate, so they are not part of the default-registry subset
#: assertions; they get their own opt-in test below.
SHELL_TOOLS = {"bash", "background_run", "cancel_background"}
NON_SHELL_FORBIDDEN = FORBIDDEN_IN_RESEARCH_ONLY - SHELL_TOOLS


def test_unmarked_registry_is_unchanged():
    full = set(build_registry().tool_names)
    profiled = set(build_registry(tool_profile=None).tool_names)

    assert full == profiled
    # The dangerous tools are all still present for unmarked sessions.
    assert NON_SHELL_FORBIDDEN <= full


def test_research_only_wins_over_shell_opt_in():
    """Even a caller that opts into shell tools gets none in a bound session."""
    marked = set(build_registry(include_shell_tools=True, tool_profile="research_only").tool_names)
    unmarked = set(build_registry(include_shell_tools=True).tool_names)

    assert SHELL_TOOLS <= unmarked
    assert not (marked & SHELL_TOOLS)
    assert not (marked & FORBIDDEN_IN_RESEARCH_ONLY)


def test_unknown_tool_profile_is_rejected():
    with pytest.raises(ValueError):
        build_registry(tool_profile="full-access-please")


def test_misclassified_tools_now_declare_not_readonly():
    """Tools that write to disk / execute must not claim is_readonly."""
    from src.tools.factor_analysis_tool import FactorAnalysisTool
    from src.tools.shadow_account_tool import (
        ExtractShadowStrategyTool,
        RenderShadowReportTool,
        RunShadowBacktestTool,
    )

    assert FactorAnalysisTool.is_readonly is False
    assert ExtractShadowStrategyTool.is_readonly is False
    assert RunShadowBacktestTool.is_readonly is False
    assert RenderShadowReportTool.is_readonly is False


# ---------------------------------------------------------------------------
# Session binding
# ---------------------------------------------------------------------------


def test_marked_session_binds_profile_outside_config(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    session = service.create_session(
        title="console",
        config={"tool_profile": None, "other": 1},
        tool_profile="research_only",
    )

    assert session.tool_profile == "research_only"
    # The binding is not in the client-writable config dict.
    assert "tool_profile" not in session.config
    stored = service.store.get_session(session.session_id)
    assert stored is not None and stored.tool_profile == "research_only"
    assert "tool_profile" not in stored.config


def test_config_smuggled_profile_does_not_bind(tmp_path, monkeypatch):
    """Passing the marker only inside config must NOT create the binding."""
    service = _service(tmp_path, monkeypatch)
    session = service.create_session(
        title="sneaky", config={"tool_profile": "research_only"}
    )
    assert session.tool_profile is None
    stored = service.store.get_session(session.session_id)
    assert stored is not None and stored.tool_profile is None


def test_unmarked_session_has_no_profile(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    session = service.create_session(title="direct")
    assert session.tool_profile is None


def test_profile_cannot_be_removed_or_changed_by_update(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    session = service.create_session(title="console", tool_profile="research_only")

    # A stale / hand-built object without the profile tries to overwrite.
    tampered = Session(
        session_id=session.session_id, title="renamed", tool_profile=None
    )
    service.store.update_session(tampered)

    stored = service.store.get_session(session.session_id)
    assert stored is not None
    assert stored.tool_profile == "research_only"
    assert stored.title == "renamed"  # the legitimate update still lands


def test_unknown_profile_rejected_at_creation(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        service.create_session(title="bad", tool_profile="admin")


def test_effective_tool_names_reflect_binding(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch)
    marked = service.create_session(title="console", tool_profile="research_only")
    unmarked = service.create_session(title="direct")

    marked_tools = service.effective_tool_names(marked.session_id)
    unmarked_tools = service.effective_tool_names(unmarked.session_id)

    assert marked_tools is not None and unmarked_tools is not None
    assert marked_tools["tool_profile"] == "research_only"
    assert unmarked_tools["tool_profile"] is None
    assert not (set(marked_tools["tools"]) & FORBIDDEN_IN_RESEARCH_ONLY)
    assert NON_SHELL_FORBIDDEN <= set(unmarked_tools["tools"])
    assert set(marked_tools["tools"]) < set(unmarked_tools["tools"])
