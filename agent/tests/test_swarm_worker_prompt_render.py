"""Regression test: system-prompt template variables must be substituted.

Root cause of the swarm target-substitution bug (observed: investment_committee
on ANET silently produced NVDA reports): ``build_worker_prompt`` only spliced
``{upstream_context}`` into the agent's system prompt, leaving ``{target}`` /
``{market}`` as literal placeholders. The worker's primary role instruction
therefore never named the actual security and it fell back to a bundled example
ticker. This pins the fix: user_vars are rendered into the system prompt.
"""

from __future__ import annotations

from src.swarm.models import SwarmAgentSpec
from src.swarm.worker import build_worker_prompt


def _spec(target_placeholder: bool = True) -> SwarmAgentSpec:
    prompt = (
        "You are a bull researcher. Build the case for {target} in market {market}."
        if target_placeholder
        else "You are a bull researcher with no placeholders."
    )
    return SwarmAgentSpec(
        id="bull_advocate",
        role="Bull-side Researcher",
        system_prompt=prompt,
        tools=["get_market_data"],
        skills=[],
    )


def test_user_vars_substituted_into_system_prompt():
    spec = _spec()
    prompt = build_worker_prompt(
        spec,
        {},
        "(no matching skills)",
        user_vars={"target": "ANET.US", "market": "US"},
    )
    assert "ANET.US" in prompt
    assert "market US" in prompt
    assert "{target}" not in prompt
    assert "{market}" not in prompt


def test_missing_user_vars_become_inference_hint_not_literal():
    # No user_vars supplied: must not leave a literal {target} (which caused the
    # NVDA fallback); instead the LLM is told to infer the value.
    spec = _spec()
    prompt = build_worker_prompt(spec, {}, "(no matching skills)", user_vars=None)
    assert "{target}" not in prompt
    assert "determine the appropriate target" in prompt


def test_no_placeholders_is_unchanged():
    spec = _spec(target_placeholder=False)
    prompt = build_worker_prompt(spec, {}, "(no matching skills)")
    assert "no placeholders" in prompt
