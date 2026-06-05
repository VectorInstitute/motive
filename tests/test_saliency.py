"""Tests for motive.saliency — all pure logic, no network calls."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from motive._types import Segment
from motive.saliency import (
    SaliencyEngine,
    SaliencyScore,
    _extract_tool_name,
    _logprob_for_tool,
    _mask_messages,
    _normalise,
    _split_sentences,
)


# ---------------------------------------------------------------------------
# Helpers to build fake OpenAI responses
# ---------------------------------------------------------------------------


def _tool_response(tool_name: str, logprobs: list[dict] | None = None) -> MagicMock:
    """Build a minimal fake chat completion response with a tool call."""
    tool_call = MagicMock()
    tool_call.function.name = tool_name

    msg = MagicMock()
    msg.tool_calls = [tool_call]
    msg.content = None

    choice = MagicMock()
    choice.message = msg

    if logprobs is not None:
        lp_content = []
        for entry in logprobs:
            td = MagicMock()
            td.token = entry["token"]
            td.logprob = entry["logprob"]
            td.top_logprobs = []
            lp_content.append(td)
        lp = MagicMock()
        lp.content = lp_content
        choice.logprobs = lp
    else:
        choice.logprobs = None

    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _text_response(text: str) -> MagicMock:
    """Build a response with no tool call (text only)."""
    msg = MagicMock()
    msg.tool_calls = None
    msg.content = text

    choice = MagicMock()
    choice.message = msg
    choice.logprobs = None

    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_engine(n_samples: int = 1, top_k: int = 2) -> tuple[SaliencyEngine, AsyncMock]:
    """Return an engine wired to a mock OpenAI client."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return SaliencyEngine(client=client, model="test-model", top_k=top_k, n_samples=n_samples), client


def _sample_segments() -> list[Segment]:
    return [
        Segment(id="system", content="You are a support agent. Escalate urgent issues.", label="System prompt"),
        Segment(id="user_msg", content="My account is locked for 3 days.", label="User message"),
        Segment(id="doc_1", content="Locked accounts require human review.", label="Escalation policy"),
    ]


def _sample_messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are a support agent. Escalate urgent issues."},
        {"role": "user", "content": "My account is locked for 3 days. Locked accounts require human review."},
    ]


def _sample_tools() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "escalate_to_human",
                "description": "Escalate.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_reminder",
                "description": "Send reminder.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]


# ---------------------------------------------------------------------------
# _extract_tool_name
# ---------------------------------------------------------------------------


class TestExtractToolName:
    def test_returns_tool_name_from_tool_call(self) -> None:
        resp = _tool_response("escalate_to_human")
        assert _extract_tool_name(resp) == "escalate_to_human"

    def test_falls_back_to_content_when_no_tool_calls(self) -> None:
        resp = _text_response("send_reminder")
        assert _extract_tool_name(resp) == "send_reminder"

    def test_empty_content_returns_empty_string(self) -> None:
        msg = MagicMock()
        msg.tool_calls = None
        msg.content = None
        resp = MagicMock()
        resp.choices = [MagicMock(message=msg)]
        assert _extract_tool_name(resp) == ""


# ---------------------------------------------------------------------------
# _logprob_for_tool
# ---------------------------------------------------------------------------


class TestLogprobForTool:
    def test_single_token_match(self) -> None:
        logprobs = [{"token": "escalate_to_human", "logprob": -0.5}]
        resp = _tool_response("escalate_to_human", logprobs)
        result = _logprob_for_tool(resp, "escalate_to_human")
        assert result == pytest.approx(-0.5)

    def test_multi_token_match_sums_logprobs(self) -> None:
        logprobs = [
            {"token": "<function=", "logprob": -0.01},
            {"token": "escal", "logprob": -0.1},
            {"token": "ate", "logprob": -0.2},
            {"token": "_to", "logprob": -0.3},
            {"token": "_human", "logprob": -0.4},
            {"token": ">", "logprob": -0.01},
        ]
        resp = _tool_response("escalate_to_human", logprobs)
        result = _logprob_for_tool(resp, "escalate_to_human")
        assert result == pytest.approx(-0.1 + -0.2 + -0.3 + -0.4)

    def test_returns_none_when_no_logprobs(self) -> None:
        resp = _tool_response("tool_a")
        assert _logprob_for_tool(resp, "tool_a") is None

    def test_returns_none_when_tool_name_not_in_tokens(self) -> None:
        logprobs = [{"token": "something_else", "logprob": -1.0}]
        resp = _tool_response("tool_a", logprobs)
        assert _logprob_for_tool(resp, "tool_a") is None

    def test_returns_none_when_logprob_content_empty(self) -> None:
        resp = _tool_response("tool_a", logprobs=[])
        assert _logprob_for_tool(resp, "tool_a") is None

    def test_partial_overlap_tokens_included(self) -> None:
        # Token spans: "too" covers start of "tool", "_x" covers end
        logprobs = [
            {"token": "too", "logprob": -0.3},
            {"token": "l_x", "logprob": -0.4},
        ]
        resp = _tool_response("tool_x", logprobs)
        result = _logprob_for_tool(resp, "tool_x")
        assert result == pytest.approx(-0.3 + -0.4)


# ---------------------------------------------------------------------------
# _mask_messages
# ---------------------------------------------------------------------------


class TestMaskMessages:
    def _segments(self) -> list[Segment]:
        return [
            Segment(id="system", content="You are a helpful agent."),
            Segment(id="user", content="My account is locked."),
            Segment(id="doc_1", content="Locked accounts require human review."),
        ]

    def test_masks_target_segment_content(self) -> None:
        segments = self._segments()
        messages = [
            {"role": "system", "content": "You are a helpful agent."},
            {"role": "user", "content": "My account is locked."},
        ]
        masked = _mask_messages(messages, segments, mask_idx=0)
        assert "[CONTENT REDACTED]" in masked[0]["content"]
        assert "You are a helpful agent." not in masked[0]["content"]

    def test_leaves_other_messages_unchanged(self) -> None:
        segments = self._segments()
        messages = [
            {"role": "system", "content": "You are a helpful agent."},
            {"role": "user", "content": "My account is locked."},
        ]
        masked = _mask_messages(messages, segments, mask_idx=0)
        assert masked[1]["content"] == "My account is locked."

    def test_masks_correct_index(self) -> None:
        segments = self._segments()
        messages = [
            {"role": "user", "content": "My account is locked. Locked accounts require human review."},
        ]
        masked = _mask_messages(messages, segments, mask_idx=2)
        assert "Locked accounts require human review." not in masked[0]["content"]
        assert "My account is locked." in masked[0]["content"]

    def test_no_match_leaves_messages_unchanged(self) -> None:
        segments = self._segments()
        messages = [{"role": "user", "content": "Something completely different."}]
        masked = _mask_messages(messages, segments, mask_idx=0)
        assert masked[0]["content"] == "Something completely different."

    def test_returns_new_list_not_mutating_original(self) -> None:
        segments = self._segments()
        messages = [{"role": "system", "content": "You are a helpful agent."}]
        original_content = messages[0]["content"]
        _mask_messages(messages, segments, mask_idx=0)
        assert messages[0]["content"] == original_content

    def test_masks_only_first_occurrence(self) -> None:
        segments = [Segment(id="s", content="repeat")]
        messages = [{"role": "user", "content": "repeat and repeat again"}]
        masked = _mask_messages(messages, segments, mask_idx=0)
        content = masked[0]["content"]
        assert "[CONTENT REDACTED]" in content
        assert content.count("[CONTENT REDACTED]") == 1
        assert "repeat again" in content


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------


class TestSplitSentences:
    def test_splits_on_period(self) -> None:
        result = _split_sentences("First sentence. Second sentence.")
        assert len(result) == 2
        assert result[0] == "First sentence."

    def test_splits_on_exclamation(self) -> None:
        result = _split_sentences("Watch out! Be careful.")
        assert len(result) == 2

    def test_splits_on_question_mark(self) -> None:
        result = _split_sentences("Is this right? Yes it is.")
        assert len(result) == 2

    def test_single_sentence_returns_one_item(self) -> None:
        result = _split_sentences("Just one sentence here.")
        assert result == ["Just one sentence here."]

    def test_empty_string_returns_empty_list(self) -> None:
        result = _split_sentences("")
        assert result == []

    def test_filters_whitespace_only_parts(self) -> None:
        result = _split_sentences("  First.   Second.  ")
        assert all(s.strip() for s in result)

    def test_three_sentences(self) -> None:
        result = _split_sentences("One. Two. Three.")
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _normalise
# ---------------------------------------------------------------------------


class TestNormalise:
    def _score(self, sid: str, importance: float) -> SaliencyScore:
        return SaliencyScore(
            segment_id=sid,
            label=sid,
            importance=importance,
            decision_changed=importance > 0,
            original_decision="tool_a",
            masked_decision="tool_b" if importance > 0 else "tool_a",
        )

    def test_max_becomes_one(self) -> None:
        scores = [self._score("a", 4.0), self._score("b", 2.0), self._score("c", 1.0)]
        normalised = _normalise(scores)
        assert max(s.importance for s in normalised) == pytest.approx(1.0)

    def test_relative_order_preserved(self) -> None:
        scores = [self._score("a", 3.0), self._score("b", 1.0), self._score("c", 0.0)]
        normalised = _normalise(scores)
        by_id = {s.segment_id: s.importance for s in normalised}
        assert by_id["a"] > by_id["b"] > by_id["c"]

    def test_all_zeros_unchanged(self) -> None:
        scores = [self._score("a", 0.0), self._score("b", 0.0)]
        normalised = _normalise(scores)
        assert all(s.importance == 0.0 for s in normalised)

    def test_empty_list_returns_empty(self) -> None:
        assert _normalise([]) == []

    def test_preserves_segment_ids(self) -> None:
        scores = [self._score("x", 2.0), self._score("y", 1.0)]
        normalised = _normalise(scores)
        assert {s.segment_id for s in normalised} == {"x", "y"}

    def test_single_nonzero_score_becomes_one(self) -> None:
        scores = [self._score("a", 0.0), self._score("b", 3.7)]
        normalised = _normalise(scores)
        by_id = {s.segment_id: s.importance for s in normalised}
        assert by_id["b"] == pytest.approx(1.0)
        assert by_id["a"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SaliencyEngine (n_samples=1, logprob mode)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSaliencyEngineAsync:
    async def test_returns_saliency_result_with_correct_decision(self) -> None:
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _tool_response("escalate_to_human")

        result = await engine.explain_async(
            messages=_sample_messages(),  # type: ignore[arg-type]
            segments=_sample_segments(),
            tools=_sample_tools(),  # type: ignore[arg-type]
        )

        assert result.original_decision == "escalate_to_human"
        assert result.model == "test-model"

    async def test_flipped_segment_gets_high_importance(self) -> None:
        engine, client = _make_engine()
        responses = [
            _tool_response("escalate_to_human"),  # original
            _tool_response("escalate_to_human"),  # mask system → same
            _tool_response("send_reminder"),  # mask user_msg → flips
            _tool_response("escalate_to_human"),  # mask doc_1 → same
        ]
        client.chat.completions.create.side_effect = responses

        result = await engine.explain_async(
            messages=_sample_messages(),  # type: ignore[arg-type]
            segments=_sample_segments(),
            tools=_sample_tools(),  # type: ignore[arg-type]
        )

        pass1 = [s for s in result.scores if "__s" not in s.segment_id]
        by_id = {s.segment_id: s for s in pass1}
        assert by_id["user_msg"].decision_changed is True
        assert by_id["user_msg"].importance == pytest.approx(1.0)
        assert by_id["system"].decision_changed is False

    async def test_logprob_drop_used_as_importance_when_available(self) -> None:
        engine, client = _make_engine()
        segments = [
            Segment(id="seg_a", content="important context", label="A"),
            Segment(id="seg_b", content="irrelevant context", label="B"),
        ]
        messages = [{"role": "user", "content": "important context irrelevant context"}]

        original_resp = _tool_response(
            "tool_a", [{"token": "tool", "logprob": -0.01}, {"token": "_a", "logprob": -0.01}]
        )
        masked_a = _tool_response("tool_a", [{"token": "tool", "logprob": -0.8}, {"token": "_a", "logprob": -0.7}])
        masked_b = _tool_response("tool_a", [{"token": "tool", "logprob": -0.01}, {"token": "_a", "logprob": -0.02}])

        client.chat.completions.create.side_effect = [original_resp, masked_a, masked_b]

        result = await engine.explain_async(
            messages=messages,  # type: ignore[arg-type]
            segments=segments,
            tools=_sample_tools(),  # type: ignore[arg-type]
        )

        pass1 = [s for s in result.scores if "__s" not in s.segment_id]
        by_id = {s.segment_id: s for s in pass1}
        assert by_id["seg_a"].importance > by_id["seg_b"].importance

    async def test_importance_clipped_at_zero_when_masked_logprob_higher(self) -> None:
        engine, client = _make_engine()
        segments = [Segment(id="seg_a", content="context", label="A")]
        messages = [{"role": "user", "content": "context"}]

        # Original logprob lower than masked (masked is more confident) — drop should clip to 0
        original_resp = _tool_response("tool_a", [{"token": "tool_a", "logprob": -1.0}])
        masked_resp = _tool_response("tool_a", [{"token": "tool_a", "logprob": -0.1}])
        client.chat.completions.create.side_effect = [original_resp, masked_resp]

        result = await engine.explain_async(
            messages=messages,  # type: ignore[arg-type]
            segments=segments,
            tools=_sample_tools(),  # type: ignore[arg-type]
        )

        assert result.scores[0].importance == pytest.approx(0.0)

    async def test_no_pass2_when_all_segments_zero_importance(self) -> None:
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _tool_response("escalate_to_human")

        result = await engine.explain_async(
            messages=_sample_messages(),  # type: ignore[arg-type]
            segments=_sample_segments(),
            tools=_sample_tools(),  # type: ignore[arg-type]
        )

        assert [s for s in result.scores if "__s" in s.segment_id] == []

    async def test_pass2_only_drills_into_nonzero_segments(self) -> None:
        engine, client = _make_engine()
        segments = [
            Segment(id="system", content="You are an agent.", label="System"),
            Segment(id="user_msg", content="Urgent issue. Account locked.", label="User"),
        ]
        messages = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Urgent issue. Account locked."},
        ]
        responses = [
            _tool_response("escalate_to_human"),  # original
            _tool_response("escalate_to_human"),  # mask system → same
            _tool_response("send_reminder"),  # mask user_msg → flips
            _tool_response("escalate_to_human"),  # pass2: mask "Urgent issue."
            _tool_response("send_reminder"),  # pass2: mask "Account locked."
        ]
        client.chat.completions.create.side_effect = responses

        result = await engine.explain_async(
            messages=messages,  # type: ignore[arg-type]
            segments=segments,
            tools=_sample_tools(),  # type: ignore[arg-type]
        )

        sentence_scores = [s for s in result.scores if "__s" in s.segment_id]
        assert all(s.segment_id.startswith("user_msg") for s in sentence_scores)
        assert len(sentence_scores) == 2

    async def test_call_count_is_1_plus_n_segments_minimum(self) -> None:
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _tool_response("escalate_to_human")

        await engine.explain_async(
            messages=_sample_messages(),  # type: ignore[arg-type]
            segments=_sample_segments(),  # 3 segments
            tools=_sample_tools(),  # type: ignore[arg-type]
        )

        assert client.chat.completions.create.call_count >= 4  # 1 original + 3 masked


class TestSaliencyEngineSync:
    def test_explain_sync_returns_result(self) -> None:
        engine, client = _make_engine()
        client.chat.completions.create.return_value = _tool_response("escalate_to_human")

        result = engine.explain(
            messages=_sample_messages(),  # type: ignore[arg-type]
            segments=_sample_segments(),
            tools=_sample_tools(),  # type: ignore[arg-type]
        )

        assert result.original_decision == "escalate_to_human"
        assert len(result.scores) > 0


# ---------------------------------------------------------------------------
# SaliencyEngine (n_samples>1, sampling mode)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSaliencyEngineSamplingMode:
    async def test_sampling_mode_uses_probability_drop(self) -> None:
        engine, client = _make_engine(n_samples=4)
        segments = [
            Segment(id="seg_a", content="key context", label="A"),
            Segment(id="seg_b", content="irrelevant", label="B"),
        ]
        messages = [{"role": "user", "content": "key context irrelevant"}]

        # Original tool determined via _get_tool (1 call at temp=0)
        # Then _sample_probability for original: 4 calls (all return tool_a) → P=1.0
        # Mask seg_a: 4 calls → 1 returns tool_a → P=0.25 → importance=0.75
        # Mask seg_b: 4 calls → all return tool_a → P=1.0 → importance=0.0
        responses = (
            [_tool_response("tool_a")]  # _get_tool
            + [_tool_response("tool_a")] * 4  # original probability sampling
            + [
                _tool_response("tool_a"),
                _tool_response("send_reminder"),
                _tool_response("send_reminder"),
                _tool_response("send_reminder"),
            ]  # mask seg_a
            + [_tool_response("tool_a")] * 4  # mask seg_b
        )
        client.chat.completions.create.side_effect = responses

        result = await engine.explain_async(
            messages=messages,  # type: ignore[arg-type]
            segments=segments,
            tools=_sample_tools(),  # type: ignore[arg-type]
        )

        pass1 = [s for s in result.scores if "__s" not in s.segment_id]
        by_id = {s.segment_id: s for s in pass1}
        assert by_id["seg_a"].importance > by_id["seg_b"].importance

    async def test_sampling_mode_importance_proportional_to_prob_drop(self) -> None:
        engine, client = _make_engine(n_samples=4)
        segments = [Segment(id="seg_a", content="context", label="A")]
        messages = [{"role": "user", "content": "context"}]

        # Original: P=1.0 (4/4 match)
        # Mask seg_a: P=0.5 (2/4 match) → raw importance = 0.5
        responses = (
            [_tool_response("tool_a")]  # _get_tool
            + [_tool_response("tool_a")] * 4  # original probability
            + [
                _tool_response("tool_a"),
                _tool_response("tool_a"),
                _tool_response("send_reminder"),
                _tool_response("send_reminder"),
            ]  # mask
        )
        client.chat.completions.create.side_effect = responses

        result = await engine.explain_async(
            messages=messages,  # type: ignore[arg-type]
            segments=segments,
            tools=_sample_tools(),  # type: ignore[arg-type]
        )

        # Only one segment, normalises to 1.0 regardless of raw value
        assert result.scores[0].importance == pytest.approx(1.0)

    async def test_sampling_mode_decision_unchanged_when_prob_majority_same(self) -> None:
        engine, client = _make_engine(n_samples=4)
        segments = [Segment(id="seg_a", content="context", label="A")]
        messages = [{"role": "user", "content": "context"}]

        # Mask: 3/4 still return tool_a → majority same → decision_changed=False
        responses = (
            [_tool_response("tool_a")]  # _get_tool
            + [_tool_response("tool_a")] * 4  # original probability
            + [
                _tool_response("tool_a"),
                _tool_response("tool_a"),
                _tool_response("tool_a"),
                _tool_response("send_reminder"),
            ]
        )
        client.chat.completions.create.side_effect = responses

        result = await engine.explain_async(
            messages=messages,  # type: ignore[arg-type]
            segments=segments,
            tools=_sample_tools(),  # type: ignore[arg-type]
        )

        assert result.scores[0].decision_changed is False

    async def test_sampling_mode_call_count(self) -> None:
        # n_samples=3, 2 segments:
        # 1 _get_tool + 3 original_prob + (3 mask_seg_a + 3 mask_seg_b) = 10 calls
        engine, client = _make_engine(n_samples=3)
        segments = [
            Segment(id="seg_a", content="a", label="A"),
            Segment(id="seg_b", content="b", label="B"),
        ]
        messages = [{"role": "user", "content": "a b"}]
        client.chat.completions.create.return_value = _tool_response("tool_a")

        await engine.explain_async(
            messages=messages,  # type: ignore[arg-type]
            segments=segments,
            tools=_sample_tools(),  # type: ignore[arg-type]
        )

        assert client.chat.completions.create.call_count == 10

    async def test_n_samples_default_is_1(self) -> None:
        engine, _ = _make_engine()
        assert engine.n_samples == 1

    async def test_sample_temperature_default(self) -> None:
        engine, _ = _make_engine()
        assert engine.sample_temperature == pytest.approx(0.7)
