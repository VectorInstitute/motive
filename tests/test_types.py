"""Tests for motive._types."""

import pytest

from motive._types import SaliencyResult, SaliencyScore, Segment, SegmentLevel


def _score(segment_id: str, importance: float, changed: bool = False) -> SaliencyScore:
    return SaliencyScore(
        segment_id=segment_id,
        label=segment_id,
        importance=importance,
        decision_changed=changed,
        original_decision="tool_a",
        masked_decision="tool_b" if changed else "tool_a",
    )


class TestSegment:
    def test_defaults(self) -> None:
        seg = Segment(id="s1", content="hello")
        assert seg.label == ""
        assert seg.level == SegmentLevel.DOCUMENT

    def test_explicit_fields(self) -> None:
        seg = Segment(id="s1", content="hello", label="My label", level=SegmentLevel.SENTENCE)
        assert seg.label == "My label"
        assert seg.level == SegmentLevel.SENTENCE

    def test_segment_level_values(self) -> None:
        assert SegmentLevel.DOCUMENT == "document"
        assert SegmentLevel.SENTENCE == "sentence"
        assert SegmentLevel.TOKEN == "token"


class TestSaliencyScore:
    def test_fields(self) -> None:
        score = _score("doc_1", 0.9, changed=True)
        assert score.segment_id == "doc_1"
        assert score.importance == pytest.approx(0.9)
        assert score.decision_changed is True
        assert score.masked_decision == "tool_b"

    def test_unchanged(self) -> None:
        score = _score("doc_2", 0.1, changed=False)
        assert score.decision_changed is False
        assert score.original_decision == score.masked_decision


class TestSaliencyResult:
    def _result(self) -> SaliencyResult:
        return SaliencyResult(
            scores=[
                _score("user_msg", 1.0, changed=True),
                _score("system", 0.0),
                _score("doc_1", 0.5),
            ],
            original_decision="escalate_to_human",
            model="test-model",
        )

    def test_top_sorted_descending(self) -> None:
        result = self._result()
        importances = [s.importance for s in result.top]
        assert importances == sorted(importances, reverse=True)

    def test_top_first_is_highest(self) -> None:
        result = self._result()
        assert result.top[0].segment_id == "user_msg"

    def test_summary_names_top_segment(self) -> None:
        result = self._result()
        summary = result.summary()
        assert "escalate_to_human" in summary
        assert "user_msg" in summary

    def test_summary_empty_scores(self) -> None:
        result = SaliencyResult(scores=[], original_decision="tool_a", model="m")
        assert result.summary() == "No segments scored."

    def test_default_method(self) -> None:
        result = self._result()
        assert result.method == "hierarchical_occlusion"
