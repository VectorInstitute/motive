"""Core data types for motive."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SegmentLevel(str, Enum):
    """Granularity level of a context segment."""

    DOCUMENT = "document"
    SENTENCE = "sentence"
    TOKEN = "token"


@dataclass
class Segment:
    """A labelled chunk of input context that can be masked."""

    id: str
    content: str
    label: str = ""
    level: SegmentLevel = SegmentLevel.DOCUMENT


@dataclass
class SaliencyScore:
    """Importance score for a single segment."""

    segment_id: str
    label: str
    importance: float  # [0, 1] normalised across segments in this pass
    decision_changed: bool
    original_decision: str
    masked_decision: str


@dataclass
class SaliencyResult:
    """Full result of a hierarchical occlusion pass."""

    scores: list[SaliencyScore]
    original_decision: str
    model: str
    method: str = "hierarchical_occlusion"

    @property
    def top(self) -> list[SaliencyScore]:
        """Segments sorted by importance descending."""
        return sorted(self.scores, key=lambda s: s.importance, reverse=True)

    def summary(self) -> str:
        """One-line human-readable summary of the top driver."""
        if not self.scores:
            return "No segments scored."
        best = self.top[0]
        return (
            f"Decision '{self.original_decision}' was most influenced by "
            f"'{best.label}' (importance {best.importance:.2f})."
        )
