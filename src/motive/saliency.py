"""Hierarchical occlusion-based saliency engine."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Literal, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from ._types import SaliencyResult, SaliencyScore, Segment, SegmentLevel


_MASK = "[CONTENT REDACTED]"
_TOP_K_DEFAULT = 2

ToolChoice = Literal["auto", "required", "none"]


def _extract_tool_name(response: Any) -> str:
    """Pull the called tool name out of a chat completion response."""
    msg = response.choices[0].message
    if msg.tool_calls:
        return msg.tool_calls[0].function.name
    return msg.content or ""


def _logprob_for_tool(response: Any, tool_name: str) -> float | None:
    """Sum logprobs over the tokens that spell out `tool_name` in the response.

    Tool names are multi-token (e.g. "escalate_to_human" → ["escal","ate","_to","_human"]),
    so we reconstruct the full generated string, locate the tool name span, and sum
    the logprobs of every token that overlaps with it.
    """
    choice = response.choices[0]
    if not getattr(choice, "logprobs", None):
        return None
    tokens_data = getattr(choice.logprobs, "content", None) or []
    if not tokens_data:
        return None

    full_text = "".join(td.token for td in tokens_data)
    start = full_text.find(tool_name)
    if start == -1:
        return None
    end = start + len(tool_name)

    total_lp = 0.0
    pos = 0
    found = False
    for td in tokens_data:
        tok_end = pos + len(td.token)
        if pos < end and tok_end > start:
            total_lp += td.logprob
            found = True
        pos = tok_end

    return total_lp if found else None


def _mask_messages(
    messages: list[ChatCompletionMessageParam],
    segments: list[Segment],
    mask_idx: int,
) -> list[ChatCompletionMessageParam]:
    """Return a copy of messages with segment[mask_idx] replaced by the mask placeholder."""
    target = segments[mask_idx]
    result: list[dict[str, Any]] = []
    for msg in messages:
        raw = cast(dict[str, Any], msg)
        content = raw.get("content") or ""
        if isinstance(content, str) and target.content in content:
            result.append({**raw, "content": content.replace(target.content, _MASK, 1)})
        else:
            result.append(raw)
    return cast(list[ChatCompletionMessageParam], result)


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter for agent context segments."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _normalise(scores: list[SaliencyScore]) -> list[SaliencyScore]:
    """Scale importance values to [0, 1] relative to the max in this pass."""
    max_val = max((s.importance for s in scores), default=1.0)
    if max_val == 0:
        return scores
    return [
        SaliencyScore(
            segment_id=s.segment_id,
            label=s.label,
            importance=s.importance / max_val,
            decision_changed=s.decision_changed,
            original_decision=s.original_decision,
            masked_decision=s.masked_decision,
        )
        for s in scores
    ]


class SaliencyEngine:
    """Hierarchical occlusion saliency for agent tool-call decisions.

    Pass 1 masks at segment (document/message) level.
    Pass 2 drills into the top_k segments at sentence level.

    Importance is the drop in log-probability of the original tool choice
    when a segment is masked.  Falls back to binary (flipped = 1.0,
    unchanged = 0.0) when logprobs are unavailable.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        top_k: int = _TOP_K_DEFAULT,
    ) -> None:
        self.client = client
        self.model = model
        self.top_k = top_k

    def explain(
        self,
        messages: list[ChatCompletionMessageParam],
        segments: list[Segment],
        tools: list[ChatCompletionToolParam],
        tool_choice: ToolChoice = "auto",
    ) -> SaliencyResult:
        """Run explanation synchronously."""
        return asyncio.run(self.explain_async(messages, segments, tools, tool_choice))

    async def explain_async(
        self,
        messages: list[ChatCompletionMessageParam],
        segments: list[Segment],
        tools: list[ChatCompletionToolParam],
        tool_choice: ToolChoice = "auto",
    ) -> SaliencyResult:
        """Async entry point."""
        original_resp = await self._call(messages, tools, tool_choice, logprobs=True)
        original_tool = _extract_tool_name(original_resp)
        original_lp = _logprob_for_tool(original_resp, original_tool)

        # Pass 1: segment level
        pass1 = await self._occlusion_pass(messages, segments, tools, tool_choice, original_tool, original_lp)
        pass1 = _normalise(pass1)

        # Pass 2: sentence level on top_k segments that actually had non-zero importance
        top_ids = {
            s.segment_id
            for s in sorted(pass1, key=lambda s: s.importance, reverse=True)[: self.top_k]
            if s.importance > 0
        }
        pass2: list[SaliencyScore] = []
        for seg in segments:
            if seg.id not in top_ids:
                continue
            sentences = _split_sentences(seg.content)
            if len(sentences) <= 1:
                continue
            sub_segments = [
                Segment(
                    id=f"{seg.id}__s{i}",
                    content=s,
                    label=s[:80],
                    level=SegmentLevel.SENTENCE,
                )
                for i, s in enumerate(sentences)
            ]
            sub_scores = await self._occlusion_pass(
                messages, sub_segments, tools, tool_choice, original_tool, original_lp
            )
            pass2.extend(_normalise(sub_scores))

        return SaliencyResult(
            scores=pass1 + pass2,
            original_decision=original_tool,
            model=self.model,
        )

    async def _occlusion_pass(
        self,
        messages: list[ChatCompletionMessageParam],
        segments: list[Segment],
        tools: list[ChatCompletionToolParam],
        tool_choice: ToolChoice,
        original_tool: str,
        original_lp: float | None,
    ) -> list[SaliencyScore]:
        tasks = [
            self._score_segment(messages, segments, i, tools, tool_choice, original_tool, original_lp)
            for i in range(len(segments))
        ]
        return list(await asyncio.gather(*tasks))

    async def _score_segment(
        self,
        messages: list[ChatCompletionMessageParam],
        segments: list[Segment],
        mask_idx: int,
        tools: list[ChatCompletionToolParam],
        tool_choice: ToolChoice,
        original_tool: str,
        original_lp: float | None,
    ) -> SaliencyScore:
        masked_msgs = _mask_messages(messages, segments, mask_idx)
        resp = await self._call(masked_msgs, tools, tool_choice, logprobs=True)
        masked_tool = _extract_tool_name(resp)

        if original_lp is not None:
            masked_lp = _logprob_for_tool(resp, original_tool)
            importance = (
                max(0.0, original_lp - masked_lp)
                if masked_lp is not None
                else (1.0 if masked_tool != original_tool else 0.0)
            )
        else:
            importance = 1.0 if masked_tool != original_tool else 0.0

        seg = segments[mask_idx]
        return SaliencyScore(
            segment_id=seg.id,
            label=seg.label or seg.id,
            importance=importance,
            decision_changed=masked_tool != original_tool,
            original_decision=original_tool,
            masked_decision=masked_tool,
        )

    async def _call(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam],
        tool_choice: ToolChoice,
        logprobs: bool = False,
    ) -> Any:
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=5 if logprobs else None,
            temperature=0,
        )
