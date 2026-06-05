"""Demo: hierarchical occlusion saliency on a support-ticket routing decision."""

import asyncio
import os

from openai import AsyncOpenAI

from motive import SaliencyEngine, Segment, SegmentLevel


PROXY_URL = "https://proxy.vectorinstitute.ai/v1"
API_KEY = os.environ["VECTOR_API_KEY"]
MODEL = "Qwen3-Coder-Next"

# --- Context segments (what the agent sees when deciding which tool to call) ---

SEGMENTS = [
    Segment(
        id="system",
        content="You are a support agent. Escalate tickets that are urgent or have been unresolved for more than 48 hours.",
        label="System prompt",
        level=SegmentLevel.DOCUMENT,
    ),
    Segment(
        id="user_msg",
        content="My payment has been failing for 3 days and my account is now locked. I need this fixed urgently.",
        label="User message",
        level=SegmentLevel.DOCUMENT,
    ),
    Segment(
        id="doc_1",
        content="Escalation policy: accounts locked for more than 48 hours require immediate human review. Do not attempt automated resolution.",
        label="Retrieved doc: escalation policy",
        level=SegmentLevel.DOCUMENT,
    ),
    Segment(
        id="doc_2",
        content="Payment retry guide: ask the user to clear browser cache and retry. Most payment failures resolve within 24 hours.",
        label="Retrieved doc: payment retry guide",
        level=SegmentLevel.DOCUMENT,
    ),
]

# --- Full messages for the decision call ---

MESSAGES = [
    {"role": "system", "content": SEGMENTS[0].content},
    {"role": "user", "content": SEGMENTS[1].content},
    {
        "role": "user",
        "content": (f"Context documents:\n\n[Doc 1] {SEGMENTS[2].content}\n\n[Doc 2] {SEGMENTS[3].content}"),
    },
]

# --- Tool definitions ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate the ticket to a human support agent.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_retry_instructions",
            "description": "Send automated payment retry instructions to the user.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "auto_resolve",
            "description": "Mark the ticket as resolved automatically.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


async def main() -> None:
    """Run the saliency demo."""
    client = AsyncOpenAI(base_url=PROXY_URL, api_key=API_KEY)
    engine = SaliencyEngine(client=client, model=MODEL, top_k=2)

    print(f"Model: {MODEL}")
    print(f"Running hierarchical occlusion on {len(SEGMENTS)} segments...\n")

    result = await engine.explain_async(
        messages=MESSAGES,  # type: ignore[arg-type]
        segments=SEGMENTS,
        tools=TOOLS,  # type: ignore[arg-type]
    )

    print(f"Original decision: {result.original_decision}\n")
    print("=== Pass 1: Segment-level importance ===")
    doc_scores = [s for s in result.top if "__s" not in s.segment_id]
    for score in doc_scores:
        bar = "█" * int(score.importance * 20)
        changed = " ← flipped" if score.decision_changed else ""
        print(f"  {score.label:<40} {score.importance:.3f}  {bar}{changed}")

    print("\n=== Pass 2: Sentence-level (top segments drilled in) ===")
    sent_scores = [s for s in result.top if "__s" in s.segment_id]
    if sent_scores:
        for score in sent_scores:
            bar = "█" * int(score.importance * 20)
            changed = " ← flipped" if score.decision_changed else ""
            print(f"  {score.label[:60]:<62} {score.importance:.3f}  {bar}{changed}")
    else:
        print("  (no multi-sentence segments found)")

    print(f"\n{result.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
