"""Demo: hierarchical occlusion saliency on a loan application review decision.

Designed to show a realistic importance distribution across multiple segments.
The decision (approve / flag / request_docs) depends on partial signals from
several sources, so masking each segment causes a different drop in confidence.
"""

import asyncio
import os

from openai import AsyncOpenAI

from motive import SaliencyEngine, Segment, SegmentLevel


PROXY_URL = "https://proxy.vectorinstitute.ai/v1"
API_KEY = os.environ["VECTOR_API_KEY"]
MODEL = "Qwen3-Coder-Next"

# --- Context segments ---
# Each segment contributes partial evidence. No single segment is decisive,
# so masking them produces a spread of importance scores rather than 0/1.

SEGMENTS = [
    Segment(
        id="system",
        content=(
            "You are a loan underwriting assistant. "
            "Approve applications that clearly meet all criteria. "
            "Flag for manual review if any criterion is borderline. "
            "Request more documents if required information is missing."
        ),
        label="System prompt",
        level=SegmentLevel.DOCUMENT,
    ),
    Segment(
        id="application",
        content=(
            "Applicant: Jordan Lee. "
            "Requested loan: $42,000. "
            "Stated annual income: $68,000. "
            "Credit score: 694. "
            "Employment: salaried, 2.5 years at current employer."
        ),
        label="Loan application",
        level=SegmentLevel.DOCUMENT,
    ),
    Segment(
        id="credit_policy",
        content=(
            "Credit score policy: scores above 720 qualify for standard approval. "
            "Scores between 660 and 720 require manual review. "
            "Scores below 660 are declined automatically."
        ),
        label="Policy: credit score",
        level=SegmentLevel.DOCUMENT,
    ),
    Segment(
        id="income_policy",
        content=(
            "Debt-to-income policy: the requested loan must not exceed 65% of annual income. "
            "Borderline cases (60-65%) require a supervisor sign-off."
        ),
        label="Policy: income ratio",
        level=SegmentLevel.DOCUMENT,
    ),
    Segment(
        id="employment_policy",
        content=(
            "Employment policy: applicants must have at least 12 months of continuous employment. "
            "Less than 24 months at the current employer is considered borderline."
        ),
        label="Policy: employment",
        level=SegmentLevel.DOCUMENT,
    ),
    Segment(
        id="prior_history",
        content=(
            "Credit history: no prior defaults. "
            "One missed payment 18 months ago, now resolved. "
            "No open collections or bankruptcies."
        ),
        label="Prior credit history",
        level=SegmentLevel.DOCUMENT,
    ),
]

# --- Messages ---

MESSAGES = [
    {"role": "system", "content": SEGMENTS[0].content},
    {
        "role": "user",
        "content": (
            f"Please review this loan application and decide on the next action.\n\n"
            f"Application:\n{SEGMENTS[1].content}\n\n"
            f"Relevant policies and history:\n"
            f"[Credit policy] {SEGMENTS[2].content}\n"
            f"[Income policy] {SEGMENTS[3].content}\n"
            f"[Employment policy] {SEGMENTS[4].content}\n"
            f"[Credit history] {SEGMENTS[5].content}"
        ),
    },
]

# --- Tools ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "approve_loan",
            "description": "Approve the loan application. All criteria are clearly met.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_for_manual_review",
            "description": "Flag the application for manual review by a senior underwriter. Use when any criterion is borderline.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_additional_documents",
            "description": "Ask the applicant for missing or incomplete documentation before proceeding.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


async def main() -> None:
    """Run the saliency demo."""
    client = AsyncOpenAI(base_url=PROXY_URL, api_key=API_KEY)
    # n_samples=5: run each masked context 5 times at temperature=0.7 and
    # measure P(original_decision | masked). Produces a genuine distribution
    # rather than near-binary logprob scores.
    engine = SaliencyEngine(client=client, model=MODEL, top_k=2, n_samples=5)

    print(f"Model: {MODEL}")
    print(f"Segments: {len(SEGMENTS)}")
    print("Running hierarchical occlusion (n_samples=5, temperature=0.7)...\n")

    result = await engine.explain_async(
        messages=MESSAGES,  # type: ignore[arg-type]
        segments=SEGMENTS,
        tools=TOOLS,  # type: ignore[arg-type]
    )

    print(f"Decision: {result.original_decision}\n")

    print("=== Pass 1: segment-level importance ===")
    doc_scores = [s for s in result.top if "__s" not in s.segment_id]
    for score in doc_scores:
        bar = "█" * int(score.importance * 30)
        changed = "  [flipped]" if score.decision_changed else ""
        print(f"  {score.label:<38} {score.importance:.3f}  {bar}{changed}")

    print("\n=== Pass 2: sentence-level (top segments) ===")
    sent_scores = [s for s in result.top if "__s" in s.segment_id]
    if sent_scores:
        for score in sent_scores:
            bar = "█" * int(score.importance * 30)
            changed = "  [flipped]" if score.decision_changed else ""
            print(f"  {score.label[:55]:<57} {score.importance:.3f}  {bar}{changed}")
    else:
        print("  (top segments were single sentences)")

    print(f"\n{result.summary()}")


if __name__ == "__main__":
    asyncio.run(main())
