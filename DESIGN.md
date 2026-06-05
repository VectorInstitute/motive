# Design Doc: `motive` — Agent Decision Explainability Layer

**Status:** Draft
**Author:** Amrit Krishnan
**Date:** 2026-06-04

---

## 1. Problem

AI agent observability platforms (Langfuse, LangSmith, Arize Phoenix, MLflow, etc.) tell you *what* an agent did — which tools were called, in what order, with what inputs and outputs. They do not tell you *why* the agent made each decision.

When an agent escalates a support ticket, calls a risky API, or selects one branch over another, developers and operators have no structured, queryable way to understand the reasoning. This makes debugging slow, audits manual, and trust low.

---

## 2. Goal

Build a Python library that computes empirically-grounded, machine-readable explanations for AI agent decisions and emits them as OpenTelemetry-compatible telemetry. The explanations cover:

- **Feature attribution (saliency)** — which parts of the input context drove the decision, and how much.
- **Counterfactual probes** — minimal input changes that would have flipped the decision.

These are emitted as OTEL span attributes and events so they flow into existing backends (Langfuse, MLflow Tracing, any OTEL collector) without new infrastructure.

> **Note on structured reasoning:** model self-reported rationales ("I chose this tool because...") are not used as an explanation method. Anthropic's interpretability research (introspection paper, NLA paper, persona vectors) shows that model self-reports are unreliable ~80% of the time, can hallucinate, and often conceal internal states. Saliency and counterfactuals are empirical — they test the model's actual input-output behaviour — and are the only methods motive relies on.

---

## 3. Non-Goals (MVP)

- No model self-reported rationales or chain-of-thought extraction.
- No mechanistic interpretability (circuits, SAEs, neuron attribution).
- No tight coupling to a specific agent framework (LangGraph, AutoGen, etc.).
- No automatic policy enforcement or risk scoring — explanations are data, not decisions.
- No new UI from scratch — the reference UI is a pluggable component, not a product.

---

## 4. Environmental Scan

**Verdict: YES — genuine gap, worth building.**

The agent observability market is crowded at the *execution trace* layer. No production tool covers the *decision explanation* layer.

| Category | Representative Tools | Gap |
|---|---|---|
| Agent tracing / observability | Langfuse, LangSmith, Arize Phoenix, Braintrust, Opik, W&B Weave, MLflow | Capture what happened; no saliency, no counterfactuals on spans |
| OTEL AI standards | OpenInference, OpenLLMetry, OTEL GenAI Semantic Conventions | Standardize span shape and cardinality; no attributes for input importance or counterfactuals |
| XAI for AI models | SHAP, LIME, Integrated Gradients, AgentSHAP (2025) | Methods exist in isolation; none integrated into observability pipelines or emitting OTEL |
| Counterfactuals for agents | Research papers (ICLR 2026, arXiv 2506.02946) | Theoretically mature; no production implementation |
| Agent explainability SDKs | None found | Direct gap |

The 2026 research literature explicitly names this gap: "traditional tracing shows what happened, not why it was decided" (Elixir Data, 2026). AgentSHAP (arXiv 2512.12597) is the closest academic analogue but is not a production library and does not emit OTEL telemetry.

---

## 5. Scope and Assumptions

**In scope (MVP)**

- Single-agent and orchestrated multi-agent systems where each model call or tool invocation is logged as a span.
- Model-agnostic: works with any model type (language, vision-language, interaction) via any OpenAI-compatible API endpoint.
- Per-span explanations for tool calls and plan/routing decisions.
- Python SDK with decorators for zero-boilerplate instrumentation.
- OTEL emission to existing backends.

**Out of scope (MVP)**

- Multi-step causal chains across spans.
- Automatic redaction / PII scrubbing (configurable hooks only).
- UI product — reference React component only.
- Circuit tracing / activation-based methods (require local model weights; not compatible with API-served models).

**Assumptions**

- The agent system calls models via an OpenAI-compatible endpoint.
- Extra compute for explainability is acceptable: each explained span costs N+M additional model calls (N = number of segments, M = sentences in top segments). Sampled and async in prod.
- The storage backend can accept extra OTEL span attributes and events.

---

## 6. Architecture

```
Agent Code
    │
    ├── @explain_tool_call / @explain_plan_decision  (SDK decorators)
    │       │
    │       └── SaliencyEngine
    │               ├── Pass 1 — segment-level occlusion
    │               │     mask each segment → re-run → measure logprob drop
    │               └── Pass 2 — sentence-level occlusion (top-k segments only)
    │                     mask each sentence → re-run → measure logprob drop
    │
    ├── CounterfactualProber  (TODO)
    │       generate minimal edits to top segments → re-run → record flips
    │
    └── TelemetryEmitter
            └── OTEL SDK  →  existing backend (Langfuse / MLflow / collector)
```

**Components**

1. **SDK** — Python library. Wraps model/tool calls via decorators. Computes and emits explanations.
2. **SaliencyEngine** — hierarchical occlusion engine. Scores input segments by how much removing them drops the model's confidence in the original decision, using logprob drop as the primary signal with binary fallback.
3. **CounterfactualProber** — takes top-importance segments, generates minimal edits, re-runs the decision, records flips as human-readable what-if statements.
4. **Telemetry Emitter** — maps results to OTEL span attributes and events using the `agent.why.*` namespace.
5. **Reference UI Component** — React component that renders a "Why?" panel from span JSON.

---

## 7. Explanation Unit: The Decision Span

The unit of explanation is a *decision span* — any span where the agent makes a discrete choice.

**Span types (MVP)**

| Type | Triggered by | What is explained |
|---|---|---|
| `tool_call` | Agent selects and calls an external tool | Which input segments drove the tool choice |
| `plan_decision` | Agent picks a branch, subagent, or subgoal | Which input segments drove the routing |

Each decision span carries:
- **Input context** — user message, agent state, retrieved documents, prior plan.
- **Segments** — the input context pre-split into logical units (one per message, retrieved doc, tool result, etc.).
- **Model output** — tool name + args, or chosen branch.
- **Explanation** — saliency scores and counterfactuals, attached as OTEL attributes and events.

---

## 8. Explanation Methods

### 8.1 Hierarchical Occlusion Saliency (primary method)

**How it works:**

The input context is split into *segments* — logical units such as the system prompt, each user message, each retrieved document, and each prior tool result. The engine runs two passes:

**Pass 1 — segment level:**
Each segment is masked (replaced with `[CONTENT REDACTED]`) one at a time. The decision call is re-run with the masked context. Importance is measured as the drop in log-probability of the original decision:

```
importance(i) = logprob(decision | full context)
              − logprob(decision | context with segment i masked)
```

Higher importance = masking that segment makes the model less confident in its original choice. Falls back to binary (decision flipped = 1.0, unchanged = 0.0) when logprobs are unavailable.

**Pass 2 — sentence level:**
For the top-k segments from Pass 1 (importance > 0), the engine repeats the same procedure at sentence granularity within each segment, identifying the specific sentences that drove the decision.

**Output:** a ranked list of `(segment, importance)` pairs at both granularity levels. The unit of attribution depends on modality: sentences/tokens for text, patches/regions for vision.

**Cost:** Pass 1 = N model calls (N = segments, typically 4–10). Pass 2 = M calls per top segment (M = sentences, typically 2–6). Total ~10–30 calls per explained span.

### 8.2 Counterfactual Probes (next to implement)

Starting from the highest-importance segments identified in §8.1:
1. Generate minimal perturbations: remove a phrase, swap a retrieved document, neutralize sentiment.
2. Re-run the decision call.
3. If the outcome changes (different tool or routing), record a human-readable description.

Example: *"If 'failing for 3 days' is removed from the user message, the agent calls `send_retry_instructions` instead of `escalate_to_human`."*

Store 1–3 counterfactuals per span.

---

## 9. Telemetry Schema (OTEL Extension)

All fields attach to the existing decision span; no new spans are created.

### Span Attributes (indexed, short strings)

| Attribute | Type | Description |
|---|---|---|
| `agent.why.type` | string | `tool_call` or `plan_decision`. |
| `agent.why.primary_factors` | JSON string | Top-k `{id, label, importance, decision_changed}` objects from Pass 1. |
| `agent.why.sentence_factors` | JSON string | Top-k `{id, label, importance, decision_changed}` objects from Pass 2. |
| `agent.why.counterfactuals` | JSON string | 1–3 `{description}` objects. |
| `agent.explainability.methods` | JSON string | Methods applied, e.g. `["saliency","counterfactual"]`. |
| `agent.explainability.version` | string | Semver of the explainability logic. |

### Span Events (richer payloads)

| Event name | Fields |
|---|---|
| `agent.why.saliency` | `{modality, pass, segments: [{id, label, importance, decision_changed, original_decision, masked_decision}]}` |
| `agent.why.counterfactual_run` | `{original_input_hash, tested_variants, outcomes}` |

---

## 10. SDK API

```python
from motive import SaliencyEngine, Segment, SegmentLevel

engine = SaliencyEngine(client=AsyncOpenAI(...), model="...", top_k=2)

result = engine.explain(
    messages=[...],   # OpenAI-format message list
    segments=[
        Segment(id="system",  content="...", label="System prompt"),
        Segment(id="user",    content="...", label="User message"),
        Segment(id="doc_1",   content="...", label="Retrieved doc 1"),
    ],
    tools=[...],
    tool_choice="auto",
)

result.top          # segments ranked by importance
result.summary()    # one-line human-readable description
```

### Sampling Policy Config (TODO)

```python
ExplainabilityConfig(
    mode="sample",           # off | dev_all | sample
    sample_rate=0.1,
    always_explain_rules=[
        OnError(),
        OnTool("escalate_to_human"),
    ],
    async_mode=True,
)
```

---

## 11. Reference UI Component

A React component `<WhyPanel span={spanJson} />` that renders:
- **Highlighted text:** input context with segments coloured by saliency importance (Pass 1), with sentence-level highlights nested inside (Pass 2).
- **Factors list:** `agent.why.primary_factors` ranked by importance.
- **What-if chips:** `agent.why.counterfactuals`.

Developer mode: raw scores. Operator mode: importance ranks only.

---

## 12. Non-Functional Requirements

| Concern | Requirement |
|---|---|
| Latency (dev) | Up to 300 ms per explained span acceptable; all occlusion calls run concurrently. |
| Latency (prod) | Default sampled + async; Pass 2 optional. |
| PII / Security | Explanations must not leak data beyond what is already in the trace. Provide a pre-emit redaction hook. |
| Extensibility | New explanation methods (e.g. SHAP, gradient-based) slot in behind the same `SaliencyResult` schema. |
| Backend compatibility | Works with any OpenAI-compatible API. No provider-specific code in core. |

---

## 13. Milestones

| Milestone | Status | Deliverable |
|---|---|---|
| M1 | ✅ Done | `Segment`, `SaliencyScore`, `SaliencyResult` types; `SaliencyEngine` with two-pass hierarchical occlusion; logprob-based scoring with binary fallback |
| M2 | Next | `CounterfactualProber` — minimal edits to top segments, flip detection, human-readable what-if output |
| M3 | — | OTEL emitter — map `SaliencyResult` to `agent.why.*` span attributes and events |
| M4 | — | `@explain_tool_call` / `@explain_plan_decision` decorators + sampling policy |
| M5 | — | Langfuse and MLflow integration tests; reference React `<WhyPanel />` |
