# motive

Answer *why did the agent do that?* with empirically-grounded explanations for AI agent decisions.

motive uses **hierarchical occlusion saliency**: mask segments of the input context, measure how much each one affects the model's decision, and surface ranked importance scores and counterfactual what-ifs. No model self-reports. No local weights required. Works with any OpenAI-compatible endpoint.

## Installation

```bash
pip install motive
```

## Quickstart

```python
from openai import AsyncOpenAI
from motive import SaliencyEngine, Segment

engine = SaliencyEngine(
    client=AsyncOpenAI(base_url="...", api_key="..."),
    model="your-model",
)

result = engine.explain(messages=messages, segments=segments, tools=tools)
print(result.summary())
```

See [`examples/demo_saliency.py`](examples/demo_saliency.py) for a full working example.
