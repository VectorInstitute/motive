# motive

[![code checks](https://github.com/VectorInstitute/motive/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/motive/actions/workflows/code_checks.yml)
[![unit tests](https://github.com/VectorInstitute/motive/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/VectorInstitute/motive/actions/workflows/unit_tests.yml)
[![codecov](https://codecov.io/github/VectorInstitute/motive/graph/badge.svg)](https://codecov.io/github/VectorInstitute/motive)
[![PyPI](https://img.shields.io/pypi/v/motive)](https://pypi.org/project/motive/)
![GitHub License](https://img.shields.io/github/license/VectorInstitute/motive)

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
