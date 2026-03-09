# LLM Translation Judge

Uses large language models as automated judges for translation quality evaluation, built around the **Multidimensional Quality Metrics (MQM)** framework ([Freitag et al., 2021](https://arxiv.org/pdf/2104.14478)).

## Project Structure

| File | Description |
|---|---|
| `phase1_tester.py` | Phase 1 — simple 0-1 holistic score using Google Gemini |
| `Phase2_tester.py` | Phase 2 — MQM-based evaluation with per-category error scores (accuracy, fluency, terminology, style, locale) using OpenAI |
| `mqm_comparison_demo.py` | Compares two MQM approaches: **(A)** simplified holistic per-category scores vs **(B)** paper-faithful individual error annotations with spans, categories, and severities |
| `MQM.pdf` | Reference paper for the MQM evaluation framework |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

**For Phase 1** (Google Gemini):
```bash
export GOOGLE_API_KEY="your-google-api-key"
```

**For Phase 2 and MQM Comparison Demo** (OpenAI):
```bash
export OPENAI_API_KEY="sk-..."
```

Optionally override the model (defaults to `gpt-4o-mini` for Phase 2, `gpt-5-mini` for the comparison demo):
```bash
export LLM_MODEL="gpt-4o"
```

### 3. Run

```bash
# Phase 1 — holistic score (Gemini)
python phase1_tester.py

# Phase 2 — MQM per-category evaluation (OpenAI)
python Phase2_tester.py

# MQM comparison demo — simplified vs paper-faithful (OpenAI)
python mqm_comparison_demo.py
```

## MQM Evaluation Approaches

### Simplified MQM (`Phase2_tester.py`)
The LLM assigns a single 0-10 severity score per broad category (accuracy, fluency, terminology, style, locale). A weighted penalty is computed and the final score is `1.0 - penalty` (higher is better). Quick quality gate, but cannot pinpoint specific errors.

### Paper-Faithful MQM (`mqm_comparison_demo.py`, Approach B)
The LLM identifies individual translation errors, each annotated with:
- **error_span** — the exact offending text
- **category / subcategory** — from the MQM hierarchy
- **severity** — Major, Minor, or Neutral

This follows the methodology from the paper and produces fine-grained, diagnosable output suitable for comparing translation systems.

## Evaluating from a JSON file

`Phase2_tester.py` supports batch evaluation. Prepare a JSON file like:

```json
[
  {
    "id": "sent-1",
    "reference": "The patient presents with acute chest pain.",
    "system_output": "The patient has acute chest pain."
  }
]
```

Then uncomment the `evaluate_from_json` call at the bottom of `Phase2_tester.py`, or use it programmatically:

```python
from Phase2_tester import evaluate_from_json
results = evaluate_from_json("translations.json", "results.json")
```
