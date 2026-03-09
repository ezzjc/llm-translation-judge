"""
MQM Comparison Demo
====================
Illustrates the difference between:
  (A) Simplified MQM — holistic per-category scores (Phase2_tester.py style)
  (B) Paper-faithful MQM — individual error annotations with spans, categories,
      severities, and a weighted penalty sum (Freitag et al., 2021)

Run:
  export OPENAI_API_KEY="sk-..."
  python mqm_comparison_demo.py
"""

from openai import OpenAI
from pydantic import BaseModel, Field
from enum import Enum
import json
import os
import textwrap

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = os.environ.get("LLM_MODEL", "gpt-5-mini")

# ═══════════════════════════════════════════════════════════════
# APPROACH A: Simplified MQM (your current Phase2_tester style)
# ═══════════════════════════════════════════════════════════════
# The LLM assigns a single 0-10 severity score per broad category.
# Final score = 1.0 - weighted penalty (higher = better).

SIMPLE_CATEGORIES = {
    "accuracy":    {"weight": 0.35},
    "fluency":     {"weight": 0.25},
    "terminology": {"weight": 0.20},
    "style":       {"weight": 0.10},
    "locale":      {"weight": 0.10},
}


class CategoryScore(BaseModel):
    score: float = Field(description="Error severity from 0 (no errors) to 10 (critical errors)")
    explanation: str = Field(description="Brief explanation justifying the score")


class SimpleMQMEvaluation(BaseModel):
    accuracy: CategoryScore = Field(description="Faithfulness to the meaning of the reference — covers additions, omissions, mistranslations")
    fluency: CategoryScore = Field(description="Grammatical correctness and naturalness in the target language")
    terminology: CategoryScore = Field(description="Correct and consistent use of domain-specific terms")
    style: CategoryScore = Field(description="Appropriateness of register, tone, and stylistic choices")
    locale: CategoryScore = Field(description="Correct handling of locale conventions such as dates, units, currency")
    overall_comment: str = Field(description="1-2 sentence summary of the overall translation quality")


def evaluate_simple(reference: str, system_output: str) -> dict:
    """Approach A: one holistic 0-10 score per category."""
    prompt = f"""You are an MQM translation evaluator.
Score the System Translation against the Reference on each category (0 = no errors, 10 = critical).

Reference: "{reference}"
System:    "{system_output}"

Categories: accuracy, fluency, terminology, style, locale.
Provide a score and brief explanation per category, plus an overall_comment."""

    resp = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format=SimpleMQMEvaluation,
    )
    evaluation = resp.choices[0].message.parsed
    eval_dict = evaluation.model_dump()

    penalty = sum(
        SIMPLE_CATEGORIES[cat]["weight"] * (eval_dict[cat]["score"] / 10.0)
        for cat in SIMPLE_CATEGORIES
    )
    score = round(max(0.0, 1.0 - penalty), 4)
    return {"mqm_score": score, "detail": eval_dict}


# ═══════════════════════════════════════════════════════════════
# APPROACH B: Paper-faithful MQM (Freitag et al., 2021)
# ═══════════════════════════════════════════════════════════════
# The LLM identifies individual errors, each with:
#   - error_span: the offending text
#   - category / subcategory from the paper's hierarchy
#   - severity: Major | Minor | Neutral
# Score = sum of severity weights (lower = better, 0 = perfect).

MQM_WEIGHTS = {
    ("Major",   "Non-translation"):       25,
    ("Major",   "_default"):               5,
    ("Minor",   "Fluency/Punctuation"):    0.1,
    ("Minor",   "_default"):               1,
    ("Neutral", "_default"):               0,
}


class Severity(str, Enum):
    MAJOR = "Major"
    MINOR = "Minor"
    NEUTRAL = "Neutral"


class MQMError(BaseModel):
    error_span: str = Field(description="The exact text in the System Translation that is wrong")
    category: str = Field(description="MQM error category: Accuracy, Fluency, Terminology, Style, Locale convention, Non-translation, or Source error")
    subcategory: str = Field(description="Specific sub-category, e.g. Mistranslation, Omission, Grammar, Punctuation, Awkward, etc.")
    severity: Severity = Field(description="Error severity: Major (affects meaning/safety), Minor (noticeable but understandable), or Neutral (stylistic preference)")
    explanation: str = Field(description="Why this is an error and how it affects the translation")


class PaperMQMEvaluation(BaseModel):
    errors: list[MQMError] = Field(description="List of individual translation errors found in the system output")
    overall_comment: str = Field(description="Summary of the overall translation quality and error patterns")


VALID_CATEGORIES = """Accuracy (Mistranslation | Omission | Addition | Untranslated text)
Fluency (Punctuation | Spelling | Grammar | Register | Inconsistency | Character encoding)
Terminology (Inappropriate for context | Inconsistent use)
Style (Awkward)
Locale convention (Address format | Currency format | Date format | Name format | Telephone format | Time format)
Non-translation
Source error"""


def error_weight(severity: str, category: str, subcategory: str) -> float:
    """Look up the weight for one error using the paper's weighting scheme."""
    key_specific = (severity, f"{category}/{subcategory}")
    if key_specific in MQM_WEIGHTS:
        return MQM_WEIGHTS[key_specific]
    key_default = (severity, "_default")
    return MQM_WEIGHTS.get(key_default, 0)


def evaluate_paper(reference: str, system_output: str, document_reference: str = "", document_system_output: str = "") -> dict:
    """Approach B: enumerate individual errors with spans and severities, using document-level context when available."""
    context_block = ""
    if document_reference or document_system_output:
        context_block = f"""
Document-level context (entire document):
Reference document:
\"\"\"{document_reference or reference}\"\"\"

System document:
\"\"\"{document_system_output or system_output}\"\"\"

Focus your evaluation on the segment shown below, but use the document-level context to resolve ambiguities and avoid isolated-segment errors.
"""

    prompt = f"""You are an expert MQM translation evaluator following Freitag et al. (2021).

Identify ALL errors in the System Translation compared to the Reference.
For each error:
  - error_span: the exact text in the System Translation that is wrong
  - category: one of {VALID_CATEGORIES}
  - subcategory: the specific sub-category
  - severity: Major | Minor | Neutral
  - explanation: why this is an error

Maximum 5 errors. If the translation is too garbled, report a single
Non-translation error spanning the whole segment.

{context_block}

Segment to evaluate:
Reference segment: "{reference}"
System segment:    "{system_output}" """

    resp = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format=PaperMQMEvaluation,
    )
    result = resp.choices[0].message.parsed
    return result.model_dump()


# ═══════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════

TEST_PAIRS = [
    {
        "id": "easy",
        "reference":     "The patient was prescribed 500 mg of amoxicillin to be taken orally three times daily.",
        "system_output": "The patient was prescribed 500 mg of amoxicillin to be taken orally three times a day.",
    },
    {
        "id": "moderate",
        "reference":     "Electrocardiogram results showed ST-segment elevation in leads II, III, and aVF, consistent with an inferior myocardial infarction.",
        "system_output": "ECG results showed ST elevation in leads II, III, and aVF, which is consistent with a lower heart attack.",
    },
    {
        "id": "severe",
        "reference":     "The patient should discontinue warfarin 5 days prior to the procedure and bridge with enoxaparin.",
        "system_output": "The sick person must stop blood medicine 5 days before the thing and use bridge with another drug.",
    },
]

# Construct simple document-level context for the demo: all segments concatenated.
FULL_DOC_REFERENCE = " ".join(p["reference"] for p in TEST_PAIRS)
FULL_DOC_SYSTEM = " ".join(p["system_output"] for p in TEST_PAIRS)


def separator(char="═", width=70):
    print(char * width)


if __name__ == "__main__":
    for pair in TEST_PAIRS:
        separator()
        print(f"  TEST CASE: {pair['id'].upper()}")
        separator()
        print(f"  Reference:  {pair['reference']}")
        print(f"  System:     {pair['system_output']}")
        separator("─")

        # --- Approach A ---
        print("\n  [A] SIMPLIFIED MQM (holistic per-category scores)")
        simple = evaluate_simple(pair["reference"], pair["system_output"])
        print(f"      Score: {simple['mqm_score']}  (0-1, higher = better)")
        for cat in SIMPLE_CATEGORIES:
            d = simple["detail"][cat]
            #print(f"      {cat:<14} {d['score']:>4}/10  — {d['explanation']}")
        print(f"      Comment: {simple['detail']['overall_comment']}")

        # --- Approach B ---
        print(f"\n  [B] PAPER-FAITHFUL MQM (individual error annotations)")
        paper = evaluate_paper(
            pair["reference"],
            pair["system_output"],
            document_reference=FULL_DOC_REFERENCE,
            document_system_output=FULL_DOC_SYSTEM,
        )
        # Numeric MQM scores are disabled in this demo; we only display errors.
        if not paper["errors"]:
            print("      No errors found.")
        for i, err in enumerate(paper["errors"], 1):
            print(f"      Error {i}:")
            print(f"        Span:     \"{err['error_span']}\"")
            print(f"        Category: {err['category']} / {err['subcategory']}")
            print(f"        Severity: {err['severity']}")
            print(f"        Why:      {err['explanation']}")
        print(f"      Comment: {paper['overall_comment']}")
        print()

    separator()
    print("  INTERPRETATION GUIDE")
    separator("─")
    print(textwrap.dedent("""\
      Approach A (Simplified):
        - Gives ONE holistic score per broad category
        - Cannot tell you WHICH words are wrong or HOW MANY errors exist
        - Two very different translations can get the same score
        - Useful as a quick quality gate, poor for diagnosis

      Approach B (Paper-faithful):
        - Enumerates each error with its exact text span
        - Each error has a specific sub-category and severity
        - The weighted sum directly reflects the paper's penalty scheme
        - You can count errors by type, analyze patterns, compare systems
        - This is what the paper shows produces "platinum standard" rankings
    """))
