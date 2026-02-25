from openai import OpenAI
import json
import os

# ============================================================
# SECTION 1: AUTHENTICATION
# ============================================================
# API key is read from the OPENAI_API_KEY environment variable.
# Set it in your shell before running:
#   export OPENAI_API_KEY="sk-..."
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ============================================================
# SECTION 2: MODEL SELECTION
# ============================================================
# Defaults to gpt-4o-mini. Override by setting LLM_MODEL:
#   export LLM_MODEL="gpt-4o"
MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# ============================================================
# SECTION 3: MQM ERROR CATEGORIES
# ============================================================
# MQM (Multidimensional Quality Metrics) is an industry-standard
# framework for evaluating translation quality. Instead of a
# single holistic score, MQM breaks quality into specific error
# categories, each with a severity weight.
#
# Categories used here:
#   - Accuracy:     Does the translation faithfully convey the
#                   meaning of the source/reference?
#   - Fluency:      Is the translation grammatically correct and
#                   natural-sounding in the target language?
#   - Terminology:  Are domain-specific terms translated correctly
#                   and consistently?
#   - Style:        Does the translation match the register, tone,
#                   and style of the reference?
#   - Locale:       Are locale conventions (dates, units, names)
#                   handled properly?
#
# Each category is scored 0-10 by the LLM, then converted to a
# penalty. The final MQM score = 1.0 minus the weighted penalty
# sum, clamped to [0, 1].

MQM_CATEGORIES = {
    "accuracy":    {"weight": 0.35, "description": "Faithfulness to the meaning of the reference translation. Covers additions, omissions, and mistranslations."},
    "fluency":     {"weight": 0.25, "description": "Grammatical correctness and naturalness in the target language. Covers grammar, spelling, and punctuation."},
    "terminology": {"weight": 0.20, "description": "Correct and consistent use of domain-specific terms and technical vocabulary."},
    "style":       {"weight": 0.10, "description": "Appropriateness of register, tone, and stylistic choices relative to the reference."},
    "locale":      {"weight": 0.10, "description": "Correct handling of locale-specific conventions such as date formats, units, currency, and proper nouns."},
}


def build_mqm_prompt(reference, system_output):
    """
    Constructs the evaluation prompt sent to OpenAI.

    The prompt instructs the model to act as an MQM-certified
    translation evaluator. It asks for:
      1. A 0-10 severity score per MQM category (0 = no errors,
         10 = critical errors).
      2. A brief explanation for each category score.
      3. An overall comment summarizing the translation quality.

    The response format is enforced as strict JSON so we can
    reliably parse it downstream.
    """
    category_block = "\n".join(
        f"    - {name} (weight {info['weight']}): {info['description']}"
        for name, info in MQM_CATEGORIES.items()
    )

    prompt = f"""You are an MQM-certified translation quality evaluator.

Evaluate the System Translation against the Reference Translation using the
Multidimensional Quality Metrics (MQM) framework.

For each of the following error categories, assign an error severity score
from 0 to 10 where:
  0  = No errors at all
  5  = Moderate errors that affect comprehension
  10 = Critical errors that make the translation unusable

Categories:
{category_block}

Reference Translation:
\"{reference}\"

System Translation:
\"{system_output}\"

Respond with ONLY valid JSON in this exact structure (no markdown, no extra text):
{{
  "accuracy":    {{"score": <0-10>, "explanation": "<brief explanation>"}},
  "fluency":     {{"score": <0-10>, "explanation": "<brief explanation>"}},
  "terminology": {{"score": <0-10>, "explanation": "<brief explanation>"}},
  "style":       {{"score": <0-10>, "explanation": "<brief explanation>"}},
  "locale":      {{"score": <0-10>, "explanation": "<brief explanation>"}},
  "overall_comment": "<1-2 sentence summary of translation quality>"
}}"""
    return prompt


def compute_mqm_score(category_scores):
    """
    Converts per-category error severity scores into a single
    MQM quality score between 0.0 and 1.0.

    Formula:
      weighted_penalty = sum(weight_i * (score_i / 10))
      mqm_score = max(0.0, 1.0 - weighted_penalty)

    A score_i of 0 (no errors) contributes 0 penalty.
    A score_i of 10 (critical) contributes the full weight as penalty.
    """
    penalty = 0.0
    for category, info in MQM_CATEGORIES.items():
        raw = category_scores.get(category, {}).get("score", 0)
        penalty += info["weight"] * (raw / 10.0)
    return round(max(0.0, 1.0 - penalty), 4)


def get_phase2_score(reference, system_output):
    """
    End-to-end MQM evaluation of a single translation pair.

    Steps:
      1. Build the MQM prompt with the reference and system output.
      2. Call OpenAI to get per-category error scores + explanations.
      3. Parse the JSON response.
      4. Compute the weighted MQM score.
      5. Return a dict with the final score, per-category breakdown,
         and the overall comment from the LLM.
    """
    prompt = build_mqm_prompt(reference, system_output)

    # Call the OpenAI Chat Completions API.
    # The "system" message sets the role; the "user" message carries
    # the actual evaluation request.
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an MQM-certified translation quality evaluator. Respond only with valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    raw_text = response.choices[0].message.content.strip()
    # LLMs sometimes wrap JSON in markdown code fences — strip them
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
        raw_text = raw_text.rsplit("```", 1)[0].strip()

    try:
        evaluation = json.loads(raw_text)
    except json.JSONDecodeError:
        print(f"Error: Could not parse LLM response as JSON:\n{raw_text}")
        return None

    mqm_score = compute_mqm_score(evaluation)

    return {
        "mqm_score": mqm_score,
        "category_scores": {
            cat: evaluation.get(cat, {}) for cat in MQM_CATEGORIES
        },
        "overall_comment": evaluation.get("overall_comment", ""),
    }


# ============================================================
# SECTION 4: JSON FILE PROCESSING
# ============================================================
# This function lets you evaluate many translation pairs at once
# by reading from a JSON file. The input format should be a list
# of objects, each with "reference" and "system_output" keys.
#
# Example input (translations.json):
# [
#   {
#     "id": "sent-1",
#     "reference": "The patient presents with acute chest pain.",
#     "system_output": "The patient has acute chest pain."
#   },
#   {
#     "id": "sent-2",
#     "reference": "Administer 500mg of amoxicillin orally.",
#     "system_output": "Give 500mg amoxicillin by mouth."
#   }
# ]

def evaluate_from_json(input_path, output_path=None):
    """
    Reads translation pairs from a JSON file, scores each one
    with MQM evaluation, and optionally writes results to an
    output JSON file.

    Parameters:
      input_path  - Path to a JSON file containing a list of
                    translation pair objects.
      output_path - (Optional) Path to write the scored results.
                    If None, results are only returned, not saved.

    Returns:
      A list of result dicts, one per translation pair.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for i, item in enumerate(data):
        reference = item.get("reference", "")
        system_output = item.get("system_output", "")
        item_id = item.get("id", f"item-{i}")

        print(f"Evaluating {item_id}...")
        score_data = get_phase2_score(reference, system_output)

        result = {
            "id": item_id,
            "reference": reference,
            "system_output": system_output,
        }
        if score_data:
            result.update(score_data)
        else:
            result["error"] = "Failed to evaluate"

        results.append(result)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults written to {output_path}")

    return results


# ============================================================
# SECTION 5: DEMO / TEST
# ============================================================
if __name__ == "__main__":
    # --- Single-pair test (like Phase 1, but with MQM breakdown) ---
    ref = "The quick brown fox jumps over the lazy dog."
    sys_trans = "A fast brown fox leaps above the lazy human."

    print("=" * 60)
    print("Phase II — MQM Translation Evaluation (Single Pair)")
    print("=" * 60)
    print(f"Reference:  {ref}")
    print(f"System:     {sys_trans}")
    print("-" * 60)

    result = get_phase2_score(ref, sys_trans)

    if result:
        print(f"\nMQM Score: {result['mqm_score']}")
        print(f"\nCategory Breakdown:")
        for cat, data in result["category_scores"].items():
            weight = MQM_CATEGORIES[cat]["weight"]
            score = data.get("score", "N/A")
            explanation = data.get("explanation", "")
            print(f"  {cat:<15} (w={weight}): {score}/10 — {explanation}")
        print(f"\nOverall: {result['overall_comment']}")
    else:
        print("Evaluation failed.")

    # --- JSON file test (uncomment when you have a file ready) ---
    # results = evaluate_from_json("translations.json", "results.json")
    # for r in results:
    #     print(f"{r['id']}: MQM={r.get('mqm_score', 'ERR')}")
