
#load python built-in JSON utilites
import json
#provide access to operating systems features
import os
import time
import google.generativeai as genai

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

# attempts to read an environmrnt variable named "GEMINI_API_KEY" and if it is not found, it defaults to
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAGT15x6iNe86NB1jJvEMXpBBPz7I47X5o")
#a configuration line before using the model
genai.configure(api_key=API_KEY)

# Initialize the model
model = genai.GenerativeModel("gemini-2.0-flash")

# # MQM penalty weights 
# MQM_PENALTIES = {
#     "Minor": 1,
#     "Major": 5,
#     "Critical": 25
# }


# ─────────────────────────────────────────────
# 2. CORE EVALUATION FUNCTION
# ─────────────────────────────────────────────

#takes reference translation and system transaltion,returns a dictionary pair (error: value)
#retries 3 time calling SAPI and parse response as valid JSON(specific format)
def get_mqm_evaluation(reference: str, system_output: str, retries: int = 3) -> dict | None:
    """
    Evaluates a system translation against a reference using MQM principles.

    Args:
        reference:     The gold-standard / reference translation.
        system_output: The translation produced by the MT system.
        retries:       Number of retry attempts on API failure.

    Returns:
        A dictionary containing errors, summary statistics, and MQM scores,
        or None if all retries fail.
    """

#builds large prompt sting using a f-string, inserts reference and system output
    prompt = f"""
    You are an expert linguistic reviewer using the Multidimensional Quality Metrics (MQM) framework.
    Evaluate the System Translation against the Reference Translation.

    Reference Translation: {reference}
    System Translation:    {system_output}

    Identify ALL errors in the System Translation based on these categories:
    - Accuracy    (sub-types: Mistranslation, Omission, Addition)
    - Fluency     (sub-types: Grammar, Spelling, Punctuation)
    - Terminology (sub-types: Inconsistent Term, Wrong Term)

    For each error, assign a severity:
    - Minor      — noticeable but does not affect meaning
    - Major     — affects meaning or usability
    - Critical  — completely changes meaning or is offensive

    Return output EXCLUSIVELY as valid JSON in this exact format:
    {{
        "errors": [
            {{
                "text_span": "the exact word or phrase in the system output that contains the error",
                "category":  "Accuracy | Fluency | Terminology",
                "sub_type":  "Mistranslation | Omission | Addition | Grammar | Spelling | Punctuation | Inconsistent Term | Wrong Term",
                "severity":  "Minor | Major | Critical",
                "explanation": "brief reason"
            }}
        ]
    }}
    If there are no errors, return: {{"errors": []}}
    Do NOT include any numeric scoring fields.
    Do NOT include any text outside the JSON object.
    """

#retry loop, where if API call fails or response not in valid JSON, tries up to 3 times (packets exist for retries)
    for attempt in range(retries):
        #starts exception handling for API call and JSON parsing
        try:
            #calls Gemnin with prompt as input
            response = model.generate_content(
                prompt,
                #response_mime_type hints return of JSON, no guarantee
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
                )
            )

            #parse model response text into a Python dictionary, if not valide JSON will raise error
            evaluation_data = json.loads(response.text)
            #reads errors list from parsed dict
            errors = evaluation_data.get("errors", [])


            ##for each error, reads severity and look up penalty weight in MQM_PENALTIES, add them up
            # # ── Calculate penalties ──────────────────────────────────────
            # total_penalty = sum(MQM_PENALTIES.get(e.get("severity"), 0) for e in errors)

            ##converts penality to a score on 100 scale (max prevents negative score)
            # # ── Flat score (100-point scale) ─────────────────────────────
            # flat_score = max(0, 100 - total_penalty)

            ##computes normalization
                    ##total penalty divided by number of words, scaled to 100 words, then rounded to 2 decimals
            # # ── Word-normalized MQM score (industry standard) ───────────
            # # Formula: penalty per 100 words; lower is better
            # word_count = max(1, len(system_output.split()))
            # normalized_penalty = round((total_penalty / word_count) * 100, 2)

            # ── Error breakdown summary ──────────────────────────────────
                ##builds a structured count summary: 
            
            summary = {
                "total_errors": len(errors),
                #total errors: number of errors found
                "by_severity": {
                    "Critical": sum(1 for e in errors if e.get("severity") == "Critical"),
                    "Major":    sum(1 for e in errors if e.get("severity") == "Major"),
                    "Minor":    sum(1 for e in errors if e.get("severity") == "Minor"),
                },
                #counts # of critical, major, minor
                "by_category": {
                    "Accuracy":    sum(1 for e in errors if e.get("category") == "Accuracy"),
                    "Fluency":     sum(1 for e in errors if e.get("category") == "Fluency"),
                    "Terminology": sum(1 for e in errors if e.get("category") == "Terminology"),
                }
            }   #counts categories

            #summary is useless
            return {
                "errors": errors,
                "error_summary": summary
            }
            # # ── Assemble final result ────────────────────────────────────
            # evaluation_data["error_summary"]              = summary
            #     #adds the computed summary into same dict
            # evaluation_data["total_penalty"]              = total_penalty
            #     #adds total penality points
            # evaluation_data["flat_mqm_score"]             = flat_score          # higher = better, max 100
            #     #adds 100 based score
            # evaluation_data["normalized_penalty_per_100w"] = normalized_penalty # lower = better
            #     #adds normalized penalty metric
            # return evaluation_data



        #error handling: if Gemini returns non-jason, print message
        except json.JSONDecodeError:
            print(f"[Attempt {attempt + 1}] Failed to decode JSON from model response.")
        except Exception as e:
            #if other errors, print that
            print(f"[Attempt {attempt + 1}] API error: {e}")

        #if there's at least one attempt remaining, wait and retry
        if attempt < retries - 1:
            wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
            #exponentially retry
            print(f"Retrying in {wait}s...")
            time.sleep(wait)
    
    #else print the retries all failed
    print("All retry attempts exhausted. Returning None.")
    return None


# ─────────────────────────────────────────────
# 3. BATCH EVALUATION FUNCTION
# ─────────────────────────────────────────────

    #function that takes many (reference, system_output) pairs and evalutes them 
    #wrapper letting you run MQM checker on many examples in one go
def batch_evaluate(pairs: list[tuple[str, str]]) -> list[dict | None]:
    """
    Evaluates multiple (reference, system_output) pairs.

    Args:
        pairs: A list of (reference, system_output) tuples.

    Returns:
        A list of evaluation result dictionaries (None for failed evaluations).
    """
    #output bucket storing each evaluation reprot
    results = []
    #loop over every pair
    for i, (ref, sys) in enumerate(pairs):
        #each pair gets evaluated and append it to the result buckt
        result = get_mqm_evaluation(ref, sys)
        results.append(result)
    return results


# ─────────────────────────────────────────────
# 4. DISPLAY HELPER
# ─────────────────────────────────────────────

def print_report(result: dict, label: str = "Evaluation Report") -> None:
    """Prints a formatted, human-readable MQM report."""
    if not result:
        print("No result to display.")
        return

    print(f"\n{'═'*55}")
    print(f"  {label}")
    print(f"{'═'*55}")

    summary = result.get("error_summary", {})
    errors = result.get("errors", [])
    # print(f"\n📊 SCORES")
    # print(f"  Flat MQM Score (100-pt scale): {result.get('flat_mqm_score')} / 100")
    # print(f"  Normalized Penalty / 100 words: {result.get('normalized_penalty_per_100w')}")
    # print(f"  Total Penalty Points:           {result.get('total_penalty')}")

    # print(f"\n📋 ERROR SUMMARY")
    # print(f"  Total Errors: {summary.get('total_errors', 0)}")
    # by_sev = summary.get("by_severity", {})
    # print(f"  ├─ Critical : {by_sev.get('Critical', 0)}")
    # print(f"  ├─ Major    : {by_sev.get('Major', 0)}")
    # print(f"  └─ Minor    : {by_sev.get('Minor', 0)}")


    #error overview
    print(f" ERROR OVERVIEW")
    print(f"  Total Errors: {summary.get('total_errors', 0)}")



    by_cat = summary.get("by_category", {})
    print(f"\n  By Category:")
    print(f"  ├─ Accuracy    : {by_cat.get('Accuracy', 0)}")
    print(f"  ├─ Fluency     : {by_cat.get('Fluency', 0)}")
    print(f"  └─ Terminology : {by_cat.get('Terminology', 0)}")

   
    if errors:
        print(f"\n🔍 DETAILED ERRORS")
        for i, err in enumerate(errors, 1):
            sev_icon = {"Critical": "🔴", "Major": "🟠", "Minor": "🟡"}.get(err.get("severity"), "⚪")
            print(f"\n  Error #{i}  {sev_icon} {err.get('severity')} | {err.get('category')} > {err.get('sub_type')}")
            print(f"    Span:    \"{err.get('text_span')}\"")
            print(f"    Reason:  {err.get('explanation')}")
    else:
        print("\n✅ No errors found.")

    print(f"\n{'═'*55}\n")


# ─────────────────────────────────────────────
# 5. EXAMPLE USAGE
# ─────────────────────────────────────────────

#run code inside block when file is executed directly, but not when imported as a module
if __name__ == "__main__":

    # ── Single evaluation ────────────────────────────────────────────────
    
    #"gold standard" sentence (correct translation)
    reference   = "The company reported a massive increase in quarterly profits."
    #system translation to judge 
    system_out  = "The company said a big go up in dumplings for the year."


   #calls main evaluator with the two strings, store returned dictionary in result
   #result contains errors (list) and error_summary (counts)
    result = get_mqm_evaluation(reference, system_out)

    #call helper function to present the result
    # Human-readable report
    print_report(result, label="Single Evaluation Report")

    #print raw JSON
    # Full raw JSON output
    print("Raw JSON output:")
    print(json.dumps(result, indent=2))


    # ── Batch evaluation ─────────────────────────────────────────────────
    #prints a separator line 
    print("\n" + "─"*55)

    pairs = [
        (
            "She left the keys on the kitchen table.",
            "She placed the keys on the kitchen table."
        ),
        (
            "The patient must take two tablets daily after meals.",
            "Patient should take 2 tablet daily."
        ),
        (
            "Please ensure all documents are signed before departure.",
            "Please ensure all documents are signed before departure."  # perfect match
        ),
    ]

    #after creating a list named pairs, we call batch_evaluate helper on them
    batch_results = batch_evaluate(pairs)

    #print each batch result
    for i, (res, pair) in enumerate(zip(batch_results, pairs)):
        print_report(res, label=f"Batch Pair {i+1}")