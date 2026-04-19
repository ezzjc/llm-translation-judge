"""
MQM Pipeline: LLM vs Human Rater Comparison
=============================================
Source-based MQM evaluation — exactly as in Freitag et al. (2021).

Key design:
  - Pick n segments present in ALL systems (strict intersection)
  - Each system is scored on the SAME n segments
  - Human avg MQM per system  → system ranking
  - LLM avg MQM per system    → system ranking
  - Kendall τ between the two rankings = main result

Files needed:
  1. mqm_newstest2020_ende.tsv    — human MQM annotations + MT target
  2. en-de.txt                    — index: line_number → doc_id, seg_id
  3. newstest2020-ende-src_en.txt — English source sentences (1 per line)

Usage:
    pip install openai pandas scipy
    export OPENAI_API_KEY="sk-..."
    python mqm_pipeline.py \
        --tsv mqm_newstest2020_ende.tsv \
        --idx en-de.txt \
        --src newstest2020-ende-src.en.txt \
        --n   50
"""

import os
import re
import json
import argparse
import threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from scipy.stats import kendalltau
from openai import OpenAI

# ─────────────────────────────────────────────────────────────
# MQM WEIGHTS  (Freitag et al. 2021, Table 4)
# ─────────────────────────────────────────────────────────────
MQM_WEIGHTS = {
    ("Major",   "Non-translation"):     25,
    ("Major",   "_default"):             5,
    ("Minor",   "Fluency/Punctuation"): 0.1,
    ("Minor",   "_default"):             1,
    ("Neutral", "_default"):             0,
}

# ── ADD YOUR API KEY HERE ──────────────────────────────────────
OPENAI_API_KEY = ""
# ──────────────────────────────────────────────────────────────
# These are overridden at runtime by --model and --n/--seed CLI args.
# Do not edit manually — use CLI args instead.
LLM_MODEL      = "gpt-5.4"
PROGRESS_FILE  = "progress_gpt-5.4_n50_seed42.jsonl"

client = OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def calc_weight(severity: str, category: str) -> float:
    severity = str(severity).strip()
    category = str(category).strip()
    if (severity, category) in MQM_WEIGHTS:
        return MQM_WEIGHTS[(severity, category)]
    return MQM_WEIGHTS.get((severity, "_default"), 0.0)


def safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def extract_json_object(text: str) -> dict:
    """Robustly extract JSON — handles markdown fences and wrapper text."""
    if not text:
        raise json.JSONDecodeError("Empty response", "", 0)
    text   = text.strip()
    fenced = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass
    start = fenced.find("{")
    end   = fenced.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(fenced[start:end + 1])
    raise json.JSONDecodeError("No valid JSON object found", text, 0)


# ─────────────────────────────────────────────────────────────
# STEP 1 — BUILD SOURCE CONTEXT LOOKUP
# ─────────────────────────────────────────────────────────────
def build_context_lookup(idx_path: str, src_path: str) -> dict:
    """
    Returns {(doc_id, seg_id): {"source_sentence": ..., "doc_src_context": ...}}
    """
    with open(idx_path) as f:
        idx_lines = [l.strip().split("\t") for l in f]
    with open(src_path) as f:
        src_lines = [l.strip() for l in f]

    assert len(idx_lines) == len(src_lines), (
        f"Line count mismatch: {len(idx_lines)} vs {len(src_lines)}"
    )

    doc_sentences: dict = defaultdict(list)
    for idx, src in zip(idx_lines, src_lines):
        doc_id = idx[3]
        seg_id = int(idx[4])
        doc_sentences[doc_id].append((seg_id, src))

    for doc_id in doc_sentences:
        doc_sentences[doc_id].sort(key=lambda x: x[0])

    doc_src_full = {
        doc_id: " ".join(s for _, s in segs)
        for doc_id, segs in doc_sentences.items()
    }

    lookup = {}
    for doc_id, segs in doc_sentences.items():
        for seg_id, src_sent in segs:
            lookup[(doc_id, seg_id)] = {
                "source_sentence": src_sent,
                "doc_src_context": doc_src_full[doc_id],
            }

    print(f"Context lookup: {len(lookup)} segments across "
          f"{len(doc_src_full)} documents")
    return lookup


# ─────────────────────────────────────────────────────────────
# STEP 2 — LOAD & MERGE
# ─────────────────────────────────────────────────────────────
def load_and_merge(tsv_path: str, context_lookup: dict,
                   exclude_systems: list = None) -> pd.DataFrame:
    """Load TSV (all systems), compute per-segment human MQM scores
    averaged across raters, and join source context.
    exclude_systems: list of system name prefixes to drop (e.g. ['Human-P'])
    """
    df = pd.read_csv(tsv_path, sep="\t")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"Loaded TSV: {len(df)} annotation rows")

    if exclude_systems:
        before = len(df)
        mask = df["system"].apply(
            lambda s: any(str(s).startswith(ex) for ex in exclude_systems)
        )
        df = df[~mask].copy()
        print(f"Excluded {exclude_systems}: {before} → {len(df)} rows")

    print(f"Systems kept: {sorted(df['system'].unique())}")

    df["penalty"] = df.apply(
        lambda r: calc_weight(
            str(r.get("severity", "")).strip(),
            str(r.get("category", "")).strip()
        ), axis=1,
    )

    # Step 1: sum penalties per (system, doc, seg_id, rater)
    per_rater = df.groupby(
        ["system", "doc", "seg_id", "rater"]
    ).agg(
        rater_score      = ("penalty",  "sum"),
        mt_target        = ("target",   "first"),
        rater_categories = ("category", list),
        rater_severities = ("severity", list),
    ).reset_index()

    # Step 2: average across raters → one human_mqm_score per (system, doc, seg_id)
    agg = per_rater.groupby(["system", "doc", "seg_id"]).agg(
        human_mqm_score  = ("rater_score",      "mean"),
        mt_target        = ("mt_target",         "first"),
        human_categories = ("rater_categories",
                            lambda x: [c for cats in x for c in cats]),
        human_severities = ("rater_severities",
                            lambda x: [s for sevs in x for s in sevs]),
        n_raters         = ("rater",             "count"),
    ).reset_index()

    agg["source_sentence"] = agg.apply(
        lambda r: context_lookup.get(
            (r["doc"], int(r["seg_id"])), {}
        ).get("source_sentence", ""), axis=1,
    )
    agg["doc_src_context"] = agg.apply(
        lambda r: context_lookup.get(
            (r["doc"], int(r["seg_id"])), {}
        ).get("doc_src_context", ""), axis=1,
    )

    print(f"After aggregation: {len(agg)} (system, doc, seg_id) rows")
    return agg


# ─────────────────────────────────────────────────────────────
# STEP 3 — PICK SHARED SEGMENTS
# ─────────────────────────────────────────────────────────────
def pick_shared_segments(agg: pd.DataFrame, n: int = 50, seed: int = 42):
    """
    Find (doc, seg_id) pairs present in EVERY system (strict intersection),
    sample n of them. Every system is evaluated on the exact same segments.
    """
    all_systems = set(agg["system"].unique())
    n_systems   = len(all_systems)
    print(f"\nSystems: {n_systems}  →  {sorted(all_systems)}")

    seg_counts = (
        agg.groupby(["doc", "seg_id"])["system"]
        .nunique()
        .reset_index(name="n_systems")
    )
    shared = seg_counts[seg_counts["n_systems"] == n_systems].copy()
    print(f"Segments present in all {n_systems} systems: {len(shared)}")

    if len(shared) == 0:
        raise ValueError("No shared segments found across all systems.")
    if len(shared) < n:
        print(f"WARNING: only {len(shared)} shared segments — "
              f"using all instead of {n}")
        n = len(shared)

    sampled = shared[["doc", "seg_id"]].sample(n=n, random_state=seed)
    result  = agg.merge(sampled, on=["doc", "seg_id"], how="inner")
    print(f"Final dataset: {len(result)} rows  "
          f"({n_systems} systems × {n} segments)")
    return result, n


# ─────────────────────────────────────────────────────────────
# STEP 4 — PROMPT + LLM EVALUATION
# ─────────────────────────────────────────────────────────────
def build_prompt(source_sentence: str,
                 mt_target: str,
                 doc_src_context: str,
                 src_lang: str = "English",
                 tgt_lang: str = "German") -> str:
    """
   
    """
    return (
        "You are a professional translation quality analyst. You will annotate "
        "a machine translation segment following the MQM (Multidimensional "
        "Quality Metrics) framework exactly as specified in Freitag et al. "
        "(2021), 'Experts, Errors, and Context', TACL.\n\n"
        f"Language pair: {src_lang} → {tgt_lang}\n\n"

        # ── Document context first — mirrors how human raters worked ──────
        "=== DOCUMENT CONTEXT ===\n"
        "Read the full document below BEFORE annotating. "
        "If a translation might be questionable on its own but is fine in the "
        "context of the document, it should not be considered erroneous; "
        "conversely, if a translation might be acceptable in some context, "
        "but not within the current document, it should be marked as wrong. "
        "Do NOT annotate errors in the context — use it only to inform "
        "your judgement on the segment below.\n\n"
        f"{doc_src_context}\n\n"

        "=== SEGMENT TO ANNOTATE ===\n"
        f"{src_lang} source:    {source_sentence}\n"
        f"{tgt_lang} MT output: {mt_target}\n\n"

        ##prompt here
    )


def evaluate_with_llm(source_sentence: str,
                      mt_target: str,
                      doc_src_context: str,
                      src_lang: str = "English",
                      tgt_lang: str = "German") -> dict:
    prompt = build_prompt(source_sentence, mt_target, doc_src_context,
                          src_lang=src_lang, tgt_lang=tgt_lang)
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            timeout=30,
        )
    except Exception as e:
        print(f"  Warning: API call failed ({e})")
        return {"errors": [], "source_errors": [], "llm_mqm_score": 0.0,
                "llm_source_score": 0.0, "llm_categories": [],
                "llm_severities": [], "overall_comment": f"api_error: {e}"}

    raw = (response.choices[0].message.content or "").strip()
    try:
        result = extract_json_object(raw)
    except json.JSONDecodeError:
        print("  Warning: JSON parse error")
        return {"errors": [], "source_errors": [], "llm_mqm_score": 0.0,
                "llm_source_score": 0.0, "llm_categories": [],
                "llm_severities": [], "overall_comment": "parse_error"}

    errors        = result.get("errors", [])
    if not isinstance(errors, list):
        errors = []

    target_errors = [e for e in errors if not e.get("is_source_error", False)][:5]
    source_errors = [e for e in errors if     e.get("is_source_error", False)]

    llm_score    = sum(calc_weight(e.get("severity",""), e.get("category",""))
                       for e in target_errors)
    source_score = sum(calc_weight(e.get("severity",""), e.get("category",""))
                       for e in source_errors)

    return {
        "errors":           target_errors,
        "source_errors":    source_errors,
        "llm_mqm_score":    llm_score,
        "llm_source_score": source_score,
        "llm_categories":   [e.get("category","") for e in target_errors],
        "llm_severities":   [e.get("severity", "") for e in target_errors],
        "overall_comment":  result.get("overall_comment", ""),
    }


# ─────────────────────────────────────────────────────────────
# PROGRESS FILE
# ─────────────────────────────────────────────────────────────
def load_completed(progress_file: str) -> set:
    """Return set of (system, doc, seg_id) already saved."""
    done = set()
    if not Path(progress_file).exists():
        return done
    with open(progress_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r      = json.loads(line)
                seg_id = safe_int(r["seg_id"])
                if seg_id is not None:
                    done.add((r["system"], r["doc"], seg_id))
            except Exception:
                print(f"  Warning: skipping malformed progress line {i}")
    return done


def load_completed_records(progress_file: str) -> list:
    records = []
    if not Path(progress_file).exists():
        return records
    with open(progress_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                print(f"  Warning: skipping malformed JSONL line {i}")
    return records


# ─────────────────────────────────────────────────────────────
# STEP 4C — RUN LLM ON PAIRS (parallel + resume + timeout)
# ─────────────────────────────────────────────────────────────
def run_llm_on_pairs(pairs_df: pd.DataFrame,
                     src_lang: str = "English",
                     tgt_lang: str = "German",
                     max_workers: int = 10) -> pd.DataFrame:
    """
    Parallel LLM evaluation with:
      - Parallelism : max_workers concurrent API calls
      - Resume      : skips pairs already in PROGRESS_FILE
      - Timeout     : 30s per call (in evaluate_with_llm)
    """
    write_lock = threading.Lock()

    done              = load_completed(PROGRESS_FILE)
    completed_records = load_completed_records(PROGRESS_FILE)

    pending = pairs_df[
        ~pairs_df.apply(
            lambda r: (r["system"], r["doc"], int(r["seg_id"])) in done,
            axis=1,
        )
    ].copy()

    total_all  = len(pairs_df)
    total_todo = len(pending)
    skipped    = total_all - total_todo

    if skipped:
        print(f"  Resuming: {skipped} pairs already done, {total_todo} remaining.")
    else:
        print(f"  Starting fresh: {total_todo} pairs to evaluate.")

    if total_todo == 0:
        print("  All pairs already evaluated — loading from progress file.")
        return pd.DataFrame(completed_records)

    counter     = {"n": skipped}
    new_records = []

    def process_row(row_dict: dict) -> dict:
        llm = evaluate_with_llm(
            source_sentence = row_dict["source_sentence"],
            mt_target       = row_dict["mt_target"],
            doc_src_context = row_dict["doc_src_context"],
            src_lang        = src_lang,
            tgt_lang        = tgt_lang,
        )
        record = {
            "system":           row_dict["system"],
            "doc":              row_dict["doc"],
            "seg_id":           int(row_dict["seg_id"]),
            "source_sentence":  row_dict["source_sentence"],
            "mt_target":        row_dict["mt_target"],
            "human_mqm_score":  float(row_dict["human_mqm_score"]),
            # ── Serialise lists as JSON strings ─────────────────────────
            # Permanent fix for "unhashable type: list":
            # JSONL reload brings these back as strings, not lists,
            # so pandas groupby never tries to hash them.
            "human_categories": json.dumps(row_dict["human_categories"]),
            "human_severities": json.dumps(row_dict["human_severities"]),
            "llm_mqm_score":    float(llm["llm_mqm_score"]),
            "llm_source_score": float(llm["llm_source_score"]),
            "llm_categories":   json.dumps(llm["llm_categories"]),
            "llm_severities":   json.dumps(llm["llm_severities"]),
            "llm_errors":       json.dumps(llm["errors"]),
            "llm_source_errors":json.dumps(llm["source_errors"]),
            "llm_comment":      llm["overall_comment"],
        }
        with write_lock:
            counter["n"] += 1
            print(f"  [{counter['n']}/{total_all}] system={row_dict['system']}  "
                  f"doc={row_dict['doc']}  seg_id={row_dict['seg_id']}")
            with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        return record

    rows = pending.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_row, row): row for row in rows}
        for future in as_completed(futures):
            try:
                new_records.append(future.result())
            except Exception as e:
                print(f"  Warning: worker failed: {e}")

    return pd.DataFrame(completed_records + new_records)


# ─────────────────────────────────────────────────────────────
# STEP 5 — SYSTEM-LEVEL RANKING ANALYSIS
# ─────────────────────────────────────────────────────────────
def analyse_system_rankings(df: pd.DataFrame) -> None:
    """
    For each system, compute avg human and avg LLM MQM score across
    the shared segments, rank systems by each, then compare with Kendall τ.
    Lower MQM = fewer/less severe errors = better → Rank 1 = best.
    """
    print("\n" + "=" * 70)
    print("  SYSTEM-LEVEL RANKING: LLM vs Human")
    print(f"  Model: {LLM_MODEL}")
    print("=" * 70)

    df = df.copy()

    # ── Coerce types — guards against old-format JSONL on reload ─────────
    df["human_mqm_score"] = pd.to_numeric(df["human_mqm_score"], errors="coerce").fillna(0.0)
    df["llm_mqm_score"]   = pd.to_numeric(df["llm_mqm_score"],   errors="coerce").fillna(0.0)
    df["seg_id"]          = pd.to_numeric(df["seg_id"],           errors="coerce")
    df = df.dropna(subset=["seg_id"]).copy()
    df["seg_id"] = df["seg_id"].astype(int)

    # ── Groupby on numeric-only subset (permanent unhashable-list fix) ───
    system_stats = (
        df[["system", "human_mqm_score", "llm_mqm_score", "seg_id"]]
        .groupby("system", as_index=False)
        .agg(
            human_avg = ("human_mqm_score", "mean"),
            llm_avg   = ("llm_mqm_score",   "mean"),
            n_segs    = ("seg_id",           "nunique"),
        )
    )

    system_stats["human_rank"] = system_stats["human_avg"].rank(
        method="average", ascending=True)
    system_stats["llm_rank"]   = system_stats["llm_avg"].rank(
        method="average", ascending=True)
    system_stats["rank_diff"]  = system_stats["llm_rank"] - system_stats["human_rank"]
    system_stats = system_stats.sort_values("human_rank").reset_index(drop=True)

    n_segs = int(system_stats["n_segs"].iloc[0])
    print(f"\n  Segments per system (shared): {n_segs}")
    print(f"  Systems evaluated:            {len(system_stats)}")
    print(f"\n  Lower score = fewer/less severe errors = better")
    print(f"  Rank 1 = best system\n")

    col_w  = max(len(s) for s in system_stats["system"]) + 2
    header = (f"  {'System':<{col_w}} {'HumanAvg':>9} {'HumanRank':>10} "
              f"{'LLMAvg':>9} {'LLMRank':>9} {'RankDiff':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for _, row in system_stats.iterrows():
        diff   = row["rank_diff"]
        marker = "  ←" if abs(diff) >= 3 else ""
        print(f"  {row['system']:<{col_w}} "
              f"{row['human_avg']:>9.3f} "
              f"{int(row['human_rank']):>10} "
              f"{row['llm_avg']:>9.3f} "
              f"{int(row['llm_rank']):>9} "
              f"{diff:>+9.1f}{marker}")

    # ── Kendall τ ─────────────────────────────────────────────────────────
    tau, p_value = kendalltau(
        system_stats["human_rank"].values,
        system_stats["llm_rank"].values,
    )
    seg_tau, seg_p = kendalltau(
        df["human_mqm_score"].values,
        df["llm_mqm_score"].values,
    )

    print(f"\n  {'─'*52}")
    print(f"  Kendall τ (system-level):  {tau:>8.4f}  p={p_value:.4f}")
    print(f"  Kendall τ (segment-level): {seg_tau:>8.4f}  p={seg_p:.4f}")
    print(f"  System-level τ is typically higher — averaging smooths noise.")

    if   tau >= 0.8: verdict = "Strong — LLM ranks systems similarly to humans."
    elif tau >= 0.5: verdict = "Moderate — LLM ranking partially matches humans."
    elif tau >= 0.2: verdict = "Weak — LLM and human rankings diverge notably."
    else:            verdict = "Poor — LLM rankings do not reflect human judgement."
    print(f"\n  Interpretation: {verdict}")

    # ── Category & severity counts ─────────────────────────────────────────
    # Normalise by avg annotations per segment so both columns represent
    # "errors per single evaluation" — apples to apples.
    def normalise_category(cat: str) -> str:
        cat = str(cat).strip()
        for top in ("Accuracy", "Fluency", "Terminology", "Style",
                    "SourceError", "Non-translation"):
            if cat.startswith(top):
                return top
        return "Other" if cat not in ("no-error", "No-error") else "No-error"

    # Parse list columns — stored as JSON strings in JSONL
    def parse_list_col(val):
        if isinstance(val, list):
            return val
        try:
            return json.loads(val)
        except Exception:
            return []

    n_rows     = len(df)
    n_raters   = df["n_raters"].median() if "n_raters" in df.columns else 3.0

    total_human_raw = sum(len(parse_list_col(r)) for r in df["human_categories"])
    total_llm_raw   = sum(len(parse_list_col(r)) for r in df["llm_categories"])

    # Error rate = errors per single (system, segment) evaluation
    # Human: divide by n_raters to get one-rater-equivalent rate
    # LLM:   already 1 call per row, no division needed
    human_rate = (total_human_raw / n_raters) / n_rows if n_rows else 0
    llm_rate   = total_llm_raw / n_rows if n_rows else 0

    llm_zero   = sum(1 for r in df["llm_categories"]   if len(parse_list_col(r)) == 0)
    human_zero = sum(1 for r in df["human_categories"] if len(parse_list_col(r)) == 0)

    human_zero_rate = (human_zero / n_raters) / n_rows if n_rows else 0
    llm_zero_rate   = llm_zero / n_rows if n_rows else 0

    print(f"\n  Rows (systems × segments):       {n_rows}")
    print(f"  Median raters per row:           {n_raters:.0f}")
    print(f"\n  Avg errors per evaluation:")
    print(f"    Human (per rater):  {human_rate:.2f}")
    print(f"    LLM   (per call):   {llm_rate:.2f}")
    print(f"    Ratio LLM/Human:    {llm_rate/human_rate:.2f}x" if human_rate else "")
    print(f"\n  Zero-error rate:")
    print(f"    Human: {human_zero_rate*100:.1f}% of rows")
    print(f"    LLM:   {llm_zero_rate*100:.1f}% of rows\n")

    from collections import defaultdict

    # Build per-category raw counts
    hc: dict = defaultdict(float)
    lc: dict = defaultdict(float)
    for _, row in df.iterrows():
        n_rat = max(row.get("n_raters", n_raters), 1)
        for c in parse_list_col(row["human_categories"]):
            hc[normalise_category(c)] += 1.0 / n_rat   # per-rater contribution
        for c in parse_list_col(row["llm_categories"]):
            lc[normalise_category(c)] += 1.0            # raw LLM count

    # Convert to rates (per evaluation)
    all_cats = sorted(set(list(hc) + list(lc)))
    print(f"  Error Category Rate  (errors per single evaluation)")
    print(f"  {'Category':<25} {'Human':>8} {'LLM':>8} {'Diff':>8}  {'Ratio (counts)':>20}")
    print(f"  {'-'*75}")
    for cat in all_cats:
        h = hc.get(cat, 0.0) / n_rows
        l = lc.get(cat, 0.0) / n_rows
        h_raw = int(round(hc.get(cat, 0.0)))
        l_raw = int(lc.get(cat, 0))
        ratio = f"{l/h:.1f}x" if h > 0 else "n/a"
        print(f"  {cat:<25} {h:>8.3f} {l:>8.3f} {l-h:>+8.3f}  {ratio:>5} ({h_raw:>4} vs {l_raw:>4})")
    print(f"  {'TOTAL (avg errors/eval)':<25} {human_rate:>8.3f} {llm_rate:>8.3f} "
          f"{llm_rate-human_rate:>+8.3f}  "
          f"{f'{llm_rate/human_rate:.1f}x' if human_rate else 'n/a':>5} "
          f"({int(round(total_human_raw/n_raters)):>4} vs {total_llm_raw:>4})")

    # Severity rates
    hs: dict = defaultdict(float)
    ls: dict = defaultdict(float)
    for _, row in df.iterrows():
        n_rat = max(row.get("n_raters", n_raters), 1)
        for s in parse_list_col(row["human_severities"]):
            hs[str(s).strip()] += 1.0 / n_rat
        for s in parse_list_col(row["llm_severities"]):
            ls[str(s).strip()] += 1.0

    all_sevs = sorted(set(list(hs) + list(ls)))
    print(f"\n  Error Severity Rate  (errors per single evaluation)")
    print(f"  {'Severity':<25} {'Human':>8} {'LLM':>8} {'Diff':>8}  {'Ratio (counts)':>20}")
    print(f"  {'-'*75}")
    for sev in all_sevs:
        h = hs.get(sev, 0.0) / n_rows
        l = ls.get(sev, 0.0) / n_rows
        h_raw = int(round(hs.get(sev, 0.0)))
        l_raw = int(ls.get(sev, 0))
        ratio = f"{l/h:.1f}x" if h > 0 else "n/a"
        print(f"  {sev:<25} {h:>8.3f} {l:>8.3f} {l-h:>+8.3f}  {ratio:>5} ({h_raw:>4} vs {l_raw:>4})")

    # ── Save ──────────────────────────────────────────────────────────────
    system_stats.to_csv("system_rankings.csv", index=False)
    df.drop(columns=["llm_errors", "llm_source_errors", "doc_src_context"],
            errors="ignore").to_csv("mqm_comparison_results.csv", index=False)
    print(f"\n  Saved: system_rankings.csv  |  mqm_comparison_results.csv")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="MQM Pipeline: LLM vs Human System Ranking")
    parser.add_argument("--tsv",      required=True,
                        help="mqm_newstest2020_ende.tsv")
    parser.add_argument("--idx",      required=True,
                        help="en-de.txt (segment index)")
    parser.add_argument("--src",      required=True,
                        help="newstest2020-ende-src_en.txt")
    parser.add_argument("--n",        type=int, default=50,
                        help="Segments to sample (must exist in ALL systems)")
    parser.add_argument("--src-lang", default="English",
                        help="Source language name (default: English)")
    parser.add_argument("--tgt-lang", default="German",
                        help="Target language name (default: German)")
    parser.add_argument("--workers",  type=int, default=10,
                        help="Parallel API workers (default: 10)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed for segment sampling (default: 42)")
    parser.add_argument("--model",    default="gpt-4.1",
                        help="OpenAI model string (default: gpt-4.1)")
    parser.add_argument("--exclude",  nargs="*", default=["Human-P"],
                        help="System name prefixes to exclude (default: Human-P)")
    args = parser.parse_args()

    # ── Set globals from CLI so all functions pick up the right model ─────
    global LLM_MODEL, PROGRESS_FILE
    LLM_MODEL     = args.model
    # Each model gets its own progress file — runs never interfere
    safe_model    = args.model.replace("/", "-").replace(":", "-")
    PROGRESS_FILE = f"progress_{safe_model}_n{args.n}_seed{args.seed}.jsonl"

    print("=" * 70)
    print("  MQM Pipeline: LLM vs Human System Ranking")
    print(f"  Model:               {LLM_MODEL}")
    print(f"  Language pair:       {args.src_lang} → {args.tgt_lang}")
    print(f"  Segments per system: {args.n}  (strict intersection)")
    print(f"  Parallel workers:    {args.workers}")
    print(f"  Random seed:         {args.seed}")
    print(f"  Excluded systems:    {args.exclude}")
    print(f"  Progress file:       {PROGRESS_FILE}")
    print("=" * 70)

    print("\n[1] Building source context lookup...")
    context_lookup = build_context_lookup(args.idx, args.src)

    print("\n[2] Loading TSV and merging context...")
    agg_df = load_and_merge(args.tsv, context_lookup,
                            exclude_systems=args.exclude)

    print(f"\n[3] Selecting {args.n} segments shared across ALL systems...")
    pairs_df, actual_n = pick_shared_segments(agg_df, n=args.n, seed=args.seed)

    print(f"\n[4] Running LLM on {len(pairs_df)} rows "
          f"({pairs_df['system'].nunique()} systems × {actual_n} segments)...")
    results_df = run_llm_on_pairs(pairs_df,
                                  src_lang=args.src_lang,
                                  tgt_lang=args.tgt_lang,
                                  max_workers=args.workers)

    print("\n[5] Analysing system rankings...")
    analyse_system_rankings(results_df)

    # ── Save per-model rankings for comparison script ──────────────────────
    out_csv = f"rankings_{safe_model}_n{args.n}_seed{args.seed}.csv"
    import shutil
    shutil.copy("system_rankings.csv", out_csv)
    print(f"  Also saved: {out_csv}")


if __name__ == "__main__":
    main()