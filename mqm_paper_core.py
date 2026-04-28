"""Core helpers for paper-faithful MQM benchmarking.

This module is intentionally API-free. It handles:
- MQM hierarchy and weighting
- parsing Google MQM TSV files
- reconstructing system-output segments with document context
- human annotation aggregation
- benchmark analysis summaries
- simple SVG report generation
"""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from enum import Enum
from html import escape
from pathlib import Path
from statistics import mean
from typing import Iterable

from pydantic import BaseModel, Field

MAX_ERRORS_PER_SEGMENT = 5
MAX_SEGMENT_MQM_SCORE = 25.0

CATEGORY_ORDER = [
    "Accuracy",
    "Fluency",
    "Terminology",
    "Style",
    "Locale convention",
    "Other",
    "Source error",
    "Non-translation",
    "No-error",
]

SEVERITY_ORDER = [
    "Major",
    "Minor",
    "Neutral",
    "no-error",
]

MQM_HIERARCHY = {
    "Accuracy": (
        "Addition",
        "Omission",
        "Mistranslation",
        "Untranslated text",
    ),
    "Fluency": (
        "Punctuation",
        "Spelling",
        "Grammar",
        "Register",
        "Inconsistency",
        "Character encoding",
    ),
    "Terminology": (
        "Inappropriate for context",
        "Inconsistent use",
    ),
    "Style": ("Awkward",),
    "Locale convention": (
        "Address format",
        "Currency format",
        "Date format",
        "Name format",
        "Telephone format",
        "Time format",
    ),
    "Other": (),
    "Source error": (),
    "Non-translation": (),
}

CATEGORY_ALIASES = {
    "accuracy": "Accuracy",
    "fluency": "Fluency",
    "terminology": "Terminology",
    "style": "Style",
    "locale": "Locale convention",
    "locale convention": "Locale convention",
    "other": "Other",
    "source error": "Source error",
    "non translation": "Non-translation",
    "no error": "No-error",
}

SUBCATEGORY_ALIASES = {
    "addition": "Addition",
    "omission": "Omission",
    "mistranslation": "Mistranslation",
    "untranslated text": "Untranslated text",
    "punctuation": "Punctuation",
    "spelling": "Spelling",
    "grammar": "Grammar",
    "register": "Register",
    "inconsistency": "Inconsistency",
    "character encoding": "Character encoding",
    "inappropriate for context": "Inappropriate for context",
    "inconsistent use": "Inconsistent use",
    "awkward": "Awkward",
    "address format": "Address format",
    "currency format": "Currency format",
    "date format": "Date format",
    "name format": "Name format",
    "telephone format": "Telephone format",
    "time format": "Time format",
    "": "",
}

SEVERITY_ALIASES = {
    "major": "Major",
    "minor": "Minor",
    "neutral": "Neutral",
    "no error": "no-error",
}

SEVERITY_WEIGHTS = {
    "Major": 5.0,
    "Minor": 1.0,
    "Neutral": 0.0,
}

VALID_CATEGORIES_TEXT = "\n".join(
    f"{category} ({' | '.join(subcategories)})" if subcategories else category
    for category, subcategories in MQM_HIERARCHY.items()
)


class Severity(str, Enum):
    MAJOR = "Major"
    MINOR = "Minor"
    NEUTRAL = "Neutral"


class MQMError(BaseModel):
    error_span: str = Field(
        description="Exact text in the system translation that contains the problem."
    )
    category: str = Field(
        description="Top-level MQM category such as Accuracy, Fluency, Terminology, Style, Locale convention, Other, Source error, or Non-translation."
    )
    subcategory: str = Field(
        default="",
        description="Specific MQM subcategory. Use an empty string for singleton categories."
    )
    severity: Severity = Field(
        description="MQM severity: Major, Minor, or Neutral."
    )
    explanation: str = Field(
        default="",
        description="Brief reason for the annotation."
    )


class PaperMQMEvaluation(BaseModel):
    errors: list[MQMError] = Field(default_factory=list)
    overall_comment: str = Field(default="")


@dataclass(frozen=True)
class SegmentKey:
    system: str
    doc: str
    doc_id: int
    seg_id: int


@dataclass
class RaterAggregate:
    rater: str
    errors: list[dict]
    segment_mqm_score: float
    category_counts: dict[str, float]
    severity_counts: dict[str, float]


@dataclass
class SegmentUnit:
    key: SegmentKey
    source_segment: str
    target_segment: str
    source_document: str
    target_document: str
    raters: list[RaterAggregate]
    human_mean_segment_score: float
    human_mean_category_counts: dict[str, float]
    human_mean_severity_counts: dict[str, float]

    @property
    def sample_key(self) -> str:
        return make_sample_key(self.key)

    @property
    def system_kind(self) -> str:
        return "human" if self.key.system.startswith("Human-") else "mt"


def canonical_key(text: str) -> str:
    """Normalize a string for dictionary lookups by lowercasing, replacing
    underscores/hyphens with spaces, and collapsing whitespace.

    Example: "Locale_Convention" -> "locale convention"
    """
    return " ".join(text.strip().lower().replace("_", " ").replace("-", " ").split())


def canonicalize_severity(severity: str) -> str:
    """Map a raw severity string to its canonical form ("Major", "Minor", "Neutral", or "no-error").

    Uses SEVERITY_ALIASES for known variants; falls back to title-casing the input.
    """
    raw = canonical_key(severity)
    return SEVERITY_ALIASES.get(raw, severity.strip().title())


def canonicalize_label(category: str, subcategory: str = "") -> tuple[str, str]:
    """Normalize an MQM category/subcategory pair to their canonical forms.

    Handles several edge cases:
      - "Accuracy/Omission" in the category field is split into category + subcategory.
      - Case-insensitive lookup via CATEGORY_ALIASES and SUBCATEGORY_ALIASES.
      - Singleton categories (No-error, Source error, Non-translation) always
        return an empty subcategory.

    Returns:
        A (category, subcategory) tuple with canonical casing.
    """
    raw_category = (category or "").strip()
    raw_subcategory = (subcategory or "").strip()

    if "/" in raw_category and not raw_subcategory:
        raw_category, raw_subcategory = [part.strip() for part in raw_category.split("/", 1)]

    canonical_category = CATEGORY_ALIASES.get(canonical_key(raw_category), raw_category)
    canonical_subcategory = SUBCATEGORY_ALIASES.get(canonical_key(raw_subcategory), raw_subcategory)

    if canonical_category in {"No-error", "Source error", "Non-translation"}:
        return canonical_category, ""
    return canonical_category, canonical_subcategory


def strip_span_markup(text: str) -> str:
    """Remove <v>...</v> markup tags from WMT annotation text.

    The WMT TSV files wrap error spans in <v> tags to highlight the
    problematic text within the full target segment.
    """
    return text.replace("<v>", "").replace("</v>", "")


def extract_marked_span(text: str) -> str:
    """Extract the error span text from a WMT target field.

    If <v> tags are present, returns the tagged text (multiple spans joined
    with " | "). If no tags are found, returns the full text with any
    leftover markup stripped — this handles rows where the entire segment
    is considered the error span.
    """
    spans = [span.strip() for span in re.findall(r"<v>(.*?)</v>", text) if span.strip()]
    if spans:
        return " | ".join(spans)
    return strip_span_markup(text).strip()


def empty_counts(labels: Iterable[str]) -> dict[str, float]:
    """Create a zero-initialized count dict for the given labels.

    Used to seed category and severity count dicts so every expected
    key is present even when no errors of that type were found.
    """
    return {label: 0.0 for label in labels}


def error_weight(severity: str, category: str, subcategory: str) -> float:
    """Compute the penalty weight for a single MQM error (Freitag et al., 2021).

    Weighting rules:
      - No-error / Source error:               0   (not counted)
      - Non-translation:                       25  (caps the segment score)
      - Neutral severity:                      0   (subjective preference)
      - Minor Fluency/Punctuation:             0.1 (cosmetic)
      - Minor (all other categories):          1
      - Major:                                 5
    """
    severity = canonicalize_severity(severity)
    category, subcategory = canonicalize_label(category, subcategory)

    if category in {"No-error", "Source error"}:
        return 0.0
    if category == "Non-translation":
        return MAX_SEGMENT_MQM_SCORE
    if severity == "Neutral":
        return 0.0
    if category == "Fluency" and subcategory == "Punctuation" and severity == "Minor":
        return 0.1
    return SEVERITY_WEIGHTS.get(severity, 0.0)


def score_mqm_errors(errors: list[dict]) -> dict:
    """Normalize a list of raw MQM error dicts and compute the segment penalty score.

    For each error:
      1. Canonicalizes category, subcategory, and severity labels.
      2. Looks up the penalty weight via error_weight().
      3. Adds the weight field to the error dict.

    Returns a dict with:
      - errors:             list of normalized error dicts (with 'weight' added)
      - segment_mqm_score:  total penalty capped at MAX_SEGMENT_MQM_SCORE (25)
      - repo_style_score:   negated segment_mqm_score (for "higher = better" contexts)
      - scored_error_count: number of errors that actually contributed penalty > 0
    """
    normalized_errors = []
    total_weight = 0.0

    for error in errors:
        category, subcategory = canonicalize_label(
            error.get("category", ""),
            error.get("subcategory", ""),
        )
        severity = canonicalize_severity(error.get("severity", ""))
        weight = error_weight(severity, category, subcategory)
        normalized = {
            **error,
            "category": category,
            "subcategory": subcategory,
            "severity": severity,
            "weight": weight,
        }
        normalized_errors.append(normalized)
        total_weight += weight

    segment_mqm_score = round(min(total_weight, MAX_SEGMENT_MQM_SCORE), 4)
    return {
        "errors": normalized_errors,
        "segment_mqm_score": segment_mqm_score,
        "repo_style_score": round(-segment_mqm_score, 4),
        "scored_error_count": sum(1 for error in normalized_errors if error["weight"] > 0),
    }


def count_categories_and_severities(errors: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    """Count how many errors fall into each category and each severity level.

    If the error list is empty, the segment is treated as "No-error" and both
    the category and severity dicts get a 1.0 for "No-error" / "no-error".

    Returns:
        A (category_counts, severity_counts) tuple — both are dicts mapping
        label names to float counts.
    """
    category_counts = empty_counts(CATEGORY_ORDER)
    severity_counts = empty_counts(SEVERITY_ORDER)

    if not errors:
        category_counts["No-error"] = 1.0
        severity_counts["no-error"] = 1.0
        return category_counts, severity_counts

    for error in errors:
        category_counts[error["category"]] = category_counts.get(error["category"], 0.0) + 1.0
        severity_counts[error["severity"]] = severity_counts.get(error["severity"], 0.0) + 1.0
    return category_counts, severity_counts


def make_sample_key(key: SegmentKey) -> str:
    """Build a unique string identifier for a segment: "system|doc|seg_id".

    Used as the cache/lookup key throughout the benchmark pipeline so we can
    match LLM predictions back to the correct human-annotated segment.
    """
    return f"{key.system}|{key.doc}|{key.seg_id}"


def average_count_dicts(dicts: list[dict[str, float]], labels: list[str]) -> dict[str, float]:
    """Compute the element-wise mean across a list of count dicts.

    Used to average category or severity distributions across multiple
    raters or multiple segments. Missing keys default to 0.0.
    """
    if not dicts:
        return empty_counts(labels)
    return {
        label: mean(d.get(label, 0.0) for d in dicts)
        for label in labels
    }


def choose_dominant_label(counts: dict[str, float], label_order: list[str]) -> str:
    """Return the label with the highest count from a count dict.

    Ties are broken by the order in label_order (earlier = lower priority,
    so a later label with the same count wins). Used to determine the
    "dominant" error category or severity for a segment.
    """
    best_label = label_order[0]
    best_value = counts.get(best_label, 0.0)
    for label in label_order[1:]:
        value = counts.get(label, 0.0)
        if value > best_value:
            best_label = label
            best_value = value
    return best_label


def build_prompt(unit: SegmentUnit) -> str:
    """Construct the full MQM evaluation prompt sent to the LLM.

    The prompt includes:
      - The complete MQM category hierarchy so the LLM knows valid labels.
      - Scoring rules from the paper (Non-translation, Source error, Neutral, etc.).
      - The full source and system documents for discourse-level context.
      - The specific focus segment to annotate.
      - The expected JSON output schema.

    The LLM is asked to do source-based evaluation (compare system output
    against the source, not a reference) with document context, matching
    how human raters worked in the WMT evaluation campaign.
    """
    return f"""You are an expert MQM translation evaluator following Freitag et al. (2021).

Evaluate exactly one system translation using source-based MQM annotation with document context.
Use the full document context below, but annotate ONLY the focus segment.

MQM hierarchy:
{VALID_CATEGORIES_TEXT}

Rules:
- Compare the system translation against the source, not against a reference translation.
- Use document context to judge consistency, discourse, referents, and style.
- Return up to {MAX_ERRORS_PER_SEGMENT} errors, choosing the most severe and consequential if there are more.
- If the translation is too garbled to analyze, return a single Non-translation error spanning the whole focus segment.
- Use Source error only if the problem is already in the source text.
- Use Neutral only for true preferences that should not affect scoring.
- Use Style/Awkward for stylistic unnaturalness or clumsy phrasing.
- Use Minor Fluency/Punctuation only for cosmetic punctuation or spacing issues.
- If the focus segment has no errors, return an empty "errors" list.

Source document:
\"\"\"{unit.source_document}\"\"\"

System document:
\"\"\"{unit.target_document}\"\"\"

Focus source segment:
\"{unit.source_segment}\"

Focus system segment:
\"{unit.target_segment}\"

Return ONLY valid JSON in this schema:
{{
  "errors": [
    {{
      "error_span": "<exact problematic text from the focus system segment>",
      "category": "<top-level category>",
      "subcategory": "<specific subcategory or empty string>",
      "severity": "Major | Minor | Neutral",
      "explanation": "<brief explanation>"
    }}
  ],
  "overall_comment": "<one short summary sentence>"
}}"""


def parse_human_row(row: dict[str, str]) -> dict | None:
    """Convert one row from the WMT MQM TSV into a normalized error dict.

    Rows marked "No-error" (either by category or severity) are not actual
    errors — they represent a rater explicitly saying the segment is fine.
    Returns None for these rows so the caller can count them separately.

    Returns:
        A dict with keys: error_span, category, subcategory, severity,
        explanation — or None if the row is a "No-error" annotation.
    """
    raw_category = row["category"].strip()
    raw_severity = row["severity"].strip()
    if raw_category == "No-error" or canonicalize_severity(raw_severity) == "no-error":
        return None

    category, subcategory = canonicalize_label(raw_category, "")
    severity = canonicalize_severity(raw_severity)
    return {
        "error_span": extract_marked_span(row["target"]),
        "category": category,
        "subcategory": subcategory,
        "severity": severity,
        "explanation": "",
    }


def aggregate_rater_rows(rater: str, rows: list[dict[str, str]]) -> RaterAggregate:
    """Aggregate all TSV rows from a single rater for a single segment.

    A rater may have annotated multiple errors on the same segment, or
    may have marked it "No-error". This function:
      1. Parses each row into a normalized error dict (skipping No-error rows).
      2. Scores the collected errors to get the rater's segment MQM score.
      3. Counts errors by category and severity for later comparison.

    Returns:
        A RaterAggregate dataclass with the rater's errors, score, and counts.
    """
    parsed_errors = []
    saw_no_error = False
    for row in rows:
        parsed = parse_human_row(row)
        if parsed is None:
            saw_no_error = True
            continue
        parsed_errors.append(parsed)

    scored = score_mqm_errors(parsed_errors)
    category_counts, severity_counts = count_categories_and_severities(scored["errors"])
    if saw_no_error and not parsed_errors:
        category_counts["No-error"] = 1.0
        severity_counts["no-error"] = 1.0

    return RaterAggregate(
        rater=rater,
        errors=scored["errors"],
        segment_mqm_score=scored["segment_mqm_score"],
        category_counts=category_counts,
        severity_counts=severity_counts,
    )


def load_wmt_mqm_dataset(tsv_path: str | Path) -> list[SegmentUnit]:
    """Load a WMT MQM TSV file and return a list of fully assembled SegmentUnits.

    The TSV contains one row per rater-error annotation. This function:
      1. Reads every row and indexes source/target segments by document.
      2. Reconstructs full source and target documents by joining segments
         in order (needed for document-level context in prompts).
      3. Groups rows by (segment, rater) and aggregates each rater's annotations.
      4. Averages rater scores and counts to produce human ground-truth per segment.

    Args:
        tsv_path: Path to the MQM TSV (e.g. mqm_newstest2020_ende.tsv).

    Returns:
        A list of SegmentUnit objects sorted by (system, doc_id, seg_id),
        each containing source/target text, document context, and human
        rater aggregates.
    """
    source_segments_by_doc: dict[str, dict[int, str]] = {}
    target_segments_by_system_doc: dict[tuple[str, str], dict[int, str]] = {}
    rows_by_unit_rater: dict[tuple[SegmentKey, str], list[dict[str, str]]] = {}

    with Path(tsv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            system = row["system"]
            doc = row["doc"]
            doc_id = int(row["doc_id"])
            seg_id = int(row["seg_id"])
            rater = row["rater"]
            key = SegmentKey(system=system, doc=doc, doc_id=doc_id, seg_id=seg_id)

            source_segments_by_doc.setdefault(doc, {})[seg_id] = row["source"]
            target_segments_by_system_doc.setdefault((system, doc), {})[seg_id] = strip_span_markup(row["target"]).strip()
            rows_by_unit_rater.setdefault((key, rater), []).append(row)

    source_documents = {
        doc: " ".join(source_segments_by_doc[doc][seg_id] for seg_id in sorted(source_segments_by_doc[doc]))
        for doc in source_segments_by_doc
    }
    target_documents = {
        system_doc: " ".join(seg_map[seg_id] for seg_id in sorted(seg_map))
        for system_doc, seg_map in target_segments_by_system_doc.items()
    }

    unit_to_raters: dict[SegmentKey, list[RaterAggregate]] = {}
    for (key, rater), rows in rows_by_unit_rater.items():
        unit_to_raters.setdefault(key, []).append(aggregate_rater_rows(rater, rows))

    units: list[SegmentUnit] = []
    for key in sorted(unit_to_raters, key=lambda item: (item.system, item.doc_id, item.seg_id)):
        raters = sorted(unit_to_raters[key], key=lambda item: item.rater)
        units.append(
            SegmentUnit(
                key=key,
                source_segment=source_segments_by_doc[key.doc][key.seg_id],
                target_segment=target_segments_by_system_doc[(key.system, key.doc)][key.seg_id],
                source_document=source_documents[key.doc],
                target_document=target_documents[(key.system, key.doc)],
                raters=raters,
                human_mean_segment_score=mean(r.segment_mqm_score for r in raters),
                human_mean_category_counts=average_count_dicts(
                    [r.category_counts for r in raters],
                    CATEGORY_ORDER,
                ),
                human_mean_severity_counts=average_count_dicts(
                    [r.severity_counts for r in raters],
                    SEVERITY_ORDER,
                ),
            )
        )
    return units


def sample_units(
    units: list[SegmentUnit],
    sample_size: int,
    seed: int,
    sampling: str,
    include_human_systems: bool,
) -> list[SegmentUnit]:
    """Select a subset of SegmentUnits for benchmarking.

    Args:
        units:                 Full list of SegmentUnits from the dataset.
        sample_size:           How many segments to select.
        seed:                  Random seed for reproducibility.
        sampling:              Strategy — "random" for uniform sampling, or
                               "stratified-system" to draw proportionally from
                               each MT system so no single system dominates.
        include_human_systems: If False, filters out Human-A/B/P references
                               and only keeps machine translation outputs.

    Returns:
        A list of sampled SegmentUnits (up to sample_size).
    """
    filtered = [
        unit for unit in units
        if include_human_systems or unit.system_kind == "mt"
    ]
    if sample_size >= len(filtered):
        return list(filtered)

    rng = random.Random(seed)
    if sampling == "random":
        return rng.sample(filtered, sample_size)

    if sampling != "stratified-system":
        raise ValueError(f"Unsupported sampling strategy: {sampling}")

    grouped: dict[str, list[SegmentUnit]] = {}
    for unit in filtered:
        grouped.setdefault(unit.key.system, []).append(unit)

    systems = sorted(grouped)
    for system in systems:
        rng.shuffle(grouped[system])

    if sample_size <= len(systems):
        chosen_systems = rng.sample(systems, sample_size)
        return [grouped[system][0] for system in sorted(chosen_systems)]

    base = sample_size // len(systems)
    remainder = sample_size % len(systems)
    sampled: list[SegmentUnit] = []
    for index, system in enumerate(systems):
        take = base + (1 if index < remainder else 0)
        sampled.extend(grouped[system][:take])
    rng.shuffle(sampled)
    return sampled


def sample_common_segments(
    units: list[SegmentUnit],
    segments_per_system: int,
    seed: int,
    include_human_systems: bool,
) -> list[SegmentUnit]:
    """Sample segments that are common across ALL systems, then return every
    system's version of those segments.

    This ensures fair system-level comparison: every system is evaluated on
    the exact same set of source segments.

    Steps:
      1. Filter out human systems if requested.
      2. Index units by (doc, seg_id) and by system.
      3. Find (doc, seg_id) pairs that exist in ALL systems.
      4. Randomly sample `segments_per_system` of those common pairs.
      5. Return all SegmentUnits (one per system) for those sampled pairs.

    Args:
        units:                 Full list of SegmentUnits from the dataset.
        segments_per_system:   Number of common segments to sample (e.g. 50).
        seed:                  Random seed for reproducibility.
        include_human_systems: If False, excludes Human-A/B/P systems.

    Returns:
        A list of SegmentUnits — segments_per_system * num_systems total,
        sorted by (system, doc_id, seg_id).
    """
    filtered = [
        unit for unit in units
        if include_human_systems or unit.system_kind == "mt"
    ]

    # Group by system and by (doc, seg_id)
    systems: set[str] = set()
    units_by_seg: dict[tuple[str, int], dict[str, SegmentUnit]] = {}
    for unit in filtered:
        seg_key = (unit.key.doc, unit.key.seg_id)
        systems.add(unit.key.system)
        units_by_seg.setdefault(seg_key, {})[unit.key.system] = unit

    # Find (doc, seg_id) pairs present in ALL systems
    common_seg_keys = [
        seg_key for seg_key, system_map in units_by_seg.items()
        if len(system_map) == len(systems)
    ]
    common_seg_keys.sort()

    if not common_seg_keys:
        raise ValueError("No segments are common across all systems.")

    # Shuffle once, then take the first N. This makes the sample *nested* across
    # N values for a fixed seed — N=200 contains N=100 contains N=50 — so caches
    # carry forward as N grows and stability curves are not confounded by
    # swapping in a different set of segments.
    rng = random.Random(seed)
    shuffled_seg_keys = list(common_seg_keys)
    rng.shuffle(shuffled_seg_keys)
    take = min(segments_per_system, len(shuffled_seg_keys))
    sampled_seg_keys = shuffled_seg_keys[:take]

    # Collect all systems' units for the sampled segments
    result: list[SegmentUnit] = []
    for seg_key in sampled_seg_keys:
        for system in sorted(systems):
            result.append(units_by_seg[seg_key][system])

    result.sort(key=lambda u: (u.key.system, u.key.doc_id, u.key.seg_id))
    return result


def compute_system_rankings(
    units: list[SegmentUnit],
    predictions: dict[str, dict] | None = None,
) -> list[dict]:
    """Compute per-system average MQM scores and return them ranked (best first).

    For each system, averages the human_mean_segment_score across all its
    segments. If LLM predictions are provided, also averages the LLM segment
    scores. Systems are sorted by human score ascending (lower = fewer errors
    = better translation quality).

    Args:
        units:       List of SegmentUnits (should be from sample_common_segments
                     so all systems have the same segments).
        predictions: Optional dict mapping sample_key -> scored prediction dict.
                     If provided, LLM scores are included in the ranking.

    Returns:
        A list of dicts sorted by human_mean_score ascending (best system first):
          - system:           system name
          - num_segments:     number of segments evaluated
          - human_mean_score: average human MQM score (lower = better)
          - llm_mean_score:   average LLM MQM score (None if no predictions)
    """
    # Group units by system
    by_system: dict[str, list[SegmentUnit]] = {}
    for unit in units:
        by_system.setdefault(unit.key.system, []).append(unit)

    rankings = []
    for system, system_units in by_system.items():
        human_scores = [u.human_mean_segment_score for u in system_units]
        human_mean = round(mean(human_scores), 4)

        llm_mean = None
        if predictions is not None:
            llm_scores = [
                predictions[u.sample_key]["segment_mqm_score"]
                for u in system_units
                if u.sample_key in predictions
            ]
            if llm_scores:
                llm_mean = round(mean(llm_scores), 4)

        rankings.append({
            "system": system,
            "num_segments": len(system_units),
            "human_mean_score": human_mean,
            "llm_mean_score": llm_mean,
        })

    # Sort by human score ascending — lower MQM penalty = better translation
    rankings.sort(key=lambda r: r["human_mean_score"])
    return rankings


def prediction_to_counts(prediction: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Extract category and severity counts from an LLM prediction dict.

    Convenience wrapper around count_categories_and_severities for use
    in the analysis pipeline.
    """
    return count_categories_and_severities(prediction["errors"])


def analyze_provider_predictions(
    provider_name: str,
    sampled_units: list[SegmentUnit],
    predictions: dict[str, dict],
) -> tuple[dict, list[dict]]:
    """Compare an LLM provider's MQM predictions against human annotations.

    For each segment that has both a human annotation and an LLM prediction,
    this function computes:
      - Segment score gap (LLM score minus human mean score).
      - Whether the dominant error category matches (e.g. both say "Accuracy").
      - Whether the dominant severity matches (e.g. both say "Major").
      - Whether both agree on the presence/absence of errors (no-error agreement).
      - Style recall: when humans flagged Style errors, did the LLM catch them?

    Args:
        provider_name:  Display name for the provider (e.g. "openai:gpt-4o-mini").
        sampled_units:  The benchmark sample with human ground truth.
        predictions:    Dict mapping sample_key -> scored LLM prediction dict.

    Returns:
        A (summary_dict, per_segment_rows) tuple:
          - summary_dict:      Aggregate metrics (MAE, match rates, mean scores).
          - per_segment_rows:  One dict per segment with all comparison fields,
                               suitable for CSV export.
    """
    completed_units = [unit for unit in sampled_units if unit.sample_key in predictions]
    human_category_means = []
    llm_category_means = []
    human_severity_means = []
    llm_severity_means = []
    per_segment_rows = []

    category_match_count = 0
    severity_match_count = 0
    no_error_match_count = 0
    style_recall_hits = 0
    style_recall_total = 0

    for unit in completed_units:
        prediction = predictions[unit.sample_key]
        llm_category_counts, llm_severity_counts = prediction_to_counts(prediction)

        human_category_means.append(unit.human_mean_category_counts)
        llm_category_means.append(llm_category_counts)
        human_severity_means.append(unit.human_mean_severity_counts)
        llm_severity_means.append(llm_severity_counts)

        human_dominant_category = choose_dominant_label(unit.human_mean_category_counts, CATEGORY_ORDER)
        llm_dominant_category = choose_dominant_label(llm_category_counts, CATEGORY_ORDER)
        human_dominant_severity = choose_dominant_label(unit.human_mean_severity_counts, SEVERITY_ORDER)
        llm_dominant_severity = choose_dominant_label(llm_severity_counts, SEVERITY_ORDER)

        category_match = human_dominant_category == llm_dominant_category
        severity_match = human_dominant_severity == llm_dominant_severity
        no_error_match = (
            unit.human_mean_category_counts["No-error"] > 0
            and llm_category_counts["No-error"] > 0
        ) or (
            unit.human_mean_category_counts["No-error"] == 0
            and llm_category_counts["No-error"] == 0
        )

        category_match_count += int(category_match)
        severity_match_count += int(severity_match)
        no_error_match_count += int(no_error_match)

        if unit.human_mean_category_counts["Style"] > 0:
            style_recall_total += 1
            style_recall_hits += int(llm_category_counts["Style"] > 0)

        per_segment_rows.append(
            {
                "sample_key": unit.sample_key,
                "provider": provider_name,
                "system": unit.key.system,
                "doc": unit.key.doc,
                "seg_id": unit.key.seg_id,
                "human_mean_segment_score": round(unit.human_mean_segment_score, 4),
                "llm_segment_score": round(prediction["segment_mqm_score"], 4),
                "segment_score_gap": round(prediction["segment_mqm_score"] - unit.human_mean_segment_score, 4),
                "human_dominant_category": human_dominant_category,
                "llm_dominant_category": llm_dominant_category,
                "dominant_category_match": category_match,
                "human_dominant_severity": human_dominant_severity,
                "llm_dominant_severity": llm_dominant_severity,
                "dominant_severity_match": severity_match,
                "no_error_agreement": no_error_match,
                "human_style_present": unit.human_mean_category_counts["Style"] > 0,
                "llm_style_present": llm_category_counts["Style"] > 0,
                "human_source_segment": unit.source_segment,
                "system_segment": unit.target_segment,
            }
        )

    human_mean_category_counts = average_count_dicts(human_category_means, CATEGORY_ORDER)
    llm_mean_category_counts = average_count_dicts(llm_category_means, CATEGORY_ORDER)
    human_mean_severity_counts = average_count_dicts(human_severity_means, SEVERITY_ORDER)
    llm_mean_severity_counts = average_count_dicts(llm_severity_means, SEVERITY_ORDER)

    if completed_units:
        score_gaps = [
            predictions[unit.sample_key]["segment_mqm_score"] - unit.human_mean_segment_score
            for unit in completed_units
        ]
        score_abs_errors = [abs(gap) for gap in score_gaps]
    else:
        score_gaps = []
        score_abs_errors = []

    summary = {
        "provider": provider_name,
        "completed_segments": len(completed_units),
        "requested_segments": len(sampled_units),
        "mean_human_segment_score": round(mean(unit.human_mean_segment_score for unit in completed_units), 4) if completed_units else None,
        "mean_llm_segment_score": round(mean(predictions[unit.sample_key]["segment_mqm_score"] for unit in completed_units), 4) if completed_units else None,
        "mean_segment_score_gap": round(mean(score_gaps), 4) if score_gaps else None,
        "segment_score_mae": round(mean(score_abs_errors), 4) if score_abs_errors else None,
        "dominant_category_match_rate": round(category_match_count / len(completed_units), 4) if completed_units else None,
        "dominant_severity_match_rate": round(severity_match_count / len(completed_units), 4) if completed_units else None,
        "no_error_agreement_rate": round(no_error_match_count / len(completed_units), 4) if completed_units else None,
        "style_presence_recall": round(style_recall_hits / style_recall_total, 4) if style_recall_total else None,
        "mean_human_category_counts": human_mean_category_counts,
        "mean_llm_category_counts": llm_mean_category_counts,
        "mean_human_severity_counts": human_mean_severity_counts,
        "mean_llm_severity_counts": llm_mean_severity_counts,
    }
    return summary, per_segment_rows


def sanitize_slug(text: str) -> str:
    """Convert arbitrary text into a filesystem-safe slug for output filenames.

    Replaces any non-alphanumeric characters (except dots, hyphens, underscores)
    with underscores. Example: "openai:gpt-4o-mini" -> "openai_gpt-4o-mini"
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def write_json(path: str | Path, payload: dict) -> None:
    """Write a dict as pretty-printed JSON to the given file path."""
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """Write a list of dicts as newline-delimited JSON (one JSON object per line)."""
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write a list of dicts as a CSV file with the given column order."""
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _nice_tick_step(max_value: float) -> float:
    """Choose a human-friendly Y-axis tick step for SVG bar charts.

    Picks from common intervals (0.1, 0.2, 0.5, 1, 2, 5, 10, ...) so that
    roughly 5 grid lines fit within the chart's data range.
    """
    if max_value <= 0:
        return 1.0
    rough = max_value / 5
    for step in (0.1, 0.2, 0.5, 1, 2, 5, 10):
        if rough <= step:
            return step
    magnitude = 10 ** (len(str(int(rough))) - 1)
    for multiple in (1, 2, 5, 10):
        step = multiple * magnitude
        if rough <= step:
            return step
    return rough


def write_grouped_bar_svg(
    path: str | Path,
    title: str,
    x_labels: list[str],
    series: dict[str, list[float]],
    y_label: str,
) -> None:
    """Generate a grouped bar chart as a standalone SVG file.

    Used to visually compare human vs LLM category/severity distributions.
    Each group on the X-axis (e.g. "Accuracy", "Fluency") gets one bar per
    series (e.g. "Human", "openai:gpt-4o-mini"), color-coded with a legend.

    Args:
        path:     Output SVG file path.
        title:    Chart title displayed at the top.
        x_labels: Labels for each group on the X-axis.
        series:   Dict mapping series name -> list of values (one per x_label).
        y_label:  Label for the Y-axis.
    """
    # Estimate legend width per entry dynamically so long model names (e.g.
    # "anthropic:claude-haiku-4-5-20251001") don't overlap at a fixed stride.
    # Roughly 7px per character at 12px font, plus swatch and gaps.
    series_names_preview = list(series.keys())
    max_label_chars = max((len(name) for name in series_names_preview), default=8)
    legend_entry_width = max(170, max_label_chars * 7 + 40)

    width = max(960, 120 * len(x_labels))
    # Figure out how many legend rows we need given the chart width; add vertical
    # room for each extra row so entries wrap cleanly instead of overlapping.
    margin_left = 80
    margin_right = 30
    entries_per_row = max(1, (width - margin_left - margin_right) // legend_entry_width)
    legend_rows = (len(series_names_preview) + entries_per_row - 1) // entries_per_row
    legend_row_height = 22
    height = 560 + max(0, legend_rows - 1) * legend_row_height
    margin_top = 70
    margin_bottom = 120 + max(0, legend_rows - 1) * legend_row_height
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]

    max_value = max((max(values) for values in series.values()), default=1.0)
    tick_step = _nice_tick_step(max_value)
    max_y = max(tick_step * 5, tick_step * ((int(max_value / tick_step) + 1) or 1))

    group_width = plot_width / max(len(x_labels), 1)
    inner_padding = group_width * 0.15
    usable_group_width = group_width - inner_padding
    bar_width = usable_group_width / max(len(series), 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="22" font-weight="bold">{escape(title)}</text>',
        f'<text x="18" y="{margin_top + plot_height / 2}" transform="rotate(-90 18 {margin_top + plot_height / 2})" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="14">{escape(y_label)}</text>',
    ]

    for index in range(6):
        value = tick_step * index
        y = margin_top + plot_height - (value / max_y) * plot_height
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" stroke="#dddddd" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 5:.1f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="12">{value:.1f}</text>')

    series_names = list(series.keys())
    for label_index, label in enumerate(x_labels):
        group_x = margin_left + label_index * group_width + inner_padding / 2
        for series_index, series_name in enumerate(series_names):
            value = series[series_name][label_index]
            bar_height = 0 if max_y == 0 else (value / max_y) * plot_height
            x = group_x + series_index * bar_width
            y = margin_top + plot_height - bar_height
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_width - 4, 8):.1f}" height="{bar_height:.1f}" fill="{colors[series_index % len(colors)]}"/>'
            )
            parts.append(
                f'<text x="{x + max(bar_width - 4, 8) / 2:.1f}" y="{max(y - 6, margin_top - 2):.1f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="11">{value:.2f}</text>'
            )
        parts.append(
            f'<text x="{group_x + usable_group_width / 2:.1f}" y="{margin_top + plot_height + 24}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12">{escape(label)}</text>'
        )

    # Legend — wraps to multiple rows when entries won't fit on one line.
    legend_x = margin_left
    legend_y_top = height - 36 - (legend_rows - 1) * legend_row_height
    for series_index, series_name in enumerate(series_names):
        color = colors[series_index % len(colors)]
        row = series_index // entries_per_row
        col = series_index % entries_per_row
        x = legend_x + col * legend_entry_width
        y = legend_y_top + row * legend_row_height
        parts.append(f'<rect x="{x}" y="{y - 12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{x + 22}" y="{y}" font-family="Helvetica, Arial, sans-serif" font-size="12">{escape(series_name)}</text>')

    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def write_provider_summary_svg(path: str | Path, provider_summaries: dict[str, dict]) -> None:
    """Generate an SVG bar chart comparing provider agreement metrics.

    Plots four key rates side-by-side for each provider:
      - Dominant category match rate
      - Dominant severity match rate
      - No-error agreement rate
      - Style recall
    """
    metric_labels = [
        "Dominant category match",
        "Dominant severity match",
        "No-error agreement",
        "Style recall",
    ]
    series = {}
    for provider_name, summary in provider_summaries.items():
        series[provider_name] = [
            summary.get("dominant_category_match_rate") or 0.0,
            summary.get("dominant_severity_match_rate") or 0.0,
            summary.get("no_error_agreement_rate") or 0.0,
            summary.get("style_presence_recall") or 0.0,
        ]
    write_grouped_bar_svg(
        path=path,
        title="Provider Agreement Summary",
        x_labels=metric_labels,
        series=series,
        y_label="Rate",
    )


def build_summary_rows(provider_summaries: dict[str, dict]) -> list[dict]:
    """Flatten the provider summary dicts into a list of flat rows for CSV export.

    Each row contains one provider's aggregate benchmark metrics (match rates,
    MAE, mean scores, etc.) with consistent column names.
    """
    rows = []
    for provider_name, summary in provider_summaries.items():
        rows.append(
            {
                "provider": provider_name,
                "completed_segments": summary["completed_segments"],
                "requested_segments": summary["requested_segments"],
                "mean_human_segment_score": summary["mean_human_segment_score"],
                "mean_llm_segment_score": summary["mean_llm_segment_score"],
                "mean_segment_score_gap": summary["mean_segment_score_gap"],
                "segment_score_mae": summary["segment_score_mae"],
                "dominant_category_match_rate": summary["dominant_category_match_rate"],
                "dominant_severity_match_rate": summary["dominant_severity_match_rate"],
                "no_error_agreement_rate": summary["no_error_agreement_rate"],
                "style_presence_recall": summary["style_presence_recall"],
            }
        )
    return rows


def write_markdown_report(
    path: str | Path,
    dataset_path: str,
    sampled_units: list[SegmentUnit],
    provider_summaries: dict[str, dict],
    run_config: dict,
) -> None:
    """Generate a human-readable Markdown report summarizing the benchmark run.

    The report includes:
      - Run configuration (dataset, sample size, sampling strategy, seed).
      - A table of all sampled segments with their human MQM scores.
      - A provider comparison table with match rates, MAE, and mean scores.
      - A list of generated artifact files for reference.

    Args:
        path:               Output .md file path.
        dataset_path:       Path to the source TSV (displayed in the report).
        sampled_units:      The benchmark sample (for the segment table).
        provider_summaries: Dict mapping provider name -> summary metrics dict.
        run_config:         Dict with sample_size, sampling, include_human_systems, seed.
    """
    lines = [
        "# Paper-Faithful MQM Benchmark Report",
        "",
        "## Run Configuration",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Sample size requested: `{run_config['sample_size']}`",
        f"- Sampling strategy: `{run_config['sampling']}`",
        f"- Include human systems: `{run_config['include_human_systems']}`",
        f"- Random seed: `{run_config['seed']}`",
        "",
        "## Sampled Units",
        "",
        "Each sampled item is a single `system × document × segment` unit, which is the paper-faithful comparison target.",
        "",
        "| System | Doc | Seg ID | Human mean MQM score |",
        "|---|---|---:|---:|",
    ]

    for unit in sampled_units:
        lines.append(
            f"| `{unit.key.system}` | `{unit.key.doc}` | {unit.key.seg_id} | {unit.human_mean_segment_score:.2f} |"
        )

    lines.extend([
        "",
        "## Provider Summary",
        "",
        "| Provider | Completed | Human mean score | LLM mean score | Score MAE | Dominant category match | Dominant severity match | No-error agreement | Style recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for provider_name, summary in provider_summaries.items():
        lines.append(
            "| "
            + " | ".join([
                f"`{provider_name}`",
                str(summary["completed_segments"]),
                f"{summary['mean_human_segment_score']:.2f}" if summary["mean_human_segment_score"] is not None else "NA",
                f"{summary['mean_llm_segment_score']:.2f}" if summary["mean_llm_segment_score"] is not None else "NA",
                f"{summary['segment_score_mae']:.2f}" if summary["segment_score_mae"] is not None else "NA",
                f"{summary['dominant_category_match_rate']:.2f}" if summary["dominant_category_match_rate"] is not None else "NA",
                f"{summary['dominant_severity_match_rate']:.2f}" if summary["dominant_severity_match_rate"] is not None else "NA",
                f"{summary['no_error_agreement_rate']:.2f}" if summary["no_error_agreement_rate"] is not None else "NA",
                f"{summary['style_presence_recall']:.2f}" if summary["style_presence_recall"] is not None else "NA",
            ])
            + " |"
        )

    lines.extend([
        "",
        "## Generated Artifacts",
        "",
        "- `sample_manifest.jsonl`: sampled system-output segments with context and human aggregates",
        "- `provider_summary.json` / `provider_summary.csv`: benchmark summary tables",
        "- `category_counts.svg`: average human vs provider category counts per evaluation",
        "- `severity_counts.svg`: average human vs provider severity counts per evaluation",
        "- `provider_summary.svg`: provider agreement overview",
        "",
    ])
    Path(path).write_text("\n".join(lines), encoding="utf-8")
