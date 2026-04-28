# MQM Reference Notes

This is a concise Markdown companion to `MQM.pdf`, focused on the parts most relevant to reproducing the WMT 2020 MQM setup used in the paper.

## Core MQM Procedure

- Annotate errors at the segment level while reading within document context.
- Highlight each error span.
- Assign each error:
  - a category / subcategory
  - a severity
- Cap annotations at 5 errors per segment.
- If a segment is too garbled to analyze reliably, assign one `Non-translation` error spanning the whole segment.
- `Source error` is tracked separately and does not count toward target-error scoring.

## Severity Levels

| Severity | Meaning |
|---|---|
| `Major` | Meaning-changing or seriously misleading / important grammatical errors |
| `Minor` | Noticeable issues that do not materially change meaning |
| `Neutral` | Comments or preferences that should not affect the score |

## Segment Weighting Scheme

| Severity | Category | Weight |
|---|---|---:|
| `Major` | `Non-translation` | 25 |
| `Major` | all others | 5 |
| `Minor` | `Fluency/Punctuation` | 0.1 |
| `Minor` | all others | 1 |
| `Neutral` | all | 0 |

Notes:

- Segment scores range from `0` to `25`.
- Final segment score is averaged over annotators.
- Document- and system-level scores are averages over segment scores.

## MQM Hierarchy Used in the Paper

### Accuracy

- `Addition`
- `Omission`
- `Mistranslation`
- `Untranslated text`

### Fluency

- `Punctuation`
- `Spelling`
- `Grammar`
- `Register`
- `Inconsistency`
- `Character encoding`

### Terminology

- `Inappropriate for context`
- `Inconsistent use`

### Style

- `Awkward`

### Locale convention

- `Address format`
- `Currency format`
- `Date format`
- `Name format`
- `Telephone format`
- `Time format`

### Special Categories

- `Other`
- `Source error`
- `Non-translation`

## Experimental Setup from the Paper

- WMT 2020 English→German:
  - `1418` segments
  - `130` documents
  - `10` systems annotated
- MQM ratings:
  - `3` professional ratings per segment
  - `6` professional raters in the MQM pool
- Annotators had access to full document context.
- Documents were assigned in round-robin fashion to different 3-rater groups.
- Each rater saw outputs from all 10 systems for their assigned documents.
- Documents and systems were anonymized and randomized for raters.

## Reproduction Implications

- The correct comparison unit is usually a `system-output segment`, not just a bare `seg_id`.
- For WMT20 `en-de`, segments are paragraph-level units, and document context matters.
- If you start from the Google MQM TSV, you already have:
  - document id
  - segment id
  - source segment
  - target translation with marked error spans
- You can reconstruct source-side document context by grouping unique rows by `(doc, seg_id)` and ordering by `seg_id`.
- If you start from WMT plain-text task files instead, the `details` file is what maps each line to `(document-id, segment-id)`.

## Common Mistakes to Avoid

- Comparing one LLM judgment for a segment to pooled human annotations from many different MT systems for that same `seg_id`.
- Ignoring system identity when sampling or aggregating.
- Using only broad top-level categories and losing the subcategory information.
- Treating `No-error` or `Source error` as if they carried penalty weight.
- Forgetting the special weights for `Minor Fluency/Punctuation` and `Non-translation`.
- Ignoring document context and evaluating segments in isolation.
