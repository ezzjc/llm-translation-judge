"""Run a paper-faithful MQM benchmark against human WMT annotations.

Example:
    python wmt20_mqm_llm_benchmark.py \
      --provider openai:gpt-4o-mini \
      --provider anthropic:claude-sonnet-4-6 \
      --provider gemini:gemini-2.0-flash \
      --segments-per-system 200

The benchmark unit is one system-output segment with document context.
It writes cached raw annotations, per-segment comparisons, summary tables,
SVG charts, and a Markdown report.

Provider / model selection is LLM-agnostic — see providers.py for the
registry. To add a new provider, register it there; no changes needed here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mqm_paper_core import (
    CATEGORY_ORDER,
    SEVERITY_ORDER,
    SegmentUnit,
    analyze_provider_predictions,
    average_count_dicts,
    build_summary_rows,
    choose_dominant_label,
    compute_system_rankings,
    load_wmt_mqm_dataset,
    sample_common_segments,
    sanitize_slug,
    write_csv,
    write_grouped_bar_svg,
    write_json,
    write_jsonl,
    write_markdown_report,
    write_provider_summary_svg,
)
from providers import (
    append_jsonl_record,
    create_provider,
    default_provider_specs,
    evaluate_with_retries,
    load_jsonl_map,
)


DEFAULT_DATASET = "wmt-mqm-human-evaluation/newstest2020/ende/mqm_newstest2020_ende.tsv"
DEFAULT_OUTPUT_DIR = "results/wmt20_en_de_benchmark"


def build_sample_manifest(sampled_units: list[SegmentUnit]) -> list[dict]:
    """Build a JSONL-ready manifest describing every sampled segment."""
    manifest = []
    for index, unit in enumerate(sampled_units, start=1):
        manifest.append(
            {
                "sample_index": index,
                "sample_key": unit.sample_key,
                "system": unit.key.system,
                "system_kind": unit.system_kind,
                "doc": unit.key.doc,
                "doc_id": unit.key.doc_id,
                "seg_id": unit.key.seg_id,
                "human_mean_segment_score": round(unit.human_mean_segment_score, 4),
                "human_dominant_category": choose_dominant_label(unit.human_mean_category_counts, CATEGORY_ORDER),
                "human_dominant_severity": choose_dominant_label(unit.human_mean_severity_counts, SEVERITY_ORDER),
                "source_segment": unit.source_segment,
                "system_segment": unit.target_segment,
                "source_document": unit.source_document,
                "system_document": unit.target_document,
            }
        )
    return manifest


def human_series(sampled_units: list[SegmentUnit]) -> tuple[dict[str, float], dict[str, float]]:
    """Average human category/severity distributions across all sampled segments."""
    category_counts = average_count_dicts(
        [unit.human_mean_category_counts for unit in sampled_units],
        CATEGORY_ORDER,
    )
    severity_counts = average_count_dicts(
        [unit.human_mean_severity_counts for unit in sampled_units],
        SEVERITY_ORDER,
    )
    return category_counts, severity_counts


def print_rank_comparison_table(
    human_rankings: list[dict],
    llm_rankings: list[dict],
    provider_name: str,
    output_dir: Path,
) -> None:
    """Print and save a side-by-side human vs LLM rank comparison table."""
    human_rank_map = {row["system"]: i for i, row in enumerate(human_rankings, 1)}

    ranked_by_llm = sorted(
        [r for r in llm_rankings if r["llm_mean_score"] is not None],
        key=lambda r: r["llm_mean_score"],
    )
    llm_rank_map = {row["system"]: i for i, row in enumerate(ranked_by_llm, 1)}
    llm_score_map = {row["system"]: row["llm_mean_score"] for row in llm_rankings}

    col = {"sys": 30, "havg": 9, "hr": 9, "lavg": 8, "lr": 7, "rd": 8}
    header = (
        f"{'System':<{col['sys']}} {'HumanAvg':>{col['havg']}} {'HumanRank':>{col['hr']}}"
        f" {'LLMAvg':>{col['lavg']}} {'LLMRank':>{col['lr']}} {'RankDiff':>{col['rd']}}"
    )
    sep = "-" * len(header)

    lines = [header, sep]
    for row in human_rankings:
        system = row["system"]
        h_avg = row["human_mean_score"]
        h_rank = human_rank_map[system]
        l_avg = llm_score_map.get(system)
        l_rank = llm_rank_map.get(system)

        if l_avg is not None and l_rank is not None:
            rank_diff = l_rank - h_rank
            rd_str = f"{rank_diff:+.1f}"
            arrow = " ←" if abs(rank_diff) >= 3 else ""
            l_avg_str = f"{l_avg:.3f}"
            l_rank_str = str(l_rank)
        else:
            rd_str, arrow, l_avg_str, l_rank_str = "N/A", "", "N/A", "N/A"

        lines.append(
            f"{system:<{col['sys']}} {h_avg:>{col['havg']}.3f} {h_rank:>{col['hr']}}"
            f" {l_avg_str:>{col['lavg']}} {l_rank_str:>{col['lr']}} {rd_str:>{col['rd']}}{arrow}"
        )

    lines.append(sep)
    table_str = "\n".join(lines)

    print(f"\n{'='*60}")
    print(f"  RANK COMPARISON TABLE — {provider_name}")
    print(f"{'='*60}")
    print(table_str)

    provider_slug = sanitize_slug(provider_name)
    (output_dir / f"rank_comparison_{provider_slug}.txt").write_text(
        table_str + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a paper-faithful MQM benchmark on WMT-style annotations. "
            "The comparison unit is one system-output segment with document context."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to an MQM TSV file.")
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        help=(
            "Provider spec 'provider:model'. Repeat to compare multiple. "
            "See providers.PROVIDER_REGISTRY for available providers."
        ),
    )
    parser.add_argument(
        "--segments-per-system",
        type=int,
        default=50,
        help="Number of common segments per system. All systems are evaluated on the same N.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--mt-only",
        action="store_true",
        help="Exclude Human-A / Human-B / Human-P and sample only machine outputs.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for artifacts.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry attempts per API call.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider_specs = args.provider or default_provider_specs()
    if not provider_specs:
        raise SystemExit(
            "No providers configured. Pass --provider openai:gpt-4o-mini (or anthropic / gemini), "
            "or set the corresponding API key env var."
        )

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    units = load_wmt_mqm_dataset(dataset_path)
    sampled_units = sample_common_segments(
        units=units,
        segments_per_system=args.segments_per_system,
        seed=args.seed,
        include_human_systems=not args.mt_only,
    )

    human_rankings = compute_system_rankings(sampled_units)
    systems_in_sample = sorted(set(u.key.system for u in sampled_units))
    seg_count = args.segments_per_system
    print(f"\nSampled {seg_count} common segments across {len(systems_in_sample)} systems")
    print(f"Total units: {len(sampled_units)}")
    print(f"\n{'='*60}")
    print(f"  HUMAN SYSTEM RANKING ({seg_count} segments per system)")
    print(f"  (lower MQM score = better translation)")
    print(f"{'='*60}")
    for rank, row in enumerate(human_rankings, 1):
        print(f"  {rank:>2}. {row['system']:<30} {row['human_mean_score']:.4f}")

    manifest = build_sample_manifest(sampled_units)
    write_jsonl(output_dir / "sample_manifest.jsonl", manifest)

    provider_summaries = {}
    for spec in provider_specs:
        provider = create_provider(spec)
        provider_slug = sanitize_slug(provider.identifier)
        raw_path = output_dir / f"raw_{provider_slug}.jsonl"
        error_path = output_dir / f"errors_{provider_slug}.jsonl"
        cache = load_jsonl_map(raw_path)
        predictions = {key: record["result"] for key, record in cache.items()}

        print(f"\n=== Provider: {provider.identifier} ===")
        print(f"Using cache file: {raw_path}")

        for index, unit in enumerate(sampled_units, start=1):
            if unit.sample_key in predictions:
                print(f"[cache] {index}/{len(sampled_units)} {unit.sample_key}")
                continue

            print(f"[eval ] {index}/{len(sampled_units)} {unit.sample_key}")
            try:
                result = evaluate_with_retries(provider, unit, retries=args.max_retries)
                record = {
                    "sample_key": unit.sample_key,
                    "provider": provider.provider_name,
                    "model": provider.model,
                    "system": unit.key.system,
                    "doc": unit.key.doc,
                    "seg_id": unit.key.seg_id,
                    "result": result,
                }
                append_jsonl_record(raw_path, record)
                predictions[unit.sample_key] = result
            except Exception as exc:  # noqa: BLE001
                append_jsonl_record(
                    error_path,
                    {
                        "sample_key": unit.sample_key,
                        "provider": provider.provider_name,
                        "model": provider.model,
                        "system": unit.key.system,
                        "doc": unit.key.doc,
                        "seg_id": unit.key.seg_id,
                        "error": str(exc),
                    },
                )
                print(f"[fail ] {unit.sample_key}: {exc}")

        summary, per_segment_rows = analyze_provider_predictions(
            provider_name=provider.identifier,
            sampled_units=sampled_units,
            predictions=predictions,
        )
        provider_summaries[provider.identifier] = summary

        llm_rankings = compute_system_rankings(sampled_units, predictions)
        print(f"\n{'='*60}")
        print(f"  LLM SYSTEM RANKING — {provider.identifier}")
        print(f"  ({seg_count} segments per system, lower = better)")
        print(f"{'='*60}")
        print(f"  {'Rank':>4}  {'System':<30} {'Human':>8} {'LLM':>8} {'Gap':>8}")
        print(f"  {'-'*4}  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
        for rank, row in enumerate(llm_rankings, 1):
            llm_str = f"{row['llm_mean_score']:.4f}" if row['llm_mean_score'] is not None else "N/A"
            gap = row['llm_mean_score'] - row['human_mean_score'] if row['llm_mean_score'] is not None else None
            gap_str = f"{gap:+.4f}" if gap is not None else "N/A"
            print(f"  {rank:>4}  {row['system']:<30} {row['human_mean_score']:>8.4f} {llm_str:>8} {gap_str:>8}")

        print_rank_comparison_table(human_rankings, llm_rankings, provider.identifier, output_dir)

        write_csv(
            output_dir / f"system_ranking_{provider_slug}.csv",
            llm_rankings,
            ["system", "num_segments", "human_mean_score", "llm_mean_score"],
        )

        if per_segment_rows:
            per_segment_fieldnames = [
                "sample_key",
                "provider",
                "system",
                "doc",
                "seg_id",
                "human_mean_segment_score",
                "llm_segment_score",
                "segment_score_gap",
                "human_dominant_category",
                "llm_dominant_category",
                "dominant_category_match",
                "human_dominant_severity",
                "llm_dominant_severity",
                "dominant_severity_match",
                "no_error_agreement",
                "human_style_present",
                "llm_style_present",
                "human_source_segment",
                "system_segment",
            ]
            write_csv(
                output_dir / f"per_segment_{provider_slug}.csv",
                per_segment_rows,
                per_segment_fieldnames,
            )

    write_csv(
        output_dir / "system_ranking_human.csv",
        human_rankings,
        ["system", "num_segments", "human_mean_score", "llm_mean_score"],
    )

    provider_summary = {
        "dataset": str(dataset_path),
        "segments_per_system": args.segments_per_system,
        "systems": systems_in_sample,
        "sampled_units": len(sampled_units),
        "seed": args.seed,
        "include_human_systems": not args.mt_only,
        "human_system_ranking": human_rankings,
        "providers": provider_summaries,
    }
    write_json(output_dir / "provider_summary.json", provider_summary)
    write_csv(
        output_dir / "provider_summary.csv",
        build_summary_rows(provider_summaries),
        [
            "provider",
            "completed_segments",
            "requested_segments",
            "mean_human_segment_score",
            "mean_llm_segment_score",
            "mean_segment_score_gap",
            "segment_score_mae",
            "dominant_category_match_rate",
            "dominant_severity_match_rate",
            "no_error_agreement_rate",
            "style_presence_recall",
        ],
    )

    human_category_counts, human_severity_counts = human_series(sampled_units)
    category_series = {"Human": [human_category_counts[label] for label in CATEGORY_ORDER]}
    severity_series = {"Human": [human_severity_counts[label] for label in SEVERITY_ORDER]}
    for provider_name, summary in provider_summaries.items():
        category_series[provider_name] = [
            summary["mean_llm_category_counts"][label]
            for label in CATEGORY_ORDER
        ]
        severity_series[provider_name] = [
            summary["mean_llm_severity_counts"][label]
            for label in SEVERITY_ORDER
        ]

    write_grouped_bar_svg(
        output_dir / "category_counts.svg",
        "Average MQM Category Counts Per Evaluation",
        CATEGORY_ORDER,
        category_series,
        "Average errors",
    )
    write_grouped_bar_svg(
        output_dir / "severity_counts.svg",
        "Average MQM Severity Counts Per Evaluation",
        SEVERITY_ORDER,
        severity_series,
        "Average errors",
    )
    write_provider_summary_svg(output_dir / "provider_summary.svg", provider_summaries)

    write_markdown_report(
        output_dir / "report.md",
        dataset_path=str(dataset_path),
        sampled_units=sampled_units,
        provider_summaries=provider_summaries,
        run_config={
            "sample_size": args.segments_per_system,
            "sampling": "common-segments",
            "include_human_systems": not args.mt_only,
            "seed": args.seed,
        },
    )

    print("\nBenchmark complete.")
    print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
