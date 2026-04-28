"""Render the N-sweep Kendall τ comparison charts for the poster.

For an experiment that varied --segments-per-system across {3, 50, 100, 200}
with all other variables held fixed (model, seed, prompt), this script reads
each per-N benchmark output and produces two poster-quality bar charts:

  - n_sweep_system_tau.png   (5 bars: Human ceiling, n=3, n=50, n=100, n=200)
  - n_sweep_segment_tau.png  (same five bars at the segment level)

Each chart's leftmost bar is the inter-human Kendall τ ceiling — the τ
between two halves of the human raters on the same segments the LLM saw.
That number is the maximum agreement any judge could achieve given human
noise. The four LLM bars on the right show how the judge's agreement with
humans scales as the sample size grows.

Usage:
    python n_sweep_chart.py \
      --results-root results/exp_N_sweep \
      --provider openai:gpt-4o-mini

Prerequisite: each N must have its own benchmark directory with plots/
already rendered. Run wmt20_mqm_llm_benchmark.py at every N first, then
graphs.py on each, then this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from graphs import PALETTE, PLOT_DPI, compute_inter_human_tau
from mqm_paper_core import load_wmt_mqm_dataset, sample_common_segments

# Match graphs.py poster defaults so the new charts share the same look.
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": PLOT_DPI,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DEFAULT_DATASET = "wmt-mqm-human-evaluation/newstest2020/ende/mqm_newstest2020_ende.tsv"
DEFAULT_N_VALUES = [3, 50, 100, 200]
DEFAULT_SEED = 42


def load_tau_for_n(results_root: Path, n: int, provider: str) -> tuple[float, float]:
    """Read system + segment Kendall τ for one N from its metrics_summary.csv."""
    csv_path = results_root / f"n{n}" / "plots" / "metrics_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}.\n"
            f"  Run: python graphs.py --results-dir {results_root}/n{n}\n"
            f"  (which itself requires wmt20_mqm_llm_benchmark.py to have produced "
            f"that results dir first)."
        )
    df = pd.read_csv(csv_path)
    row = df[df["provider"] == provider]
    if row.empty:
        raise ValueError(
            f"Provider '{provider}' not found in {csv_path}. "
            f"Providers present: {df['provider'].tolist()}"
        )
    sys_tau = row["system_kendall_tau"].iloc[0]
    seg_tau = row["segment_kendall_tau"].iloc[0]
    return float(sys_tau), float(seg_tau)


def _bar_colors(n_bars: int, human_index: int) -> list:
    """Black for the Human bar; graded blue for the N bars (lighter = smaller N)."""
    colors: list = []
    n_llm_bars = n_bars - 1
    llm_position = 0
    for i in range(n_bars):
        if i == human_index:
            colors.append(PALETTE[0])  # "#111111"
            continue
        # graded blue from light (small N) to dark (large N)
        shade = 0.35 + 0.55 * (llm_position / max(n_llm_bars - 1, 1))
        colors.append(plt.cm.Blues(shade))
        llm_position += 1
    return colors


def render_bar_chart(
    out_path: Path,
    title: str,
    subtitle: str,
    bar_labels: list[str],
    values: list[float],
    human_index: int,
    y_label: str,
) -> None:
    """Draw one Human ceiling bar plus the N-sweep bars, with value labels and a
    dashed line marking the ceiling across the chart."""
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(bar_labels))
    colors = _bar_colors(len(bar_labels), human_index)
    bars = ax.bar(x, values, color=colors, width=0.62, edgecolor="white", linewidth=0.5)

    # Dashed horizontal line at the Human τ ceiling.
    human_tau = values[human_index]
    if not np.isnan(human_tau):
        ax.axhline(human_tau, color=PALETTE[0], linestyle="--", linewidth=1.2, alpha=0.6)
        ax.text(
            len(bar_labels) - 0.55,
            human_tau + 0.015,
            "human ceiling",
            ha="right",
            va="bottom",
            fontsize=9,
            color=PALETTE[0],
            alpha=0.8,
        )

    # Value labels on top of (or below, for negative) each bar.
    for bar, v in zip(bars, values):
        if v is None or np.isnan(v):
            label_text = "N/A"
            ypos = 0.02
            va = "bottom"
        else:
            label_text = f"{v:+.2f}" if v < 0 else f"{v:.2f}"
            if v >= 0:
                ypos = v + 0.02
                va = "bottom"
            else:
                ypos = v - 0.02
                va = "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            label_text,
            ha="center",
            va=va,
            fontsize=11,
            fontweight="medium",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if subtitle:
        ax.text(
            0.5, 1.01, subtitle,
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color="#555555",
        )
    ax.axhline(0, color="#888", linewidth=0.8)

    finite_vals = [v for v in values if v is not None and not np.isnan(v)]
    y_lo = min(-0.15, min(finite_vals) - 0.1) if finite_vals else -0.15
    y_hi = max(1.05, max(finite_vals) + 0.12) if finite_vals else 1.05
    ax.set_ylim(y_lo, y_hi)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.savefig(str(out_path))
    plt.close(fig)
    print(f"  wrote {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-root", required=True,
        help="Directory containing n3/, n50/, n100/, n200/ subdirs.",
    )
    parser.add_argument(
        "--provider", required=True,
        help="Provider identifier from the benchmark, e.g. openai:gpt-4o-mini",
    )
    parser.add_argument(
        "--n-values", type=int, nargs="+", default=DEFAULT_N_VALUES,
        help=f"N values to plot (default: {' '.join(str(n) for n in DEFAULT_N_VALUES)})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Seed used in the benchmark runs (must match for the inter-human ceiling to use the same segments).",
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help="MQM TSV path (used for the inter-human ceiling).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to write the two PNGs. Defaults to <results-root>/charts.",
    )
    parser.add_argument(
        "--ceiling-n", type=int, default=None,
        help="Which N to use for the Human ceiling sample. Defaults to max(--n-values).",
    )
    parser.add_argument(
        "--mt-only", action="store_true",
        help="If the benchmark was run with --mt-only, set this so the ceiling uses the same systems.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir) if args.output_dir else results_root / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_values = sorted(set(args.n_values))
    ceiling_n = args.ceiling_n if args.ceiling_n is not None else max(n_values)

    # 1. Inter-human Kendall τ on the same segments the LLM saw at ceiling_n.
    print(f"Loading dataset: {args.dataset}")
    units = load_wmt_mqm_dataset(args.dataset)
    sampled_units = sample_common_segments(
        units=units,
        segments_per_system=ceiling_n,
        seed=args.seed,
        include_human_systems=not args.mt_only,
    )
    n_systems_in_sample = len(set(u.key.system for u in sampled_units))
    print(
        f"Computing inter-human τ on N={ceiling_n} ({len(sampled_units)} units, "
        f"{n_systems_in_sample} systems, seed={args.seed})"
    )
    ceiling = compute_inter_human_tau(sampled_units)
    print(
        f"  rater split: half-A={ceiling['n_raters_half_a']}, "
        f"half-B={ceiling['n_raters_half_b']} "
        f"(of {ceiling['n_raters_total']} total)"
    )
    print(
        f"  system τ ceiling:  {ceiling['system_tau']:+.3f} "
        f"(p={ceiling['system_p']:.3g}, over {ceiling['n_systems']} systems)"
    )
    print(
        f"  segment τ ceiling: {ceiling['segment_tau']:+.3f} "
        f"(p={ceiling['segment_p']:.3g}, over {ceiling['n_segments']} segments)"
    )

    # 2. Read LLM τ values from each per-N metrics_summary.csv.
    print(f"\nReading per-N metrics for provider '{args.provider}':")
    sys_taus: list[float] = []
    seg_taus: list[float] = []
    for n in n_values:
        sys_t, seg_t = load_tau_for_n(results_root, n, args.provider)
        sys_taus.append(sys_t)
        seg_taus.append(seg_t)
        print(f"  N={n:>3}: system τ = {sys_t:+.3f}, segment τ = {seg_t:+.3f}")

    # 3. Build the bar arrays — Human ceiling on the left, then n=3 -> n=200.
    bar_labels = ["Human\n(ceiling)"] + [f"n={n}" for n in n_values]
    sys_values = [ceiling["system_tau"]] + sys_taus
    seg_values = [ceiling["segment_tau"]] + seg_taus

    subtitle = f"{args.provider}  ·  seed={args.seed}  ·  ceiling on N={ceiling_n}"

    # 4. Render the two charts.
    print("\nRendering charts:")
    render_bar_chart(
        output_dir / "n_sweep_system_tau.png",
        title="System-Level Kendall τ vs Sample Size",
        subtitle=subtitle,
        bar_labels=bar_labels,
        values=sys_values,
        human_index=0,
        y_label="System-level Kendall τ (higher = closer to humans)",
    )
    render_bar_chart(
        output_dir / "n_sweep_segment_tau.png",
        title="Segment-Level Kendall τ vs Sample Size",
        subtitle=subtitle,
        bar_labels=bar_labels,
        values=seg_values,
        human_index=0,
        y_label="Segment-level Kendall τ (higher = closer to humans)",
    )

    # 5. Persist a flat CSV so the same numbers can land on the poster verbatim.
    csv_rows = [{"bar": bar_labels[i].replace("\n", " "),
                 "system_kendall_tau": sys_values[i],
                 "segment_kendall_tau": seg_values[i]}
                for i in range(len(bar_labels))]
    csv_path = output_dir / "n_sweep_tau_values.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"  wrote {csv_path}")

    print(f"\nDone. Charts written to: {output_dir}")


if __name__ == "__main__":
    main()
