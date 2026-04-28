"""Poster-quality visualizations for the MQM benchmark (pandas + matplotlib → PNG).

Reads an output directory written by wmt20_mqm_llm_benchmark.py and produces
PNGs at 300 DPI suitable for print posters. Every figure is a self-contained
function taking a DataFrame, so adding a new chart is just "write a function,
call it from generate_all_graphs()".

Usage (CLI):
    python graphs.py --results-dir results/wmt20_en_de_benchmark

Usage (programmatic):
    from graphs import load_results, generate_all_graphs
    tables = load_results("results/wmt20_en_de_benchmark")
    generate_all_graphs(tables, "results/wmt20_en_de_benchmark/plots")
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from mqm_paper_core import (
    CATEGORY_ORDER,
    SEVERITY_ORDER,
    SegmentUnit,
    choose_dominant_label,
    count_categories_and_severities,
    load_wmt_mqm_dataset,
)

# Poster-quality defaults. Type 42 keeps fonts editable in PDFs, which
# print shops prefer over rasterized text.
PLOT_DPI = 300
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": PLOT_DPI,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Stable, colorblind-friendly palette (Okabe-Ito). Human is always first (black-ish).
PALETTE = [
    "#111111",  # Human
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
]


@dataclass
class ResultTables:
    """All DataFrames + metadata for one benchmark run."""
    summary: dict                            # provider_summary.json
    manifest: pd.DataFrame                   # one row per sampled segment
    per_segment: pd.DataFrame                # long-form: one row per (segment, provider)
    human_rankings: pd.DataFrame             # human system ranking
    llm_rankings: pd.DataFrame               # long-form: one row per (system, provider)
    category_counts: pd.DataFrame            # long: provider × category → mean count
    severity_counts: pd.DataFrame            # long: provider × severity → mean count
    providers: list[str]                     # provider identifiers in display order
    # Per-segment raw signals — populated lazily for per-category τ analysis.
    raw_by_provider: dict[str, list[dict]]   # provider id -> list of raw_<slug>.jsonl records
    human_category_counts_by_key: dict[str, dict[str, float]]  # sample_key -> {cat: count}
    human_severity_counts_by_key: dict[str, dict[str, float]]  # sample_key -> {sev: count}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_results(results_dir: str | Path) -> ResultTables:
    """Load all artifacts a benchmark run wrote and shape them into DataFrames."""
    results_dir = Path(results_dir)

    summary_path = results_dir / "provider_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing provider_summary.json in {results_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    providers = list(summary["providers"].keys())

    manifest_rows = _load_jsonl(results_dir / "sample_manifest.jsonl")
    manifest = pd.DataFrame(manifest_rows)

    per_segment_frames = []
    for per_seg_path in sorted(results_dir.glob("per_segment_*.csv")):
        frame = pd.read_csv(per_seg_path)
        per_segment_frames.append(frame)
    per_segment = (
        pd.concat(per_segment_frames, ignore_index=True)
        if per_segment_frames
        else pd.DataFrame()
    )

    human_rankings = pd.DataFrame(summary.get("human_system_ranking", []))

    llm_rank_frames = []
    for ranking_path in sorted(results_dir.glob("system_ranking_*.csv")):
        if ranking_path.name == "system_ranking_human.csv":
            continue
        provider_slug = ranking_path.stem.replace("system_ranking_", "")
        frame = pd.read_csv(ranking_path)
        frame["provider_slug"] = provider_slug
        llm_rank_frames.append(frame)
    llm_rankings = (
        pd.concat(llm_rank_frames, ignore_index=True)
        if llm_rank_frames
        else pd.DataFrame()
    )

    # Attach readable provider identifiers by matching the slug to the summary.
    slug_to_identifier = {_slug_for(p): p for p in providers}
    if not llm_rankings.empty:
        llm_rankings["provider"] = llm_rankings["provider_slug"].map(slug_to_identifier)

    # Build long-form category / severity count tables with Human as one provider.
    count_rows_cat = []
    count_rows_sev = []
    for label in CATEGORY_ORDER:
        human_val = (
            np.mean([s["mean_human_category_counts"].get(label, 0.0) for s in summary["providers"].values()])
            if summary["providers"] else 0.0
        )
        count_rows_cat.append({"provider": "Human", "category": label, "mean_count": human_val})
    for provider, s in summary["providers"].items():
        for label in CATEGORY_ORDER:
            count_rows_cat.append({
                "provider": provider,
                "category": label,
                "mean_count": s["mean_llm_category_counts"].get(label, 0.0),
            })
    category_counts = pd.DataFrame(count_rows_cat)

    for label in SEVERITY_ORDER:
        human_val = (
            np.mean([s["mean_human_severity_counts"].get(label, 0.0) for s in summary["providers"].values()])
            if summary["providers"] else 0.0
        )
        count_rows_sev.append({"provider": "Human", "severity": label, "mean_count": human_val})
    for provider, s in summary["providers"].items():
        for label in SEVERITY_ORDER:
            count_rows_sev.append({
                "provider": provider,
                "severity": label,
                "mean_count": s["mean_llm_severity_counts"].get(label, 0.0),
            })
    severity_counts = pd.DataFrame(count_rows_sev)

    # Load raw LLM predictions per provider (for per-segment category/severity analysis).
    raw_by_provider: dict[str, list[dict]] = {}
    for provider in providers:
        raw_path = results_dir / f"raw_{_slug_for(provider)}.jsonl"
        raw_by_provider[provider] = _load_jsonl(raw_path)

    # Load per-segment human category/severity counts by re-parsing the TSV the
    # benchmark used. This is the only source of truth for per-segment human
    # distributions (the manifest only stores the dominant label).
    human_category_counts_by_key: dict[str, dict[str, float]] = {}
    human_severity_counts_by_key: dict[str, dict[str, float]] = {}
    dataset_path = summary.get("dataset")
    if dataset_path and Path(dataset_path).exists():
        units = load_wmt_mqm_dataset(dataset_path)
        for unit in units:
            human_category_counts_by_key[unit.sample_key] = dict(unit.human_mean_category_counts)
            human_severity_counts_by_key[unit.sample_key] = dict(unit.human_mean_severity_counts)

    return ResultTables(
        summary=summary,
        manifest=manifest,
        per_segment=per_segment,
        human_rankings=human_rankings,
        llm_rankings=llm_rankings,
        category_counts=category_counts,
        severity_counts=severity_counts,
        providers=providers,
        raw_by_provider=raw_by_provider,
        human_category_counts_by_key=human_category_counts_by_key,
        human_severity_counts_by_key=human_severity_counts_by_key,
    )


def _slug_for(provider_identifier: str) -> str:
    """Mirror mqm_paper_core.sanitize_slug so we can match ranking CSV filenames."""
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", provider_identifier).strip("_")


def compute_kendall_tau(tables: ResultTables) -> pd.DataFrame:
    """Compute system-level and segment-level Kendall tau for every provider.

    System-level: rank MT systems by mean MQM score; correlate human vs LLM ranks.
                  This is the MT-eval headline — answers "does this judge pick
                  the same winner as humans?"
    Segment-level: correlate raw human and LLM segment scores across all segments.
                  Higher statistical power per run, finer-grained quality signal.

    Returns one row per provider with both tau values and their p-values.
    """
    rows = []
    llm_rankings = tables.llm_rankings
    per_segment = tables.per_segment

    for provider in tables.providers:
        sys_rank = llm_rankings[llm_rankings["provider"] == provider].copy()
        sys_rank = sys_rank.dropna(subset=["human_mean_score", "llm_mean_score"])
        if len(sys_rank) >= 2:
            sys_result = kendalltau(sys_rank["human_mean_score"], sys_rank["llm_mean_score"])
            sys_tau = float(sys_result.statistic)
            sys_p = float(sys_result.pvalue)
        else:
            sys_tau, sys_p = (float("nan"), float("nan"))

        seg_rows = per_segment[per_segment["provider"] == provider]
        if len(seg_rows) >= 2:
            seg_result = kendalltau(
                seg_rows["human_mean_segment_score"],
                seg_rows["llm_segment_score"],
            )
            seg_tau = float(seg_result.statistic)
            seg_p = float(seg_result.pvalue)
        else:
            seg_tau, seg_p = (float("nan"), float("nan"))

        rows.append({
            "provider": provider,
            "n_systems": len(sys_rank),
            "n_segments": len(seg_rows),
            "system_kendall_tau": round(sys_tau, 4) if not np.isnan(sys_tau) else None,
            "system_kendall_p": round(sys_p, 4) if not np.isnan(sys_p) else None,
            "segment_kendall_tau": round(seg_tau, 4) if not np.isnan(seg_tau) else None,
            "segment_kendall_p": round(seg_p, 4) if not np.isnan(seg_p) else None,
        })
    return pd.DataFrame(rows)


def compute_inter_human_tau(units: list[SegmentUnit]) -> dict:
    """Estimate the inter-human Kendall τ ceiling by splitting raters in half.

    Strategy:
      1. Pool every unique rater ID across the supplied units.
      2. Deterministically split them by sorted-index parity (even -> A, odd -> B).
      3. For each unit, average segment_mqm_score across half-A's raters and
         half-B's raters separately. Skip units where either side has no rater.
      4. system_tau:  correlate per-system mean(half-A) vs mean(half-B).
         segment_tau: correlate per-segment half-A vs half-B scores.

    The result is the *upper bound* of agreement any judge can hit with these
    humans: even humans don't fully agree with one another, and τ between two
    halves of them is the ceiling for any LLM judge evaluated against them.
    """
    all_raters: set[str] = set()
    for unit in units:
        for r in unit.raters:
            all_raters.add(r.rater)
    sorted_raters = sorted(all_raters)
    half_a = set(sorted_raters[::2])
    half_b = set(sorted_raters[1::2])

    segment_a: list[float] = []
    segment_b: list[float] = []
    system_a: dict[str, list[float]] = {}
    system_b: dict[str, list[float]] = {}

    for unit in units:
        a_scores = [r.segment_mqm_score for r in unit.raters if r.rater in half_a]
        b_scores = [r.segment_mqm_score for r in unit.raters if r.rater in half_b]
        if not a_scores or not b_scores:
            continue
        a_mean = float(np.mean(a_scores))
        b_mean = float(np.mean(b_scores))
        segment_a.append(a_mean)
        segment_b.append(b_mean)
        system_a.setdefault(unit.key.system, []).append(a_mean)
        system_b.setdefault(unit.key.system, []).append(b_mean)

    common_systems = sorted(set(system_a) & set(system_b))
    sys_a_means = [float(np.mean(system_a[s])) for s in common_systems]
    sys_b_means = [float(np.mean(system_b[s])) for s in common_systems]

    if len(common_systems) >= 2:
        sys_result = kendalltau(sys_a_means, sys_b_means)
        sys_tau = float(sys_result.statistic)
        sys_p = float(sys_result.pvalue)
    else:
        sys_tau, sys_p = (float("nan"), float("nan"))

    if len(segment_a) >= 2:
        seg_result = kendalltau(segment_a, segment_b)
        seg_tau = float(seg_result.statistic)
        seg_p = float(seg_result.pvalue)
    else:
        seg_tau, seg_p = (float("nan"), float("nan"))

    return {
        "system_tau": sys_tau,
        "system_p": sys_p,
        "segment_tau": seg_tau,
        "segment_p": seg_p,
        "n_systems": len(common_systems),
        "n_segments": len(segment_a),
        "n_raters_total": len(sorted_raters),
        "n_raters_half_a": len(half_a),
        "n_raters_half_b": len(half_b),
    }


def plot_category_counts(tables: ResultTables, out_path: Path) -> None:
    """Grouped bar: human vs each LLM, one group per MQM category."""
    df = tables.category_counts
    providers = ["Human"] + tables.providers

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(CATEGORY_ORDER))
    bar_w = 0.8 / len(providers)

    for i, provider in enumerate(providers):
        vals = [
            df.loc[(df["provider"] == provider) & (df["category"] == cat), "mean_count"].iloc[0]
            if not df.loc[(df["provider"] == provider) & (df["category"] == cat)].empty else 0.0
            for cat in CATEGORY_ORDER
        ]
        ax.bar(x + i * bar_w, vals, width=bar_w, label=provider, color=PALETTE[i % len(PALETTE)])

    ax.set_xticks(x + bar_w * (len(providers) - 1) / 2)
    ax.set_xticklabels(CATEGORY_ORDER, rotation=30, ha="right")
    ax.set_ylabel("Mean errors per segment")
    ax.set_title("MQM Category Counts — Human vs LLM Judges")
    ax.legend(loc="upper right", ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.savefig(str(out_path))
    plt.close(fig)


def plot_severity_counts(tables: ResultTables, out_path: Path) -> None:
    """Grouped bar: human vs each LLM, one group per severity level."""
    df = tables.severity_counts
    providers = ["Human"] + tables.providers

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(SEVERITY_ORDER))
    bar_w = 0.8 / len(providers)

    for i, provider in enumerate(providers):
        vals = [
            df.loc[(df["provider"] == provider) & (df["severity"] == sev), "mean_count"].iloc[0]
            if not df.loc[(df["provider"] == provider) & (df["severity"] == sev)].empty else 0.0
            for sev in SEVERITY_ORDER
        ]
        ax.bar(x + i * bar_w, vals, width=bar_w, label=provider, color=PALETTE[i % len(PALETTE)])

    ax.set_xticks(x + bar_w * (len(providers) - 1) / 2)
    ax.set_xticklabels(SEVERITY_ORDER)
    ax.set_ylabel("Mean errors per segment")
    ax.set_title("MQM Severity Distribution — Human vs LLM Judges")
    ax.legend(loc="upper right", ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.savefig(str(out_path))
    plt.close(fig)


def plot_system_rankings(tables: ResultTables, out_path: Path, tau_df: "pd.DataFrame | None" = None) -> None:
    """Horizontal grouped bars: systems ordered by human score, human vs each LLM.

    Shows *where* each LLM breaks from the human ranking — the taller the LLM
    bar vs its human counterpart, the harsher that judge on that system. If a
    tau_df is supplied, the system-level Kendall τ appears in the legend label
    for each LLM so the agreement number sits next to the bar it describes.
    """
    llm_rankings = tables.llm_rankings
    if llm_rankings.empty:
        return

    ordered = llm_rankings.sort_values("human_mean_score")
    systems = ordered["system"].drop_duplicates().tolist()
    providers = tables.providers

    # Build a provider -> system-level tau lookup for legend annotation.
    sys_tau_by_provider: dict[str, float | None] = {}
    if tau_df is not None and not tau_df.empty:
        for _, row in tau_df.iterrows():
            sys_tau_by_provider[str(row["provider"])] = row["system_kendall_tau"]

    fig, ax = plt.subplots(figsize=(12, max(6, 0.45 * len(systems) * (len(providers) + 1))))
    y = np.arange(len(systems))
    bar_h = 0.8 / (len(providers) + 1)

    human_vals = [
        ordered[ordered["system"] == s]["human_mean_score"].iloc[0] for s in systems
    ]
    ax.barh(y, human_vals, height=bar_h, label="Human", color=PALETTE[0])

    for i, provider in enumerate(providers, start=1):
        vals = [
            ordered[(ordered["system"] == s) & (ordered["provider"] == provider)]["llm_mean_score"].iloc[0]
            if not ordered[(ordered["system"] == s) & (ordered["provider"] == provider)].empty else np.nan
            for s in systems
        ]
        tau = sys_tau_by_provider.get(provider)
        label = provider if tau is None or pd.isna(tau) else f"{provider}  (τ_sys = {float(tau):.2f})"
        ax.barh(y + i * bar_h, vals, height=bar_h, label=label, color=PALETTE[i % len(PALETTE)])

    ax.set_yticks(y + bar_h * len(providers) / 2)
    ax.set_yticklabels(systems)
    ax.invert_yaxis()
    ax.set_xlabel("Mean MQM segment score (lower = better)")
    ax.set_title("System Ranking Alignment — Human vs LLM Judges")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.savefig(str(out_path))
    plt.close(fig)


def plot_segment_score_scatter(tables: ResultTables, out_path: Path, tau_df: "pd.DataFrame | None" = None) -> None:
    """One subplot per LLM: human segment score (x) vs LLM segment score (y), with y=x line.

    If a tau_df is supplied, each subplot title shows that model's segment-level
    Kendall τ — the figure-appropriate agreement number for a per-segment view.
    """
    per_segment = tables.per_segment
    if per_segment.empty:
        return

    providers = tables.providers
    seg_tau_by_provider: dict[str, float | None] = {}
    if tau_df is not None and not tau_df.empty:
        for _, row in tau_df.iterrows():
            seg_tau_by_provider[str(row["provider"])] = row["segment_kendall_tau"]

    n_cols = min(3, len(providers))
    n_rows = (len(providers) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.2 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    max_score = max(
        per_segment["human_mean_segment_score"].max(),
        per_segment["llm_segment_score"].max(),
        25.0,
    )

    for i, provider in enumerate(providers):
        ax = axes[i]
        rows = per_segment[per_segment["provider"] == provider]
        ax.scatter(
            rows["human_mean_segment_score"],
            rows["llm_segment_score"],
            s=22,
            alpha=0.55,
            color=PALETTE[(i + 1) % len(PALETTE)],
            edgecolors="none",
        )
        ax.plot([0, max_score], [0, max_score], color="#888", linestyle="--", linewidth=1)
        tau = seg_tau_by_provider.get(provider)
        title = provider if tau is None or pd.isna(tau) else f"{provider}\nτ_seg = {float(tau):.2f}"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Human mean segment score")
        ax.set_ylabel("LLM segment score")
        ax.set_xlim(0, max_score)
        ax.set_ylim(0, max_score)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

    for j in range(len(providers), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Per-Segment Score Alignment (y = x is perfect agreement)", fontsize=14)
    fig.savefig(str(out_path))
    plt.close(fig)


def plot_kendall_tau(tau_df: pd.DataFrame, out_path: Path) -> None:
    """Side-by-side bars: system-level tau and segment-level tau per provider."""
    if tau_df.empty:
        return

    providers = tau_df["provider"].tolist()
    x = np.arange(len(providers))
    bar_w = 0.38

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(providers)), 6))
    ax.bar(
        x - bar_w / 2,
        tau_df["system_kendall_tau"].fillna(0),
        width=bar_w,
        label="System-level τ",
        color=PALETTE[1],
    )
    ax.bar(
        x + bar_w / 2,
        tau_df["segment_kendall_tau"].fillna(0),
        width=bar_w,
        label="Segment-level τ",
        color=PALETTE[2],
    )

    sys_vals = tau_df["system_kendall_tau"].tolist()
    seg_vals = tau_df["segment_kendall_tau"].tolist()
    for i, (sys_val, seg_val) in enumerate(zip(sys_vals, seg_vals)):
        if sys_val is not None and not pd.isna(sys_val):
            v = float(sys_val)  # type: ignore[arg-type]
            ax.text(i - bar_w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        if seg_val is not None and not pd.isna(seg_val):
            v = float(seg_val)  # type: ignore[arg-type]
            ax.text(i + bar_w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(providers, rotation=20, ha="right")
    ax.set_ylabel("Kendall τ (higher = closer to humans)")
    ax.set_title("Agreement With Human MQM — Kendall τ by Provider")
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.set_ylim(-0.2, 1.05)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.savefig(str(out_path))
    plt.close(fig)


def plot_kendall_tau_by_category(tables: ResultTables, out_path: Path) -> None:
    """Per-category segment-level Kendall τ.

    For each (model, category), correlate the human and LLM per-segment
    counts of that category across all sampled segments. The plot answers:
    which MQM categories does each model *track* well, and which does it
    systematically miss or over-call?

    X = MQM categories. Y = segment-level Kendall τ. Bars = models.
    Constant series (e.g. all-zero for a rare category) yield NaN τ, which
    we draw as a zero-height bar with a small marker.
    """
    if not tables.providers or not tables.human_category_counts_by_key:
        return

    providers = tables.providers
    human_counts = tables.human_category_counts_by_key

    # Build per-segment LLM category counts for every provider.
    llm_counts_by_provider: dict[str, dict[str, dict[str, float]]] = {}
    for provider in providers:
        counts_by_key: dict[str, dict[str, float]] = {}
        for record in tables.raw_by_provider.get(provider, []):
            result = record.get("result", {})
            cat_counts, _ = count_categories_and_severities(result.get("errors", []))
            counts_by_key[record["sample_key"]] = dict(cat_counts)
        llm_counts_by_provider[provider] = counts_by_key

    # Compute τ per (provider, category) over the intersection of sample_keys.
    per_category_tau: dict[str, dict[str, float | None]] = {p: {} for p in providers}
    for provider in providers:
        llm_counts = llm_counts_by_provider[provider]
        shared_keys = [k for k in llm_counts if k in human_counts]
        for cat in CATEGORY_ORDER:
            if len(shared_keys) < 2:
                per_category_tau[provider][cat] = None
                continue
            human_vals = [human_counts[k].get(cat, 0.0) for k in shared_keys]
            llm_vals = [llm_counts[k].get(cat, 0.0) for k in shared_keys]
            if len(set(human_vals)) < 2 or len(set(llm_vals)) < 2:
                # Constant series → τ undefined. Render as zero with a marker.
                per_category_tau[provider][cat] = None
                continue
            result = kendalltau(human_vals, llm_vals)
            tau = float(result.statistic)
            per_category_tau[provider][cat] = None if np.isnan(tau) else tau

    fig, ax = plt.subplots(figsize=(max(12, 1.0 * len(CATEGORY_ORDER)), 6))
    x = np.arange(len(CATEGORY_ORDER))
    bar_w = 0.8 / max(len(providers), 1)

    for i, provider in enumerate(providers):
        vals: list[float] = []
        undef_positions: list[float] = []
        for j, cat in enumerate(CATEGORY_ORDER):
            t = per_category_tau[provider].get(cat)
            if t is None:
                vals.append(0.0)
                undef_positions.append(x[j] + i * bar_w)
            else:
                vals.append(float(t))
        color = PALETTE[(i + 1) % len(PALETTE)]
        ax.bar(x + i * bar_w, vals, width=bar_w, label=provider, color=color)
        # Mark undefined τ with a subtle 'x' above the baseline so viewers don't
        # read a zero-height bar as "zero correlation".
        if undef_positions:
            ax.scatter(undef_positions, [0.02] * len(undef_positions), marker="x", s=30, color="#888", zorder=3)

    ax.set_xticks(x + bar_w * (len(providers) - 1) / 2)
    ax.set_xticklabels(CATEGORY_ORDER, rotation=30, ha="right")
    ax.set_ylabel("Segment-level Kendall τ")
    ax.set_xlabel("MQM category")
    ax.set_title("Per-Category Segment Agreement — Human vs LLM (× = τ undefined, constant series)")
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.set_ylim(-0.6, 1.05)
    ax.legend(loc="upper right", ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    fig.savefig(str(out_path))
    plt.close(fig)


def plot_category_delta_heatmap(tables: ResultTables, out_path: Path) -> None:
    """Diverging heatmap: signed delta (LLM mean − human mean) per category × provider.

    Red = LLM over-calls this category, blue = under-calls. Zero = matches humans.
    Makes prompt-tuning targets (e.g. under-detected Style) visible at a glance.
    """
    if not tables.providers:
        return

    rows = []
    human_by_cat = {
        cat: tables.category_counts.loc[
            (tables.category_counts["provider"] == "Human")
            & (tables.category_counts["category"] == cat),
            "mean_count",
        ].iloc[0]
        for cat in CATEGORY_ORDER
    }
    for provider in tables.providers:
        for cat in CATEGORY_ORDER:
            llm_val = tables.category_counts.loc[
                (tables.category_counts["provider"] == provider)
                & (tables.category_counts["category"] == cat),
                "mean_count",
            ]
            llm_mean = llm_val.iloc[0] if not llm_val.empty else 0.0
            rows.append({
                "provider": provider,
                "category": cat,
                "delta": llm_mean - human_by_cat[cat],
            })
    delta_df = pd.DataFrame(rows)
    matrix = delta_df.pivot(index="provider", columns="category", values="delta")
    matrix = matrix[CATEGORY_ORDER]

    arr = np.asarray(matrix.values, dtype=float)
    vmax = max(abs(float(np.nanmin(arr))), abs(float(np.nanmax(arr))), 0.1)

    fig, ax = plt.subplots(figsize=(max(10, 1.0 * len(CATEGORY_ORDER)), 1.1 * len(tables.providers) + 2))
    im = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(CATEGORY_ORDER)))
    ax.set_xticklabels(CATEGORY_ORDER, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(tables.providers)))
    ax.set_yticklabels(tables.providers)
    ax.set_title("LLM − Human Error Count (red = LLM over-calls, blue = under-calls)")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = float(arr[i, j])
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(val) > vmax * 0.55 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Δ mean count per segment")
    fig.savefig(str(out_path))
    plt.close(fig)


def write_all_data(tables: ResultTables, out_path: Path) -> None:
    """Consolidated long-form CSV: one row per (sampled segment, provider).

    Contains every human ground-truth signal and every LLM signal on the same
    row so Excel/pandas analyses can start from this one file. Fields:

    - identity:  sample_key, provider, system, doc, seg_id, source/system text
    - human:     mean_segment_score, dominant_category, dominant_severity,
                 per-category counts (human_count_<Cat>), per-severity counts
    - LLM:       segment_score, dominant_category, dominant_severity,
                 per-category counts (llm_count_<Cat>), per-severity counts,
                 scored_error_count, overall_comment, full errors JSON
    - deltas:    segment_score_gap (llm − human)
    """
    if tables.manifest.empty:
        return

    # Build manifest index once; dict preserves manifest row order.
    manifest_rows = tables.manifest.to_dict("records")

    # Index LLM predictions by (provider, sample_key).
    pred_by_provider: dict[str, dict[str, dict]] = {}
    for provider in tables.providers:
        pred_by_provider[provider] = {
            record["sample_key"]: record
            for record in tables.raw_by_provider.get(provider, [])
        }

    rows: list[dict] = []
    for provider in tables.providers:
        for manifest_row in manifest_rows:
            sample_key = manifest_row["sample_key"]
            pred_record = pred_by_provider[provider].get(sample_key)
            human_cats = tables.human_category_counts_by_key.get(sample_key, {})
            human_sevs = tables.human_severity_counts_by_key.get(sample_key, {})

            row: dict = {
                "sample_key": sample_key,
                "provider": provider,
                "system": manifest_row.get("system"),
                "doc": manifest_row.get("doc"),
                "seg_id": manifest_row.get("seg_id"),
                "source_segment": manifest_row.get("source_segment"),
                "system_segment": manifest_row.get("system_segment"),
                "human_mean_segment_score": manifest_row.get("human_mean_segment_score"),
                "human_dominant_category": manifest_row.get("human_dominant_category"),
                "human_dominant_severity": manifest_row.get("human_dominant_severity"),
            }
            for cat in CATEGORY_ORDER:
                row[f"human_count_{cat}"] = human_cats.get(cat, 0.0)
            for sev in SEVERITY_ORDER:
                row[f"human_sev_{sev}"] = human_sevs.get(sev, 0.0)

            if pred_record is not None:
                result = pred_record.get("result", {})
                errors = result.get("errors", [])
                llm_cats, llm_sevs = count_categories_and_severities(errors)
                llm_score = result.get("segment_mqm_score")
                row["llm_segment_score"] = llm_score
                row["llm_scored_error_count"] = result.get("scored_error_count")
                row["llm_overall_comment"] = result.get("overall_comment", "")
                row["llm_dominant_category"] = choose_dominant_label(llm_cats, CATEGORY_ORDER)
                row["llm_dominant_severity"] = choose_dominant_label(llm_sevs, SEVERITY_ORDER)
                for cat in CATEGORY_ORDER:
                    row[f"llm_count_{cat}"] = llm_cats.get(cat, 0.0)
                for sev in SEVERITY_ORDER:
                    row[f"llm_sev_{sev}"] = llm_sevs.get(sev, 0.0)
                if llm_score is not None and row["human_mean_segment_score"] is not None:
                    row["segment_score_gap"] = round(float(llm_score) - float(row["human_mean_segment_score"]), 4)
                else:
                    row["segment_score_gap"] = None
                # Full errors payload — JSON-encoded so it survives a CSV round-trip.
                row["llm_errors_json"] = json.dumps(errors, ensure_ascii=False)
            else:
                # Provider didn't produce a prediction for this segment
                # (e.g. failure after retries). Leave LLM fields empty.
                row["llm_segment_score"] = None
                row["llm_scored_error_count"] = None
                row["llm_overall_comment"] = ""
                row["llm_dominant_category"] = None
                row["llm_dominant_severity"] = None
                for cat in CATEGORY_ORDER:
                    row[f"llm_count_{cat}"] = None
                for sev in SEVERITY_ORDER:
                    row[f"llm_sev_{sev}"] = None
                row["segment_score_gap"] = None
                row["llm_errors_json"] = None

            rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_metrics_summary(tau_df: pd.DataFrame, tables: ResultTables, out_path: Path) -> None:
    """One-row-per-provider CSV including Kendall tau + the existing summary metrics.

    This is the master table plots read from, and the row you append to
    experiments.csv when sweeping across N / model / prompt.
    """
    summary_rows = []
    for provider in tables.providers:
        s = tables.summary["providers"][provider]
        tau_row = tau_df[tau_df["provider"] == provider].iloc[0] if not tau_df[tau_df["provider"] == provider].empty else None
        summary_rows.append({
            "provider": provider,
            "segments_per_system": tables.summary.get("segments_per_system"),
            "seed": tables.summary.get("seed"),
            "n_systems": tau_row["n_systems"] if tau_row is not None else None,
            "n_segments": tau_row["n_segments"] if tau_row is not None else None,
            "system_kendall_tau": tau_row["system_kendall_tau"] if tau_row is not None else None,
            "system_kendall_p": tau_row["system_kendall_p"] if tau_row is not None else None,
            "segment_kendall_tau": tau_row["segment_kendall_tau"] if tau_row is not None else None,
            "segment_kendall_p": tau_row["segment_kendall_p"] if tau_row is not None else None,
            "segment_score_mae": s.get("segment_score_mae"),
            "mean_segment_score_gap": s.get("mean_segment_score_gap"),
            "dominant_category_match_rate": s.get("dominant_category_match_rate"),
            "dominant_severity_match_rate": s.get("dominant_severity_match_rate"),
            "no_error_agreement_rate": s.get("no_error_agreement_rate"),
            "style_presence_recall": s.get("style_presence_recall"),
            "mean_human_segment_score": s.get("mean_human_segment_score"),
            "mean_llm_segment_score": s.get("mean_llm_segment_score"),
        })
    pd.DataFrame(summary_rows).to_csv(out_path, index=False)


def generate_all_graphs(tables: ResultTables, output_dir: str | Path) -> pd.DataFrame:
    """Write every PNG + the metrics CSV for one benchmark run. Returns the tau DataFrame."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tau_df = compute_kendall_tau(tables)

    plot_category_counts(tables, output_dir / "category_counts.png")
    plot_severity_counts(tables, output_dir / "severity_counts.png")
    plot_system_rankings(tables, output_dir / "system_rankings.png", tau_df=tau_df)
    plot_segment_score_scatter(tables, output_dir / "segment_score_scatter.png", tau_df=tau_df)
    plot_kendall_tau(tau_df, output_dir / "kendall_tau.png")
    plot_kendall_tau_by_category(tables, output_dir / "kendall_tau_by_category.png")
    plot_category_delta_heatmap(tables, output_dir / "category_delta_heatmap.png")

    write_metrics_summary(tau_df, tables, output_dir / "metrics_summary.csv")
    write_all_data(tables, output_dir / "all_data.csv")
    tau_df.to_csv(output_dir / "kendall_tau.csv", index=False)

    return tau_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Render poster-ready PNG plots from a benchmark run.")
    parser.add_argument("--results-dir", required=True, help="Directory written by wmt20_mqm_llm_benchmark.py.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where PNGs land. Defaults to <results-dir>/plots.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "plots"

    tables = load_results(results_dir)
    tau_df = generate_all_graphs(tables, output_dir)

    print(f"\nWrote plots and metrics to: {output_dir}")
    print("\nKendall τ by provider:")
    print(tau_df.to_string(index=False))


if __name__ == "__main__":
    main()
