#!/bin/bash
# Runs the remaining N-sweep work after N=200 is already done.
# All smaller-N runs are cache hits (no API calls), so this is fast.
#
# Usage:
#   source .venv/bin/activate
#   export OPENAI_API_KEY=sk-...
#   bash run_n_sweep.sh

set -e

PROVIDER="openai:gpt-4o-mini"
SEED=42
SWEEP_DIR="results/exp_N_sweep"

if [ -z "$OPENAI_API_KEY" ]; then
  echo "ERROR: OPENAI_API_KEY is not set in this shell."
  echo "Run:  export OPENAI_API_KEY=sk-..."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "ERROR: .venv directory missing."
  exit 1
fi

if [ ! -f "$SWEEP_DIR/n200/raw_openai_gpt-4o-mini.jsonl" ]; then
  echo "ERROR: $SWEEP_DIR/n200/raw_openai_gpt-4o-mini.jsonl not found."
  echo "       The N=200 run must complete first."
  exit 1
fi

# 1. Run smaller-N benchmarks (all cache hits — fast)
for N in 100 50 3; do
  echo ""
  echo "===================================================================="
  echo "  N=$N benchmark (should be all cache hits)"
  echo "===================================================================="
  python wmt20_mqm_llm_benchmark.py \
    --provider "$PROVIDER" \
    --segments-per-system "$N" \
    --seed "$SEED" \
    --output-dir "$SWEEP_DIR/n$N"
done

# 2. Render per-run plots and metrics_summary.csv for every N
for N in 200 100 50 3; do
  echo ""
  echo "===================================================================="
  echo "  Rendering plots for N=$N"
  echo "===================================================================="
  python graphs.py --results-dir "$SWEEP_DIR/n$N"
done

# 3. Render the two final 5-bar comparison charts
echo ""
echo "===================================================================="
echo "  Rendering N-sweep comparison charts"
echo "===================================================================="
python n_sweep_chart.py \
  --results-root "$SWEEP_DIR" \
  --provider "$PROVIDER"

echo ""
echo "===================================================================="
echo "  All done."
echo "  Comparison charts: $SWEEP_DIR/charts/"
echo "    - n_sweep_system_tau.png"
echo "    - n_sweep_segment_tau.png"
echo "    - n_sweep_tau_values.csv"
echo "===================================================================="
