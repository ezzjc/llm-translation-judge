#!/bin/bash
# Full N-sweep for one provider end-to-end:
#   1. N=200 benchmark (the only billable step)
#   2. Copy cache to n100/n50/n3 dirs
#   3. N=100/50/3 benchmarks (cache hits, no API calls)
#   4. graphs.py for every N
#   5. n_sweep_chart.py to produce the 5-bar comparison charts
#
# Usage:
#   source .venv/bin/activate
#   export OPENAI_API_KEY=sk-...
#   bash run_full_sweep.sh openai:gpt-5.4-mini
#   bash run_full_sweep.sh openai:gpt-5.4

set -e

PROVIDER="${1:-openai:gpt-4o-mini}"
SEED=42
SWEEP_DIR="results/exp_N_sweep"

# Mirror mqm_paper_core.sanitize_slug so we know the cache filename.
SLUG=$(echo "$PROVIDER" | sed 's/[^A-Za-z0-9._-]/_/g' | sed 's/^_*//' | sed 's/_*$//')
CACHE_FILE="$SWEEP_DIR/n200/raw_${SLUG}.jsonl"

# Sanity checks
if [ -z "$OPENAI_API_KEY" ]; then
  echo "ERROR: OPENAI_API_KEY is not set in this shell."
  echo "Run:  export OPENAI_API_KEY=sk-..."
  exit 1
fi
if [ ! -d ".venv" ]; then
  echo "ERROR: .venv directory missing."
  exit 1
fi

echo "===================================================================="
echo "  Full N-sweep for provider:  $PROVIDER"
echo "  Cache slug:                 $SLUG"
echo "  Output root:                $SWEEP_DIR"
echo "===================================================================="

# Step 1: N=200 (billable)
mkdir -p "$SWEEP_DIR/n200"
echo ""
echo "===================================================================="
echo "  [1/5] N=200 benchmark for $PROVIDER (this is the billable step)"
echo "===================================================================="
python wmt20_mqm_llm_benchmark.py \
  --provider "$PROVIDER" \
  --segments-per-system 200 \
  --seed "$SEED" \
  --output-dir "$SWEEP_DIR/n200"

# Step 2: Copy the new model's cache to smaller-N dirs
echo ""
echo "===================================================================="
echo "  [2/5] Copying cache to n100, n50, n3"
echo "===================================================================="
for N in 100 50 3; do
  mkdir -p "$SWEEP_DIR/n$N"
  cp "$CACHE_FILE" "$SWEEP_DIR/n$N/"
done

# Step 3: Smaller-N benchmarks (cache hits, fast)
echo ""
echo "===================================================================="
echo "  [3/5] Smaller-N benchmarks for $PROVIDER (should be all cache hits)"
echo "===================================================================="
for N in 100 50 3; do
  python wmt20_mqm_llm_benchmark.py \
    --provider "$PROVIDER" \
    --segments-per-system "$N" \
    --seed "$SEED" \
    --output-dir "$SWEEP_DIR/n$N"
done

# Step 4: Render graphs.py for every N
echo ""
echo "===================================================================="
echo "  [4/5] Rendering per-run plots for n200 / n100 / n50 / n3"
echo "===================================================================="
for N in 200 100 50 3; do
  python graphs.py --results-dir "$SWEEP_DIR/n$N"
done

# Step 5: Render the 5-bar comparison charts for this provider
echo ""
echo "===================================================================="
echo "  [5/5] Rendering N-sweep comparison charts for $PROVIDER"
echo "===================================================================="
python n_sweep_chart.py \
  --results-root "$SWEEP_DIR" \
  --provider "$PROVIDER"

echo ""
echo "===================================================================="
echo "  Done.  Comparison charts in $SWEEP_DIR/charts/"
echo "  Look for files containing the slug: $SLUG"
echo "===================================================================="
