#!/usr/bin/env bash
# ==============================================================================
# Step 6: Run Analytics - 4-Dimension Evaluation (Section 3.5)
# Container: app
# Usage: ./scripts/06_run_analytics.sh [--datasets sst2,ag_news]
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults
DATASETS="sst2,ag_news"
EXPLANATIONS_DIR="outputs/explanations"
OUTPUT_DIR="outputs/analytics"
DRY_RUN=false

show_help() {
    cat <<'USAGE'
Usage: 06_run_analytics.sh [options]

Run 4-dimension evaluation framework (Paper Section 3.5)

Dimensions:
  1. AUC Discriminative Capacity
  2. Behavioral Faithfulness (Fidelity)
  3. Consistency Across Outcomes
  4. Feature Ranking Stability (Progression)

Options:
  --datasets LIST        Comma-separated: sst2,ag_news (default: both)
  --explanations-dir     Input explanations directory
  --output-dir           Analytics output directory
  --dry-run              Print commands without executing
  --help                 Show this help
USAGE
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets) DATASETS="$2"; shift 2 ;;
        --explanations-dir) EXPLANATIONS_DIR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --help) show_help ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

cd "$REPO_ROOT"

echo "=========================================="
echo "Step 6: Running 4-Dimension Evaluation"
echo "  Explanations: $EXPLANATIONS_DIR"
echo "  Output: $OUTPUT_DIR"
echo "=========================================="

# Run each dimension analysis
DIMENSIONS=("auc" "fidelity" "consistency" "progression")

for dim in "${DIMENSIONS[@]}"; do
    echo ""
    echo "--- Dimension: $dim ---"
    
    CMD="python -m src.Analytics.$dim \
        --inputs $EXPLANATIONS_DIR \
        --output $OUTPUT_DIR/$dim"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "[dry-run] $CMD"
    else
        eval $CMD || echo "[warn] $dim analysis failed or module not found"
    fi
done

echo ""
echo "--- Running Logistic Regression (Section 3.6) ---"

CMD="python -m src.use_case.save_logistic_coefficients \
    --input $OUTPUT_DIR \
    --output $OUTPUT_DIR/logistic_regression"

if [[ "$DRY_RUN" == true ]]; then
    echo "[dry-run] $CMD"
else
    eval $CMD || echo "[warn] Logistic regression failed"
fi

echo ""
echo "Analytics complete. Results in: $OUTPUT_DIR/"
