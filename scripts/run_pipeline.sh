#!/usr/bin/env bash
# End-to-end pipeline: dataset -> experiments -> sweeps -> ablations ->
# baselines -> PQC/TPM benchmarks -> figures + tables.
# Usage:  bash scripts/run_pipeline.sh [quick|full]
# "quick" uses small sizes for a fast smoke run; "full" reproduces paper numbers.
set -euo pipefail

cd "$(dirname "$0")/.."
MODE="${1:-quick}"

if [[ "$MODE" == "full" ]]; then
  NPC=1500; CRYPTO=48; TWIN=256; NSESS=100; EPOCHS=120; NSWEEP=40; MC=5000; TPMN=100
  SEEDS="0 1 2 3 4"; PQITERS=200
else
  NPC=400;  CRYPTO=16; TWIN=96;  NSESS=25;  EPOCHS=60;  NSWEEP=20; MC=2000; TPMN=80
  SEEDS="0 1 2"; PQITERS=80
fi

echo "=== [1/7] generate dataset (mode=$MODE) ==="
python data_generator/generate_dataset.py --n-per-class "$NPC" \
  --crypto-sessions "$CRYPTO" --transport-windows "$TWIN" --tpm-N "$TPMN"

echo "=== [2/7] five-configuration comparison ==="
python experiments/run_all.py --n-sessions "$NSESS" --tpm-N "$TPMN" \
  --mc-trials "$MC" --fnn-epochs "$EPOCHS"

echo "=== [3/7] sweeps (alpha, K, TPM-N, and (T,N) threshold frontier) ==="
python experiments/sweep.py --n-sessions "$NSWEEP" --fnn-epochs "$EPOCHS" \
  --N-values 40 60 80 100

echo "=== [4/7] ablations (feature groups + architecture + early termination) ==="
python experiments/ablation.py --n-sessions "$NSWEEP" --tpm-N "$TPMN" \
  --fnn-epochs "$EPOCHS"

echo "=== [5/7] baselines + significance (RF/SVM/GB/LogReg vs WKNN/FNN/Hybrid) ==="
python experiments/baselines.py --seeds $SEEDS --fnn-epochs "$EPOCHS"

echo "=== [6/7] key-establishment benchmarks (ML-KEM vs TPM vs hybrid; TPM micro) ==="
python scripts/benchmark_pqc.py --iters "$PQITERS" --tpm-N "$TPMN" --tpm-reps 8
python scripts/benchmark_tpm.py --N "$TPMN" --round-iters 4000 --session-reps 8

echo "=== [7/7] figures + LaTeX tables ==="
python scripts/make_figures.py --tpm-N "$TPMN" --fnn-epochs "$EPOCHS"
python scripts/make_tables.py

echo "=== DONE. Results in results/, figures in figures/ and paper/figures/,"
echo "    tables in paper/tables/. Build the paper: cd paper && pdflatex main &&"
echo "    bibtex main && pdflatex main && pdflatex main ==="
