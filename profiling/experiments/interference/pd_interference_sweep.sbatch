#!/bin/bash
#SBATCH --job-name=pd_interference_sweep
#SBATCH --partition=seec-contrib
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --exclude=r003
#SBATCH --time=4:00:00
#SBATCH --chdir=/home/eicchen/conserve_project/BMPD/profiling/experiments/interference
#SBATCH --output=/home/eicchen/conserve_project/BMPD/profiling/output/interference/pd_interference/logs/slurm-%j.out
#
# Run the pd_interference input and output length sweep at vLLM's default max_num_seqs. Submit from anywhere:
#
#   sbatch profiling/experiments/interference/pd_interference_sweep.sbatch

set -euo pipefail

set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate conserve
set -u

python - <<'EOF'
from pd_interference import run_sweep

run_sweep(
    "length_sweep2",
    [512, 1024, 2048, 4096, 8192, 16384, 32768],
    [128, 256, 512, 1024, 2048],
    32,
)
EOF
