#!/bin/bash
#SBATCH --job-name=prefill_length
#SBATCH --partition=seec-contrib
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --chdir=/home/eicchen/conserve_project/BMPD/profiling/experiments/prefill_length
#SBATCH --output=/home/eicchen/conserve_project/BMPD/profiling/output/prefill_length/logs/slurm-%j.out
#
# Run the prefill length sweep, then plot it. Submit from anywhere:
#
#   sbatch profiling/experiments/prefill_length/prefill_length.sbatch

set -euo pipefail

set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate conserve
set -u

python prefill_length.py
python plot.py
