#!/bin/bash
#SBATCH --job-name=jaqsi-data
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --time=06:00:00
#SBATCH --partition cpu
#SBATCH --mem=200GB
#SBATCH --output="logs/slurm/slurm-%j-%x.out"

module load compiler/llvm
module load devel/python/3.12.3

cd ~/jaqsi-data
uv sync
uv run python -m benchmark
