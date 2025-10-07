#!/bin/sh

#SBATCH --partition=ampere
#SBATCH --account=mli:bes-anomaly
#SBATCH --job-name=dama_bax
#SBATCH --output=logs/output-dama-bax-%j.txt
#SBATCH --error=logs/error-dama-bax-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --mem-per-cpu=4gb
#SBATCH --gpus=1
#SBATCH --time=12:00:00

# Activate conda env
source /sdf/group/mli/zhezhang/conda/bin/activate
conda activate ml

# Run code
python run_3.py
