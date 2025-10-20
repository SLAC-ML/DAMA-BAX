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

# Configuration: Set these via environment variables when submitting
# Example: CASE_DIR=examples/dama RUN_ID=3 MAX_ITER=100 sbatch job.sh
CASE_DIR=${CASE_DIR:-"examples/dama"}
RUN_ID=${RUN_ID:-3}
MAX_ITER=${MAX_ITER:-3200}
N_SAMPLING=${N_SAMPLING:-50}
DEVICE=${DEVICE:-"auto"}

# Run via unified runner
python run.py --case "${CASE_DIR}" \
              --run-id ${RUN_ID} \
              --max-iter ${MAX_ITER} \
              --n-sampling ${N_SAMPLING} \
              --device ${DEVICE}
