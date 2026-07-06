#!/bin/bash
#SBATCH --job-name=trash_rfdetr
#SBATCH --partition=gpu-4-a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=0-06:00
#SBATCH --output=/home/vfdabreu/logs/treino_rfdetr_%j.log
#SBATCH --error=/home/vfdabreu/logs/treino_rfdetr_%j.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /etc/profile.d/conda.sh
conda activate rfdetr_env

python - << 'PYTHON'
from rfdetr import RFDETRBase

model = RFDETRBase()
model.train(
    dataset_dir="/home/vfdabreu/trash-detection",
    epochs=50,
    batch_size=8,
    grad_accumulation_steps=2,
    lr=1e-4,
    output_dir="/home/vfdabreu/resultados/trash_rfdetr",
)
PYTHON

echo "Treino RF-DETR finalizado em $(date)"
