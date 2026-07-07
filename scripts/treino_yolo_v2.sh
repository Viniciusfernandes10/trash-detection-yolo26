#!/bin/bash
#SBATCH --job-name=trash_yolo_v2
#SBATCH --partition=gpu-4-a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=0-04:00
#SBATCH --output=$HOME/logs/treino_v2_%j.log
#SBATCH --error=$HOME/logs/treino_v2_%j.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source activate gpu
pip install -q ultralytics

yolo detect train \
    model=$HOME/resultados/trash_baseline/weights/best.pt \
    data=$HOME/trash-detection/data.yaml \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    name=trash_v2_100ep \
    seed=42 \
    project=$HOME/resultados

echo "Treino v2 finalizado em $(date)"
