#!/bin/bash
#SBATCH --job-name=trash_yolo
#SBATCH --partition=gpu-4-a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=0-03:00
#SBATCH --output=$HOME/logs/treino_%j.log
#SBATCH --error=$HOME/logs/treino_%j.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source activate gpu

echo "Iniciando treino em $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

pip install -q ultralytics

mkdir -p $HOME/resultados

yolo detect train \
    model=yolo26s.pt \
    data=$HOME/trash-detection/data.yaml \
    epochs=50 \
    imgsz=640 \
    batch=16 \
    name=trash_baseline \
    seed=42 \
    project=$HOME/resultados

echo "Treino finalizado em $(date)"
