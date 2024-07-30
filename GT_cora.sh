#!/bin/bash
#SBATCH -p sfscai
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1

module purge
module load miniconda
source activate graphT

PYTHON_PATH="/gpfsnyu/home/ys6310/.conda/envs/graphT/bin/python"

$PYTHON_PATH -u trainGT.py > /gpfsnyu/scratch/ys6310/GraphT/cora_gt_new.txt 2>&1