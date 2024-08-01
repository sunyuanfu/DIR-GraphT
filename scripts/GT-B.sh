#!/bin/bash
#SBATCH -p sfscai
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:1

module purge
module load miniconda
source activate graphT

cd /gpfsnyu/scratch/ys6310/GraphT

PYTHON_PATH="/gpfsnyu/home/ys6310/.conda/envs/graphT/bin/python"

$PYTHON_PATH -u trainGT.py dataset pubmed gt.train.n_layers 6 > /gpfsnyu/scratch/ys6310/GraphT/scripts/GT-B-Pubmed.txt 2>&1