#!/bin/bash
#SBATCH --job-name="Overfit novel view synthsis single example Kubric, disentangle"
#SBATCH --cpus-per-task=40
#SBATCH --mem=512G
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute_gpu
#SBATCH --get-user-env
#SBATCH --account=pn76ne-c
#SBATCH -D /dss/dsshome1/00/di97nip/master_thesis/gcd/data-gen
#SBATCH --output=test_bake_colours_%j.out
#SBATCH --error=test_bake_colours_%j.out

module load slurm_setup
module load miniconda3
# replace the output of conda info --base below
source activate kubric_creator 
nvidia-smi
python test_bake_colours.py

