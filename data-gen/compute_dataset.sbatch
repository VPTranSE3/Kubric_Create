#!/bin/bash
#SBATCH --job-name=kubric-dgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-999%38
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --time=72:00:00
#SBATCH --partition=hpda2_compute
#SBATCH --account=pn76ne-c
#SBATCH --chdir=/dss/dsshome1/00/di97nip/master_thesis/gcd/data-gen/
#SBATCH --output=jobs/compute_dataset_%A_%a.out
#SBATCH --error=jobs/compute_dataset_%A_%a.err

module load slurm_setup
module load miniconda3

conda run -n kubric_creator python export_kub_mv_slurm.py \
  --mass_est_fp=gpt_mass_v4.txt \
  --root_dp=/dss/dsstbyfs02/pn52ko/pn52ko-dss-0000/di97nip/Kubric_generation \
  --num_scenes=3000 \
  --start_idx=$((SLURM_ARRAY_TASK_ID + 2000)) \
  --end_idx=$((SLURM_ARRAY_TASK_ID + 2000 + 1)) \
  --num_workers=1 \
  --restart_count=1 \
  --seed=900000 \
  --num_views=16 \
  --frame_width=576 \
  --frame_height=384 \
  --num_frames=60 \
  --frame_rate=24 \
  --save_depth=1 \
  --save_coords=1 \
  --render_samples_per_pixel=16 \
  --focal_length=32 \
  --fixed_alter_poses=1 \
  --few_views=4