#!/bin/bash
#SBATCH --job-name="Calculate Kubric dataset"
#SBATCH --mem=500G
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --clusters=hpda2
#SBATCH --partition=hpda2_compute
#SBATCH --get-user-env
#SBATCH --account=pn76ne-c
#SBATCH -D /dss/dsshome1/00/di97nip/master_thesis/gcd/data-gen/
#SBATCH --output=compute_dataset_%j.out
#SBATCH --error=compute_dataset_error_%j.out

module load slurm_setup
module load miniconda3
# replace the output of conda info --base below
source activate kubric_creator 
for i in {1..110}
do
python export_kub_mv_new.py --mass_est_fp=gpt_mass_v4.txt \
--root_dp=/dss/dsstbyfs02/pn52ko/pn52ko-dss-0000/di97nip/Kubric_generation \
--num_scenes=3000 --num_workers=20 --restart_count=60 --end_idx=1 \
--seed=900000 --num_views=16 --frame_width=576 --frame_height=384 \
--num_frames=60 --frame_rate=24 --save_depth=1 --save_coords=1 \
--render_samples_per_pixel=16 --focal_length=32 \
--fixed_alter_poses=1 --few_views=4
rm -rf /dss/dsstbyfs02/scratch/00/di97nip/tmp/
done

