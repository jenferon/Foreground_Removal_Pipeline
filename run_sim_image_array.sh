#!/bin/bash
#SBATCH --partition=shortq,devq,defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --array=1-101
module load singularity/3.8.5

#export CUDA_VISIBLE_DEVICES=`/gpfs01/software/gpucheck/gpuuse.sh`

application="python $/gpfs01/home/ppxjf3/OSKAR/OSKAR_example_settings_script.py"

#------------------------------------------------------------------------------
# This script takes 3 arguments!
#   1. Frequency channel [0-169]
#   2. Run mode [1 = fg, 2 = cs_sf_tapered, 3 = noise]
#   3. Filename
#------------------------------------------------------------------------------

channel_i="140.0"
delf="0.1"
run_mode=3
fname="Noise_N512_FOV1.000"
fov="1.0" 
pix_side="512"
telescope="SKA"
field="EOR0"
telescope_model="SKA_all_stations_Rev3"
max_bl=3400
workdir="./"
weighting="natural"

freq=$(echo "scale=3;$channel_i+$delf*($SLURM_ARRAY_TASK_ID-1)" | bc)
echo $freq
seed=$(echo "scale=0;$SLURM_ARRAY_TASK_ID*$RANDOM" | bc)
echo $seed
printf -v fname1 "_%04.1fMHz" $freq
#fname1="_%04.1fMHz" $freq

fname2=$fname$fname1
options="$freq $run_mode $fname2 $fov $pix_side $telescope $field $telescope_model $max_bl $seed"
tm=".tm"                     
tel_model=$telescope_model$tm
fits=".fits"
fname_name=$fname2$fits
osm=".osm"
fname_osm=$fname2$osm

export OMP_NUM_THREADS=20
cd /gpfs01/home/ppxjf3/OSKAR/
singularity exec --nv /gpfs01/software/augusta/oskar-2.8.3/singularity/OSKAR-2.8.3-Python3.sif python3 OSKAR_example_settings_script.py $options


#image_type="I"
#options="$image_type $freq $fname2 $fov $pix_side $telescope $field $telescope_model $weighting"
#singularity exec --nv /gpfs01/software/augusta/oskar-2.8.3/singularity/OSKAR-2.8.3-Python3.sif python3 image.py $options

#image_type="PSF"
#options="$image_type $freq $fname2 $fov $pix_side $telescope $field $telescope_model $weighting"
#singularity exec --nv /gpfs01/software/augusta/oskar-2.8.3/singularity/OSKAR-2.8.3-Python3.sif python3 image.py $options

ms=".ms"
rm -rf "/gpfs01/home/ppxjf3/peculiar_vel/data/NOISE/vis/${fname2}_${telescope}_${telescope_model}_${field}_${fov}_${pix_side}${ms}"
ms=".vis"
rm -rf "/gpfs01/home/ppxjf3/peculiar_vel/data/NOISE/vis/${fname2}_${telescope}_${telescope_model}_${field}_${fov}_${pix_side}${ms}"
ms=".ini"
rm -rf "/gpfs01/home/ppxjf3/peculiar_vel/data/NOISE/ini/${fname2}_${telescope}_${telescope_model}_${field}_${fov}_${pix_side}${ms}"
echo "/gpfs01/home/ppxjf3/peculiar_vel/data/NOISE/vis/${fname2}_${telescope}_${telescope_model}_${field}_${fov}_${pix_side}${ms}"
