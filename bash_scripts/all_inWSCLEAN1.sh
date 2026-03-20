#!/bin/bash
#SBATCH --partition=shortq,devq,defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80g
#SBATCH --time=05:00:00
#SBATCH --array=1-10%4
#SBATCH --output=log/%x.o%j
#SBATCH --error=log/%x.e%j

module load wsclean-uoneasy/3.5-foss-2023b


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd ${SLURM_SUBMIT_DIR}


size=2048  # dimensions of output image
temp=temp_files/
int_idx=566
idx=$((int_idx + SLURM_ARRAY_TASK_ID - 1))
num=$((SLURM_ARRAY_TASK_ID))
inp=/gpfs01/share/SKADataChallenge/MS/ZW3_IFRQ_0"$idx".ms
echo $inp

#remove bright point sources and then diffuse to stop over counting - try use wsclean predict and taql everytime
wsclean -weight uniform -size 2048 2048 -scale 16asec \
-taper-gaussian 60 -taper-edge 100 -padding 2 -wstack-nwlayers 10000 \
-wstack-oversampling 4095 -wstack-grid-mode kb -wstack-kernel-size  15 \
-mgain 0.8 -niter 1000000 -auto-threshold 1 -auto-mask 4 \
-pol xx -temp-dir $temp -name initial_uniform"$idx" $inp 

#copy measurement set so can overwrite data column 
cp -r $inp temp"$idx".ms
#predict visibilities from model of bright point sources
wsclean -predict -name initial_uniform"$idx" temp"$idx".ms

rm initial_uniform"$idx"*

#remove bright point sources
taql update temp"$idx".ms set DATA=DATA-MODEL_DATA
