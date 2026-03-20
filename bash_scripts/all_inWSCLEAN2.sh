#!/bin/bash
#SBATCH --partition=shortq,devq,defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80g
#SBATCH --time=05:00:00
#SBATCH --array=1-1%4
#SBATCH --output=log/%x.o%j
#SBATCH --error=log/%x.e%j

module load wsclean-uoneasy/3.5-foss-2023b


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd ${SLURM_SUBMIT_DIR}


size=2048  # dimensions of output image
temp=temp_files/
int_idx=556
idx=$((int_idx + SLURM_ARRAY_TASK_ID - 1))
num=$((SLURM_ARRAY_TASK_ID))
inp=/gpfs01/share/SKADataChallenge/MS/ZW3_IFRQ_0"$idx".ms
echo $inp

#copy measurement set so can overwrite data column 
cp -r temp"$idx".ms ../BSS/ZW3_BSS_0"$idx".ms

#predict visibilities from model of bright point sources
wsclean -predict -name initial_natural-000"$num" ../BSS/ZW3_BSS_0"$idx".ms

#remove bright point sources
taql update BSS/ZW3_BSS_0"$idx".ms set DATA=DATA-MODEL_DATA

#image result - naturally for blind source removal
wsclean -weight natural -size 1024 1024 -scale 32asec \
-taper-gaussian 60 -taper-edge 100 -padding 2 -wstack-nwlayers 10000 \
-wstack-oversampling 4095 -wstack-grid-mode kb -wstack-kernel-size  15 \
-mgain 0.8 -auto-threshold 1 -auto-mask 4 \
-pol xx -temp-dir $temp -name ../BSS/ZW3_BSS_0"$idx" ../BSS/ZW3_BSS_0"$idx".ms

#remove files to save space
#rm -r temp"$idx".ms
rm -r ../BSS/ZW3_BSS_0"$idx".ms
rm initial_natural"$idx"*
rm ../BSS/ZW3_BSS_0"$idx"-dirty.fits
