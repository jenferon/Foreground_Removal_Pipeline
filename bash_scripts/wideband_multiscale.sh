#!/bin/bash
#SBATCH --partition=shortq,devq,defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=log/%x.o%j
#SBATCH --error=log/%x.e%j

module load wsclean-uoneasy/3.5-foss-2023b

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd ${SLURM_SUBMIT_DIR}

temp=${SLURM_SUBMIT_DIR}/temp_files/

# Measurement sets
ms_list=()
for i in {525..565}; do
    ms_list+=("temp${i}.ms")
done

wsclean \
-weight natural \
-size 1024 1024 \
-scale 32asec \
-multiscale \
-taper-gaussian 60 \
-taper-edge 100 \
-padding 2 \
-wstack-nwlayers 1000 \
-wstack-oversampling 4095 \
-wstack-grid-mode kb \
-wstack-kernel-size 15 \
-mgain 0.8 \
-niter 1000000 \
-auto-threshold 1 \
-auto-mask 4 \
-pol xx \
-temp-dir $temp \
-join-channels \
-channels-out 40 \
-name initial_natural \
-save-source-list \
-deconvolution-channels 8 \
-fit-spectral-pol 2 \
"${ms_list[@]}"


#rm -r temp*.ms
rm initial_natural*image.fits
rm initial_natural*dirty.fits
rm initial_natural*residual.fits
rm initial_natural*psf.fits
rm temp_files/*

