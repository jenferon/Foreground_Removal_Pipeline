#! /usr/bin/bash

# Initialize a variable to hold the job ID of the previous job
PREV_JOB_ID=""

# Loop to submit jobs, alternating between job1.sh and job2.sh
for i in {1..151}; do
    # Alternate between job1.sh and job2.sh based on the iteration number
    if (( $i % 2 == 1 )); then
        JOB_SCRIPT="/gpfs01/home/ppxjf3/OSKAR/run_sim_image_array.sh"
	rm "/gpfs01/home/ppxjf3/peculiar_vel/log/*.o"
	rm "/gpfs01/home/ppxjf3/peculiar_vel/log/*.e"
    else
        JOB_SCRIPT="/gpfs01/home/ppxjf3/peculiar_vel/run_python.sh"
	rm "/gpfs01/home/ppxjf3/OSKAR/*.log"
	rm "/gpfs01/home/ppxjf3/OSKAR/*.out"
    fi

    # Submit the job
    if [ -z "$PREV_JOB_ID" ]; then
        # Submit the first job without any dependency
        PREV_JOB_ID=$(sbatch ${JOB_SCRIPT}| awk '{print $4}')
    else
        # Submit subsequent jobs with dependency on the previous job
        PREV_JOB_ID=$(sbatch --dependency=afterok:${PREV_JOB_ID} ${JOB_SCRIPT}| awk '{print $4}')
    fi
done

