#!/bin/bash

# Set your strict cluster job limit here
MAX_JOBS=2
USER_ID=$(whoami)

# Define the seeds needed to fill out the table
SEEDS=(2 3)

mkdir -p slurm-logs

for seed in "${SEEDS[@]}"; do
    while true; do
        # Count the number of jobs currently queued or running for your user
        CURRENT_JOBS=$(squeue -u $USER_ID -h | wc -l)
        
        if [ "$CURRENT_JOBS" -lt "$MAX_JOBS" ]; then
            break
        fi
        
        echo "Job limit reached ($CURRENT_JOBS/$MAX_JOBS). Waiting 60 seconds..."
        sleep 60
    done

    echo "Submitting job for seed $seed..."
    
    # Submit the worker script, dynamically naming the job and output file
    sbatch -A bera89 dsharsa_p45_100m_gilbreth.sub $seed
           
    # Give SLURM a few seconds to register the submission before looping
    sleep 5
done

echo "All seed runs have been successfully dispatched to the queue!"
