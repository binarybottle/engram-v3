#!/bin/bash
# Script to submit all batches of the keyboard layout optimization

# Total number of configurations
TOTAL_CONFIGS=21840

# SLURM array limit per batch
BATCH_SIZE=1000

# Calculate number of batches needed (ceiling division)
NUM_BATCHES=$(( (TOTAL_CONFIGS + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "Submitting $NUM_BATCHES batches for $TOTAL_CONFIGS configurations"

for ((batch=0; batch<NUM_BATCHES; batch++)); do
    # Calculate start and end indices for this batch
    START_IDX=$((batch * BATCH_SIZE + 1))
    END_IDX=$((START_IDX + BATCH_SIZE - 1))
    
    # Make sure we don't exceed the total
    if [ $END_IDX -gt $TOTAL_CONFIGS ]; then
        END_IDX=$TOTAL_CONFIGS
    fi
    
    # Calculate array size for this batch
    ARRAY_SIZE=$((END_IDX - START_IDX + 1))
    
    echo "Submitting batch $((batch+1))/$NUM_BATCHES: configs $START_IDX-$END_IDX"
    
    # Submit with appropriate array range for the last batch
    if [ $batch -eq $((NUM_BATCHES-1)) ] && [ $ARRAY_SIZE -lt $BATCH_SIZE ]; then
        # For the last batch, we might need fewer array indices
        LAST_ARRAY_IDX=$((ARRAY_SIZE-1))
        JOB_ID=$(sbatch --array=0-$LAST_ARRAY_IDX%1000 --export=BATCH_NUM=$batch slurm_batchmaking.sh | awk '{print $4}')
    else
        # Full batch
        JOB_ID=$(sbatch --export=BATCH_NUM=$batch slurm_batchmaking.sh | awk '{print $4}')
    fi
    
    echo "  Submitted job $JOB_ID"
    
    # Optional: add delay between submissions to avoid overwhelming scheduler
    sleep 2
done

echo "All batches submitted!"
