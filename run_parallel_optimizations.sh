#!/bin/bash
#SBATCH --time=08:00:00            # Time limit (max hours per job)
#SBATCH --array=0-6383%100         # Job array: #configs %simultaneous jobs
#SBATCH --ntasks-per-node=1        # Run one task per job
#SBATCH --cpus-per-task=2          # Cores per optimization process
#SBATCH --mem=4GB                  # Memory per job
#SBATCH --job-name=layouts         # Job name
#SBATCH --output=layouts_%A_%a.out # Output file with job and array IDs
#SBATCH --error=layouts_%A_%a.err  # Error file with job and array IDs
#SBATCH -p RM-shared               # Partition (queue) - use shared to save SUs
#SBATCH -A <YOUR_ALLOCATION_ID>    # Replace with your allocation ID

# Load required modules
module purge
module load python/3.9.0

# Activate virtual environment
source $HOME/keyboard_optimizer/keyboard_env/bin/activate

# Set working directory
cd $HOME/keyboard_optimizer/optimize_layouts

# Create a directory for this specific job's output
mkdir -p output/layouts/config_${SLURM_ARRAY_TASK_ID}

# Echo job information
echo "Starting job array ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "Running on node: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Configuration file: configs/config_${SLURM_ARRAY_TASK_ID}.yaml"

# Run the optimization with the specific configuration
python optimize_layout.py --config configs/config_${SLURM_ARRAY_TASK_ID}.yaml 

# Save completion status
if [ $? -eq 0 ]; then
    echo "Job completed successfully at $(date)" > output/layouts/config_${SLURM_ARRAY_TASK_ID}/job_completed.txt
else
    echo "Job failed with error code $? at $(date)" > output/layouts/config_${SLURM_ARRAY_TASK_ID}/job_failed.txt
fi

echo "Job finished at: $(date)"