#!/bin/sh
## This is the settings for Slurm that we used
## This is only a template that you need to modify.
#SBATCH --chdir ##### <- should be the path to project's root
#SBATCH --job-name="PINN4Rei"
#SBATCH --nodes 1
#SBATCH --time 00:30:00
#SBATCH --account sk02
#SBATCH --output logs/cscs/slurm-%j.out
#SBATCH --error logs/cscs/slurm-%j.err
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --mem 62464
#SBATCH --array=0-1000%100

echo STARTING AT `date`
echo "Setting up enviroment to use MPI..."

# Set how you want to split the cube.
CUBESIZE=30
NDOMAINS=1000

# Here we load the different (you will have to change this, this is CSCS specific)
# module load daint-gpu
# export CRAY_CUDA_MPS=1

# Activate your python environnement here
source /path/to/python/env/activate

# The path to the root the project to load the python modules
export PYTHONPATH="/path/to/project/root"

# Define the task number
TASK_ID=$(( ${SLURM_ARRAY_TASK_ID} - 1 ))
echo "Task ${TASK_ID}/${NDOMAINS}"

# Execute the script for subvolume $TASK_ID
srun python prediction/subvolume_prediction.py $CUBESIZE $TASK_ID &> logs/cscs/slurm-${SLURM_ARRAY_JOB_ID}_${TASK_ID}-stdout.out

echo FINISHED AT `date`