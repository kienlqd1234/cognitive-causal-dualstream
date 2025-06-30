#!/bin/bash
#SBATCH --job-name=pen_tuning    # Job name
#SBATCH --output=tuning_manual_job_%j.out  # Output file
#SBATCH --error=tuning_manual_job_%j.err   # Error file
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=4        # CPU cores per task
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --time=12:00:00          # Time limit hrs:min:sec
#SBATCH --mem=16G                # Memory per node

# Print some information about the job
echo "Running on host: $(hostname)"
echo "Running on account: $(whoami)"
echo "Job starting time: $(date)"
echo "Current working directory: $(pwd)"

# Set up environment variables for CUDA
export CUDA_HOME=/media02/tvquy/tmp_cuda_install
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Load conda
echo "Activating conda environment"
source $HOME/miniconda/etc/profile.d/conda.sh
conda activate kien_py36

# Print environment info
echo "Python version:"
python --version
echo "Conda environment:"
conda info

# Print CUDA info
echo "CUDA environment:"
echo "CUDA_HOME: $CUDA_HOME"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"


# Install required packages
echo "Installing required packages"
pip install optuna tensorflow-gpu==1.4.0

# Check if TensorFlow can find the GPU
echo "Checking if TensorFlow can detect GPU:"
python -c "
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Enable all log output
print('GPU available:', tf.test.is_gpu_available())
try:
    with tf.Session() as sess:
        devices = sess.list_devices()
        print('Available devices:')
        for d in devices:
            print(f'  {d}')
except Exception as e:
    print(f'Error creating session: {e}')
"

# Run the script
echo "Running hyperparameter tuning script"
cd /media02/tvquy/PEN
python src/tune_hyperparameters.py

echo "Job finished at: $(date)"
