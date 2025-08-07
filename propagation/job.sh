#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=nc624 # required to send email notifcations - please replace <your_username> with your college login name or email address
# Activate the virtual environment
source /vol/bitbucket/${USER}/sam2/venv/bin/activate
source /vol/cuda/12.0.0/setup.sh

python main.py

#srun --pty --gres=gpu:1 bash