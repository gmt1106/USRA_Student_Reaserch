#!/bin/bash
module load cuda scipy-stack python/3
source /home/$USER/juho-usra/bin/activate
cd /home/$USER/scratch/$USER/juho-usra
srun python3 main.py --function=sr --w0=5.0 --learning_rate=0.001 --case_num=10