#!/bin/bash
module load cuda scipy-stack python/3
source /home/$USER/juho-usra/bin/activate
cd /home/$USER/scratch/$USER/juho-usra
srun python3 main.py --function=d --w0=125.0 --learning_rate=0.01 --case_num=7