#!/bin/bash
module load cuda scipy-stack python/3
cd /home/$USER/scratch/juho-usra/train
python3 main.py --function=d --w0=22.0 --learning_rate=0.002