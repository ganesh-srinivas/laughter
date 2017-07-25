#!/bin/bash
srun -p gpuk40 --gres=gpu:1 --pty bash
module load singularity/2.3.1
singularity shell --nv /home/mxp523/singularity/rh_laughter_20170710.img
python scripts/convnet_laughter_classify.py > 20july2017.txt
