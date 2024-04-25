#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=1

# python3 ./tools/train.py --dataset linemod\
#   --dataset_root ./datasets/linemod/Linemod_preprocessed\
#   --nepoch 10

python3 ./tools/train.py --dataset linemod\
  --dataset_root ./datasets/linemod/LINEMOD
  
# 2022-09-20 19:27:28,799 : Test time 04h 19m 47s Epoch 1 TEST FINISH Avg dis: 0.0310234678651674
