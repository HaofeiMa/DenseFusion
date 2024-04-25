#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0


# LINEMOD 测试
python3 ./tools/visualize.py --dataset_root ./datasets/linemod/LINEMOD\
  --model ./trained_models/linemod/pose_model_current.pth\
  --refine_model ./trained_models/linemod/pose_refine_model_current.pth\

# Linemod_preprocessed 测试
# python3 ./tools/visualize.py --dataset_root ./datasets/linemod/Linemod_preprocessed\
#   --model ./trained_checkpoints/linemod/pose_model_9_0.01310166542980859.pth\
#   --refine_model ./trained_checkpoints/linemod/pose_refine_model_493_0.006761023565178073.pth\

# parser.add_argument('--dataset_root', type=str, default = './datasets/linemod/LINEMOD', help='dataset root dir')
# parser.add_argument('--model', type=str, default = 'trained_models/linemod/pose_model_current.pth',  help='resume PoseNet model')
# parser.add_argument('--refine_model', type=str, default = 'trained_models/linemod/pose_refine_model_current.pth',  help='resume PoseRefineNet model')
