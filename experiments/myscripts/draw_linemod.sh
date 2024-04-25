#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

# python3 ./tools/draw_linemod.py --data_root ./datasets/linemod/Linemod_preprocessed/data/09\
#   --item 0107\
#   --seg_root ./datasets/linemod/Linemod_preprocessed/segnet_results/09_label\
#   --obj 9\
#   --model_root ./datasets/linemod/Linemod_preprocessed/models\
#   --model ./trained_checkpoints/linemod/pose_model_9_0.01310166542980859.pth\
#   --refine_model ./trained_checkpoints/linemod/pose_refine_model_493_0.006761023565178073.pth\
#   --output ./

# LINEMOD 测试
python3 ./mytools/draw_linemod.py --data_root ./datasets/linemod/LINEMOD/data/04\
  --item 0666\
  --seg_root ./datasets/linemod/LINEMOD/segnet_results/04_label\
  --obj 1\
  --model_root ./datasets/linemod/LINEMOD/models\
  --model ./trained_models/linemod/pose_model_current.pth\
  --refine_model ./trained_models/linemod/pose_refine_model_current.pth\
  --output ./

# Temp Pth
# python3 ./tools/draw_linemod.py --data_root ./datasets/linemod/LINEMOD/data/03\
#   --item 0167\
#   --seg_root ./datasets/linemod/LINEMOD/segnet_results/03_label\
#   --obj 3\
#   --model_root ./datasets/linemod/LINEMOD/models\
#   --model ./trained_models/pose_model_current.pth\
#   --refine_model ./trained_models/pose_refine_model_current.pth\
#   --output ./

# --data_root：要选择的图像类别文件夹路径  
# --item：选择的图像编号  
# --seg_root：因为是eval模式，需要用到标准分割的标签，这里输入选择的图像类别的语义分割路径  
# --obj：图像类别编号  
# --model_root：图像类别元数据模型路径  
# --model：训练好的PoseNet路径  
# --refine_model：训练好的PoseRefineNet路径
# --output：可视化保存路径