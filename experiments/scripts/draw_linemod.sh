#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

# python3 ./tools/draw_linemod.py --data_root ./datasets/linemod/Linemod_preprocessed/data/01\
#   --item 0107\
#   --seg_root ./datasets/linemod/Linemod_preprocessed/segnet_results/01_label\
#   --obj 1\
#   --model_root ./datasets/linemod/Linemod_preprocessed/models\
#   --model ./trained_models/linemod/pose_model_9_0.013053627398030966.pth\
#   --refine_model ./trained_models/linemod/pose_refine_model_current.pth\
#   --output ./

# Linemod_processed 测试
python3 ./tools/draw_linemod.py --data_root ./datasets/linemod/Linemod_preprocessed/data/08\
  --item 0707\
  --seg_root ./datasets/linemod/Linemod_preprocessed/segnet_results/08_label\
  --obj 8\
  --model_root ./datasets/linemod/Linemod_preprocessed/models\
  --model ./trained_models/linemod/pose_model_current.pth\
  --refine_model ./trained_models/linemod/pose_refine_model_current.pth\
  --output ./

# --data_root：要选择的图像类别文件夹路径  
# --item：选择的图像编号  
# --seg_root：因为是eval模式，需要用到标准分割的标签，这里输入选择的图像类别的语义分割路径  
# --obj：图像类别编号  
# --model_root：图像类别元数据模型路径  
# --model：训练好的PoseNet路径  
# --refine_model：训练好的PoseRefineNet路径
# --output：可视化保存路径