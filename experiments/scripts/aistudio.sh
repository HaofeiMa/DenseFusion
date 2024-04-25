touch env.sh
vim env.sh

#!/bin/sh
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --set show_channel_urls yes

sh env.sh
conda init
conda env list
conda create --name densefusion python=3.6
# 重新打开终端
conda activate densefusion
# conda install pytorch==1.0.0 torchvision==0.2.1 cuda100 -c pytorch
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/pytorch-1.0.0-py3.6_cuda10.0.130_cudnn7.4.1_1.tar.bz2
conda install --use-local ./pytorch/linux-64/pytorch-1.0.0-py3.6_cuda10.0.130_cudnn7.4.1_1.tar.bz2


##########################################3
wget -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-4.7.12.1-Linux-x86_64.sh --no-check-certificate
bash Miniconda3-4.7.12.1-Linux-x86_64.sh
source ~/miniconda3/bin/activate
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
conda install pytorch==1.0.1 cudatoolkit=10.0
conda install torchvision
pip install opencv-python
conda install scipy pyyaml
cd data/data169516/
unzip Linemod_preprocessed.zip
unzip LINEMOD.zip
unzip trained_checkpoints.zip
cd ~
git clone https://github.com/HuffieMa/DenseFusion.git
cd DenseFusion/
mv ~/data/data169516/Linemod_preprocessed ./datasets/linemod/
mv ~/data/data169516/trained_checkpoints ./

mkdir trained_models
mkdir ./datasets/linemod/backup
mv Linemod_preprocessed/data/01 backup/
mv Linemod_preprocessed/data/04 backup/
mv Linemod_preprocessed/data/05 backup/
mv Linemod_preprocessed/data/06 backup/
mv Linemod_preprocessed/models/obj_01.ply backup/
mv Linemod_preprocessed/models/obj_04.ply backup/
mv Linemod_preprocessed/models/obj_05.ply backup/
mv Linemod_preprocessed/models/obj_06.ply backup/
mv Linemod_preprocessed/segnet_results/01_label backup/
mv Linemod_preprocessed/segnet_results/04_label backup/
mv Linemod_preprocessed/segnet_results/05_label backup/
mv Linemod_preprocessed/segnet_results/06_label backup/
mv ~/data/data169516/data/* ./Linemod_preprocessed/data/
mv ~/data/data169516/models/* ./Linemod_preprocessed/models/
mv ~/data/data169516/segnet_results/* ./Linemod_preprocessed/segnet_results/

