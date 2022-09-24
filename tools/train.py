# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------

import _init_paths
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger

parser = argparse.ArgumentParser()
# 数据集，选择YCB或者LineMOD，默认为YCB
parser.add_argument('--dataset', type=str, default = 'ycb', help='ycb or linemod')
# 数据集的路径
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
# 批量大小
parser.add_argument('--batch_size', type=int, default = 8, help='batch size')
# 读取数据的进程数量，PyTorch的 DataLoader 允许使用多进程来加速数据读取
parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
# 学习率
parser.add_argument('--lr', default=0.0001, help='learning rate')
# 学习率衰减率
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
# 权重参数
parser.add_argument('--w', default=0.015, help='learning rate')
# 权重衰减率
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
# 衰减阈值
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
# 开始迭代自优化refine的阈值
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
# 添加到训练数据中随机噪声的范围
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
# 迭代自优化的次数
parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
# 最大训练周期
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
# 之前训练已经保存的posenet模型
parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model')
# 之前训练已经保存的refinenet模型
parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
# 开始训练的epoch
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
opt = parser.parse_args()


def main():
    # 设置随机种子，用于参数初始化
    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # 定义数据集，包括物体类别数，随机选取的点云数、保存路径等
    if opt.dataset == 'ycb':
        opt.num_objects = 21 # 物体类别数
        opt.num_points = 1000 # 输入点云的点数量
        opt.outf = 'trained_models/ycb' # 保存训练模型的路径
        opt.log_dir = 'experiments/logs/ycb' # 保存log文件的路径
        opt.repeat_epoch = 1 # 每个epoch重复训练的次数
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        # opt.num_objects = 1
        opt.num_points = 500
        opt.outf = 'trained_models/linemod'
        opt.log_dir = 'experiments/logs/linemod'
        opt.repeat_epoch = 20
    else:
        print('Unknown dataset')
        return

    # estimator为PoseNet网络，即用于预测姿态的主干网络
    estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
    estimator.cuda()
    # refiner为PoseRefineNet网络，用于后续迭代自优化
    refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
    refiner.cuda()

    # 保存的模型路径
    # 由于训练过程可能中断，但会保存训练的模型，因此可以通过--resume_posenet和--resume_refinenet
    # 指定先前训练的模型地址，就会加载先前训练的模型继续训练
    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

    if opt.resume_refinenet != '':
        refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))
        opt.refine_start = True
        opt.decay_start = True
        opt.lr *= opt.lr_rate
        opt.w *= opt.w_rate
        opt.batch_size = int(opt.batch_size / opt.iteration)
        optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
    else:
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

    # 加载训练数据集和测试数据集
    if opt.dataset == 'ycb':
        dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    elif opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    # dataloader使用opt.workers个进程加速读取数据
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    
    # 获取对陈物体的编号列表，以及mesh点
    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    # 对loss进行初始化
    criterion = Loss(opt.num_points_mesh, opt.sym_list)
    criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

    # 将初始的loss值best_test设置成无穷大
    best_test = np.Inf

    # 如果开始训练的epoch为1，则视为重头开始训练，就将之前训练的log文件全都删除
    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    # 开始记录时间
    st_time = time.time()

    for epoch in range(opt.start_epoch, opt.nepoch):
        # 保存每次训练的log文件
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        train_dis_avg = 0.0
        # 是否开始refine过程
        if opt.refine_start:
            # 如果开始refine了，那么posenet开始eval，poserefinenet开始train
            estimator.eval()
            refiner.train()
        else:
            # 否则posenet继续train
            estimator.train()
        # 梯度初始化为0
        optimizer.zero_grad()

        # repeat_epoch每个epoch训练多少次
        for rep in range(opt.repeat_epoch):
            # 将dataloader数据对象组合为一个索引数列
            for i, data in enumerate(dataloader, 0):
                # points：由深度图转换成点云并随机筛选500个点，相机坐标系。
                # choose：所选择500个点云的索引，[bs, 1, 500]
                # img：通过语义分割之后剪切下来的RGB图像
                # target：根据model_points点云信息，以及标准旋转偏移矩阵转换过的目标点云[bs,500,3]
                # model_points：目标初始帧（模型）对应的点云信息[bs,500,3]
                # idx：目标物体类别
                points, choose, img, target, model_points, idx = data
                points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                                 Variable(choose).cuda(), \
                                                                 Variable(img).cuda(), \
                                                                 Variable(target).cuda(), \
                                                                 Variable(model_points).cuda(), \
                                                                 Variable(idx).cuda()
                # 将截取的RGB图像、筛选的点云、索引和物体类别输入到PoseNet姿态估计网络中进行训练
                # pred_r: 预测的旋转参数[bs, 500, 4]，每个像素都有一个预测
                # pred_t: 预测的偏移参数[bs, 500, 3]，每个像素都有一个预测
                # pred_c: 预测的置信度[bs, 500, 1]，置信度，每个像素都有一个预测
                # emb: 经过choose操作之后的img，与点云一一对应
                pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
                # 计算loss，将预测值、目标点云、初始帧点云模型、编号、筛选的500个点云、权重参数等作为输入计算loss
                loss, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)
                
                if opt.refine_start:
                    # 如果开始了refine过程
                    for ite in range(0, opt.iteration):
                        # 将上述loss计算输出的由预测pose和points逆转而来的new_points作为PoseRefineNet网络的输入，与经过choose之后的rbg图像一起进行训练
                        pred_r, pred_t = refiner(new_points, emb, idx)
                        # 计算refine过程的loss
                        dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)
                        # 进行反向传播，refine过程的次数通过opt.iteration设置
                        dis.backward()
                else:
                    # 如果没有开始refine就直接对loss进行反向传播
                    loss.backward()

                train_dis_avg += dis.item()
                train_count += 1

                # 每一个batch输出log信息
                if train_count % opt.batch_size == 0:
                    logger.info('Train time {0} Epoch {1} Batch {2} Frame {3} Avg_dis:{4}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, int(train_count / opt.batch_size), train_count, train_dis_avg / opt.batch_size))
                    optimizer.step()
                    optimizer.zero_grad()
                    train_dis_avg = 0

                # 每1000次训练保存一次模型
                if train_count != 0 and train_count % 1000 == 0:
                    if opt.refine_start:
                        # 如果已有refine过程则保存refine模型
                        torch.save(refiner.state_dict(), '{0}/pose_refine_model_current.pth'.format(opt.outf))
                    else:
                        # 如果没有则保存estimator模型
                        torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))


        # 保存每次的log文件
        logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
        logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
        test_dis = 0.0
        test_count = 0
        # 将模型设置为eval模式，否则再有输入数据时，权值也会改变
        estimator.eval()
        refiner.eval()

        # 对测试数据进行集进行测试
        for j, data in enumerate(testdataloader, 0):
            # 获取测试数据的各个值
            points, choose, img, target, model_points, idx = data
            points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                             Variable(choose).cuda(), \
                                                             Variable(img).cuda(), \
                                                             Variable(target).cuda(), \
                                                             Variable(model_points).cuda(), \
                                                             Variable(idx).cuda()
            # 使用PoseNet计算姿态
            pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
            # 然后计算loss
            _, dis, new_points, new_target = criterion(pred_r, pred_t, pred_c, target, model_points, idx, points, opt.w, opt.refine_start)

            # 如果有refine
            if opt.refine_start:
                for ite in range(0, opt.iteration):
                    # 则将上一次预测姿态逆转的点云作为输入，用PoseRefineNet计算新的pose
                    pred_r, pred_t = refiner(new_points, emb, idx)
                    # 然后计算refine过程的loss
                    dis, new_points, new_target = criterion_refine(pred_r, pred_t, new_target, model_points, idx, new_points)

            # 输出log
            test_dis += dis.item()
            logger.info('Test time {0} Test Frame No.{1} dis:{2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count, dis))

            test_count += 1

        # 计算测试过程的平均distance
        test_dis = test_dis / test_count
        # 输出log
        logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_dis))
        # 测试的distance小于最好的distance
        if test_dis <= best_test:
            # 将test_dis作为best_dis
            best_test = test_dis
            # 保存本次epoch模型
            if opt.refine_start:
                torch.save(refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            else:
                torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_dis))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

        # 是否进行学习率和权重衰减
        if best_test < opt.decay_margin and not opt.decay_start:
            opt.decay_start = True
            opt.lr *= opt.lr_rate
            opt.w *= opt.w_rate
            optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)

        # 是否开始refine过程
        if best_test < opt.refine_margin and not opt.refine_start:
            opt.refine_start = True
            opt.batch_size = int(opt.batch_size / opt.iteration)
            optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)

            if opt.dataset == 'ycb':
                dataset = PoseDataset_ycb('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            elif opt.dataset == 'linemod':
                dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
            if opt.dataset == 'ycb':
                test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
            elif opt.dataset == 'linemod':
                test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
            testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
            
            opt.sym_list = dataset.get_sym_list()
            opt.num_points_mesh = dataset.get_num_points_mesh()

            print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

            criterion = Loss(opt.num_points_mesh, opt.sym_list)
            criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

if __name__ == '__main__':
    main()
