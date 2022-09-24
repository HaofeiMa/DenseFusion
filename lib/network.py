import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pdb
import torch.nn.functional as F
from lib.pspnet import PSPNet

psp_models = {
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

# 提取颜色特征的网络
class ModifiedResnet(nn.Module):

    def __init__(self, usegpu=True):
        super(ModifiedResnet, self).__init__()

        # 上面定义了不同参数的网络组合，可以自行选择
        # 编码器是resnet18，解码器是4个上采样层PSPNet
        self.model = psp_models['resnet18'.lower()]()
        # DataParallel函数用于实现多个GPU加速训练
        self.model = nn.DataParallel(self.model)

    def forward(self, x):
        x = self.model(x)
        return x

class PoseNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points
    # 输入的x为点云数据，大小为[bs, 3, 500]，emb为对应颜色特征[bs, 32, 500]
    def forward(self, x, emb):
        # 首先对x和emb分别使用1*1卷积和relu激活函数，输出64维特征
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        # 然后在深度维度上将x和emb融合形成pointfeat1，[bs, 128. 500]
        pointfeat_1 = torch.cat((x, emb), dim=1)

        # 继续对x和emb使用1*1卷积和relu激活函数，输出128维特征
        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        # 然后在深度维度上将x和emb融合形成pointfeat1，[bs, 256. 500]
        pointfeat_2 = torch.cat((x, emb), dim=1)

        # 对pointfeat2使用1*1卷积 -> relu，得到[bs, 512, 500]
        x = F.relu(self.conv5(pointfeat_2))
        # 继续使用1*1卷积 -> relu，得到[bs, 1024, 500]
        x = F.relu(self.conv6(x))

        # 使用平均池化，对numpoints秋平均，ap_x为[bs, 1024, 1]
        ap_x = self.ap1(x)

        # 将特征复制500份，然后将pointfeat_1，pointfeat_2，ap_x在通道维度上融合，[bs, 128+256+1024, 500]
        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) #128 + 256 + 1024

class PoseNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseNet, self).__init__()
        self.num_points = num_points
        self.cnn = ModifiedResnet()
        self.feat = PoseNetFeat(num_points)
        
        self.conv1_r = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_t = torch.nn.Conv1d(1408, 640, 1)
        self.conv1_c = torch.nn.Conv1d(1408, 640, 1)

        self.conv2_r = torch.nn.Conv1d(640, 256, 1)
        self.conv2_t = torch.nn.Conv1d(640, 256, 1)
        self.conv2_c = torch.nn.Conv1d(640, 256, 1)

        self.conv3_r = torch.nn.Conv1d(256, 128, 1)
        self.conv3_t = torch.nn.Conv1d(256, 128, 1)
        self.conv3_c = torch.nn.Conv1d(256, 128, 1)

        self.conv4_r = torch.nn.Conv1d(128, num_obj*4, 1) #quaternion
        self.conv4_t = torch.nn.Conv1d(128, num_obj*3, 1) #translation
        self.conv4_c = torch.nn.Conv1d(128, num_obj*1, 1) #confidence

        self.num_obj = num_obj
    # 输入为与处理后的随机选择的500个点云x，物体对应的图片img，随机选确定像素序列choose，物体的类别编号obj
    def forward(self, img, x, choose, obj):
        out_img = self.cnn(img) # 输入网络的img，[bs, 3, h, w]，其中self.cnn是ModifiedResnet，用于提取颜色特征
        
        bs, di, _, _ = out_img.size()   # out_img是提取之后的颜色特征，bs是批量大小，di是通道数

        emb = out_img.view(bs, di, -1)  # 将输出的特征转换成[bs, 32, -1]，其中-1用于自动计算剩余维度
        choose = choose.repeat(1, di, 1)    # 将choose复制di遍，每个通道豆腐之一个
        emb = torch.gather(emb, 2, choose).contiguous() # 收集输入的特征维度指定位置的数值，2表示在深度维度上，即选取choose点云对应位置的颜色特征。
        
        x = x.transpose(2, 1).contiguous()  # 交换第1维和第2维，[bs, 500, 3] -> [bs, 3, 500]
        ap_x = self.feat(x, emb)    # 使用PoseNetFeat进行稠密融合

        # 为每个像素回归旋转r、平移t、置信度c
        # 输入的ap_x为[bs, 1408, 500]，连续四次卷积后rx为[bs, num_obj*4, 500]，tx为[bs, num*3, 500]，c为[bs, num_obj*1, 500]
        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))
        cx = F.relu(self.conv1_c(ap_x))      

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))
        cx = F.relu(self.conv2_c(cx))

        rx = F.relu(self.conv3_r(rx))
        tx = F.relu(self.conv3_t(tx))
        cx = F.relu(self.conv3_c(cx))

        # 然后对rx和tx即你想嗯重构得到[bs, num_obj, 4, 500]和[bs, num_obj, 3, 500]，cx重构为[bs, num_obj, 1, 500]，每个像素有每个类别对应的姿态和置信度
        rx = self.conv4_r(rx).view(bs, self.num_obj, 4, self.num_points)
        tx = self.conv4_t(tx).view(bs, self.num_obj, 3, self.num_points)
        cx = torch.sigmoid(self.conv4_c(cx)).view(bs, self.num_obj, 1, self.num_points)
        
        # obj是物体类别[bs, 1]
        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])
        out_cx = torch.index_select(cx[b], 0, obj[b])
        
        out_rx = out_rx.contiguous().transpose(2, 1).contiguous()
        out_cx = out_cx.contiguous().transpose(2, 1).contiguous()
        out_tx = out_tx.contiguous().transpose(2, 1).contiguous()
        
        # 输出预测的每个像素的旋转r、平移t、置信度c、随机选择之后的500像素的RGB图像
        return out_rx, out_tx, out_cx, emb.detach()
 


class PoseRefineNetFeat(nn.Module):
    def __init__(self, num_points):
        super(PoseRefineNetFeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(32, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(384, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points

    # 有上一步预测的RT转换后的点云x和posenet输出的500像素的rgb图像emb
    def forward(self, x, emb):
        # 首先分别用1*1卷积和relu激活函数，输出64维度的特征
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        # 然后在深度维上将x和emb融合，形成pointfeat_1为[bs, 128, 500]
        pointfeat_1 = torch.cat([x, emb], dim=1)

        # 继续对x和emb使用1*1卷积和relu激活函数，输出128维度的特征
        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        # 然后在深度维上进行融合形成pointfeat_2，得到[bs, 256, 500]
        pointfeat_2 = torch.cat([x, emb], dim=1)

        # 下面把pointfeat_1和pointfeat_2进行cat，形成pointfeat_3，[bs,128+256, 500]
        pointfeat_3 = torch.cat([pointfeat_1, pointfeat_2], dim=1)

        # 对pointfeat_3使用1*1卷积—relu—1*1卷积—relu，得到[bs, 1024, 500]
        x = F.relu(self.conv5(pointfeat_3))
        x = F.relu(self.conv6(x))

        # 使用全局平均池化输出ap_x，[bs, 1024, 1]
        ap_x = self.ap1(x)
        
        # 转换成大小[bs, 1024]后直接输出
        ap_x = ap_x.view(-1, 1024)
        return ap_x

class PoseRefineNet(nn.Module):
    def __init__(self, num_points, num_obj):
        super(PoseRefineNet, self).__init__()
        self.num_points = num_points
        self.feat = PoseRefineNetFeat(num_points)
        
        self.conv1_r = torch.nn.Linear(1024, 512)
        self.conv1_t = torch.nn.Linear(1024, 512)

        self.conv2_r = torch.nn.Linear(512, 128)
        self.conv2_t = torch.nn.Linear(512, 128)

        self.conv3_r = torch.nn.Linear(128, num_obj*4) #quaternion
        self.conv3_t = torch.nn.Linear(128, num_obj*3) #translation

        self.num_obj = num_obj

    def forward(self, x, emb, obj):
        bs = x.size()[0]
        
        x = x.transpose(2, 1).contiguous()
        ap_x = self.feat(x, emb)

        rx = F.relu(self.conv1_r(ap_x))
        tx = F.relu(self.conv1_t(ap_x))   

        rx = F.relu(self.conv2_r(rx))
        tx = F.relu(self.conv2_t(tx))

        rx = self.conv3_r(rx).view(bs, self.num_obj, 4)
        tx = self.conv3_t(tx).view(bs, self.num_obj, 3)

        b = 0
        out_rx = torch.index_select(rx[b], 0, obj[b])
        out_tx = torch.index_select(tx[b], 0, obj[b])

        return out_rx, out_tx
