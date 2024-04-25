#!/usr/bin/env python3
import sys
import os

sys.path.append('./')
# import _init_paths
import numpy as np
import yaml
import copy
import random
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
from PIL import Image
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix

# from lib.knn.__init__ import KNearestNeighbor
'''
—————————————————————————————————— 参数定义 ——————————————————————————————————————————————
'''
################### 固定参数 #######################
# 相机内参
cam_cx = 321.6173095703125
cam_cy = 237.4153594970703
cam_fx = 605.1395263671875
cam_fy = 604.8554077148438
cam_scale = 1.0

# 边界列表，固定了一系列包围框的尺寸，目的是使用标准的包围框大小
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

##################### 模型位置 ######################
model = "trained_models/linemod/pose_model_current.pth"
refine_model = "trained_models/linemod/pose_refine_model_current.pth"
root = "datasets/linemod/LINEMOD"

##################### 默认变量 ######################
# num_objects = 13
# objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
num_objects = 4
objlist = [1, 2, 3, 4]
sym_list = []
num_points = 500
num_points_mesh = 500
iteration = 4
bs = 1
# dataset_config_dir = 'datasets/linemod/dataset_config'
# output_result_dir = 'experiments/eval_result/linemod'
# knn = KNearestNeighbor(1)
'''
—————————————————————————————————— 常用函数 ——————————————————————————————————————————————
'''


# 输入：RGB图像、深度图、掩码图
# 返回：点云
def rgbd2cloud(img, depth, label, obj):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])  # 归一化。通过归一化可以是模型在训练的时候更快的收敛，并降低对输入数据的敏感度

    ################## 获取掩码 ##################
    # label = cv2.cvtColor(label ,cv2.COLOR_GRAY2BGR)
    mask_depth = np.ma.getmaskarray(np.ma.masked_not_equal(depth, 0))  # 获取深度图的掩码，确定深度图上的哪些像素值有效True，哪些像素值无效False
    mask_label = np.ma.getmaskarray(np.ma.masked_equal(label, np.array(255)))
    # mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]
    mask = mask_label * mask_depth  # 取label和depth都为true的像素作为mask,depth为true代表深度值有效，label为true代表此处有物体

    img = np.array(img)[:, :, :3]  # 提取前三个通道rgb
    img = np.transpose(img, (2, 0, 1))  # 转换通道，将图像的维度从(height, width, channel)转换为(channel, height, width)
    img_masked = img

    rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))

    img_masked = img_masked[:, rmin:rmax, cmin:cmax]

    ############### 获得choose列表 ##################
    # 随机选择500个点云，将物体所在标准包围框区域的mask展开，选取为true的部分的下标
    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    # 如果索引个数为0，则返回0张量，表示没有物体
    if len(choose) == 0:
        cc = torch.LongTensor([0])
        return (cc, cc, cc, cc, cc, cc)
    # 如果索引个数大于500，则随机选取500个作为choose
    if len(choose) > num_points:
        c_mask = np.zeros(len(choose), dtype=int)
        c_mask[:num_points] = 1
        np.random.shuffle(c_mask)
        choose = choose[c_mask.nonzero()]
    # 如果索引个数大于0小于500，则用warp模式填充，[1,2,3] -> [1,2,3,1,2,3,...]
    else:
        choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

    ################### 获取点云坐标 ################
    # xmap 数组的第 i 行和第 j 列的值为j，ymap 数组的第 i 行和第 j 列的值为i
    xmap = np.array([[j for i in range(640)] for j in range(480)])
    ymap = np.array([[i for i in range(640)] for j in range(480)])
    # 根据choose，获取对应的深度值，x坐标，y坐标
    depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    choose = np.array([choose])

    pt2 = depth_masked * cam_scale  # z轴（深度值单位默认为mm）
    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx  # x轴，直接计算得到的xy单位也是mm
    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy  # y轴
    cloud = np.concatenate((pt0, pt1, pt2), axis=1)  # 将三个列向量(500,1)拼接为(500,3)的点云
    cloud = cloud / 1000.0

    return torch.from_numpy(cloud.astype(np.float32)), \
        torch.LongTensor(choose.astype(np.int32)), \
        norm(torch.from_numpy(img_masked.astype(np.float32))), \
        torch.LongTensor([objlist.index(obj)])


# 使用findContours找到mask物体的最小包围框，返回包围框左上角的[x, y, w, h]
def mask_to_bbox(mask):
    # 将mask数组True/False，转换为uint8类型(255/0)
    mask = mask.astype(np.uint8)
    # _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.findContours找到mask的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        # 找到mask中面积最大的轮廓
        # 将找到的轮廓用最小矩形包起来，并返回左上角坐标，宽高
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]


# 输入一个包围框bbox = [x, y, w, h]
# 返回一个标准包围框 rmin, rmax, cmin, cmax
def get_bbox(bbox):
    # bbox = [x, y, w, h]
    # bbx = [左上y, 左下y, 左上x, 右上x]
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639
        # row_min, row_max, col_min, col_max
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
    # 找到稍大于bbox的包围框尺寸c_b, r_b
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    # 确定bbox的中心
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    # 确定标准包围框的位置
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    # 如果包围框的某一个边超出边界，就将对面的边往反方向平移
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax


def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)


if __name__ == '__main__':
    '''
    —————————————————————————————————— 网络定义 ——————————————————————————————————————————————
    '''

    # 网络定义
    estimator = PoseNet(num_points=num_points, num_obj=num_objects)
    estimator.cuda()
    refiner = PoseRefineNet(num_points=num_points, num_obj=num_objects)
    refiner.cuda()
    estimator.load_state_dict(torch.load(model))
    refiner.load_state_dict(torch.load(refine_model))
    estimator.eval()
    refiner.eval()

    # 损失函数定义
    criterion = Loss(num_points_mesh, sym_list)
    criterion_refine = Loss_refine(num_points_mesh, sym_list)

    '''
    —————————————————————————————————— 预测位姿 ——————————————————————————————————————————————
    '''
    # 要识别的物体是哪个
    obj = 1
    item = 66

    pt = ply_vtx('{0}/models/obj_{1}.ply'.format(root, '%02d' % obj))

    cv2.namedWindow("image")

    img_path = os.path.join(root, "test", "%02d" % obj, "rgb")
    img_list = os.listdir(img_path)

for item in img_list:

    # 点云文件读取
    model_points = pt / 1000.0
    dellist = [j for j in range(0, len(model_points))]
    dellist = random.sample(dellist, len(model_points) - num_points_mesh)
    model_points = np.delete(model_points, dellist, axis=0)
    model_points = torch.from_numpy(model_points.astype(np.float32))

    # 图像读取
    image = Image.open('{0}/test/{1}/rgb/{2}'.format(root, '%02d' % obj, item))  # 读取图片
    depth = np.array(Image.open('{0}/test/{1}/depth/{2}'.format(root, '%02d' % obj, item)))  # 读取深度
    label = np.array(Image.open('{0}/test/{1}/mask/{2}'.format(root, '%02d' % obj, item)))  # 读取标签
    # if label.shape[-1] == 1:
    # label = cv2.cvtColor(label, cv2.COLOR_BGRA2GRAY)

    # 获取深度点云数据
    points, choose, img, idx = rgbd2cloud(image, depth, label, obj)
    if len(points.size()) == 2:
        print('NOT Pass! Lost detection!')

    points = points.unsqueeze(0)
    choose = choose.unsqueeze(0)
    img = img.unsqueeze(0)
    idx = idx.unsqueeze(0)

    points, choose, img, model_points, idx = Variable(points).cuda(), \
        Variable(choose).cuda(), \
        Variable(img).cuda(), \
        Variable(model_points).cuda(), \
        Variable(idx).cuda()

    model_points_ori = model_points

    # 利用网络获得预测的参数
    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)  # 四元数归一化
    pred_c = pred_c.view(bs, num_points)
    how_max, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(bs * num_points, 1, 3)

    # 获得置信度最高的预测参数
    my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
    my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
    my_pred = np.append(my_r, my_t)

    for ite in range(0, iteration):
        # 将四元数转换为旋转矩阵
        T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points,
                                                                                         1).contiguous().view(1,
                                                                                                              num_points,
                                                                                                              3)
        my_mat = quaternion_matrix(my_r)
        R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
        my_mat[0:3, 3] = my_t

        # 进行refine过程
        new_points = torch.bmm((points - T), R).contiguous()
        # 得到refine后的参数
        pred_r, pred_t = refiner(new_points, emb, idx)
        pred_r = pred_r.view(1, 1, -1)
        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
        # 得到refine后的真正参数
        my_r_2 = pred_r.view(-1).cpu().data.numpy()
        my_t_2 = pred_t.view(-1).cpu().data.numpy()
        my_mat_2 = quaternion_matrix(my_r_2)
        my_mat_2[0:3, 3] = my_t_2

        # 将refine结果添加到之前的预测结果上，进行修正
        my_mat_final = np.dot(my_mat, my_mat_2)
        my_r_final = copy.deepcopy(my_mat_final)
        my_r_final[0:3, 3] = 0
        my_r_final = quaternion_from_matrix(my_r_final, True)
        my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

        # f赋值最终结果
        my_pred = np.append(my_r_final, my_t_final)
        my_r = my_r_final
        my_t = my_t_final

    # model_points = model_points[0].cpu().detach().numpy()
    model_points = model_points.cpu().detach().numpy()
    my_r = quaternion_matrix(my_r)[:3, :3]  # 将四元数转换为旋转矩阵
    pred = np.dot(model_points, my_r.T) + my_t  # 得到预测的位姿

    '''
    —————————————————————————————————————————— 可视化 ————————————————————————————————————————————
    '''
    # 预测点云的深度、x坐标、y坐标
    depth_masked = pred[:, 2] * cam_scale
    ymap_masked_pred = pred[:, 0] * cam_fx / pred[:, 2] + cam_cx
    xmap_masked_pred = pred[:, 1] * cam_fy / pred[:, 2] + cam_cy

    image = cv2.imread('{0}/test/{1}/rgb/{2}'.format(root, '%02d' % obj, item))  # 读取图片
    point_size = 1
    point_color = (0, 0, 255)  # BGR
    point_thickness = 4  # 可以为 0 、4、8
    points_list = [(int(ymap_masked_pred[i]), int(xmap_masked_pred[i])) for i in range(len(xmap_masked_pred))]
    for point in points_list:
        cv2.circle(image, point, point_size, point_color, point_thickness)
    cv2.imshow("image", image)
    # cv2.waitKey(-1)
    cv2.waitKey(1)
cv2.destroyAllWindows()