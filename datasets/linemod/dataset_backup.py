import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import yaml
import cv2


class PoseDataset(data.Dataset):
    def __init__(self, mode, num, add_noise, root, noise_trans, refine):
        # mode：模式，共三种，train、test、eval
        # num：输入的点云个数（这里linemod默认500个）
        # add_noise：是否加入噪声（实验中将train加入噪声，test和eval不加入）
        # root：数据集的根目录
        # noise_trans：噪声超参数（默认0.03，这个参数可以在后续eval的时候自己设置）
        # refine：refine过程是否开始
        # self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]    # 物体类别编号
        self.objlist = [1, 4, 5, 6, 7]    # 物体类别编号
        self.mode = mode

        self.list_rgb = []      #存放rgb图路径
        self.list_depth = []    #存放深度图路径
        self.list_label = []    #存放语义分割mask图路径
        self.list_obj = []      #存放读取的类别编号
        self.list_rank = []     #存放读取的图片编号
        self.meta = {}          #存放每个类别的元数据信息
        self.pt = {}            #存放每个类别的点云信息
        self.root = root        #数据集根目录
        self.noise_trans = noise_trans  #是否加入噪声
        self.refine = refine            #是否开始了refine过程

        item_count = 0
        for item in self.objlist:
            # 找到对应的文件夹，开始读取数据集
            if self.mode == 'train':
                input_file = open('{0}/data/{1}/train.txt'.format(self.root, '%02d' % item))
            else:
                input_file = open('{0}/data/{1}/test.txt'.format(self.root, '%02d' % item))
            while 1:
                item_count += 1
                input_line = input_file.readline()
                if self.mode == 'test' and item_count % 10 != 0:
                    continue
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                # 添加RGB图路径
                self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(self.root, '%02d' % item, input_line))
                # 添加深度图路径
                self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(self.root, '%02d' % item, input_line))
                # 如果模式为eval
                if self.mode == 'eval':
                    # 添加语义分割之后的标签
                    self.list_label.append('{0}/segnet_results/{1}_label/{2}_label.png'.format(self.root, '%02d' % item, input_line))
                else:
                    # 否则添加标准的mask
                    self.list_label.append('{0}/data/{1}/mask/{2}.png'.format(self.root, '%02d' % item, input_line))
                # 添加类别号
                self.list_obj.append(item)
                # 添加图片编号
                self.list_rank.append(int(input_line))

            # 元数据信息
            meta_file = open('{0}/data/{1}/gt.yml'.format(self.root, '%02d' % item), 'r')
            # 加载元数据信息
            self.meta[item] = yaml.load(meta_file)
            # 加载三维点云数据模型
            self.pt[item] = ply_vtx('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % item))
            # 打印该类别加载完毕
            print("Object {0} buffer loaded".format(item))

        # 所有物体的数量
        self.length = len(self.list_rgb)

        # 相机中心坐标
        self.cam_cx = 321.6173095703125
        self.cam_cy = 237.4153594970703
        # 相机的焦距
        self.cam_fx = 605.1395263671875
        self.cam_fy = 604.8554077148438

        # 相机分辨率
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        
        self.num = num      # 输入点云的个数，500
        self.add_noise = add_noise  # 添加噪声
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)    # 改变图像的亮度，
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 归一化
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]  # 边界列表，将图像分割成了多个坐标
        self.num_pt_mesh_large = 500    # 点云最大数量
        self.num_pt_mesh_small = 500    # 点云最小数量
        self.symmetry_obj_idx = [7, 8]  # 对陈物体的序号

    def __getitem__(self, index):       # 初始化部分
        # 读取图片信息
        img = Image.open(self.list_rgb[index])  # 读取图片
        ori_img = np.array(img)     # 转换成矩阵
        depth = np.array(Image.open(self.list_depth[index]))    # 读取深度
        label = np.array(Image.open(self.list_label[index]))    # 读取标签
        obj = self.list_obj[index]      # 读取类别号
        # if (obj == 1) or (obj == 4) or (obj == 5) or (obj == 6):
            # label = cv2.cvtColor(label ,cv2.COLOR_GRAY2BGR)
        # label = cv2.cvtColor(label ,cv2.COLOR_GRAY2BGR)
        rank = self.list_rank[index]    # 读取图片序号

        # 读取元数据信息
        if obj == 2:
            # 因为2物体文件夹内除了该物体的mask，还有其他物体的mask，所以需要单独处理找到2物体的mask
            for i in range(0, len(self.meta[obj][rank])):
                if self.meta[obj][rank][i]['obj_id'] == 2:
                    meta = self.meta[obj][rank][i]
                    break
        else:
            meta = self.meta[obj][rank][0]
        '''
        self.meta
        { 物体类别 (比如1):
            { 图像编号 (比如0004,取int为4)
                [
                    { 'cam_R_m2c': [ 旋转矩阵,9个数的列表 ] ,
                        'cam_t_m2c': [ 平移矩阵,3个数的列表 ] ,
                        'obj_bb': [ 标准的bounding box,4个数的列表 ] ,
                        'obj_id': 物体类别号
                    }
                ]
            }
        }
        '''
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0)) # 获取掩码
        if self.mode == 'eval':
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
        else:
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]
        
        # 取label和depth都为true的像素作为mask
        mask = mask_label * mask_depth

        if self.add_noise:
            img = self.trancolor(img)   # 加入噪声

        img = np.array(img)[:, :, :3]   # 提取前三个通道rgb
        img = np.transpose(img, (2, 0, 1))  # 转换通道
        img_masked = img

        # 计算物体所在区域的角点坐标
        if self.mode == 'eval':
            # mask_to_bbox,根据mask计算bounding box
            rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))
        else:
            rmin, rmax, cmin, cmax = get_bbox(meta['obj_bb'])

        img_masked = img_masked[:, rmin:rmax, cmin:cmax]
        #p_img = np.transpose(img_masked, (1, 2, 0))
        #scipy.misc.imsave('evaluation_result/{0}_input.png'.format(index), p_img)

        # 获取元数据中真实旋转R，转换成数组类，然后resize为3*3矩阵
        target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
        # 获取真实平移，转换成数组类
        target_t = np.array(meta['cam_t_m2c'])
        # 定义噪声，并分别给xyz坐标添加噪声
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        # 随机选择500个点云，将物体所在区域mask展开，选取为true的部分
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        # 如果索引个数为0，则返回0，表示没有物体
        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return(cc, cc, cc, cc, cc, cc)
        # 如果索引个数大于500，则随机选取500个作为choose
        if len(choose) > self.num:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        # 如果索引个数大于0小于500，则用warp模式填充，[1,2,3] -> [1,2,3,1,2,3,...]
        else:
            choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')
        
        # 根据choose，获取对应的深度值，x坐标，y坐标
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        # 将depth转换成点云数据
        ############################################################################################
        cam_scale = 1.0
        ############################################################################################
        pt2 = depth_masked / cam_scale  # z轴
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx   # x轴
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy   # y轴
        cloud = np.concatenate((pt0, pt1, pt2), axis=1) #拼接
        cloud = cloud / 1000.0

        # 如果添加噪声，则每个点的三个坐标都添加add_t
        if self.add_noise:
            cloud = np.add(cloud, add_t)

        #fw = open('evaluation_result/{0}_cld.xyz'.format(index), 'w')
        #for it in cloud:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        #fw.close()

        # 获取物体的真实点云
        model_points = self.pt[obj] / 1000.0
        # 随机删除点云
        dellist = [j for j in range(0, len(model_points))]
        dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_small)
        model_points = np.delete(model_points, dellist, axis=0)

        #fw = open('evaluation_result/{0}_model_points.xyz'.format(index), 'w')
        #for it in model_points:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        #fw.close()

        # target是由model_points经过标准姿态转换后相机坐标系下的点
        target = np.dot(model_points, target_r.T)
        if self.add_noise:
            target = np.add(target, target_t / 1000.0 + add_t)
            out_t = target_t / 1000.0 + add_t
        else:
            target = np.add(target, target_t / 1000.0)
            out_t = target_t / 1000.0

        #fw = open('evaluation_result/{0}_tar.xyz'.format(index), 'w')
        #for it in target:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        #fw.close()

        # 返回点云，点选区的点云索引，物体所在图像区域、目标点云、物体第一帧点云、物体类别，并转换成tensor形式
        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([self.objlist.index(obj)])

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

    def get_img(self):
        return self.list_rgb
    
    def get_mask(self):
        return self.list_label

    def get_ssp_label(self):
        label_file = []
        for item in self.objlist:
            label_path = '{0}/data/{1}/labels'.format(self.root, '%02d' % item)
            for file in os.listdir(label_path):
                # f = open(label_path + "/" + file)
                label_file.append(label_path + "/" + file)
        return label_file
        
    def get_diameter(self):
        diameter = []
        dataset_config_dir = 'datasets/linemod/dataset_config'
        objlist = [1, 4, 5, 6, 7]
        meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
        # meta = yaml.load(meta_file)
        meta = yaml.load(meta_file,Loader=yaml.FullLoader)
        for obj in objlist:
            diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
        return diameter
        # 相机中心坐标

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640


def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    # _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.findContours找到mask的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        # 将找到的轮廓用最小矩形包起来，并返回左上角坐标，宽高
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]


def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639                
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
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
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
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
