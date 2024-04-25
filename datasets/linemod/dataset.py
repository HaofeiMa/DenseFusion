import torch.utils.data as data
from PIL import Image   # Python Imaging Library 一个用于处理图像的 Python 库，Image 模块提供了读取、保存图像、对图像进行基本操作、调整图像大小等功能。
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
        '''
        —————————————————————————————————————————— 参数输入 ————————————————————————————————————
        - mode：模式，共有三种，train、test、eval
        - num：输入的点云个数（这里linemod默认500个）
        - add_noise：是否加入噪声（实验中将train加入噪声，test和eval不加入）
        - root：数据集的根目录
        - noise_trans：噪声超参数
        - refine：是否启用refine过程
        '''
        #####################################################################################
        # self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]    # 物体类别编号
        self.objlist = [1, 2, 3, 4]    # 物体类别编号
        #####################################################################################
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

        '''
        —————————————————————————————————————— 加载数据集文件 ————————————————————————————————————
        针对数据集中每一个文件夹（物体类别序号），开始读取。
        首先根据mode模式参数不同，获取train.txt或者test.txt，即得到哪些图片模型文件是要被读取来进行训练或测试的。
        这里train读取train.txt，test和eval模式都读取test.txt。每读取一行train.txt或test.txt：
        - 添加RGB图、深度图、mask标签图的路径到三个列表（list_rgb、list_depth、list_label）中，
            其中mask标签图如果是eval模式，则添加语义分割后的mask标签，如果是train或test模式，就添加标准的mask。
        - 添加类别号list_obj
        - 添加图片编号list_rank
        （这里的几个列表按顺序保存了所有物体的txt对应的图片）
        然后加载该物体类别的元数据信息meta，包括每一个图像的旋转矩阵、平移矩阵、包围矩形框的坐标、物体类别号。（每个meta\[item]保存了一个物体item的元数据）
        最后加载该物体的三维点云文件。（每个pt\[item]保存了一个物体item的点云）
        '''
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
                # test模式读test.txt时，每隔10行读取一次
                if self.mode == 'test' and item_count % 10 != 0:
                    continue
                if not input_line:
                    break
                # 去除末尾的换行符
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

        '''
        —————————————————————————————————————— 初始化参数 ————————————————————————————————————
        初始化设置一些参数：
        - 相机内参、分辨率
        - 图像调整参数trancolor，通过颜色变换可以使图像变得更随机，从而防止模型对数据集过度拟合，作为噪声项根据需要添加。
        - 图像归一化参数，通过归一化可以是模型在训练的时候更快的收敛，并降低对输入数据的敏感度
        - 图像分割列表，将图像分成多块
        - 点云最大数量、最小数量
        - 对陈物体的序号
        '''
        # 所有物体的数量
        self.length = len(self.list_rgb)

        # 相机中心坐标
        self.cam_cx = 321.6173095703125
        self.cam_cy = 237.4153594970703
        # 相机的焦距
        self.cam_fx = 605.1395263671875
        self.cam_fy = 604.8554077148438

        # 相机分辨率
        # xmap 数组的第 i 行和第 j 列的值为j，ymap 数组的第 i 行和第 j 列的值为i
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        
        self.num = num      # 输入点云的个数，500
        self.add_noise = add_noise  # 添加噪声
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)    # 改变图像参数：亮度、对比度、饱和度、色调。通过颜色变换可以使图像变得更随机，从而防止模型对数据集过度拟合
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 归一化。通过归一化可以是模型在训练的时候更快的收敛，并降低对输入数据的敏感度
        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]  # 边界列表，将图像分割成了多个坐标
        self.num_pt_mesh_large = 500    # 点云最大数量
        self.num_pt_mesh_small = 500    # 点云最小数量
        self.symmetry_obj_idx = []  # 对陈物体的序号

    def __getitem__(self, index):       # 初始化部分
        '''
        —————————————————————————————————— （1）加载图片 ——————————————————————————————————————
        使用PIL的Image模块加载当前索引批次的RGB图片、深度图、mask标签图、类别号、图片序号信息。并且要保证mask标签图也是BRG三通道图。
        根据物体类别号和图片序号，在meta中找到当前数据批次对应的元数据
        '''
        # 读取图片信息
        img = Image.open(self.list_rgb[index])  # 读取图片
        ori_img = np.array(img)     # 转换成矩阵
        depth = np.array(Image.open(self.list_depth[index]))    # 读取深度
        label = np.array(Image.open(self.list_label[index]))    # 读取标签
        obj = self.list_obj[index]      # 读取类别号
        ##########################################################################
        # if (obj == 1) or (obj == 2) or (obj == 3):
        if label.shape[-1] != 3:
            label = cv2.cvtColor(label ,cv2.COLOR_GRAY2BGR)
        ##########################################################################
        rank = self.list_rank[index]    # 读取图片序号

        # 读取元数据信息
        # if obj == 2:
        #     # 因为2物体文件夹内除了该物体的mask，还有其他物体的mask，所以需要单独处理找到2物体的mask
        #     for i in range(0, len(self.meta[obj][rank])):
        #         if self.meta[obj][rank][i]['obj_id'] == 2:
        #             meta = self.meta[obj][rank][i]
        #             break
        # else:
        #     meta = self.meta[obj][rank][0]
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

        '''
        ———————————————————————————————————— （2）获取掩码（得到物体所在的标准包围框区域） ——————————————————————————————————————
        获取深度图的掩码，根据深度图上每个像素的值是否为0，来构造一个掩码，以确定深度图上的哪些像素值有效，哪些像素值无效。掩码是一个布尔数组，其中的值为True的元素表示相应位置的值有效，值为False的元素表示相应位置的值无效。
        获取mask掩码，将实例分割后的结果(eval)，或者标准mask(train/test)中值为255的像素作为掩码保存在mask_label数组中，并转换为1通道。
        取两个掩码图的交集作为mask，其中depth为true代表深度值有效，label为true代表此处有物体
        根据超参数确定是否为RGB添加噪声（改变亮度、对比度、饱和度、色调），然后提取RGB图的前三个通道，并将图像的维度从(height, width, channel)转换为(channel, height, width)。很多图像处理的库都把通道数放在第1维，这样可以方便地对每一个通道进行操作。
        根据元数据中的包围框坐标，或eval时使用mask_to_bbox获取mask图像中物体的包围框，利用get_bbox将包围框转换为标准大小的包围框（标准大小在border_list中定义）。
        利用标准大小的包围框获取RGB图像中物体所在的矩形区域。
        '''
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0)) # 获取深度图的掩码，确定深度图上的哪些像素值有效True，哪些像素值无效False
        # 读取mask图片，将255的像素作为掩码保存在mask_label中，并转换为1通道
        # 如果是eval，那么添加的是语义分割之后的图片
        if self.mode == 'eval':
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
        # 否则添加的是标准mask，需要保存第一个通道
        else:
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]
        
        # 取label和depth都为true的像素作为mask
        # depth为true代表深度值有效，label为true代表此处有物体
        mask = mask_label * mask_depth

        if self.add_noise:
            img = self.trancolor(img)   # 加入噪声

        img = np.array(img)[:, :, :3]   # 提取前三个通道rgb
        img = np.transpose(img, (2, 0, 1))  # 转换通道，将图像的维度从(height, width, channel)转换为(channel, height, width)
        # 很多图像处理的库都把通道数放在第1维，这样可以方便地对每一个通道进行操作。
        img_masked = img

        # 计算物体所在区域的角点坐标
        if self.mode == 'eval':
            # mask_to_bbox：使用findContours找到mask物体的最小包围框，返回包围框左上角的[x, y, w, h]
            # get_bbox：输入一个包围框bbox = [x, y, w, h]，返回一个标准大小的包围框 rmin, rmax, cmin, cmax
            rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))
        else:
            rmin, rmax, cmin, cmax = get_bbox(meta['obj_bb'])

        img_masked = img_masked[:, rmin:rmax, cmin:cmax]
        #p_img = np.transpose(img_masked, (1, 2, 0))
        #scipy.misc.imsave('evaluation_result/{0}_input.png'.format(index), p_img)


        '''
        ————————————————————————————————————— （3）获得物体上500个随机点的x、y、深度值 ——————————————————————————————————————
        从元数据中读取真实的旋转矩阵、平移矩阵。
        将物体所在标准包围框区域的mask展开到1维，选取为true的部分的下标，赋值给choose，此时choose就是所有非零元素的下标的数组。
        - 如果choose长为0，则返回6个0张量
        - 如果choose超过500，则随机选取500个作为choose
        - 如果choose介于0-500个，则使用warp模式填充\[1,2,3] -> \[1,2,3,1,2,3,...]
        根据物体所在标准包围框，获得对应区域的深度图，并展开至1维，然后使用choose进行索引，得到choose的深度值，然后转换为列向量。对xmap和ymap进行同样的操作。
        利用相机参数，以及choose点的x、y、深度，计算choose的500个点的相机坐标系下的三维坐标，并将x、y、深度三个列向量(500,1)拼接为(500,3)的点云cloud。
        根据超参数判断是否为x、y、depth点云添加噪声

        注意：这些点云都是当前数据姿态
        '''
        # 相机坐标系与世界坐标系之间的旋转矩阵，转换成数组类，然后resize为3*3矩阵
        target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
        # 相机在世界坐标系下的平移，转换成数组类
        target_t = np.array(meta['cam_t_m2c'])

        # 定义噪声，并分别给xyz坐标添加噪声（生成一个长度为3的数组，每个元素是+-noise_trans范围内的随机数）
        add_t = np.array([random.uniform(self.noise_trans, self.noise_trans) for i in range(3)])
        ################################################################################################
        # add_t = add_t / 1000.0
        ################################################################################################

        # 随机选择500个点云，将物体所在标准包围框区域的mask展开，选取为true的部分的下标
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        # 如果索引个数为0，则返回0张量，表示没有物体
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
        ##############################################################
        cam_scale = 1.0
        # cam_scale = 0.0010000000474974513
        # pt2 = depth_masked / cam_scale  # z轴
        pt2 = depth_masked * cam_scale  # z轴（深度值单位默认为mm）
        ##############################################################
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx   # x轴，直接计算得到的xy单位也是mm
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy   # y轴
        cloud = np.concatenate((pt0, pt1, pt2), axis=1) # 将三个列向量(500,1)拼接为(500,3)的点云
        ##############################################################
        cloud = cloud / 1000.0
        # cloud = cloud
        ##############################################################

        # 如果添加噪声，则每个点的三个坐标都添加add_t
        if self.add_noise:
            cloud = np.add(cloud, add_t)

        #fw = open('evaluation_result/{0}_cld.xyz'.format(index), 'w')
        #for it in cloud:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        #fw.close()

        '''
        —————————————————————————————— （4）读取模型点云和转换到相机坐标系后的模型点云 ————————————————————————————————
        读取该物体的模型点云（从ply文件中读取），在点云中随机删除，最后只留下500个点云。
        从元数据中读取真实的旋转矩阵、平移矩阵。利用齐次变换得到当前姿态的点云target。
        根据超参数判断是否为模型变换得到的点云target添加噪声
        '''
        
        # 获取物体的模型点云
        ###############################################################################################################
        # model_points = self.pt[obj]
        model_points = self.pt[obj] / 1000.0
        ###############################################################################################################
        # 随机删除点云，只保留num_pt_mesh_small(500)个
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
            #####################################################################################
            target = np.add(target, target_t / 1000.0 + add_t)
            # target = np.add(target, target_t + add_t)
            #####################################################################################
            out_t = target_t / 1000.0 + add_t
        else:
            #####################################################################################
            target = np.add(target, target_t / 1000.0)
            # target = np.add(target, target_t)
            #####################################################################################
            out_t = target_t / 1000.0
        #fw = open('evaluation_result/{0}_tar.xyz'.format(index), 'w')
        #for it in target:
        #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
        #fw.close()

        ####################################################3
        # target = target / 1000
        # print("\nmodel_points")
        # print(model_points)
        # print("\ntarget")
        # print(target)
        # print("\ntarget_r.T")
        # print(target_r.T)
        # print("\ntarget_t")
        # print(target_t)
        ####################################

        '''
        —————————————————————————————— 返回值 ——————————————————————————————————————
        - cloud：深度图+内参计算的点云
        - choose：用于选取的choose列表
        - img_masked：物体所在标准包围框的RGB图
        - target：模型点云齐次变换后的点云
        - model_points：模型点云
        - self.objlist.index(obj)：该物体在objlist的索引
        '''
        # 返回点云，点选区的点云索引，物体所在图像区域、目标点云、物体第一帧点云、物体类别
        # 将个个变量转换成tensor形式，张量（Tensor）是一种高效的多维数组，用于存储和操作大量数字
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
        objlist = [1, 2, 3]
        meta_file = open('{0}/models_info.yml'.format(dataset_config_dir), 'r')
        # meta = yaml.load(meta_file)
        meta = yaml.load(meta_file,Loader=yaml.FullLoader)
        for obj in objlist:
            diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
        return diameter
        # 相机中心坐标

# 边界列表，固定了一系列包围框的尺寸，目的是使用标准的包围框大小
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640


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
