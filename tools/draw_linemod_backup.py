import _init_paths
import argparse
import os
from PIL import Image
import numpy as np
import numpy.ma as ma
import torch
import random
import torchvision.transforms as transforms
import yaml
import cv2
from torch.autograd import Variable
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
from lib.loss_refiner import Loss_refine
from lib.knn.__init__ import KNearestNeighbor
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
import copy
import os
from PIL import Image
import matplotlib.pyplot as plt
import json

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

###################################################################################################################
def get_3D_corners(mesh):
    Tform = mesh.apply_obb()
    points = mesh.bounding_box.vertices
    center = mesh.centroid
    min_x = np.min(points[:,0])
    min_y = np.min(points[:,1])
    min_z = np.min(points[:,2])
    max_x = np.max(points[:,0])
    max_y = np.max(points[:,1])
    max_z = np.max(points[:,2])
    corners = np.array([[min_x, min_y, min_z], [min_x, min_y, max_z], [min_x, max_y, min_z],
                        [min_x, max_y, max_z], [max_x, min_y, min_z], [max_x, min_y, max_z],
                        [max_x, max_y, min_z], [max_x, max_y, max_z]])
    corners = np.concatenate((np.transpose(corners), np.ones((1,8)) ), axis=0)
    return corners

def get_camera_intrinsic(u0, v0, fx, fy):
    return np.array([[fx, 0.0, u0], [0.0, fy, v0], [0.0, 0.0, 1.0]])

def compute_projection(points_3D, transformation, internal_calibration):
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d

def visualize_labels(img, mask, labelfile):

    cv2.addWeighted(mask, 0.4, img, 0.6, 0, img)
    if os.path.exists(labelfile):
        with open(labelfile, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                info = line.split()
        info = [float(i) for i in info]
        width, length = img.shape[:2]
        one = (int(info[3]*length),int(info[4]*width))
        two = (int(info[5]*length),int(info[6]*width))
        three = (int(info[7]*length),int(info[8]*width))
        four = (int(info[9]*length),int(info[10]*width))
        five = (int(info[11]*length),int(info[12]*width))
        six = (int(info[13]*length),int(info[14]*width))
        seven = (int(info[15]*length),int(info[16]*width))
        eight =  (int(info[17]*length),int(info[18]*width))

        cv2.line(img,one,two,(255,0,0),3)
        cv2.line(img,one,three,(255,0,0),3)
        cv2.line(img,two,four,(255,0,0),3)
        cv2.line(img,three,four,(255,0,0),3)
        cv2.line(img,one,five,(255,0,0),3)
        cv2.line(img,three,seven,(255,0,0),3)
        cv2.line(img,five,seven,(255,0,0),3)
        cv2.line(img,two,six,(255,0,0),3)
        cv2.line(img,four,eight,(255,0,0),3)
        cv2.line(img,six,eight,(255,0,0),3)
        cv2.line(img,five,six,(255,0,0),3)
        cv2.line(img,seven,eight,(255,0,0),3)

    return img
####################################################################################################################

# 将mask转换为边界框
def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]


# 获取边界框坐标
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


# 读取ply角点坐标
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

# 读取数据
# data_root数据路径，item编号，seg_root分割路径，obj物体序号，model_root模型路径，添加噪音，噪音矩阵
def get_data(data_root, item, seg_root, obj, model_root, add_noise, noise_trans):
    # 读取图片（RGB、深度图、标签图）
    img = Image.open('{0}/rgb/{1}.png'.format(data_root, item))
    ori_img = np.array(img)
    depth = np.array(Image.open('{0}/depth/{1}.png'.format(data_root, item)))
    label = np.array(Image.open('{0}/{1}_label.png'.format(seg_root, item)))
    print('{0}/depth/{1}.png'.format(data_root, item))
    print('{0}/{1}_label.png'.format(seg_root, item))
    print('{0}/gt.yml'.format(data_root))
    print('{0}/obj_{1}.ply'.format(model_root, '%02d' % obj))
    
    # 读取元数据
    meta_file = open('{0}/gt.yml'.format(data_root), 'r')
    meta = yaml.load(meta_file, Loader=yaml.FullLoader)
    pt = ply_vtx('{0}/obj_{1}.ply'.format(model_root, '%02d' % obj))    # 读取ply角点坐标
    
    # 定义参数
    trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05) # 定义颜色和透明度
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    num = 500   # 点云数量
    xmap = np.array([[j for i in range(640)] for j in range(480)])  # 图像范围
    ymap = np.array([[i for i in range(640)] for j in range(480)])
    ######################################################################
    # cam_cx = 325.26110  # 相机参数
    # cam_cy = 242.04899
    # cam_fx = 572.41140
    # cam_fy = 573.57043
    # 相机中心坐标
    cam_cx = 321.6173095703125
    cam_cy = 237.4153594970703
    # 相机的焦距
    cam_fx = 605.1395263671875
    cam_fy = 604.8554077148438
    ######################################################################
    num_pt_mesh_large = 500
    num_pt_mesh_small = 500
    rank = int(item)    # 图像编号
    if obj == 2:
        for i in range(0, len(meta[obj])):
            if meta[rank][i]['obj_id'] == 2:
                meta = meta[rank][i]
                break
    else:
        meta = meta[rank][0]

    # 获取掩码（深度图和标签图同时为true的像素为掩码）
    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
    mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
    mask = mask_label * mask_depth

    if add_noise:
        img = trancolor(img)

    # 获取掩码mask对应范围的img
    img = np.array(img)[:, :, :3]
    img = np.transpose(img, (2, 0, 1))
    img_masked = img

    # 获取边界框
    rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))
    img_masked = img_masked[:, rmin:rmax, cmin:cmax]
    #p_img = np.transpose(img_masked, (1, 2, 0))
    #scipy.misc.imsave('evaluation_result/{0}_input.png'.format(index), p_img)

    # 获取真实的旋转和平移变换（从meta中读取）
    target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
    target_t = np.array(meta['cam_t_m2c'])
    add_t = np.array([random.uniform(-noise_trans, noise_trans) for i in range(3)])

    # 从掩码mask的true值中随机选取500个
    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    if len(choose) == 0:
        cc = torch.LongTensor([0])
        return(cc, cc, cc, cc, cc, cc)

    if len(choose) > num:
        c_mask = np.zeros(len(choose), dtype=int)
        c_mask[:num] = 1
        np.random.shuffle(c_mask)
        choose = choose[c_mask.nonzero()]
    else:
        choose = np.pad(choose, (0, num - len(choose)), 'wrap')
    
    # 获取choose的500个掩码mask的点对应的深度图、x坐标、y坐标
    depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    choose = np.array([choose])

    # 将选区的500个mask的点的深度图转换成500个点云
    ###############################################################
    # cam_scale = 1
    cam_scale = 0.0010000000474974513
    # pt2 = depth_masked / cam_scale
    pt2 = depth_masked * cam_scale
    ###############################################################
    
    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
    cloud = np.concatenate((pt0, pt1, pt2), axis=1)
    cloud = cloud / 1000.0
    print(cloud.shape)

    if add_noise:
        cloud = np.add(cloud, add_t)

    # 读取模型真实点云，并进行随机删除
    #fw = open('evaluation_result/{0}_cld.xyz'.format(index), 'w')
    #for it in cloud:
    #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
    #fw.close()
    ############################################################################
    # model_points = pt / 1000.0
    model_points = pt
    ############################################################################
    dellist = [j for j in range(0, len(model_points))]
    dellist = random.sample(dellist, len(model_points) - num_pt_mesh_small)
    model_points = np.delete(model_points, dellist, axis=0)

    #fw = open('evaluation_result/{0}_model_points.xyz'.format(index), 'w')
    #for it in model_points:
    #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
    #fw.close()

    # 计算当前帧的真实点云，通过模型点云*真实变换矩阵
        # target是由model_points经过标准姿态转换后相机坐标系下的点
    target = np.dot(model_points, target_r.T)
    if add_noise:
        #####################################################################################
        # target = np.add(target, target_t / 1000.0 + add_t)
        target = np.add(target, target_t + add_t * 1000)
        #####################################################################################
        out_t = target_t / 1000.0 + add_t
    else:
        #####################################################################################
        # target = np.add(target, target_t / 1000.0)
        target = np.add(target, target_t)
        #####################################################################################
        out_t = target_t / 1000.0

    #fw = open('evaluation_result/{0}_tar.xyz'.format(index), 'w')
    #for it in target:
    #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
    #fw.close()
    
    ###########################################################
    # model_points = model_points*1000
    print("\nmodel_points")
    print(model_points)
    print("\ntarget")
    print(target)
    print("\ntarget_r.T")
    print(target_r.T)
    print("\ntarget_t")
    print(target_t)
    ###########################################################

    return torch.from_numpy(cloud.astype(np.float32)), \
           torch.LongTensor(choose.astype(np.int32)), \
           norm(torch.from_numpy(img_masked.astype(np.float32))), \
           torch.from_numpy(target.astype(np.float32)), \
           torch.from_numpy(model_points.astype(np.float32)), \
           torch.LongTensor([obj])


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default = '', help='ycb or linemod')
parser.add_argument('--seg_root', type=str, default = '', help='ycb or linemod')
parser.add_argument('--model_root', type=str, default = '', help='ycb or linemod')
parser.add_argument('--item', type=str, default = '', help='')
parser.add_argument('--obj', type=int, default=1, help='')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--output', type=str, default = '',  help='resume PoseRefineNet model')
opt = parser.parse_args() 

##################################################################################
# 定义PoseNet和PoseRefineNet网络
# estimator = PoseNet(num_points = 500, num_obj = 13)
estimator = PoseNet(num_points = 500, num_obj = 5)
estimator.cuda()
# refiner = PoseRefineNet(num_points = 500, num_obj = 13)
refiner = PoseRefineNet(num_points = 500, num_obj = 5)
refiner.cuda()
##################################################################################


estimator.load_state_dict(torch.load(opt.model), strict=False)
refiner.load_state_dict(torch.load(opt.refine_model))
# pose_model = torch.load(opt.model)
# pose_model.pop('conv4_r.weight')
# pose_model.pop('conv4_r.bias')
# pose_model.pop('conv4_t.weight')
# pose_model.pop('conv4_t.bias')
# pose_model.pop('conv4_c.weight')
# pose_model.pop('conv4_c.bias')
# refine_pose_model = torch.load(opt.refine_model)
# refine_pose_model.pop('conv3_r.weight')
# refine_pose_model.pop('conv3_r.bias')
# refine_pose_model.pop('conv3_t.weight')
# refine_pose_model.pop('conv3_t.bias')
# estimator.load_state_dict(pose_model, strict=False)
# refiner.load_state_dict(refine_pose_model, strict=False)

bs = 1
num_points = 500
iteration = 4
knn = KNearestNeighbor(1)

for i in range(1):
    # points深度图转换的点云，choose序列，mask区域的img图像，target当前帧真实点云，model_points模型点云，物体类别号
    points, choose, img, target, model_points, idx = get_data(opt.data_root, opt.item, opt.seg_root, opt.obj, opt.model_root, False, 0.0)
  
    if len(points.size()) == 1:
        print('No.{0} NOT Pass! Lost detection!'.format(i))
        continue
    points, choose, img, target, model_points, idx = Variable(points.unsqueeze(0)).cuda(), \
                                                        Variable(choose.unsqueeze(0)).cuda(), \
                                                        Variable(img.unsqueeze(0)).cuda(), \
                                                        Variable(target.unsqueeze(0)).cuda(), \
                                                        Variable(model_points.unsqueeze(0)).cuda(), \
                                                        Variable(idx.unsqueeze(0)).cuda()
    # 通过PoseNet计算预测的旋转，平移，置信度，颜色特征
    pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
    pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
    pred_c = pred_c.view(bs, num_points)
    how_max, which_max = torch.max(pred_c, 1)
    pred_t = pred_t.view(bs * num_points, 1, 3)

    print("\npred_c")
    print(how_max)

    # 根据最大置信度的点，找到对应的旋转和平移变换，组合成变换四元数pred
    my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
    my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
    my_pred = np.append(my_r, my_t)
    # # 迭代次数
    # for ite in range(0, iteration):
    #     # 将四元数转换为变换矩阵
    #     Tt = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
    #     my_mat = quaternion_matrix(my_r)
    #     R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
    #     my_mat[0:3, 3] = my_t
        
    #     # 根据现在的点，反推更新初始点的位姿，再次计算变换矩阵
    #     new_points = torch.bmm((points - Tt), R).contiguous()
    #     pred_r, pred_t = refiner(new_points, emb, idx)
    #     pred_r = pred_r.view(1, 1, -1)
    #     pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
    #     my_r_2 = pred_r.view(-1).cpu().data.numpy()
    #     my_t_2 = pred_t.view(-1).cpu().data.numpy()
    #     my_mat_2 = quaternion_matrix(my_r_2)
    #     my_mat_2[0:3, 3] = my_t_2

    #     # 获得最终的变换矩阵，并将变换矩阵转换为四元数进行下一次循环
    #     my_mat_final = np.dot(my_mat, my_mat_2)
    #     my_r_final = copy.deepcopy(my_mat_final)
    #     my_r_final[0:3, 3] = 0
    #     my_r_final = quaternion_from_matrix(my_r_final, True)
    #     my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

    #     my_pred = np.append(my_r_final, my_t_final)
    #     my_r = my_r_final
    #     my_t = my_t_final
    
    

    # 读取模型点云
    model_points = model_points[0].cpu().detach().numpy()
    ###########################################################
    # model_points = model_points*1000
    ###########################################################

    # 获取变换矩阵的旋转部分
    my_r = quaternion_matrix(my_r)[:3, :3]
    # print("\nmy_r")
    # print(my_r)
    # print("\nmy_t")
    # print(my_t)
    # 计算预测的点云
    # print("\nmodel_points")
    # print(model_points)
    pred = np.dot(model_points, my_r.T) + my_t  
    # print("\npred before *1000")
    # print(pred)
    target = target[0].cpu().detach().numpy()

    ########################################################################
    # 定义相机参数
    # pred = pred*1000.0
    pred = pred/1000
    # print("\npred after *1000")
    # print(pred)

    target = target*1000.0
    # print("\ntarget")
    # print(target)

    
    # cam_scale = 1.0
    cam_scale = 0.0010000000474974513
    # 相机中心坐标
    cam_cx = 321.6173095703125
    cam_cy = 237.4153594970703
    # 相机的焦距
    cam_fx = 605.1395263671875
    cam_fy = 604.8554077148438
    ########################################################################

    # 预测点云的深度、x坐标、y坐标
    depth_masked =  pred[:,2] * cam_scale
    ymap_masked_pred =  pred[:,0] * cam_fx / pred[:,2] + cam_cx
    xmap_masked_pred = pred[:,1] * cam_fy / pred[:,2] + cam_cy

    # 当前帧真实点云的深度、x坐标、y坐标
    depth_masked =  target[:,2] * cam_scale
    ymap_masked_target =  target[:,0] * cam_fx / target[:,2] + cam_cx
    xmap_masked_target = target[:,1] * cam_fy / target[:,2] + cam_cy

    image = Image.open('{0}/rgb/{1}.png'.format(opt.data_root, opt.item))

    plt.figure(figsize=(8,12)) # 图像窗口名称
    plt.imshow(image)
    plt.scatter(ymap_masked_pred,xmap_masked_pred,marker='.',c='r',alpha=1)
    plt.axis('on') # 关掉坐标轴为 off
    plt.title('image') # 图像题目
    plt.show()
    plt.savefig('{0}/test_pred.png'.format(opt.output))

    plt.figure(figsize=(8,12)) # 图像窗口名称
    plt.imshow(image)
    plt.scatter(ymap_masked_target,xmap_masked_target,marker='.',c='r',alpha=1)
    plt.axis('on') # 关掉坐标轴为 off
    plt.title('image') # 图像题目
    plt.show() 
    plt.savefig('{0}/test_target.png'.format(opt.output))

    ########################################################################################################

    import trimesh

    image = cv2.imread('{0}/rgb/{1}.png'.format(opt.data_root, opt.item))
    mask = cv2.imread('{0}/mask/{1}.png'.format(opt.data_root, opt.item))
    ori_label_path = '{0}/labels/{1}.txt'.format(opt.data_root, str(int(opt.item)))

    edges_corners = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    meshname = '{0}/obj_{1}.ply'.format(opt.model_root, '%02d' % opt.obj)
    mesh = trimesh.load(meshname)
    vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners3D = get_3D_corners(mesh)
    intrinsic_calibration = get_camera_intrinsic(cam_cx,cam_cy,cam_fx,cam_fy)

    meta_file = open('{0}/gt.yml'.format(opt.data_root), 'r')
    meta = yaml.load(meta_file, Loader=yaml.FullLoader)
    rank = int(opt.item)
    meta = meta[rank][0]
    target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
    target_t = np.array(meta['cam_t_m2c'])

    print("\ntarget_r")
    print(target_r)
    print("\ntarget_t")
    print(target_t)
    ########################################################################################################

    # cvimg = cv2.imread('{0}/rgb/{1}.png'.format(opt.data_root, opt.item))
    # point_size = 1
    # point_color = (0, 0, 255) # BGR
    # point_thickness = 4 # 可以为 0 、4、8
    # points_list = [(int(ymap_masked_target[i]), int(xmap_masked_target[i])) for i in range(len(xmap_masked_target))]
    # for point in points_list:
    #     cv2.circle(cvimg, point, point_size, point_color, point_thickness)
    # cv2.namedWindow("image")
    # cv2.imshow("image", cvimg)
    # cv2.waitKey()
    # cv2.destroyAllWindows()