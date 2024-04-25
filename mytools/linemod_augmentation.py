import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import cv2
import os
import numpy as np

# 图像数据增强
def data_augmentation(data_path, output_path):
    # 定义数据增强操作
    seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),   # 方法执行的概率，添加高斯模糊增强
        # iaa.Affine(
        #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},   # 添加仿射变换增强，scale表示缩放比例，translate_percent表示平移比例，shear表示剪切角度
        #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        #     rotate=(-45, 45),
        #     shear=(-16, 16)
        # ),
        iaa.SomeOf((0, 5), [    # 定义任意k个增强方法被随机执行
            iaa.OneOf([
                iaa.Dropout(p=(0, 0.1)),    # 添加高斯随机删除像素增强，p表示删除像素的概率
                iaa.CoarseDropout(p=(0, 0.02), size_percent=(0.02, 0.05)),   # 添加块状随机删除像素增强，p表示删除像素的概率，size_percent表示每个块的大小（占图像总像素的百分比）
            ]),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),  # 添加颜色偏移增强，value表示偏移量
            iaa.LinearContrast(alpha=(0.95, 1.05), per_channel=True),   # 添加对比度变化增强，alpha表示对比度的缩放因子。
            iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255), per_channel=True), # 添加高斯噪声增强，scale表示噪声的标准差
            iaa.Multiply((0.8, 1.2), per_channel=0.2),  # 添加像素值乘法增强，乘数的范围是(Minimum 0.8, Maximum 1.2)
        ], random_order=True)
    ])

    # 获取图片列表
    obj_list = os.listdir(data_path)
    obj_list.sort()
    for obj in obj_list:
        print(obj, "is working.")
        img_list = os.listdir(os.path.join(data_path, obj, "rgb"))
        img_list.sort()

        # 路径检查
        if not os.path.exists(os.path.join(output_path, obj, "rgb")):  # 如果路径不存在
            os.makedirs(os.path.join(output_path, obj, "rgb"))  # 创建路径文件夹
        if not os.path.exists(os.path.join(output_path, obj, "mask")):  # 如果路径不存在
            os.makedirs(os.path.join(output_path, obj, "mask"))  # 创建路径文件夹

        for img_name in img_list:
            print(obj, "/", img_name, "is working.")
            img_path = os.path.join(data_path, obj, "rgb", img_name)
            mask_path = os.path.join(data_path, obj, "mask", img_name)
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)

            segmap_mask = SegmentationMapsOnImage(mask, shape=img.shape)
            # 对 RGB 图像和 mask 掩码图同时进行增强
            aug_det = seq.to_deterministic()
            aug_img = aug_det.augment_image(img)
            segmap_aug = aug_det.augment_segmentation_maps(segmap_mask)
            aug_mask = segmap_aug.get_arr()
            aug_mask = np.array(aug_mask, dtype=np.uint8)
            thresh, aug_mask_bin = cv2.threshold(aug_mask, 127, 255, cv2.THRESH_BINARY)
            # 定义结构元素
            kernel = np.ones((5, 5), np.uint8)

            # 对掩码图像进行腐蚀操作
            aug_mask_bin = cv2.erode(aug_mask_bin, kernel, iterations=1)

            # 保存增强后的图像
            output_img_path = os.path.join(output_path, obj, "rgb", img_name)
            output_mask_path = os.path.join(output_path, obj, "mask", img_name)
            cv2.imwrite(output_img_path, aug_img)
            cv2.imwrite(output_mask_path, aug_mask_bin)

# 测试
if __name__ == '__main__':
    data_path = 'datasets/linemod/LINEMOD/data'
    output_path = 'datasets/linemod/LINEMOD/data_aug'

    data_augmentation(data_path, output_path)