import os
import numpy as np
import natsort
from PIL import Image
import random
import cv2
from torchvision import transforms
import torchvision.transforms.functional as tf
from tqdm import tqdm
# 读入一张图片

def RandomRotation(image, mask):
    # 拿到角度的随机数。angle是一个-180到180之间的一个数
    angle = transforms.RandomRotation.get_params([-180, 180])
    # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
    image = image.rotate(angle)
    mask = mask.rotate(angle)

    return image, mask


def random_flip(image, mask):
    # 50%的概率应用垂直，水平翻转。
    if random.random() > 0.5:
        image = tf.hflip(image)
        mask = tf.hflip(mask)
    if random.random() > 0.5:
        image = tf.vflip(image)
        mask = tf.vflip(mask)
    return image, mask


def RandomResizedCrop(image, mask):
    # 随机裁剪
    i, j, h, w = transforms.RandomResizedCrop.get_params(
        image, scale=(0.7, 1.0), ratio=(1, 1))
    image = tf.resized_crop(image, i, j, h, w, 48)
    mask = tf.resized_crop(mask, i, j, h, w, 48)
    return image, mask


labels = ['random_crop_1','random_crop_2',
         'random_crop_3','random_crop_4','center_crop','h_flip','v_flip', 'rotation', 'affine']
Input = r'D:\code\pclassified\bengkou/Images/'
Input2 = r'D:\code\pclassified\bengkou/Masks/'
folders = os.listdir(Input)
folders = natsort.natsorted(folders)
folders_2 = os.listdir(Input2)
folders_2 = natsort.natsorted(folders_2)
index = 0
for folder in tqdm(folders):

    im = Image.open(Input + folder)
    mask = Image.open(Input2 + folders_2[index])
    random_im1,random_mask1 = RandomResizedCrop(im,mask)
    random_im2,random_mask2 = RandomResizedCrop(im,mask)
    random_im3,random_mask3 = RandomResizedCrop(im,mask)
    random_im4,random_mask4 = RandomResizedCrop(im,mask)
    center_im = transforms.CenterCrop(size=1000)(im)
    center_mask = transforms.CenterCrop(size=1000)(mask)
    flip_1, flip_mask_1 = random_flip(im, mask)
    flip_2, flip_mask_2 = random_flip(im, mask)
    rotation, rotation_mask = RandomRotation(im, mask)
    affine, affine_mask = RandomRotation(im, mask)
    imgs = [random_im1,random_im2,random_im3,random_im4,center_im,
            flip_1, flip_2, rotation,affine]
    masks = [random_mask1,random_mask2,random_mask3,random_mask4,
             center_mask,flip_mask_1,
             flip_mask_2,rotation_mask,affine_mask]
    for i,label in enumerate(labels):
        imgs[i].save(Input + '/' + folder.split('.')[0] + '_' +
                    label + '.png')
        masks[i].save(Input2 + '/' + folders_2[index].split('.')[0] + '_' +
                      label + '.png')
    index += 1
# # 亮度
# bright_im = tfs.ColorJitter(brightness=1)(im) # 随机从 0 ~ 2 之间亮度变化，1 表示原图
# # 对比度
# contrast_im = tfs.ColorJitter(contrast=1)(im) # 随机从 0 ~ 2 之间对比度变化，1 表示原图
# # 颜色
# color_im = tfs.ColorJitter(hue=0.5)(im) # 随机从 -0.5 ~ 0.5 之间对颜色变化


