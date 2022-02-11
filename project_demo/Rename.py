import json
import re
from glob import glob
from os.path import join

import numpy as np
import cv2
import os
import natsort

Input = r'D:\code\laser_distance_measuring\small'
mask_folder = Input + r'/Masks/'
img_folder = Input + r'/Images/'

if not os.path.exists(mask_folder):
    os.makedirs(mask_folder)

if not os.path.exists(img_folder):
    os.makedirs(img_folder)

folders = os.listdir(Input)
folders = natsort.natsorted(folders)
# prefix = ['12L.jpg','12M.jpg','12R.jpg']
# json_prefix = ['12L.json','12M.json','12R.json']
for root, dirs, files in os.walk(Input):
    for i, file in enumerate(files):
        file_name = os.path.join(root, file)
        file_name = file_name.replace('\\', '/')
        root = root.replace('\\', '/')
        name = root.split('/')[-1]
        name = re.sub('_json', '', name)
        name = root.split('/')[-2] + '_' + name
        if file.split('.')[0] == 'img':
            new_name = name + '.png'
            os.rename(file_name, os.path.join(img_folder, new_name))
        elif file.split('.')[0] == 'label':
            new_name = name + '.png'
            # new_name = file_name.split('/')[4] + '_' + \
            #            file_name.split('/')[3] + '_' + f'{i+1}.jpg'
            os.rename(file_name, os.path.join(mask_folder, new_name))
        print(file_name)

        # print(img_path.split('\\'))
        # if img_path.split('\\')[1] == "img.png":
        #     os.rename(img_path, os.path.join(img_folder, folder+".png"))
        # if img_path.split('\\')[1] == 'label.png':
        #     os.rename(img_path, os.path.join(mask_folder, folder+'_mask.png'))
    # with open(json_paths[i], 'r', encoding='utf8') as fp:
    # json_file = json.load(fp)
    # json_file['imagePath'] = folder + "_" + prefix[i]
    # with open(json_paths[i], 'w', encoding='utf8') as fp2:
    # json.dump(json_file,fp2)
    # os.rename(img_path,os.path.join(Input, folder+'_'+prefix[i]))
    # os.rename(json_paths[i],os.path.join(Input, folder+'_'+json_prefix[i]))
