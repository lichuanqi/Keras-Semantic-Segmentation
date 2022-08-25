#coding=utf-8
import os
# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import argparse
import glob
import itertools
import random
from time import time
import yaml

import cv2
import numpy as np
from keras.models import load_model

import data
import Models
from Models import build_model
from metrics import metrics

from utils.utils import mk_if_not_exits


EPS = 1e-12

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="unet")
parser.add_argument("--weights_path", type=str,
                    default="runs/20220823_MD500_unet_train/weights/epoch16_acc0.986904_valacc0.979002.hdf5")
parser.add_argument("--output_path", type=str, 
                    default="runs/20220823_MD500_unet_test")
parser.add_argument("--input_height", type=int, default=640)
parser.add_argument("--input_width", type=int, default=640)
parser.add_argument("--resize_op", type=int, default=2)
parser.add_argument("--classes", type=int, default=3)
# streetscape(12)(320x640), helen_small(11)(512x512), bbufdataset(2)
parser.add_argument("--mIOU", type=bool, default=True)
parser.add_argument("--val_images",type=str, 
                    default="D:/CodePost/Miandan500/test_image/")
parser.add_argument("--val_masks",type=str,
                    default="D:/CodePost/Miandan500/test_label/")
parser.add_argument("--image_init", type=str, default="divide")
args = parser.parse_args()

output_path = args.output_path
save_weights_path = args.weights_path
model_name = args.model_name
input_height = args.input_height
input_width = args.input_width
resize_op = args.resize_op
n_class = args.classes
iou = args.mIOU
image_init = args.image_init

# color
random.seed(0)
colors = [(random.randint(0, 255), random.randint(0,255), random.randint(0, 255)) for _ in range(10)]

# model
model = build_model(model_name,
                    n_class,
                    input_height=input_height,
                    input_width=input_width)

model.load_weights(save_weights_path)
output_height = model.outputHeight
output_width = model.outputWidth

# 输出
mk_if_not_exits(output_path)

# mIOU
if iou:

    print('PA and IoU Start.')

    # 计算结果保存至 output.yaml
    yaml_save_name = os.path.join(output_path, 'output.yaml')
    output_dic = {}
    
    tp = np.zeros(n_class)
    fp = np.zeros(n_class)
    fn = np.zeros(n_class)
    n_pixels = np.zeros(n_class)

    images_path = args.val_images
    segs_path = args.val_masks

    assert images_path[-1] == '/'
    assert segs_path[-1] == '/'
    
    images = glob.glob(images_path + "*.jpg") + \
             glob.glob(images_path + "*.png") + \
             glob.glob(images_path + "*.jpeg")
    images.sort()
    segmentations = glob.glob(segs_path + "*.jpg") + \
                    glob.glob(segs_path + "*.png") + \
                    glob.glob(segs_path + "*.jpeg")
    segmentations.sort()
    assert len(images) == len(segmentations)

    print('===================== test data info =======================')
    print('test images path : {}'.format(images))
    print('test images num  : {}'.format(len(images)))
    
    zipped = itertools.cycle(zip(images, segmentations))
    for _ in range(len(images)):
        img_path, seg_path = next(zipped)
        # get origin h, w
        img = data.getImage(img_path, 
                            input_width, input_height, 
                            image_init,resize_op)
        gt = data.getLable(seg_path, n_class, 
                           output_width, output_height,
                           resize_op)
        pr = model.predict(np.array([img]))[0]
        gt = gt.argmax(axis=-1)
        pr = pr.argmax(axis=-1)
        gt = gt.flatten()
        pr = pr.flatten()

        for c in range(n_class):
            tp[c] += np.sum((pr == c) * (gt == c))
            fp[c] += np.sum((pr == c) * (gt != c))
            fn[c] += np.sum((pr != c) * (gt == c))
            n_pixels[c] += np.sum(gt == c)

    print('TP : {}'.format(tp))
    print('FP : {}'.format(fp))
    print('FN : {}'.format(fn))

    # ================= PA 像素准确率 ================
    # class_PA: 每个类别的 PA
    # mean_PA : 所有类别的平均 PA
    class_PA = tp / (tp + fp + EPS)
    mean_PA = np.mean(class_PA)

    # ================== IOU 交并比 ==================
    # class_IoU: 每个类别的 IoU
    # mean_IOU : 所有类别的平均 IoU
    class_IoU = tp / (tp + fp + fn + EPS)
    mean_IoU = np.mean(class_IoU)

    # ============= frequency weighted IoU ==========
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IU = np.sum(class_IoU * n_pixels_norm)
    
    print('Class PA: {}'.format(class_PA))
    print('Mean PA : {:.5f}'.format(mean_PA))
    print("Class IoU : {}".format(class_IoU))
    print("Mean IoU  : {:.5f}".format(mean_IoU))
    print("Frequency Weighted IOU: {:.5f}".format(frequency_weighted_IU))

    # 保存输出结果
    output_dic['Class PA'] = '%s'%class_PA
    output_dic['Mean PA'] = '%.5f'%mean_PA
    output_dic['Class IoU'] = '%s'%class_IoU
    output_dic['Mean IoU'] = '%.5f'%mean_IoU
    output_dic['Frequency Weighted IOU'] = '%.5f'%frequency_weighted_IU

    with open(yaml_save_name, 'w') as f:
        f.write(yaml.dump(output_dic, allow_unicode=True))
    
