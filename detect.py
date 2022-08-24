#coding=utf-8
import argparse
import glob,os
import itertools
import random
from time import time
from utils.utils import mk_if_not_exits

import cv2
import numpy as np
from keras.models import load_model

import data
import Models
from Models import build_model
from metrics import metrics

EPS = 1e-12

parser = argparse.ArgumentParser()
parser.add_argument("--test_images", type=str, 
                    default="D:/CodePost/Miandan500/test_image/")
parser.add_argument("--output_path", type=str, 
                    default="D:/Code/Keras-Semantic-Segmentation/runs/20220823_MD500_unet_detect/")
parser.add_argument("--weights_path", type=str,
                    default="runs/20220823_MD500_unet_train/weights/epoch16_acc0.986904_valacc0.979002.hdf5")
parser.add_argument("--model_name", type=str, default="unet")
parser.add_argument("--input_height", type=int, default=640)
parser.add_argument("--input_width", type=int, default=640)
parser.add_argument("--resize_op", type=int, default=1)
parser.add_argument("--classes", type=int, default=3)
parser.add_argument("--image_init", type=str, default="divide")
args = parser.parse_args()

images_path = args.test_images
output_path = args.output_path
save_weights_path = args.weights_path
model_name = args.model_name
input_height = args.input_height
input_width = args.input_width
resize_op = args.resize_op
n_class = args.classes
image_init = args.image_init

# color
random.seed(0)
colors = [(random.randint(0, 255), random.randint(0,255), random.randint(0, 255)) for _ in range(100)]

# model
model = build_model(model_name,
                    n_class,
                    input_height=input_height,
                    input_width=input_width)

model.load_weights(save_weights_path)
output_height = model.outputHeight
output_width = model.outputWidth

# look up test images
if images_path.endswith(".jpg") or images_path.endswith(".png"):
    images = [images_path]
else:
    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png")
img_num = len(images)

# 输出
mk_if_not_exits(output_path)

print('===================== lc info =======================')
print('test images path : {}'.format(images_path))
print('test images num  : {}'.format(img_num))

for i in range(img_num):
    
    origin_img = cv2.imread(images[i])
    origin_h = origin_img.shape[0]
    origin_w = origin_img.shape[1]
    
    # 保存结果时的文件名 - Linux
    # imgName = images[i].split('/')[-1].split('.')[0]
    # pr_name = os.path.join(output_path, '{}_pr.png'.format(imgName))
    # org_Name = os.path.join(output_path, '{}_org.png'.format(imgName))
    # mask_Name = os.path.join(output_path, '{}_mask.png'.format(imgName))
    # viz_Name = os.path.join(output_path, '{}_viz.jpg'.format(imgName))
    
    # 保存结果时的文件名 - Windows
    imgName = images[i].split('\\')[-1].split('.')[0]
    pr_name = os.path.join(output_path, '{}_pr.png'.format(imgName))
    pr_viz_name = os.path.join(output_path, '{}_pr_viz.jpg'.format(imgName))

    print('{}/{}: {} in processing'.format(i+1, img_num, images[i]))

    X = data.getImage(images[i], input_width, input_height, image_init, resize_op)
    pr = model.predict(np.array([X]))[0]
    pr = pr.reshape((output_height, output_width, n_class)).argmax(axis=2)

    # 预测结果可视化
    if n_class > 2:
        pr_viz = np.zeros((output_height, output_width, 3))
        for c in range(n_class):
            pr_viz[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
            pr_viz[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
            pr_viz[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
    else:
        pr_viz = (pr * 255).astype('uint8')

    # 预测结果与原图叠加
    aaa = np.zeros((output_height, output_width, n_class))
    for i in range(n_class):
        aaa[:, :, i] = (pr[:, :] == i)

    class_1 = aaa[:, :, 1] * 255
    class_2 = aaa[:, :, 2] * 255

    # 保存结果
    cv2.imwrite(pr_name, pr)
    cv2.imwrite(pr_viz_name, pr_viz)

    class1_name = os.path.join(output_path, '{}_class1.png'.format(imgName))
    cv2.imwrite(class1_name, class_1)

    class2_name = os.path.join(output_path, '{}_class2.png'.format(imgName))
    cv2.imwrite(class2_name, class_2)

print('Finished')
