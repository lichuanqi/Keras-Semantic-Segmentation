#coding=utf-8
# 毕业论文第三章代码
# 输入：
# 输出：

import argparse
import glob,os
import itertools
import random
from utils.utils import mk_if_not_exits

import cv2
import numpy as np
from keras.models import load_model

import data
import Models
from Models import build_model
from metrics import metrics

from postProcessing import png_alignment, png_viz

EPS = 1e-12

parser = argparse.ArgumentParser()
parser.add_argument("--test_images", type=str, 
                    default="D:/Code/1_video/M/")
parser.add_argument("--weights_path",type=str, 
                    default="expdata/0721_R660_unet_3/epoch56_acc0.996290_valacc0.989380.hdf5")
parser.add_argument("--output_path", type=str, 
                    default="expdata/20220310")
parser.add_argument("--model_name", type=str, default="unet")
parser.add_argument("--input_height", type=int, default=640)
parser.add_argument("--input_width", type=int, default=640)
parser.add_argument("--resize_op", type=int, default=2)
parser.add_argument("--classes", type=int, default=2)
# streetscape(12)(320x640), helen_small(11)(512x512), bbufdataset(2)
parser.add_argument("--mIOU", type=bool, default=False)
parser.add_argument("--val_images",type=str, 
                    default="/media/lcq/Data/modle_and_code/DataSet/RailGuard/bj_jpgs/")
parser.add_argument("--val_annotations",type=str,
                    default="/media/lcq/Data/modle_and_code/DataSet/RailGuard/bj_masks/")
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
iou = args.mIOU
image_init = args.image_init

# color
random.seed(0)
colors = [(random.randint(0, 255), random.randint(0,255), random.randint(0, 255))
          for _ in range(100)]

# model
model = build_model(model_name,
                    n_class,
                    input_height=input_height,
                    input_width=input_width)

model.load_weights(save_weights_path)
output_height = model.outputHeight
output_width = model.outputWidth

# look up test images
images = glob.glob(images_path + "*.jpg") + \
         glob.glob(images_path + "*.png") + \
         glob.glob(images_path + "*.jpeg")
images.sort()
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
    
    # 保存结果时的文件名
    imgName = images[i].split('\\')[-1].split('.')[0]
    
    org_Name = os.path.join(output_path, 'org_{}.png'.format(imgName))
    org_Name = org_Name.replace("/", "\\")
    mask_Name = os.path.join(output_path, 'mask_{}.png'.format(imgName))
    viz_Name = os.path.join(output_path, 'viz_{}.jpg'.format(imgName))

    print('{}/{}: {} in processing'.format(i+1,img_num, imgName))

    X = data.getImage(images[i], input_width, input_height, image_init, resize_op)
    pr = model.predict(np.array([X]))[0]
    pr = pr.reshape((output_height, output_width, n_class)).argmax(axis=2)
    
    # 二分类时输出二值化图像
    if n_class > 2:
        seg_img = np.zeros((output_height, output_width, 3))
        for c in range(n_class):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
    else:
        seg_img = (pr * 255).astype('uint8')
   
    # 后处理
    png_crop = png_alignment(origin_img, seg_img)
    img_viz = png_viz(origin_img, png_crop)
    
    # 保存掩码图片和可视化结果
    print(org_Name)
    cv2.imwrite(org_Name, seg_img)
    cv2.imwrite(mask_Name, png_crop)
    cv2.imwrite(viz_Name, img_viz)

print('Finished')
 
# mIOU
if iou:

    print('PA and IoU Start.')
    
    tp = np.zeros(n_class)
    fp = np.zeros(n_class)
    fn = np.zeros(n_class)
    n_pixels = np.zeros(n_class)

    images_path = args.val_images
    segs_path = args.val_annotations
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