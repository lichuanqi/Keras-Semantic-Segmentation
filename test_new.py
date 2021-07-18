#coding=utf-8
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
                    default="/media/lcq/Data/modle_and_code/DataSet/RailGuard/bj_jpgs/")
parser.add_argument("--output_path", type=str, 
                    default="/media/lcq/Data/modle_and_code/DataSet/RailGuard/bj_output")
parser.add_argument("--weights_path",type=str, 
                    default="weights/0706R2646-o_unet/epoch042_acc0.990559.hdf5")
parser.add_argument("--model_name", type=str, default="unet")
parser.add_argument("--input_height", type=int, default=512)
parser.add_argument("--input_width", type=int, default=512)
parser.add_argument("--resize_op", type=int, default=2)
parser.add_argument("--classes", type=int, default=2)
# streetscape(12)(320x640), helen_small(11)(512x512), bbufdataset(2)
parser.add_argument("--mIOU", type=bool, default=True)
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
output_mask = os.path.join(output_path, 'mask')
mk_if_not_exits(output_mask)
output_viz = os.path.join(output_path, 'viz')
mk_if_not_exits(output_viz)

print('========================= lc info ===========================')
print('test images path : {}'.format(images_path))
print('test images num  : {}'.format(img_num))

for i in range(img_num):
    
    origin_img = cv2.imread(images[i])
    origin_h = origin_img.shape[0]
    origin_w = origin_img.shape[1]
	
    imgName = images[i].split('/')[-1].split('.')[0]
    mask_Name = '{}/{}_{}.png'.format(output_mask, imgName, model_name)
    viz_Name = '{}/{}_{}.png'.format(output_viz, imgName, model_name)

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
        
    # seg_img = cv2.resize(seg_img, (origin_w, origin_w), interpolation=cv2.INTER_NEAREST)
    
    # 后处理
    png_crop = png_alignment(origin_img, seg_img)
    img_viz = png_viz(origin_img, png_crop)
    
    # 保存掩码图片和可视化结果
    cv2.imwrite(mask_Name, png_crop)
    cv2.imwrite(viz_Name, img_viz)


print("======================= Test Success! =======================")
 
# mIOU
if iou:
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
    
    print('=================== iou info ===================')

    print('TP : {}'.format(tp))
    print('FP : {}'.format(fp))
    print('FN : {}'.format(fn))

    cl_wise_score = tp / (tp + fp + fn + EPS)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score * n_pixels_norm)
    mean_IOU = np.mean(cl_wise_score)

    print("frequency_weighted_IU: ", frequency_weighted_IU)
    print("mean IOU: ", mean_IOU)
    print("class_wise_IOU:", cl_wise_score)