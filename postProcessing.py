#==============================================
# 语义分割检测结果后处理和可视化
# lichuan
# lc@dlc618.com
# 2021.7.2
#==============================================

import os,glob

import cv2
import numpy as np
import tensorflow as tf


def png_alignment(jpg, png):
    '''
    功能：
        图片和检测结果尺寸对齐
    输入：
        jpg：检测图片
        png：检测结果
    输出：
        png：对齐后的掩码图片
    '''

    png_resize = cv2.resize(png, (1920,1920), interpolation=cv2.INTER_NEAREST)
    png_crop = png_resize[419:1499,:]

    return png_crop


def png_viz(jpg,png_crop):

    # 平滑一下
    png_crop = cv2.medianBlur(png_crop, 5)

    # 连通域
    contours, hierarchy = cv2.findContours(png_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # ============== 联通域单独填充 ==============
    # img_fill = jpg.copy()
    # for i in range(len(contours)):
    #     # 轮廓
    #     # cv2.drawContours(img, contours[i], -1, (0, 0, 255), 3)
    #     # 填充
    #    cv2.fillConvexPoly(img_fill, contours[i], (0, 0, 255))
    
    # # 按权重叠加
    # img_viz = cv2.addWeighted(jpg, 0.6, img_fill, 0.4, 0)

    # ================ 最大联通域 ================
    img_fill = jpg.copy()
    areas = []
    for i in range(len(contours)):
        areas.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(areas)
    
    # 填充
    cv2.fillConvexPoly(img_fill, contours[max_idx], (0, 0, 255))
    
    # 按权重叠加
    img_viz = cv2.addWeighted(jpg, 0.6, img_fill, 0.4, 0)

    # 获取轮廓凸包绘制
    hull_points = cv2.convexHull(contours[max_idx])
    cv2.polylines(img_viz, [hull_points], True, (0, 255, 0), 2)

    
    # =============== 合并所有联通域 ===============
    # 待实现


    return img_viz


def main(jpg_path, png_path):

    jpg = cv2.imread(jpg_path)
    png = cv2.imread(png_path, 0)

    png_crop = png_alignment(jpg, png)
    img_viz = png_viz(jpg, png_crop)

    cv2.imshow('png_crop', img_viz)

    if cv2.waitKey(-1) and 0xFF == ord('q'):
        cv2.destroyAllWindows()

    # 

if __name__=='__main__':
    main(
        jpg_path = 'data/RailGuard/test/bj00008.jpg',
        png_path = 'data/RailGuard/output-0709/bj00008_unet.png'
    )