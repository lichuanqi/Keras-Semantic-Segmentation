# coding=utf-8
import argparse
import glob
import os
from re import T
import yaml

import keras
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau)
import keras as K
import data
import Models
from Models import build_model
from utils.utils import *
from metrics import metrics
from losses import LOSS_FACTORY

from keras.callbacks import History

history = History()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="unet")
parser.add_argument("--dataset", type=str,default="D:/CodePost/Miandan500") 
parser.add_argument("--n_classes", type=int, default=3)
parser.add_argument("--epochs", type=int, default=100)

parser.add_argument("--input_height", type=int, default=640)
parser.add_argument("--input_width", type=int, default=640)

parser.add_argument('--validate', type=bool, default=True)
parser.add_argument("--resize_op", type=int, default=2)

parser.add_argument("--train_batch_size", type=int, default=4)
parser.add_argument("--val_batch_size", type=int, default=4)

parser.add_argument("--train_save_path", type=str, default="runs/")
parser.add_argument("--exp_name", type=str, default='20220824_MD500x10')
# parser.add_argument("--resume", type=str, default="weights/0705_unet_R294/unet.48-0.987546.hdf5")
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--optimizer_name", type=str, default="adam")
parser.add_argument("--image_init", type=str, default="divide")
parser.add_argument("--multi_gpus", type=bool, default=False)
parser.add_argument("--gpu_count", type=int, default=1)
parser.add_argument("--loss", type=str, default='dice')
args = parser.parse_args()

exp_name = args.exp_name
train_save_path = args.train_save_path
epochs = args.epochs
load_weights = args.resume
model_name = args.model_name
optimizer_name = args.optimizer_name
image_init = args.image_init
multi_gpus = args.multi_gpus
gpu_count = args.gpu_count

# 设置log的存储位置
log_dir = os.path.join(train_save_path, '{}_{}'.format(exp_name, model_name))
mk_if_not_exits(log_dir)

# 保存参数信息至args.yaml
args_save_path = os.path.join(log_dir, 'args.yaml')
args_dict = args.__dict__
with open(args_save_path, 'w') as f:
    f.write(yaml.dump(args_dict, allow_unicode=True))

# 训练集和验证集路径
train_images = os.path.join(args.dataset, "train_image_aug")
train_segs = os.path.join(args.dataset, "train_label_aug")
train_batch_size = args.train_batch_size

validate = args.validate
if validate:
    val_images = os.path.join(args.dataset, "val_image")
    val_segs = os.path.join(args.dataset, "val_label")
    val_batch_size = args.val_batch_size

# 统计一下训练集/验证集样本数
images = glob.glob(os.path.join(train_images, "*.jpg")) + \
         glob.glob(os.path.join(train_images, "*.png")) 
num_train = len(images)

images = glob.glob(os.path.join(val_images, "*.jpg")) + \
         glob.glob(os.path.join(val_images, "*.png"))
num_val = len(images)

# 数据参数
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
resize_op = args.resize_op

model = build_model(model_name,
                    n_classes,
                    input_height=input_height,
                    input_width=input_width)

# 需要保证脚本开头指定的gpu个数和现在要使用的gpu数量相等
if multi_gpus == True:
	parallel_model = multi_gpu_model(model, gpus=gpu_count)

print('========================= lc info ===========================')
print('Train path and num : {}/{}'.format(train_images, num_train))
print('Val path and num   : {}/{}'.format(val_images, num_val))
print('Model flops        : {}'.format(get_flops(model)))
print('Log save path      : {}'.format(log_dir))

# 将网络权值以图片格式保持在tensorboard中显示
tb_cb = keras.callbacks.TensorBoard(log_dir=log_dir,
                                    histogram_freq=0,
                                    write_images=True,
                                    embeddings_freq=0, embeddings_layer_names=None,
                                    update_freq=500)

# 学习率
early_stop = EarlyStopping(monitor='val_acc', 
                           min_delta=0.0001, 
                           patience=30, 
                           verbose=1,
                           mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_acc',
                              factor=0.1,
                              patience=6,
                              mode='auto',
                              min_delta=0.0001,
                              verbose=1,
                              min_lr=0)
# 保存日志
log_file_path = os.path.join(log_dir, 'log.csv')
csv_logger = CSVLogger(log_file_path, append=False)

# 保存权重文件的路径
weights_dir = os.path.join(train_save_path, '{}_{}/weights'.format(exp_name, model_name))
mk_if_not_exits(weights_dir)
model_names = weights_dir + '/epoch{epoch:02d}_acc{acc:2f}_valacc{val_acc:2f}.hdf5'

# 保存权重
if multi_gpus == True:
    model_checkpoint = ParallelModelCheckpoint(model, filepath=model_names,
                                    monitor='val_acc',
                                    save_best_only=True,
                                    save_weights_only=True,
                                    mode='auto')
else:
    model_checkpoint = ParallelModelCheckpoint(model, filepath=model_names,
                                   monitor='val_acc',
                                   save_best_only=True,
                                   save_weights_only=True,
                                   mode='auto')

call_backs = [model_checkpoint, csv_logger, early_stop, reduce_lr, tb_cb]

# 损失函数
loss_func  = LOSS_FACTORY[args.loss]

if multi_gpus == True:
	parallel_model.compile(loss=loss_func,
              	optimizer=optimizer_name,
              	metrics=['accuracy', 'iou_score', 'dice_score', 'f1_score', 'f2_score'])

else:
	model.compile(loss=loss_func,
              	optimizer=optimizer_name,
              	metrics=['accuracy', 'iou_score', 'dice_score', 'f1_score', 'f2_score'])

# 加载预训练权重
if len(load_weights) > 0:
    if multi_gpus == True:
        parallel_model.load_weights(load_weights)
    else:
        model.load_weights(load_weights)

# 打印网络结构
# print("Model output shape : ", model.output_shape)
model.summary() 

output_height = model.outputHeight
output_width = model.outputWidth

# data generator
train_ge = data.imageSegmentationGenerator(train_images, train_segs,
                                           train_batch_size, n_classes,
                                           input_height, input_width, resize_op,
                                           output_height, output_width,
                                           image_init)

if validate:
    val_ge = data.imageSegmentationGenerator(val_images, val_segs,
                                             val_batch_size, n_classes,
                                             input_height, input_width, resize_op,
                                             output_height, output_width,
                                             image_init)

# 开始训练
if not validate:
    history = model.fit_generator(train_ge,
                        epochs=epochs,
                        callbacks=call_backs,
                        steps_per_epoch=int(num_train / train_batch_size),
                        verbose=1,
                        shuffle=True,
                        max_q_size=10,
                        workers=1)
else:
    history = model.fit_generator(train_ge,
                        validation_data=val_ge,
                        epochs=epochs,
                        callbacks=call_backs,
                        verbose=1,
                        steps_per_epoch=int(num_train / train_batch_size),
                        shuffle=True,
                        validation_steps=int(num_val / val_batch_size),
                        max_q_size=10,
                        workers=1)