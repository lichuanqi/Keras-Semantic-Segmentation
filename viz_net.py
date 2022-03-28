# ===================================
# 网络结构可视化及参数统计
# 2021.07.19
# ===================================

import argparse
import os

import tensorflow as tf

from Models import build_model
from utils.utils import get_flops, mk_if_not_exits
from keras.utils import plot_model

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="nunet")
parser.add_argument("--n_classes", type=int, default=2)
parser.add_argument("--input_height", type=int, default=640)
parser.add_argument("--input_width", type=int, default=640)
parser.add_argument("--save", action='store_false')
parser.add_argument("--save_path", type=str, default="expdata/0328_nunet_viz")
args = parser.parse_args()

model_name = args.model_name
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
save_jpg = args.save
save_path = args.save_path

# 检测 cuda 是否可用
device = 'gpu' if tf.test.is_gpu_available() else 'cpu'
print('Device: {}'.format(device))

model = build_model(model_name,
					n_classes,
					input_height=input_height,
					input_width=input_width)
# 网络结构
model.summary() 

if save_jpg == True:

	# 保存路径不存在的话新建一个
	mk_if_not_exits(save_path)

	# 保存网络结构到 YAML
	yaml_string = model.to_yaml()
	save_yaml_name = save_path + '/{}.yaml'.format(model_name)
	# Linux
	# save_yaml_name = os.path.join(save_path, '{}.yaml'.format(model_name))
	with open(save_yaml_name, "w") as f:
		f.write(yaml_string)

	# 保存网络结构图
	save_jpg_name = save_path + '/{}_network.jpg'.format(model_name)
	# Linux
	save_jpg_name = os.path.join(save_path, '{}_network.jpg'.format(model_name))
	plot_model(model, to_file=save_jpg_name, show_shapes=True)

	print('已保存到: {}'.format(save_path))