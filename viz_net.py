# ===================================
# 网络结构可视化及参数统计
# 2021.07.19
# ===================================

import argparse
import os

from Models import build_model
from utils.utils import get_flops
from keras.utils import plot_model

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="attunet")
parser.add_argument("--n_classes", type=int, default=2)
parser.add_argument("--input_height", type=int, default=640)
parser.add_argument("--input_width", type=int, default=640)
parser.add_argument("--save_path", type=str, default="expdata/")
args = parser.parse_args()

model_name = args.model_name
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
save_path = args.save_path

model = build_model(model_name,
                    n_classes,
                    input_height=input_height,
                    input_width=input_width)

print("Model output shape : ", model.output_shape)
model.summary() 

# save_name = os.path.join(save_path, '{}_network.jpg'.format(model_name))
# plot_model(model, to_file=save_name)