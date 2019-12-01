import copy
import numpy as np
import transformations as trans
import open3d as o3
from probreg import gmmtree, cpd, l2dist_regs
from probreg import callbacks
import hgmm_utils as utils

filepath = '../data/bunny.pcd'

#TEST TRANSFORMS
TRANSLATE_02x = np.identity(4)
TRANSLATE_02x[0, 3] = 0.2

TRANSLATE_005x = np.identity(4)
TRANSLATE_005x[0, 3] = 0.05

th = np.deg2rad(30)
ROT_30z = np.identity(4)
ROT_30z = np.array([[np.cos(th), -np.sin(th), 0.0, 0.0],
                           [np.sin(th), np.cos(th), 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])

th = np.deg2rad(15)
ROT_15z = np.identity(4)
ROT_15z = np.array([[np.cos(th), -np.sin(th), 0.0, 0.0],
                           [np.sin(th), np.cos(th), 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])

all_trans = []

#1: a)Load Source & Target Pointclouds b) Transform Target c) Downsample for speed
source = o3.read_point_cloud(filepath)
target = copy.deepcopy(source)
target.transform(TRANSLATE_02x)
target.transform(ROT_30z)
source = o3.voxel_down_sample(source, voxel_size=0.003)
target = o3.voxel_down_sample(target, voxel_size=0.003)

#2: Compute L2 Gaussian Registration 
# cbs = [callbacks.Open3dVisualizerCallback(source, target)]
tf_param = l2dist_regs.registration_svr(source, target)
result = copy.deepcopy(source)
# result.points = tf_param.transform(result.points)

for i in range(4):
    result.transform(TRANSLATE_005x)
result.transform(ROT_15z)
result.transform(ROT_15z)

print(tf_param.rot)
print(tf_param.t)

#3: draw result
source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
result.paint_uniform_color([0, 0, 1])
o3.draw_geometries([source, target, result])