import copy
import numpy as np
import transformations as trans
import open3d as o3
from probreg import gmmtree, cpd, l2dist_regs
from probreg import callbacks
import hgmm_utils as utils

filepath = '../data/bunny.pcd'

#TEST TRANSFORMS
TRANSLATE_1x = np.identity(4)
TRANSLATE_1x[0, 3] = 0.2

#1: a)Load Source & Target Pointclouds b) Transform Target c) Downsample for speed
source = o3.read_point_cloud(filepath)
target = copy.deepcopy(source)
target.transform(TRANSLATE_1x)
source = o3.voxel_down_sample(source, voxel_size=0.003)
target = o3.voxel_down_sample(target, voxel_size=0.003)

#2: Compute L2 Gaussian Registration
# cbs = [callbacks.Open3dVisualizerCallback(source, target)]
tf_param = l2dist_regs.registration_svr(source, target)

print(tf_param.rot)
print(tf_param.t)