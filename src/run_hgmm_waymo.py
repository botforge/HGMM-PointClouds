import copy
from waymoutils import WaymoLIDARPair, convert_np_to_pc, WaymoLIDARVisCallback
import open3d as o3
import numpy as np
import transformations as trans
from probreg import gmmtree, cpd, l2dist_regs
from probreg import callbacks
from collections import deque
import hgmm_utils

waymopair = WaymoLIDARPair(max_frames=150, as_pc = True, voxel_size = 0.5, filename='/home/mlab-dhruv/Desktop/pointclouds/waymodata/segment-10206293520369375008_2796_800_2816_800_with_camera_labels.tfrecord')

done = False
all_trans = []
vis = WaymoLIDARVisCallback() 
idx = 0
while True:
    prev_np, curr_np, prev_pc, curr_pc, done = waymopair.next_pair()
    if done:
        break

    #1: Compute L2 Gaussian Registration & Transform chain
    tf_param = l2dist_regs.registration_svr(prev_pc, curr_pc)
    all_trans.append(copy.deepcopy(tf_param.inverse()))
    if idx == 0:
        result = copy.deepcopy(curr_pc)
    else:
        result = convert_np_to_pc(np.linspace(-0.03, 0.01, num=20*3).reshape(-1, 3))
        result.colors = o3.utility.Vector3dVector(np.ones((20*3)).reshape(-1, 3) * np.array([1, 0, 0]))
    # idx+=1
    for tf in reversed(all_trans):
        result.points = tf.transform(result.points)
    vis(result, addpc=True)