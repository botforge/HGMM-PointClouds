import copy
from waymoutils import WaymoLIDARPair, convert_np_to_pc, WaymoLIDARVisCallback
import open3d as o3
import numpy as np
import transformations as trans
from probreg import gmmtree, cpd, l2dist_regs
from probreg import callbacks
from collections import deque
import hgmm_utils

waymopair = WaymoLIDARPair(max_frames=50, as_pc = True, voxel_size = 2.0, filename='/home/mlab-dhruv/Desktop/pointclouds/waymodata/segment-10206293520369375008_2796_800_2816_800_with_camera_labels.tfrecord')
TRANSLATE_02x = np.identity(4)
TRANSLATE_02x[0, 3] = 0.5

done = False
all_trans = deque([])
vis = WaymoLIDARVisCallback() 
while True:
    prev_np, curr_np, prev_pc, curr_pc, done = waymopair.next_pair()
    if done:
        break

    #1: Compute L2 Gaussian Registration & Transform chain
    tf_param = l2dist_regs.registration_svr(curr_pc, prev_pc)
    all_trans.append(tf_param)
    result = copy.deepcopy(curr_pc)
    # result.points = tf_param.transform(result.points)

    for tf in all_trans.reverse():
        result.points = tf.transform(result.points)
    vis(result, addpc=True)