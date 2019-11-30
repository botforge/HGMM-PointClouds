from waymoutils import WaymoLIDARPair, convert_np_to_pc, WaymoLIDARVisCallback
import open3d as o3
import numpy as np
import transformations as trans
from probreg import gmmtree
from probreg import callbacks
import hgmm_utils

waymopair = WaymoLIDARPair(max_frames=150, as_pc = True, voxel_size = 2.0, filename='/home/mlab-dhruv/Desktop/pointclouds/waymodata/segment-10206293520369375008_2796_800_2816_800_with_camera_labels.tfrecord')

done = False
trans_so_far = np.identity(4)
vis = WaymoLIDARVisCallback() 
while True:
    prev_np, curr_np, prev_pc, curr_pc, done = waymopair.next_pair()
    if done:
        break
    cbs = [callbacks.Open3dVisualizerCallback(curr_pc, prev_pc)]
    tf_param, _ = gmmtree.registration_gmmtree(curr_pc, prev_pc, callbacks=cbs)
    # cbs[0].__del__()
    #Update Running Transformation Matrix
    # homo = trans.identity_matrix()
    # homo[:3, :3] = tf_param.rot
    # homo[:3, 3] = tf_param.t
    # trans_so_far = np.dot(trans_so_far, homo)

    # #Update curr_pc
    # homo_curr_np = np.ones((curr_np.shape[0], 4))
    # homo_curr_np[:, :3] = curr_np
    # homo_curr_np = np.dot(homo_curr_np, trans_so_far.T)
    # curr_np = homo_curr_np[:, :3]
    # print(homo_curr_np)
    # vis(curr_np)