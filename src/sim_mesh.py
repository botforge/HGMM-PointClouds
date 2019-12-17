import numpy as np
import open3d as o3d
import trimesh


WAYMO=False
F110=True

if WAYMO or F110:
    from waymoutils import f110LIDARPair, WaymoLIDARPair, convert_np_to_pc, WaymoLIDARVisCallback

if WAYMO:
    waymopair = WaymoLIDARPair(max_frames=100, as_pc = True, voxel_size = 0.7, filename='/home/mlab-dhruv/Desktop/pointclouds/waymodata/segment-10206293520369375008_2796_800_2816_800_with_camera_labels.tfrecord')
    vis = WaymoLIDARVisCallback()
elif F110:
    waymopair = f110LIDARPair(max_frames=2, as_pc = True, voxel_size = 0.001)
    vis = WaymoLIDARVisCallback(asmesh=True)

while True:
    if F110:
        source_np, _, source, _, done = waymopair.next_pair()
        if done:
            break 
    vis(source_np)