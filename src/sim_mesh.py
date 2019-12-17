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
    waymopair = f110LIDARPair(max_frames=100, as_pc = True, voxel_size = 0.001)

while True:
    if F110:
        source_np, _, source, _, done = waymopair.next_pair()
        if done:
            break 

    pcd = convert_np_to_pc(source_np.copy())

    pcd.estimate_normals()

    # estimate radius for rolling ball
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist   

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))

    tmesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), vertex_normals=np.asarray(mesh.vertex_normals))
    tmesh.show()