import numpy as np
from probreg import features as ft
import time
import open3d as o3d
from scipy.stats import multivariate_normal
import trimesh

filepath = '../data/bunny.pcd'
pcd = o3d.io.read_point_cloud(filepath)
pcd.estimate_normals()

# estimate radius for rolling ball
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 1.5 * avg_dist   

mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))

trimesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles), vertex_normals=np.asarray(mesh.vertex_normals))
trimesh.show()