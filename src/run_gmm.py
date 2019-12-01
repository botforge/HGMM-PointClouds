import numpy as np
from probreg import features as ft
import open3d as o3
filepath = '../data/bunny.pcd'


#1: a)Load Source & Target Pointclouds b) Transform Target c) Downsample for speed
source = o3.read_point_cloud(filepath)
source = o3.voxel_down_sample(source, voxel_size=0.003)
source_np = np.asarray(source.points)

#2: Runn GMM on Source
gmm = ft.GMM()
gmm.init()
means, weights, covs = gmm.compute_cov(source_np)
# print(f"means:{means.shape}, weights:{weights.shape}, covs:{covs.shape}")