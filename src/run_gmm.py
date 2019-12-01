import numpy as np
from probreg import features as ft
import open3d as o3
from scipy.stats import multivariate_normal

filepath = '../data/bunny.pcd'

def map_points_to_gmm(pts, means, weights, covs):
    logprobs = np.zeros((pts.shape[0], means.shape[0]))
    for i in range(means.shape[0]):
        mvnorm = multivariate_normal(means[i, :], np.identity(3)*covs[i])
        logprobs[:, i] = mvnorm.logpdf(pts) * weights[i]

    #Each Point Gets Assigned to one GMM
    gmm_idxs = np.argmax(logprobs, axis=1)
    return gmm_idxs

# 1: a)Load Source & Target Pointclouds b) Transform Target c) Downsample for speed
source = o3.read_point_cloud(filepath)
source = o3.voxel_down_sample(source, voxel_size=0.003)
source_np = np.asarray(source.points)

#2: Runn GMM on Source
gmm = ft.GMM(n_gmm_components=100)
gmm.init()
means, weights, covs = gmm.compute_cov(source_np)
gmm_idxs = map_points_to_gmm(source_np, means, weights, covs)

#3: Visualize
vis = o3.Visualizer()
vis.create_window()
opt = vis.get_render_option()
opt.background_color = np.asarray([0.0, 0.0267, 0.1286])

all_colors = np.array([np.random.uniform(0, 1, 3) for _ in range(means.shape[0])])
source_colors = all_colors[gmm_idxs, :]
source.colors = o3.utility.Vector3dVector(source_colors)

vis.add_geometry(source)
vis.run()