import numpy as np
from probreg import features as ft
import time
import open3d as o3
from scipy.stats import multivariate_normal

filepath = '../data/bunny.pcd'
WAYMO=False

if WAYMO:
    from waymoutils import WaymoLIDARPair, convert_np_to_pc, WaymoLIDARVisCallback

def map_points_to_gmm(pts, means, weights, covs):
    dists = np.linalg.norm(means, axis=1)
    neworder = np.argsort(dists)
    temp_means = means[neworder, :]
    means = temp_means
    temp_weights = weights[neworder]
    weights = temp_weights
    temp_covs = covs[neworder]
    covs = temp_covs

    logprobs = np.zeros((pts.shape[0], means.shape[0]))
    for i in range(means.shape[0]):
        mvnorm = multivariate_normal(means[i, :], np.identity(3)*covs[i])
        logprobs[:, i] = mvnorm.logpdf(pts) * weights[i]

    #Each Point Gets Assigned to one GMM
    gmm_idxs = np.argmax(logprobs, axis=1)
    return gmm_idxs

if WAYMO:
    waymopair = WaymoLIDARPair(max_frames=100, as_pc = True, voxel_size = 0.7, filename='/home/mlab-dhruv/Desktop/pointclouds/waymodata/segment-10206293520369375008_2796_800_2816_800_with_camera_labels.tfrecord')
    vis = WaymoLIDARVisCallback()

n_gmm_components = 50
all_colors = np.array([np.random.uniform(0, 1, 3) for _ in range(n_gmm_components)])
fit_gmm_every = 10
num_iters = 0
while True:
    # 1: a)Load Source & Target Pointclouds b) Transform Target c) Downsample for speed
    if WAYMO:
        source_np, _, source, _, done = waymopair.next_pair()
        if done:
            break
    else:
        source = o3.read_point_cloud(filepath)
        source = o3.voxel_down_sample(source, voxel_size=0.003)
        source_np = np.asarray(source.points)

    #2: Runn GMM on Source
    if num_iters % fit_gmm_every == 0:
        start = time.time()
        gmm = ft.GMM(n_gmm_components=n_gmm_components)
        gmm.init()
        means, weights, covs = gmm.compute_cov(source_np)
        end = time.time()
        print(f"Num Points:{source_np.shape[0]}, Num GMM Components:{n_gmm_components}, Time Taken:{end - start}")
    gmm_idxs = map_points_to_gmm(source_np, means, weights, covs)

    #3: Visualize
    source_colors = all_colors[gmm_idxs, :]

    if WAYMO:
        vis(source_np, source_colors)
    else:
        vis = o3.Visualizer()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.0, 0.0167, 0.1186])

        source.colors = o3.utility.Vector3dVector(source_colors)

        vis.add_geometry(source)
        vis.run()
        break
    num_iters+=1