try:
    import tensorflow as tf
    tf.enable_eager_execution()
except:
    print("NO TF")
import numpy as np
import pickle
import open3d as o3

try:
    from waymo_open_dataset.utils import range_image_utils
    from waymo_open_dataset.utils import transform_utils
    from waymo_open_dataset.utils import  frame_utils
    from waymo_open_dataset import dataset_pb2 as open_dataset
except:
    print("NO WAYMOOD")

class WaymoLIDARVisCallback(object):
    """Display Stream of LIDAR Points & GMM

    Args:
        save (bool, optional): If this flag is True,
            each iteration image is saved in a sequential number.
        keep_window (bool, optional): If this flag is True,
            the drawing window blocks after registration is finished.
    """
    def __init__(self, save=False,
                 keep_window=True):
        #Create Visualizer Window etc
        self._vis = o3.Visualizer()
        self._vis.create_window()
        opt = self._vis.get_render_option()
        opt.background_color = np.asarray([0.0, 0.0167, 0.1186])
        opt.point_size = 1.5
        self._save = save
        self._keep_window = keep_window
        self._cnt = 0
        self._currpc = o3.geometry.PointCloud()

    def __del__(self):
        if self._keep_window:
            self._vis.run()
        self._vis.destroy_window()

    def np_to_pc(self, pts, colors=None):
        self._currpc.points = o3.utility.Vector3dVector(pts)
        if colors is not None:
            self._currpc.colors = o3.utility.Vector3dVector(colors)
        
    def __call__(self, newpoints, colors=None, addpc=False):
        if addpc:
            self._vis.add_geometry(newpoints)
        else:
            #Convert Points into pointcloud 
            self.np_to_pc(newpoints, colors)
            if(self._cnt == 0):
                self._vis.add_geometry(self._currpc)
        self._vis.update_geometry()
        self._vis.poll_events()
        self._vis.update_renderer()
        if self._save:
            self._vis.capture_screen_image("image_%04d.jpg" % self._cnt)
        self._cnt += 1
    
def convert_np_to_pc(np_pts):
    pc = o3.geometry.PointCloud() 
    pc.points = o3.utility.Vector3dVector(np_pts)
    return pc

"""
Show Camera Image Along with Labels - Copied from Waymo Open Dataset Tutorial Colab
"""
def show_camera_image(camera_image, camera_labels, layout):
    ax = plt.subplot(*layout)
    # Draw the camera labels.
    for camera_labels in frame.camera_labels:
        # Ignore camera labels that do not correspond to this camera.
        if camera_labels.name != camera_image.name:
            continue

        # Iterate over the individual labels.
        for label in camera_labels.labels:
          # Draw the object bounding box.
          ax.add_patch(patches.Rectangle(
            xy=(label.box.center_x - 0.5 * label.box.length,
                label.box.center_y - 0.5 * label.box.width),
            width=label.box.length,
            height=label.box.width,
            linewidth=1,
            edgecolor='red',
            facecolor='none'))
        
        # Show the camera image.
        ax.imshow(tf.image.decode_jpeg(camera_image.image))
        plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
        plt.grid(False)
        plt.axis('off')
        
"""
Stream Camera Images inside a list of Frame Protobufs Across Cameras
"""
def stream_camera_images(framelist):
    for frame_idx, frame in enumerate(framelist):
        for index, image in enumerate(frame.images):
            show_camera_image(image, frame.camera_labels, [3, 3, index+1])
        plt.pause(0.1)

def extract_waymo_data(filename, max_frames=150):
    #FILENAME = '../waymodata/segment-10206293520369375008_2796_800_2816_800_with_camera_labels.tfrecord'
    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    #EXTRACT FRAME AND RANGE DATA
    framelist = []
    for i, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        framelist.append(frame)
        if i > max_frames: break

    range_imagelist = []
    camera_imagelist = []
    range_image_toplist = []
    for frame in framelist:
        range_images, camera_images, range_image_top  = frame_utils.parse_range_image_and_camera_projection(
        frame)
        range_imagelist.append(range_images)
        camera_imagelist.append(camera_images)
        range_image_toplist.append(range_image_top)
    return framelist, range_imagelist, camera_imagelist, range_image_toplist

class f110LIDARPair(object):
    """Get a pair of LIDAR frames from the F110 Simulator (prev, curr)"""
    def __init__(self, as_pc = False, voxel_size=1.0, skip=0, max_frames = 150, filename='lidarlist.pkl'
):
        self._ptr = 1
        #Open LiDAR Range Frames
        with open(filename, 'rb') as f:
            self.framelist = pickle.load(f)
        self.as_pc = as_pc

        #Put all Points into a list as pointclouds
        self.points_list = []
        self.pc_list = []
        for i in range(len(self.framelist)):
            pc_np = self.get_pc(i)
            if as_pc:
                pc = convert_np_to_pc(pc_np)
                pc = o3.voxel_down_sample(pc, voxel_size=voxel_size)
                self.pc_list.append(pc)
                self.points_list.append(np.asarray(pc.points))
            else:
                self.points_list.append(pc_np)

    def get_pc(self, i):
        return self.framelist[i]

    def next_pair(self):
        if(self._ptr < len(self.points_list)):
            ret_np = (self.points_list[self._ptr - 1], self.points_list[self._ptr])
            if self.as_pc:
                ret_pc = (self.pc_list[self._ptr - 1], self.pc_list[self._ptr])
            else:
                ret_pc = (None, None)
            self._ptr+=1
            return ret_np[0], ret_np[1], ret_pc[0], ret_pc[1], False
        else:
            return None, None, None, None, True


class WaymoLIDARPair(object):
    """Get a pair of LIDAR frames (prev, curr)"""
    def __init__(self, as_pc = False, voxel_size=1.0, skip=0, max_frames = 150, filename='../waymodata/segment-10206293520369375008_2796_800_2816_800_with_camera_labels.tfrecord'
):
        self._ptr = 1
        self.framelist, self.range_imagelist, self.camera_imagelist, self.range_image_toplist = extract_waymo_data(filename, max_frames=max_frames)
        self.as_pc = as_pc

        #Put all Points into a list
        self.points_list = []
        self.pc_list = []
        for i in range(len(self.framelist)):
            pc_np = self.get_pc(i)
            if as_pc:
                pc = convert_np_to_pc(pc_np)
                pc = o3.voxel_down_sample(pc, voxel_size=voxel_size)
                self.pc_list.append(pc)
                self.points_list.append(np.asarray(pc.points))
            else:
                self.points_list.append(pc_np)

    def get_pc(self, i):
        points, _ = frame_utils.convert_range_image_to_point_cloud(
                                self.framelist[i],
                                self.range_imagelist[i],
                                self.camera_imagelist[i],
                                self.range_image_toplist[i])
        points_all = np.concatenate(points, axis=0)
        return points_all

    def next_pair(self):
        if(self._ptr < len(self.points_list)):
            ret_np = (self.points_list[self._ptr - 1], self.points_list[self._ptr])
            if self.as_pc:
                ret_pc = (self.pc_list[self._ptr - 1], self.pc_list[self._ptr])
            else:
                ret_pc = (None, None)
            self._ptr+=1
            return ret_np[0], ret_np[1], ret_pc[0], ret_pc[1], False
        else:
            return None, None, None, None, True

    # callback = WaymoLIDARVisCallback()
    # for i in range(len(framelist)):
    #     show_pc(framelist[i], range_imagelist[i], camera_imagelist[i], range_image_toplist[i], callback)
    #     # time.sleep(0.01)