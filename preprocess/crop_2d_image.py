
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mmengine

from collections import defaultdict
from functools import partial
from tqdm import tqdm

import random

import psutil
import time


class PointCloudToImageMapper(object):

    def __init__(self, visibility_threshold, cut_bound):
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound
        

    def compute_mapping(self, camera_to_world, coords, depth, image_dim, intrinsic):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        
        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = np.round(p).astype(int)     # simply round the projected coordinates
        inside_mask = (pi[0] >= self.cut_bound) * (pi[1] >= self.cut_bound) \
                    * (pi[0] < image_dim[0]-self.cut_bound) \
                    * (pi[1] < image_dim[1]-self.cut_bound)
        # print(inside_mask)
        if depth is not None:
            depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]]
                                    - p[2][inside_mask]) <= \
                                    self.vis_thres * depth_cur

            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2] > 0     # make sure the depth is in front
            inside_mask = front_mask * inside_mask
        mapping[0][inside_mask] = pi[1][inside_mask]
        mapping[1][inside_mask] = pi[0][inside_mask]
        mapping[2][inside_mask] = 1
        return mapping.T


def load_image(f):
    # print(f)
    img = cv2.imread(f + '.jpg')
    image_dim = (img.shape[1], img.shape[0])

    depth = cv2.imread(f + '.png', cv2.IMREAD_UNCHANGED) / 1000.0     # convert to meter
    depth = cv2.resize(depth, image_dim, interpolation=cv2.INTER_NEAREST)

    pose = np.loadtxt(f + '.txt')
    return img, depth, pose, image_dim


def plot_scene(pts, fig_name):    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    pts = np.vstack(pts)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.001)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(f'./{fig_name}.jpg', dpi=600)


def process_scene(scene_id, image_root, point2img_mapper, save_dir):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_cpu = process.cpu_percent(interval=None)  # 第一次调用返回0，需要等下一次
    start_mem = process.memory_info().rss / (1024 * 1024)  # MB


    print(f'Processing {scene_id} ...')
    scene_dir = os.path.join(image_root, scene_id)

    fs = os.listdir(scene_dir)
    image_list = list(set(f.split('.')[0] for f in fs if f.startswith('00')))
    image_list = sorted(image_list)
    print('Number of images: {}.'.format(len(image_list)))

    # load point cloud
    pts_dir = '../data/mask3d/scannet200'
    data = np.load(os.path.join(pts_dir, scene_id + '.npz'), allow_pickle=True)
    
    # load axis align matrix
    # line = open(f'/data1/ljw/VLM-Grounder-main/data/scannet/scans/{scene_id}/{scene_id}.txt').readline()
    with open(f'path_to_scannet/scannet/scans/{scene_id}/{scene_id}.txt', 'r') as file:
        line = file.readline()
        while not line.startswith('axisAlignment'):
            line = file.readline()
    axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    axis_inv_matrix = np.linalg.inv(axis_align_matrix)

    imgs = defaultdict(list)
    areas = defaultdict(list)

    intrinsic = np.loadtxt(f'{scene_dir}/intrinsic.txt')

    for image_name in image_list:
        f = os.path.join(scene_dir, image_name)
        img, depth, pose, image_dim = load_image(f)
        
        for obj_id, pts in enumerate(data['ins_pcds']):            
            pts = pts[:, :3]
            # recover to the initial point cloud coordinates
            pts = np.hstack((pts, np.ones((pts.shape[0], 1))))
            orig_pts = np.dot(pts, axis_inv_matrix.T)[:, :3]
           
            link = np.ones([orig_pts.shape[0], 4], dtype=int)
            link[:, 1:4] = point2img_mapper.compute_mapping(pose, orig_pts, depth, image_dim, intrinsic)
            link = link[:, 1:]
            valid_map = link[link[:, -1] != 0]

            empty = np.zeros((image_dim[1], image_dim[0], 3))
            empty[valid_map[:, 0], valid_map[:, 1]] = np.array([255, 255, 255])

            indices = np.nonzero(empty != 0)
            if len(indices[0]) < 20:
                continue

            crop_h = indices[0].max() - indices[0].min()
            crop_w = indices[1].max() - indices[1].min()

            if crop_h < 20 or crop_w < 20:
                continue

            crop_img = img[max(indices[0].min() - crop_h // 4, 0):indices[0].max() + crop_h // 4,
                           max(indices[1].min() - crop_w // 4, 0):indices[1].max() + crop_w // 4]
            
            imgs[str(obj_id)].append((crop_img, image_name))
            areas[str(obj_id)].append(crop_h * crop_w)


    print(f'Crop {len(imgs)} objects')
    
    k = 5
    for obj_id, area in areas.items():
        # select top-k
        num_imgs = len(imgs[obj_id])
        if num_imgs <= k:
            img_list = imgs[obj_id]  # 全部选中
        else:
            idx = random.sample(range(num_imgs), k)  # 随机选k个索引
            img_list = [imgs[obj_id][i] for i in idx]
        # idx = np.argsort(area)[-k:]
        # img_list = [imgs[obj_id][i] for i in idx]
        
        obj_path = os.path.join(save_dir, 'crop_images', scene_id, str(obj_id))
        os.makedirs(obj_path, exist_ok=True)
        
        for crop_img, image_name in img_list:
            cv2.imwrite(os.path.join(obj_path, f'img_{image_name}.jpg'), crop_img)
        
    end_time = time.time()
    end_cpu = process.cpu_percent(interval=None)  # 非精确，但可做相对参考
    end_mem = process.memory_info().rss / (1024 * 1024)
    print(f"[{scene_id}] Time: {end_time - start_time:.2f}s | "
        f"Memory: {start_mem:.2f} MB → {end_mem:.2f} MB | "
        f"CPU: {end_cpu:.1f}%")
            

if __name__ == '__main__':
    image_root = '../data/posed_images_20frame'
    save_dir = '../data'
    scan_id_file = "../data/scannetv2_val.txt"
    # scene_list = ['scene0015_00']
    scene_list = list(set([x.strip() for x in open(scan_id_file, 'r')]))

    visibility_threshold = 0.25     # threshold for the visibility check
    cut_num_pixel_boundary = 0     # do not use the features on the image boundary
    
    # calculate image pixel-3D points correspondances
    point2img_mapper = PointCloudToImageMapper(visibility_threshold=visibility_threshold,
                                               cut_bound=cut_num_pixel_boundary)

    process_func = partial(process_scene,
                           image_root=image_root,
                           point2img_mapper=point2img_mapper,
                           save_dir=save_dir)
    

    # for scene_id in tqdm(scene_list):
    #     process_func(scene_id)
    mmengine.track_parallel_progress(
        func=process_func,
        tasks=scene_list,
        nproc=1  
    )
