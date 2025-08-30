import base64
from collections import defaultdict
import json
import logging
import numpy as np
import os
import torch

from PIL import Image
from io import BytesIO


def load_seg_inst(scene_id):
    root_dir = '../data/mask3d/scannet200'
    data = np.load(os.path.join(root_dir, scene_id + '.npz'), allow_pickle=True)
    ins_labels = list(data['ins_labels'])
    ins_scores = [float(x) for x in data['ins_scores']]
    
    ins_locs = []
    scene_pc = []
    for obj in data['ins_pcds']:
        if obj.shape[0] == 0:
            obj = np.zeros((1, 6))
        obj_pcd = obj[:, :3]
        scene_pc.append(obj_pcd)
        obj_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
        obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
        ins_locs.append(np.concatenate([obj_center, obj_size], 0))

    scene_pc = np.concatenate(scene_pc, 0)
    center = (scene_pc.max(0) + scene_pc.min(0)) / 2
    
    return ins_labels, ins_locs, ins_scores, center


def load_pc(scene_id):
    root_dir = '../data/referit3d/scan_data'
    pcds, _, _, instance_labels = torch.load(
        os.path.join(root_dir, 'pcd_with_global_alignment', '%s.pth' % scene_id))
    inst_to_name = json.load(open(os.path.join(root_dir, 'instance_id_to_name', '%s.json' % scene_id)))

    obj_labels = []
    inst_locs = []
    obj_ids = []
    
    for i, obj_label in enumerate(inst_to_name):
        if obj_label in ['wall', 'floor', 'ceiling']:
            continue
        mask = instance_labels == i
        assert np.sum(mask) > 0, 'scene: %s, obj %d' % (scene_id, i)
        
        obj_pcd = pcds[mask]
        obj_center = (obj_pcd[:, :3].max(0) + obj_pcd[:, :3].min(0)) / 2
        obj_size = obj_pcd[:, :3].max(0) - obj_pcd[:, :3].min(0)
        inst_locs.append(np.concatenate([obj_center, obj_size], 0))

        obj_labels.append(obj_label)
        obj_ids.append(i)

    return obj_ids, obj_labels, inst_locs


def calc_iou(box_a, box_b):
    max_a = box_a[0:3] + box_a[3:6] / 2
    max_b = box_b[0:3] + box_b[3:6] / 2
    min_max = np.array([max_a, max_b]).min(0)

    min_a = box_a[0:3] - box_a[3:6] / 2
    min_b = box_b[0:3] - box_b[3:6] / 2
    max_min = np.array([min_a, min_b]).max(0)
    if not ((min_max > max_min).all()):
        return 0.0

    intersection = (min_max - max_min).prod()
    vol_a = box_a[3:6].prod()
    vol_b = box_b[3:6].prod()
    union = vol_a + vol_b - intersection
    
    return 1.0 * intersection / union


def create_logger(exp_name):
    # Create a custom logger
    logger = logging.getLogger('seq-vlm')

    # Set the log level of logger to DEBUG
    logger.setLevel(logging.DEBUG)

    # Create handlers for writing to a file and logging to console
    file_handler = logging.FileHandler(f'../logs/{exp_name}.log', mode='w')
    console_handler = logging.StreamHandler()

    # Create formatters and add them to the handlers
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

    return logger


def encode_image_to_base64(image_path):
    with Image.open(image_path) as image:
        buf = BytesIO()
        image.save(buf, format='JPEG')
        byte_data = buf.getvalue()
        base64_str = base64.b64encode(byte_data).decode('utf-8')
        return base64_str