import json
import os
from itertools import accumulate

import cv2
import numpy as np
import torch.utils.data as data

from utils.preproc import image_net_center as center_image, mask_depth_image, to_channel_first, resize, random_crop, center_crop, recursive_apply
from utils.preproc import random_brightness, random_contrast, motion_blur
from utils.io_utils import load_cam, load_pfm
from data.data_utils import dict_collate, Until, Cycle


def load_pair(file: str):
    with open(file) as f:
        lines = f.readlines()
    n_cam = int(lines[0])
    pairs = {}
    img_ids = []
    for i in range(1, 1+2*n_cam, 2):
        pair = []
        score = []
        img_id = lines[i].strip()
        pair_str = lines[i+1].strip().split(' ')
        n_pair = int(pair_str[0])
        for j in range(1, 1+2*n_pair, 2):
            pair.append(pair_str[j])
            score.append(float(pair_str[j+1]))
        img_ids.append(img_id)
        pairs[img_id] = {'id': img_id, 'index': i//2, 'pair': pair, 'score': score}
    pairs['id_list'] = img_ids
    return pairs


class Blended(data.Dataset):

    def __init__(self, root, list_file, num_src, read, transforms):
        super().__init__()
        self.root = root
        self.num_src = num_src
        with open(os.path.join(root, list_file)) as f:
            self.scene_list = [line.strip() for line in f.readlines()]
        self.pair_list = [
            load_pair(os.path.join(root, scene, 'cams', 'pair.txt'))
            for scene in self.scene_list
            ]
        self.index2scene = [[(i, j) for j in range(len(self.pair_list[i]['id_list']))] for i in range(len(self.scene_list))]
        self.index2scene = sum(self.index2scene, [])
        self.read = read
        self.transforms = transforms
        print(f'Number of samples: {len(self.index2scene)}')
    
    def _idx2filename(self, scene_idx, img_id, file_type):
        if img_id == 'dummy': return 'dummy'
        img_id = img_id.zfill(8)
        if file_type == 'img':
            return os.path.join(self.root, self.scene_list[scene_idx], 'blended_images', f'{img_id}.jpg')
        if file_type == 'cam':
            return os.path.join(self.root, self.scene_list[scene_idx], 'cams', f'{img_id}_cam.txt')
        if file_type == 'gt':
            return os.path.join(self.root, self.scene_list[scene_idx], 'rendered_depth_maps', f'{img_id}.pfm')
    
    def __len__(self):
        return len(self.index2scene)
    
    def __getitem__(self, i):
        scene_idx, ref_idx = self.index2scene[i]
        ref_id = self.pair_list[scene_idx]['id_list'][ref_idx]
        skip = 0
        if len(self.pair_list[scene_idx][ref_id]['pair']) < self.num_src:
            skip = 1
            print(f'sample {i} does not have enough sources')
        src_ids = self.pair_list[scene_idx][ref_id]['pair'][:self.num_src]
        if skip: src_ids += ['dummy'] * (self.num_src - len(src_ids))
        ref = self._idx2filename(scene_idx, ref_id, 'img')
        ref_cam = self._idx2filename(scene_idx, ref_id, 'cam')
        srcs = [self._idx2filename(scene_idx, src_id, 'img') for src_id in src_ids]
        srcs_cam = [self._idx2filename(scene_idx, src_id, 'cam') for src_id in src_ids]
        gt = self._idx2filename(scene_idx, ref_id, 'gt')
        filenames = {
            'ref': ref, 
            'ref_cam': ref_cam, 
            'srcs': srcs, 
            'srcs_cam': srcs_cam, 
            'gt': gt, 
            'skip': skip
        }

        sample = self.read(filenames)
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def read(filenames, max_d, interval_scale):
    ref_name, ref_cam_name, srcs_name, srcs_cam_name, gt_name, skip = [filenames[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'gt', 'skip']]
    ref, *srcs = [cv2.imread(fn) if fn != 'dummy' else None for fn in [ref_name] + srcs_name]
    srcs = [src if src is not None else np.ones_like(ref, dtype=np.uint8) for src in srcs]
    ref_cam, *srcs_cam = [load_cam(fn, max_d, interval_scale) if fn != 'dummy' else None for fn in [ref_cam_name] + srcs_cam_name]
    srcs_cam = [src_cam if src_cam is not None else np.ones_like(ref_cam, dtype=np.float32) for src_cam in srcs_cam]
    gt = np.expand_dims(load_pfm(gt_name), -1)
    masks = [(np.ones_like(gt)*255).astype(np.uint8) for _ in range(len(srcs))]
    if ref_cam[1,3,0] <= 0:
        skip = 1
        print(f'depth start <= 0')
    return {
        'ref': ref,
        'ref_cam': ref_cam,
        'srcs': srcs,
        'srcs_cam': srcs_cam,
        'gt': gt,
        'masks': masks,
        'skip': skip
    }


def train_preproc(sample, preproc_args):
    ref, ref_cam, srcs, srcs_cam, gt, masks, skip = [sample[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'gt', 'masks', 'skip']]

    ref, *srcs = [random_contrast(img, strength_range=[0.3, 1.5]) for img in [ref] + srcs]
    ref, *srcs = [random_brightness(img, max_abs_change=50) for img in [ref] + srcs]
    ref, *srcs = [motion_blur(img, max_kernel_size=3) for img in [ref] + srcs]

    ref, *srcs = [center_image(img) for img in [ref] + srcs]
    ref, ref_cam, srcs, srcs_cam, gt, masks = resize([ref, ref_cam, srcs, srcs_cam, gt, masks], preproc_args['resize_width'], preproc_args['resize_height'])
    ref, ref_cam, srcs, srcs_cam, gt, masks = center_crop([ref, ref_cam, srcs, srcs_cam, gt, masks], preproc_args['crop_width'], preproc_args['crop_height'])
    ref, *srcs, gt = to_channel_first([ref] + srcs + [gt])
    masks = to_channel_first(masks)

    srcs, srcs_cam, masks = [np.stack(arr_list, axis=0) for arr_list in [srcs, srcs_cam, masks]]

    return {
        'ref': ref,  # 3hw
        'ref_cam': ref_cam,  # 244
        'srcs': srcs,  # v3hw
        'srcs_cam': srcs_cam,  # v244
        'gt': gt,  # 1hw
        'masks': masks,  # v1hw
        'skip': skip  # scalar
    }


def get_train_loader(root, num_src, total_steps, batch_size, preproc_args, num_workers=0):
    dataset = Blended(
        root, 'training_list.txt', num_src,
        read=lambda filenames: read(filenames, preproc_args['max_d'], preproc_args['interval_scale']),
        transforms=[lambda sample: train_preproc(sample, preproc_args)]
    )
    loader = data.DataLoader(dataset, batch_size, collate_fn=dict_collate, shuffle=True, num_workers=num_workers, drop_last=True)
    cyclic_loader = Until(loader, total_steps)
    return dataset, cyclic_loader


def val_preproc(sample, preproc_args):
    ref, ref_cam, srcs, srcs_cam, gt, masks, skip = [sample[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'gt', 'masks', 'skip']]

    ref, *srcs = [center_image(img) for img in [ref] + srcs]
    ref, ref_cam, srcs, srcs_cam, gt, masks = resize([ref, ref_cam, srcs, srcs_cam, gt, masks], preproc_args['resize_width'], preproc_args['resize_height'])
    ref, ref_cam, srcs, srcs_cam, gt, masks = center_crop([ref, ref_cam, srcs, srcs_cam, gt, masks], preproc_args['crop_width'], preproc_args['crop_height'])
    ref, *srcs, gt = to_channel_first([ref] + srcs + [gt])
    masks = to_channel_first(masks)

    srcs, srcs_cam, masks = [np.stack(arr_list, axis=0) for arr_list in [srcs, srcs_cam, masks]]

    return {
        'ref': ref,  # 3hw
        'ref_cam': ref_cam,  # 244
        'srcs': srcs,  # v3hw
        'srcs_cam': srcs_cam,  # v244
        'gt': gt,  # 1hw
        'masks': masks,  # v1hw
        'skip': skip  # scalar
    }


def get_val_loader(root, num_src, preproc_args):
    dataset = Blended(
        root, 'validation_list.txt', num_src,
        read=lambda filenames: read(filenames, preproc_args['max_d'], preproc_args['interval_scale']),
        transforms=[lambda sample: val_preproc(sample, preproc_args)]
    )
    loader = data.DataLoader(dataset, 1, collate_fn=dict_collate, shuffle=False)
    return dataset, loader