import json
import os

import cv2
import numpy as np
import torch.utils.data as data

from utils.preproc import image_net_center as center_image, mask_depth_image, to_channel_first, resize, random_crop, center_crop, recursive_apply
from utils.io_utils import load_cam, load_pfm
from data.data_utils import dict_collate, Until, Cycle


class DTU(data.Dataset):

    def __init__(self, root, list_file, pair_file, num_src, read, transforms, fix_light=None):
        self.root = root
        with open(list_file) as f:
            self.data_list = json.load(f)
        with open(pair_file) as f:
            self.pair = json.load(f)
        self.num_scan = 1
        self.num_light = 7 if fix_light is None else 1
        self.fix_light = fix_light
        self.num_view = 49
        self.num_src = num_src
        self.total = self.num_scan * self.num_light * self.num_view
        self.read = read
        self.transforms = transforms
        print(f'Number of samples: {self.total}')

    def __len__(self):
        return self.total

    def __getitem__(self, i):
        ref_idx = i % self.num_view#31#
        src_idxs = self.pair[ref_idx][:self.num_src]#[22, 48, 0]#
        light_idx = i // self.num_view % self.num_light if self.fix_light is None else self.fix_light#3#
        scan_idx = int(os.environ['SCAN']) #i // self.num_view // self.num_light#2#
        ref = self.data_list[scan_idx][light_idx][ref_idx]
        scan = int(ref[0].split('/')[1].split('_')[0][4:])
        srcs = [self.data_list[scan_idx][light_idx][source_idx] for source_idx in src_idxs]
        masks = [f'occlusion2/scan{scan}/{ref_idx}_{src_idx}.png' for src_idx in src_idxs]
        skip = 0

        filenames = {'ref':ref[0], 'ref_cam':ref[1], 'srcs':[srcs[i][0] for i in range(self.num_src)], 'srcs_cam':[srcs[i][1] for i in range(self.num_src)], 'gt':ref[2], 'masks':masks}
        recursive_apply(filenames, lambda fn: os.path.join(self.root, fn))
        filenames['skip'] = skip

        sample = self.read(filenames)
        for transform in self.transforms:
            sample = transform(sample)
        return sample


def read(filenames, max_d, interval_scale):
    ref_name, ref_cam_name, srcs_name, srcs_cam_name, gt_name, masks_name, skip = [filenames[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'gt', 'masks', 'skip']]
    ref, *srcs = [cv2.imread(fn) for fn in [ref_name] + srcs_name]
    ref_cam, *srcs_cam = [load_cam(fn, max_d, interval_scale) for fn in [ref_cam_name] + srcs_cam_name]
    gt = np.expand_dims(load_pfm(gt_name), -1)
    # masks = [np.expand_dims(cv2.imread(fn, cv2.IMREAD_GRAYSCALE), -1) for fn in masks_name]
    masks = [(np.ones_like(gt)*255).astype(np.uint8) for fn in masks_name]
    return {
        'ref': ref,
        'ref_cam': ref_cam,
        'srcs': srcs,
        'srcs_cam': srcs_cam,
        'gt': gt,
        'masks': masks,
        'skip': skip
    }


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
    dataset = DTU(
        root, 'list/dtu_o_eval.json', 'list/pair.json', num_src,
        read=lambda filenames: read(filenames, preproc_args['max_d'], preproc_args['interval_scale']),
        transforms=[lambda sample: val_preproc(sample, preproc_args)],
        fix_light=3
    )
    loader = data.DataLoader(dataset, 1, collate_fn=dict_collate, shuffle=False)
    return dataset, loader
