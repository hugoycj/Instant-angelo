import os

import cv2
import numpy as np
import torch.utils.data as data

from utils.io_utils import load_cam, load_pfm, cam_adjust_max_d
from utils.preproc import to_channel_first, resize, center_crop, image_net_center as center_image
from data.data_utils import dict_collate


def load_pair(file: str, min_views: int=None):
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
        if min_views is not None and n_pair < min_views: continue
        for j in range(1, 1+2*n_pair, 2):
            pair.append(pair_str[j])
            score.append(float(pair_str[j+1]))
        img_ids.append(img_id)
        pairs[img_id] = {'id': img_id, 'index': i//2, 'pair': pair, 'score': score}
    pairs['id_list'] = img_ids
    return pairs


class Depth2pcd(data.Dataset):

    def __init__(self, root, pair, num_src, read, transforms):
        self.root = root
        self.num_src = num_src
        self.read = read
        self.transforms = transforms
        self.pair = load_pair(pair, min_views=num_src)

    def __len__(self):
        return len(self.pair['id_list'])

    def __getitem__(self, i):
        ref_idx = self.pair['id_list'][i]
        src_idxs = self.pair[ref_idx]['pair'][:self.num_src]

        ref, *srcs = [os.path.join(self.root, f'{idx.zfill(8)}.jpg') for idx in [ref_idx] + src_idxs]
        ref_cam, *srcs_cam = [os.path.join(self.root, f'cam_{idx.zfill(8)}_flow3.txt') for idx in [ref_idx] + src_idxs]
        ref_depth, *srcs_depth = [os.path.join(self.root, f'{idx.zfill(8)}_flow3.pfm') for idx in [ref_idx] + src_idxs]
        ref_probs = [os.path.join(self.root, f'{ref_idx.zfill(8)}_flow{k+1}_prob.pfm') for k in range(3)]
        skip = 0

        sample = self.read({
            'ref': ref, 
            'ref_cam': ref_cam, 
            'srcs': srcs, 
            'srcs_cam': srcs_cam, 
            'ref_depth': ref_depth,
            'srcs_depth': srcs_depth,
            'ref_probs': ref_probs,
            'skip':skip,
            'id': ref_idx
        })
        for t in self.transforms:
            sample = t(sample)
        return sample


def read(filenames):
    ref_name, ref_cam_name, srcs_name, srcs_cam_name, ref_depth_name, srcs_depth_name, ref_probs_name = [filenames[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'ref_depth', 'srcs_depth', 'ref_probs']]
    ref, *srcs = [cv2.imread(fn) for fn in [ref_name] + srcs_name]
    ref_cam, *srcs_cam = [load_cam(fn, 0, 1) for fn in [ref_cam_name] + srcs_cam_name]
    ref_depth, *srcs_depth = [np.expand_dims(load_pfm(fn), axis=-1) for fn in [ref_depth_name]+srcs_depth_name]
    ref_probs = [np.expand_dims(load_pfm(fn), axis=-1) for fn in ref_probs_name]
    return {
        'ref': ref,
        'ref_cam': ref_cam,
        'ref_depth': ref_depth,
        'srcs': srcs,
        'srcs_cam': srcs_cam,
        'srcs_depth': srcs_depth,
        'ref_probs': ref_probs,
        'skip': filenames['skip'],
        'id': filenames['id']
    }


def val_preproc(sample, preproc_args):
    ref, ref_cam, srcs, srcs_cam, ref_depth, srcs_depth, ref_probs = [sample[attr] for attr in ['ref', 'ref_cam', 'srcs', 'srcs_cam', 'ref_depth', 'srcs_depth', 'ref_probs']]

    ref, *srcs = to_channel_first([ref] + srcs)
    ref_depth, *srcs_depth = to_channel_first([ref_depth] + srcs_depth)
    ref_probs = to_channel_first(ref_probs)

    srcs, srcs_cam, srcs_depth, ref_probs = [np.stack(arr_list, axis=0) for arr_list in [srcs, srcs_cam, srcs_depth, ref_probs]]

    return {
        'id': sample['id'],
        'ref': ref,  # 3hw
        'ref_cam': ref_cam,  # 244
        'ref_depth': ref_depth,  # 1hw
        'srcs': srcs,  # v3hw
        'srcs_cam': srcs_cam,  # v244
        'srcs_depth': srcs_depth,  # v1hw
        'ref_probs': ref_probs,  # 31hw
        'skip': sample['skip']  # scalar
    }


def get_val_loader(root, pair, num_src, preproc_args):
    dataset = Depth2pcd(
        root, pair, num_src,
        read=lambda filenames: read(filenames),
        transforms=[lambda sample: val_preproc(sample, preproc_args)]
    )
    loader = data.DataLoader(dataset, batch_size=4, num_workers=8, collate_fn=dict_collate, shuffle=False, drop_last=False)
    return dataset, loader
