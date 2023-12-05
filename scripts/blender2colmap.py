import os
import cv2
from tqdm import tqdm
import numpy as np
import struct
from typing import Tuple
import time
from argparse import ArgumentParser
import shutil
import json
import math

# from https://github.com/colmap/colmap/tree/dev/scripts/python
from utils.read_write_model import read_images_binary, rotmat2qvec
from utils.read_write_dense import read_array
from utils.read_write_model import read_model, write_model
from utils.database import COLMAPDatabase

colmap_bin = 'colmap'

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data', help='path to data dir')
    parser.add_argument('--output-dir', help='path to output dir')
    parser.add_argument('--matching-method', default='exhaustive_matcher', choices=['vocab_tree_matcher', 'exhaustive_matcher', 'spatial_matcher', 'squential_matcher'], help='matching method to use (default: vocab_tree_matcher)')
    args = parser.parse_args()
    return args

def run_command(command, log_message):
    print(log_message)
    start_time = time.time()
    return_code = os.system(command)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{log_message} completed in {elapsed_time:.2f} seconds")
    return elapsed_time, return_code

def create_cameras_and_points_txt(target: str, intrinsic_dict):
    os.makedirs(f'{target}/model', exist_ok=True)
    os.system(f'touch {target}/model/points3D.txt')
    with open(f'{target}/model/cameras.txt', 'w') as f:
        f.write(f'1 PINHOLE {intrinsic_dict["width"]} {intrinsic_dict["height"]} {intrinsic_dict["fx"]} {intrinsic_dict["fy"]} {intrinsic_dict["cx"]} {intrinsic_dict["cy"]}')

def create_images_txt(target: str, id_list: list, pose_dict: dict, images: list):
    data_list = []
    for image in images:
        id = image[1][:-4]
        rt = pose_dict[int(id)]
        rt = np.linalg.inv(rt)
        r = rt[:3, :3]
        t = rt[:3, 3]
        q = rotmat2qvec(r)
        data = [image[0], *q, *t, 1, f'{id}.png']
        data = [str(_) for _ in data]
        data = ' '.join(data)
        data_list.append(data)

    with open(f'{target}/model/images.txt', 'w') as f:
        for data in data_list:
            f.write(data)
            f.write('\n\n')

def load_K_Rt_from_P(P=None):
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def load_from_blender(source: str) -> Tuple[list, dict, dict, dict]:
    with open(os.path.join(source, f"transforms_train.json"), 'r') as f:
        meta = json.load(f)
    
    if 'w' in meta and 'h' in meta:
        W, H = int(meta['w']), int(meta['h'])
    else:
        W, H = 800, 800
    focal = 0.5 * W / math.tan(0.5 * meta['camera_angle_x'])
    
    id_list = []
    pose_dict = dict()
    image_dict = dict()
    mask_dict = dict()
    for i, frame in enumerate(meta['frames']):
        id_list.append(i)
        c2w = np.array(frame['transform_matrix'])
        c2w[:,1:3] *= -1
        pose_dict[i] = c2w
        
        image = cv2.imread(os.path.join(source, f"{frame['file_path']}.png"))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)
        foreground_mask = cv2.bitwise_not(binary)

        # Convert image to grayscale
        
        image_dict[i] = image
        mask_dict[i] = foreground_mask

    intrinsic_dict = {'width': W, 'height': H, 'fx': focal, 'fy': focal, 'cx': W//2, 'cy': H//2}
    return id_list, pose_dict, image_dict, mask_dict, intrinsic_dict
        
import requests
from rich.progress import track
from pathlib import Path
def get_vocab_tree():
    """Return path to vocab tree. Downloads vocab tree if it doesn't exist.

    Returns:
        The path to the vocab tree.
    """
    vocab_tree_filename =  Path("/tmp/vocab_tree.fbow")

    if not vocab_tree_filename.exists():
        r = requests.get("https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin", stream=True)
        vocab_tree_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_tree_filename, "wb") as f:
            total_length = r.headers.get("content-length")
            assert total_length is not None
            for chunk in track(
                r.iter_content(chunk_size=1024),
                total=int(total_length) / 1024 + 1,
                description="Downloading vocab tree...",
            ):
                if chunk:
                    f.write(chunk)
                    f.flush()
    return vocab_tree_filename


def create_masked_image(image: str, mask) -> np.ndarray:
    """
    Create a masked image by applying a mask to an image and making the background white.

    :param image_path: Path to the image file.
    :param mask_path: Path to the mask file.
    :return: Masked image with white background.
    """
    # Resize the mask to match the size of the image
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Invert the mask to get the background
    background_mask = cv2.bitwise_not(mask)

    # Make the background white
    white_background = np.full(image.shape, 255, dtype=np.uint8)

    # Combine the white background and the mask
    background = cv2.bitwise_and(white_background, white_background, mask=background_mask)

    # Apply mask to image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Combine the masked image and the background
    final_image = cv2.addWeighted(masked_image, 1, background, 1, 0)

    return final_image

def main(args):
    source = args.data
    target = args.output_dir

    # Create a sparse reconstruction folder
    sparse_reconstruction_folder = os.path.join(target, 'sparse')
    os.makedirs(sparse_reconstruction_folder, exist_ok=True)

    # Load data
    id_list, pose_dict, image_dict, mask_dict, intrinsic_dict = load_from_blender(args.data)
    
    # Cache data
    images_folder = os.path.join(target, 'images')
    mask_folder = os.path.join(target, 'mask')
    masked_image_folder = os.path.join(target, 'masked_image')
    
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(masked_image_folder, exist_ok=True)
    for _id in id_list:
        _image = image_dict[_id]
        _, _mask = cv2.threshold(mask_dict[_id], 1, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(images_folder, f'{_id}.png'), _image)
        cv2.imwrite(os.path.join(mask_folder, f'{_id}.png'), _mask)
        masked_image = create_masked_image(_image, _mask)
        cv2.imwrite(os.path.join(masked_image_folder, f'{_id}.png'), masked_image)
    
    # Create cameras.txt file
    create_cameras_and_points_txt(target, intrinsic_dict)

    # Run feature_extractor to extract feature
    if os.path.exists(f'{target}/database.db'):
        os.remove(f'{target}/database.db')
    feature_extractor_command = f'{colmap_bin} feature_extractor --database_path {target}/database.db --image_path {target}/images > {target}/log.txt'
    feature_extraction_time, feature_extraction_return_code = run_command(feature_extractor_command, "Running feature_extractor...")

    if args.matching_method == 'vocab_tree_matcher':
        vocab_tree_filename = get_vocab_tree()
        matcher_command = f'{colmap_bin} vocab_tree_matcher --database_path {target}/database.db --VocabTreeMatching.vocab_tree_path {str(vocab_tree_filename)} >> {target}/log.txt'
    elif args.matching_method == 'exhaustive_matcher':
        matcher_command = f'{colmap_bin} exhaustive_matcher --database_path {target}/database.db >> {target}/log.txt'
    elif args.matching_method == 'spatial_matcher':
        matcher_command = f'{colmap_bin} spatial_matcher --database_path {target}/database.db --SpatialMatching.is_gps 0 --SpatialMatching.ignore_z 0 >> {target}/log.txt'
    elif args.matching_method == 'squential_matcher':
        matcher_command = f'{colmap_bin} sequential_matcher --database_path {target}/database.db >> {target}/log.txt'

    matching_time, matching_return_code = run_command(matcher_command, f"Running {args.matching_method}...")

    db = COLMAPDatabase.connect(f'{target}/database.db')
    images = list(db.execute('select * from images'))
    create_images_txt(target, id_list, pose_dict, images)

    # Run point_triangulator to compute the 3D points
    point_triangulator_command = f'{colmap_bin} point_triangulator --database_path {target}/database.db --image_path {target}/masked_image \
                                        --input_path {target}/model --output_path {sparse_reconstruction_folder} \
                                        --Mapper.tri_min_angle 10 --Mapper.tri_merge_max_reproj_error 1 >> {target}/log.txt'
    point_triangulation_time, point_triangulation_return_code = run_command(point_triangulator_command, "Running point_triangulator...")
    
    # Run model_converter to convert the sparse points results
    model_converter_command = f'{colmap_bin} model_converter --input_path {sparse_reconstruction_folder} --output_path {target}/sparse/points3D.ply --output_type PLY'
    run_command(model_converter_command, "Exporting sparse points...")
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)