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
import click

# from https://github.com/colmap/colmap/tree/dev/scripts/python
from utils.read_write_model import read_images_binary, rotmat2qvec
from utils.read_write_dense import read_array
from utils.read_write_model import read_model, write_model
from utils.database import COLMAPDatabase

colmap_bin = 'colmap'

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

def project_pinhole(camera, point3D_camera):
    x_normalized = point3D_camera[0] / point3D_camera[2]
    y_normalized = point3D_camera[1] / point3D_camera[2]
    u = camera.params[0] * x_normalized + camera.params[2]
    v = camera.params[1] * y_normalized + camera.params[3]
    return np.array([u, v])


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

def load_from_dtu(source: str) -> Tuple[list, dict, dict, dict]:
    cams = np.load(os.path.join(source, 'cameras_sphere.npz'))
    id_list = list(cams.keys())
    n_images = max([int(k.split('_')[-1]) for k in id_list]) + 1
    
    pose_dict = dict()
    image_dict = dict()
    for i in range(n_images):
        world_mat, scale_mat = cams[f'world_mat_{i}'], cams[f'scale_mat_{i}']
        P = (world_mat @ scale_mat)[:3,:4]
        K, c2w = load_K_Rt_from_P(P)
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        # c2w[:3,1:3] *= -1. # flip input sign
        
        # pose_dict[i] = np.linalg.inv(c2w)
        pose_dict[i] = c2w
        image_dict[i] = os.path.join(source, 'image', f'{i:03d}.png')
    
    intrinsic_dict = {'width': 1920, 'height': 1080, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
    return id_list, pose_dict, image_dict, intrinsic_dict
        
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

def create_masked_image(image_path: str, mask_path: str) -> np.ndarray:
    """
    Create a masked image by applying a mask to an image and making the background white.

    :param image_path: Path to the image file.
    :param mask_path: Path to the mask file.
    :return: Masked image with white background.
    """
    # Load image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Ensure mask is binary so multiply operation works as expected
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

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

@click.command()
@click.option('--data', type=str, help='path to data dir')
@click.option('--output-dir', type=str, help='path to output dir')
@click.option('--matching-method', type=str, default='squential_matcher', help='matching method to use (default: squential_matcher)')
def main(data, output_dir, matching_method):
    source = data
    target = output_dir
    os.makedirs(target, exist_ok=True)
    
    # Copy images and mask
    source_images_dir  = os.path.join(source, 'images')
    target_images_dir = os.path.join(target, 'images')
    source_mask_dir = os.path.join(source, 'mask')
    target_mask_dir = os.path.join(target, 'mask')
    
    assert os.path.exists(source_mask_dir), "Mask directory does not exist"
    assert os.path.exists(source_images_dir), "Image directory does not exist"
    
    if os.path.exists(target_images_dir):
        shutil.rmtree(target_images_dir)
    if os.path.exists(target_mask_dir):
        shutil.rmtree(target_mask_dir)
        
    shutil.copytree(source_images_dir, target_images_dir)
    shutil.copytree(source_mask_dir, target_mask_dir)
        
    # Generate masked_images
    masked_image_dir = os.path.join(target, 'masked_image')
    os.makedirs(masked_image_dir, exist_ok=True)
    # Iterate over images in the image directory
    for image_file in tqdm(os.listdir(target_images_dir)):
        # Construct full image path
        image_path = os.path.join(target_images_dir, image_file)

        # Construct corresponding mask path
        mask_file = image_file  # Assumes mask and image files have the same name
        mask_path = os.path.join(target_mask_dir, mask_file)

        # Create masked image
        masked_image = create_masked_image(image_path, mask_path)

        # Save masked image
        masked_image_path = os.path.join(masked_image_dir, image_file)
        cv2.imwrite(masked_image_path, masked_image)
    
    # Run feature_extractor to extract feature
    if os.path.exists(f'{target}/database.db'):
        os.remove(f'{target}/database.db')
        
    feature_extractor_command = f'{colmap_bin} feature_extractor --database_path {target}/database.db --image_path {source}/masked_image > {target}/log.txt'
    feature_extraction_time, feature_extraction_return_code = run_command(feature_extractor_command, "Running feature_extractor...")

    if matching_method == 'vocab_tree_matcher':
        vocab_tree_filename = get_vocab_tree()
        matcher_command = f'{colmap_bin} vocab_tree_matcher --database_path {target}/database.db --VocabTreeMatching.vocab_tree_path {str(vocab_tree_filename)} >> {target}/log.txt'
    elif matching_method == 'exhaustive_matcher':
        matcher_command = f'{colmap_bin} exhaustive_matcher --database_path {target}/database.db >> {target}/log.txt'
    elif matching_method == 'spatial_matcher':
        matcher_command = f'{colmap_bin} spatial_matcher --database_path {target}/database.db --SpatialMatching.is_gps 0 --SpatialMatching.ignore_z 0 >> {target}/log.txt'
    elif matching_method == 'squential_matcher':
        matcher_command = f'{colmap_bin} sequential_matcher --database_path {target}/database.db >> {target}/log.txt'

    matching_time, matching_return_code = run_command(matcher_command, f"Running {matching_method}...")

    source_reconstruction_folder = os.path.join(source, 'sparse/0')
    sparse_reconstruction_folder = os.path.join(target, 'sparse/0')
    os.makedirs(sparse_reconstruction_folder, exist_ok=True)
    # Run point_triangulator to compute the 3D points
    point_triangulator_command = f'{colmap_bin} point_triangulator --database_path {target}/database.db --image_path {source}/masked_image \
                                        --input_path {source_reconstruction_folder} --output_path {sparse_reconstruction_folder} \
                                        --Mapper.tri_min_angle 15 --Mapper.tri_merge_max_reproj_error 0.5 --clear_points 1 >> {target}/log.txt'
    point_triangulation_time, point_triangulation_return_code = run_command(point_triangulator_command, "Running point_triangulator...")
    
    # Run model_converter to convert the sparse points results
    model_converter_command = f'{colmap_bin} model_converter --input_path {sparse_reconstruction_folder} --output_path {target}/sparse/0/points3D.ply --output_type PLY'
    run_command(model_converter_command, "Exporting sparse points...")
    
if __name__ == '__main__':
    main()