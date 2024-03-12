import sys
import argparse
import os
import time
import logging
from datetime import datetime
import trimesh
import numpy as np

logging.basicConfig(level=logging.INFO)

    
def main():
    logging.info("Start exporting.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--color', action='store_true')
    parser.add_argument('--align', action='store_true')
    parser.add_argument('--res', default=1024)
    parser.add_argument('--output-dir', default='results')
    args, extras = parser.parse_known_args()

    # set CUDA_VISIBLE_DEVICES then import pytorch-lightning
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = len(args.gpu.split(','))

    # code_dir = os.path.join(args.exp_dir, 'code')
    ckpt_dir = os.path.join(args.exp_dir, 'ckpt')
    latest_ckpt = sorted(os.listdir(ckpt_dir), key=lambda s: int(s.split('-')[0].split('=')[1]), reverse=True)[0]
    latest_ckpt = os.path.join(ckpt_dir, latest_ckpt)
    config_path = os.path.join(args.exp_dir, 'config', 'parsed.yaml')
    
    # logging.info(f"Importing modules from cached code: {code_dir}")
    # sys.path.append(code_dir)
    import systems
    import pytorch_lightning as pl
    from utils.misc import load_config    

    # parse YAML config to OmegaConf
    logging.info(f"Loading configuration: {config_path}")
    config = load_config(config_path, cli_args=extras)
    
    # Update level of ProgressiveBandHashGrid
    if  config.model.geometry.xyz_encoding_config.otype == 'ProgressiveBandHashGrid':
        config.model.geometry.xyz_encoding_config.start_level = config.model.geometry.xyz_encoding_config.n_levels
    config.model.geometry.isosurface.resolution = args.res
    config.export.export_vertex_color = args.color
    config.cmd_args = vars(args)
    
    if 'seed' not in config:
        pl.seed_everything(config.seed)

    logging.info(f"Creating system: {config.system.name}")
    system = systems.make(config.system.name, config, load_from_checkpoint=latest_ckpt)
    system.model.cuda()
    mesh_properties = system.model.export(config.export)
    
    
    if config.export.export_vertex_color:
        mesh = trimesh.Trimesh(
                vertices=mesh_properties['v_pos'],
                faces=mesh_properties['t_pos_idx'],
                vertex_colors=mesh_properties['v_rgb'].numpy(),
                vertex_normals=mesh_properties['v_norm'].numpy()
            )
    else:
        mesh = trimesh.Trimesh(
                vertices=mesh_properties['v_pos'],
                faces=mesh_properties['t_pos_idx'],
            )
    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        metallicFactor=0.25,
        roughnessFactor=0.25
    )
    
    if args.align:
        import datasets
        dm = datasets.make(config.dataset.name, config.dataset)
        dm.setup('predict')
        inv_scale = dm.predict_dataset.inv_scale.numpy()
        mesh.vertices = mesh.vertices * inv_scale
        mesh.apply_transform(dm.predict_dataset.inv_trans.inverse())
    mesh.faces = np.fliplr(mesh.faces)
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info("Exporting mesh.")
    mesh.export(os.path.join(args.output_dir, f'{config.name}.glb'))
    mesh.export(os.path.join(args.output_dir, f'{config.name}.obj'))
    logging.info("Export finished successfully.")
    
if __name__ == '__main__':
    main()