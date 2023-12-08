import os
import trimesh
import argparse
import numpy as np
from loguru import logger
from pydlutils.torch.seed import set_seed

import systems
from utils.misc import load_config


def main():
    logger.info("Start exporting.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0", help="GPU(s) to be used")
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--res", default=1024)
    parser.add_argument("--output-dir", default="results")
    args, extras = parser.parse_known_args()

    ckpt_dir = os.path.join(args.exp_dir, "ckpt")
    latest_ckpt = sorted(
        os.listdir(ckpt_dir),
        key=lambda s: int(s.split("-")[0].split("=")[1]),
        reverse=True,
    )[0]
    latest_ckpt = os.path.join(ckpt_dir, latest_ckpt)
    config_path = os.path.join(args.exp_dir, "config", "parsed.yaml")

    # parse YAML config to OmegaConf
    logger.info(f"Loading configuration: {config_path}")
    config = load_config(config_path, cli_args=extras)

    # Update level of ProgressiveBandHashGrid
    if config.model.geometry.xyz_encoding_config.otype == "ProgressiveBandHashGrid":
        config.model.geometry.xyz_encoding_config.start_level = (
            config.model.geometry.xyz_encoding_config.n_levels
        )
    config.model.geometry.isosurface.resolution = args.res
    config.export.export_vertex_color = True
    config.cmd_args = vars(args)

    if "seed" not in config:
        set_seed(config.seed)

    logger.info(f"Creating system: {config.system.name}")
    system = systems.make(config.system.name, config, load_from_checkpoint=latest_ckpt)
    system.model.cuda()
    mesh = system.model.export(config.export)

    mesh["v_pos"] = mesh["v_pos"][:, [0, 2, 1]].numpy()
    if args.flip:
        mesh["t_pos_idx"] = mesh["t_pos_idx"].numpy()[:, [0, 2, 1]]
    else:
        mesh["t_pos_idx"] = np.fliplr(mesh["t_pos_idx"].numpy())[:, [0, 2, 1]]

    mesh = trimesh.Trimesh(
        vertices=mesh["v_pos"],
        faces=mesh["t_pos_idx"],
        vertex_colors=mesh["v_rgb"].numpy(),
        vertex_normals=mesh["v_norm"].numpy(),
    )
    mesh.visual.material = trimesh.visual.material.PBRMaterial(
        metallicFactor=0.25, roughnessFactor=0.25
    )

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Exporting mesh.")
    mesh.export(os.path.join(args.output_dir, f"{config.name}.glb"))
    mesh.export(os.path.join(args.output_dir, f"{config.name}.obj"))
    logger.info("Export finished successfully.")


if __name__ == "__main__":
    main()
