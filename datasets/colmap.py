import os
import math
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import open3d as o3d

import datasets
from datasets.colmap_utils import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)
from models.ray_utils import get_ray_directions
from models.utils import scale_anything
from nerfacc import ContractionType


def get_center(pts):
    center = pts.mean(0)
    dis = (pts - center[None, :]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (
        (dis > mean - 1.5 * std)
        & (dis < mean + 1.5 * std)
        & (dis > mean - (q75 - q25) * 1.5)
        & (dis < mean + (q75 - q25) * 1.5)
    )
    center = pts[valid].mean(0)
    return center


def normalize_poses(poses, pts, up_est_method, center_est_method, pts3d_normal=None):
    if center_est_method == "camera":
        # estimation scene center as the average of all camera positions
        center = poses[..., 3].mean(0)
    elif center_est_method == "lookat":
        # estimation scene center as the average of the intersection of selected pairs of camera rays
        cams_ori = poses[..., 3]
        cams_dir = poses[:, :3, :3] @ torch.as_tensor([0.0, 0.0, -1.0])
        cams_dir = F.normalize(cams_dir, dim=-1)
        A = torch.stack([cams_dir, -cams_dir.roll(1, 0)], dim=-1)
        b = -cams_ori + cams_ori.roll(1, 0)
        t = torch.linalg.lstsq(A, b).solution
        center = (
            torch.stack([cams_dir, cams_dir.roll(1, 0)], dim=-1) * t[:, None, :]
            + torch.stack([cams_ori, cams_ori.roll(1, 0)], dim=-1)
        ).mean((0, 2))
    elif center_est_method == "point":
        # first estimation scene center as the average of all camera positions
        # later we'll use the center of all points bounded by the cameras as the final scene center
        center = poses[..., 3].mean(0)
    else:
        raise NotImplementedError(
            f"Unknown center estimation method: {center_est_method}"
        )

    if up_est_method == "ground":
        # estimate up direction as the normal of the estimated ground plane
        # use RANSAC to estimate the ground plane in the point cloud
        import pyransac3d as pyrsc

        ground = pyrsc.Plane()
        plane_eq, inliers = ground.fit(
            pts.numpy(), thresh=0.01
        )  # TODO: determine thresh based on scene scale
        plane_eq = torch.as_tensor(plane_eq)  # A, B, C, D in Ax + By + Cz + D = 0
        z = F.normalize(plane_eq[:3], dim=-1)  # plane normal as up direction
        signed_distance = (
            torch.cat([pts, torch.ones_like(pts[..., 0:1])], dim=-1) * plane_eq
        ).sum(-1)
        if signed_distance.mean() < 0:
            z = -z  # flip the direction if points lie under the plane
    elif up_est_method == "camera":
        # estimate up direction as the average of all camera up directions
        z = F.normalize((poses[..., 3] - center).mean(0), dim=0)
    else:
        raise NotImplementedError(f"Unknown up estimation method: {up_est_method}")

    # new axis
    y_ = torch.as_tensor([z[1], -z[0], 0.0])
    x = F.normalize(y_.cross(z), dim=0)
    y = z.cross(x)

    if center_est_method == "point":
        # rotation
        Rc = torch.stack([x, y, z], dim=1)
        R = Rc.T
        poses_homo = torch.cat(
            [
                poses,
                torch.as_tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(
                    poses.shape[0], -1, -1
                ),
            ],
            dim=1,
        )
        inv_trans = torch.cat(
            [
                torch.cat([R, torch.as_tensor([[0.0, 0.0, 0.0]]).T], dim=1),
                torch.as_tensor([[0.0, 0.0, 0.0, 1.0]]),
            ],
            dim=0,
        )
        poses_norm = (inv_trans @ poses_homo)[:, :3]
        pts = (
            inv_trans
            @ torch.cat([pts, torch.ones_like(pts[:, 0:1])], dim=-1)[..., None]
        )[:, :3, 0]

        # translation and scaling
        poses_min, poses_max = (
            poses_norm[..., 3].min(0)[0],
            poses_norm[..., 3].max(0)[0],
        )
        pts_fg = pts[
            (poses_min[0] < pts[:, 0])
            & (pts[:, 0] < poses_max[0])
            & (poses_min[1] < pts[:, 1])
            & (pts[:, 1] < poses_max[1])
        ]
        center = get_center(pts_fg)
        tc = center.reshape(3, 1)
        t = -tc
        poses_homo = torch.cat(
            [
                poses_norm,
                torch.as_tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(
                    poses_norm.shape[0], -1, -1
                ),
            ],
            dim=1,
        )
        inv_trans = torch.cat(
            [
                torch.cat([torch.eye(3), t], dim=1),
                torch.as_tensor([[0.0, 0.0, 0.0, 1.0]]),
            ],
            dim=0,
        )
        poses_norm = (inv_trans @ poses_homo)[:, :3]
        scale = poses_norm[..., 3].norm(p=2, dim=-1).min()
        poses_norm[..., 3] /= scale
        pts = (
            inv_trans
            @ torch.cat([pts, torch.ones_like(pts[:, 0:1])], dim=-1)[..., None]
        )[:, :3, 0]
        # apply the rotation to the point cloud normal
        if pts3d_normal is not None:
            pts3d_normal = (R @ pts3d_normal.T).T
        pts = pts / scale
    else:
        # rotation and translation
        Rc = torch.stack([x, y, z], dim=1)
        tc = center.reshape(3, 1)
        R, t = Rc.T, -Rc.T @ tc
        poses_homo = torch.cat(
            [
                poses,
                torch.as_tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(
                    poses.shape[0], -1, -1
                ),
            ],
            dim=1,
        )
        inv_trans = torch.cat(
            [torch.cat([R, t], dim=1), torch.as_tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0
        )
        poses_norm = (inv_trans @ poses_homo)[:, :3]  # (N_images, 4, 4)

        # scaling
        scale = poses_norm[..., 3].norm(p=2, dim=-1).min()
        poses_norm[..., 3] /= scale

        # apply the transformation to the point cloud
        pts = (
            inv_trans
            @ torch.cat([pts, torch.ones_like(pts[:, 0:1])], dim=-1)[..., None]
        )[:, :3, 0]
        # apply the rotation to the point cloud normal
        if pts3d_normal is not None:
            pts3d_normal = (R @ pts3d_normal.T).T
        pts = pts / scale

    return poses_norm, pts, pts3d_normal


def create_spheric_poses(cameras, n_steps=120):
    center = torch.as_tensor(
        [0.0, 0.0, 0.0], dtype=cameras.dtype, device=cameras.device
    )
    mean_d = (cameras - center[None, :]).norm(p=2, dim=-1).mean()
    mean_h = cameras[:, 2].mean()
    r = (mean_d**2 - mean_h**2).sqrt()
    up = torch.as_tensor([0.0, 0.0, 1.0], dtype=center.dtype, device=center.device)

    all_c2w = []
    for theta in torch.linspace(0, 2 * math.pi, n_steps):
        cam_pos = torch.stack([r * theta.cos(), r * theta.sin(), mean_h])
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:, None]], axis=1)
        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0)

    return all_c2w


def contract_to_unisphere(x, radius, contraction_type):
    if contraction_type == ContractionType.AABB:
        x = scale_anything(x, (-radius, radius), (0, 1))
    elif contraction_type == ContractionType.UN_BOUNDED_SPHERE:
        x = scale_anything(x, (-radius, radius), (0, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        raise NotImplementedError
    return x


def error_to_confidence(error):
    # Here smaller_beta means slower transition from 0 to 1.
    # Increasing beta raises steepness of the curve.
    beta = 1
    # confidence = 1 / np.exp(beta*error)
    confidence = 1 / (1 + np.exp(beta * error))
    return confidence


class ColmapDataset(Dataset):
    # the data only has to be processed once
    initialized = False
    properties = {}

    def __init__(self, config, split):
        self.config = config
        self.split = split

        if not ColmapDataset.initialized:
            camdata = read_cameras_binary(
                os.path.join(self.config.root_dir, "sparse/0/cameras.bin")
            )

            H = int(camdata[1].height)
            W = int(camdata[1].width)

            if "img_wh" in self.config:
                w, h = self.config.img_wh
                assert round(W / w * h) == H
            elif "img_downscale" in self.config:
                w, h = int(W / self.config.img_downscale + 0.5), int(
                    H / self.config.img_downscale + 0.5
                )
            else:
                raise KeyError("Either img_wh or img_downscale should be specified.")

            img_wh = (w, h)
            factor = w / W

            if camdata[1].model == "SIMPLE_RADIAL":
                fx = fy = camdata[1].params[0] * factor
                cx = camdata[1].params[1] * factor
                cy = camdata[1].params[2] * factor
            elif camdata[1].model in ["PINHOLE", "OPENCV"]:
                fx = camdata[1].params[0] * factor
                fy = camdata[1].params[1] * factor
                cx = camdata[1].params[2] * factor
                cy = camdata[1].params[3] * factor
            elif camdata[1].model == "SIMPLE_PINHOLE":
                fx = fy = camdata[1].params[0] * factor
                cx = camdata[1].params[1] * factor
                cy = camdata[1].params[2] * factor
            else:
                raise ValueError(
                    f"Please parse the intrinsics for camera model {camdata[1].model}!"
                )

            directions = get_ray_directions(w, h, fx, fy, cx, cy)

            imdata = read_images_binary(
                os.path.join(self.config.root_dir, "sparse/0/images.bin")
            )

            all_c2w, all_images = [], []
            for i, d in enumerate(imdata.values()):
                R = d.qvec2rotmat()
                t = d.tvec.reshape(3, 1)
                c2w = torch.from_numpy(np.concatenate([R.T, -R.T @ t], axis=1)).float()
                c2w[:, 1:3] *= -1.0  # COLMAP => OpenGL
                all_c2w.append(c2w)
                if self.split in ["train", "val"]:
                    img_path = os.path.join(self.config.root_dir, "images", d.name)
                    img = Image.open(img_path)
                    img = img.resize(img_wh, Image.BICUBIC)
                    img = TF.to_tensor(img).permute(1, 2, 0)[..., :3]
                    all_images.append(img)

            all_c2w = torch.stack(all_c2w, dim=0)

            if self.config.dense_pcd_path is not None:
                dense_points_path = os.path.join(
                    self.config.root_dir, self.config.dense_pcd_path
                )
                assert os.path.exists(
                    dense_points_path
                ), f"Please check whether {dense_points_path} exists"
                print(f"Loading dense prior from {dense_points_path}")
                import plyfile

                plydata = plyfile.PlyData.read(dense_points_path)
                pts3d = np.vstack(
                    [
                        plydata["vertex"]["x"],
                        plydata["vertex"]["y"],
                        plydata["vertex"]["z"],
                    ]
                ).T
                pts3d_normal = np.vstack(
                    [
                        plydata["vertex"]["nx"],
                        plydata["vertex"]["ny"],
                        plydata["vertex"]["nz"],
                    ]
                ).T
                if "confidence" in plydata["vertex"]:
                    pts3d_confidence = plydata["vertex"]["confidence"]
                else:
                    pts3d_confidence = np.ones([pts3d.shape[0]])
            else:
                sparse_points_path = os.path.join(
                    self.config.root_dir, "sparse/0/points3D.bin"
                )
                print(f"Loading sparse prior from {sparse_points_path}")
                pts3d = read_points3d_binary(sparse_points_path)
                pts3d_confidence = np.array(
                    [error_to_confidence(pts3d[k].error) for k in pts3d]
                )
                pts3d = np.array([pts3d[k].xyz for k in pts3d])
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts3d)
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=30
                    )
                )
                pts3d_normal = np.asarray(pcd.normals)

            pts3d = torch.from_numpy(pts3d).float()
            pts3d_confidence = torch.from_numpy(pts3d_confidence).float()
            all_c2w, pts3d, pts3d_normal = normalize_poses(
                all_c2w,
                pts3d,
                up_est_method=self.config.up_est_method,
                center_est_method=self.config.center_est_method,
                pts3d_normal=pts3d_normal,
            )

            ColmapDataset.properties = {
                "w": w,
                "h": h,
                "img_wh": img_wh,
                "factor": factor,
                "directions": directions,
                "pts3d": pts3d,
                "pts3d_confidence": pts3d_confidence,
                "pts3d_normal": pts3d_normal,
                "all_c2w": all_c2w,
                "all_images": all_images,
            }

            ColmapDataset.initialized = True

        for k, v in ColmapDataset.properties.items():
            setattr(self, k, v)

        if self.split == "test":
            self.all_c2w = create_spheric_poses(
                self.all_c2w[:, :, 3], n_steps=self.config.n_test_traj_steps
            )
            self.all_images = torch.zeros(
                (self.config.n_test_traj_steps, self.h, self.w, 3), dtype=torch.float32
            )
            self.all_points = torch.tensor([])
            self.all_points_confidence = torch.tensor([])
        else:
            self.all_images = torch.stack(self.all_images, dim=0).float()
            self.all_points = self.pts3d
            self.all_points_confidence = self.pts3d_confidence

        self.all_c2w = self.all_c2w.float()
        self.all_images = self.all_images
        self.all_points_confidence = self.all_points_confidence.float()
        self.all_points = self.all_points.float()
        self.pts3d_normal = self.pts3d_normal.float()
        self.all_points_ = contract_to_unisphere(
            self.all_points, 1.0, ContractionType.AABB
        )  # points normalized to (0, 1)

    def to_device(self, device: torch.device):
        self.all_images = self.all_images.to(device)
        self.all_c2w = self.all_c2w.to(device)
        self.all_points = self.all_points.to(device)
        self.pts3d_normal = self.pts3d_normal.to(device)
        self.all_points_confidence = self.all_points_confidence.to(device)
        self.directions = self.directions.to(device)

    def query_radius_occ(self, query_points, radius=0.01):
        # Compute minimum distances
        min_dist, _ = torch.cdist(query_points, self.all_points_).min(dim=1)
        # Create occupancy masks based on min dist
        occ_mask = min_dist < radius
        return occ_mask

    def __len__(self):
        if self.split == "train":
            return 99999999
        else:
            return len(self.all_images)

    def __getitem__(self, index):
        if self.split == "train":
            return {}
        else:
            return {"index": index}


@datasets.register("colmap")
class ColmapDataModule:
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = device

    def setup(self, stage=None, device=None):
        if device is None:
            device = self.device
        if stage in [None, "train"]:
            self.train_dataset = ColmapDataset(self.config, "train")
            self.train_dataset.to_device(device)
        if stage in [None, "test"]:
            self.test_dataset = ColmapDataset(self.config, "test")
            self.test_dataset.to_device(device)

    def general_loader(self, dataset, batch_size):
        return DataLoader(
            dataset,
            num_workers=4,
            batch_size=batch_size,
            pin_memory=True,
            sampler=None,
        )

    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)
