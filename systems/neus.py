import torch
import torch.nn.functional as F
from loguru import logger
from torch_efficient_distloss import flatten_eff_distloss

import systems
from models.ray_utils import get_rays
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy


@systems.register("neus-system")
class NeuSSystem(BaseSystem):
    def prepare(self):
        self.criterions = {"psnr": PSNR()}
        self.train_num_samples = self.config.model.train_num_rays * (
            self.config.model.num_samples_per_ray
            + self.config.model.get("num_samples_per_ray_bg", 0)
        )
        self.train_num_rays = self.config.model.train_num_rays
        self.sample_foreground_ratio = self.config.dataset.get(
            "sample_foreground_ratio", 1.0
        )

    def forward(self, batch):
        return self.model(batch["rays"])

    def preprocess_data(self, batch, stage):
        if stage in ["test"]:
            index = batch["index"]
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(
                    0,
                    len(self.dataset.all_images),
                    size=(self.train_num_rays,),
                    device=self.device,
                )
                x = torch.randint(
                    0,
                    self.dataset.w,
                    size=(self.train_num_rays,),
                    device=self.device,
                )
                y = torch.randint(
                    0,
                    self.dataset.h,
                    size=(self.train_num_rays,),
                    device=self.device,
                )
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,))
                x = torch.randint(0, self.dataset.w, size=(self.train_num_rays,))
                y = torch.randint(0, self.dataset.h, size=(self.train_num_rays,))

        if stage in ["train"]:
            c2w = self.dataset.all_c2w[index]

            # sample the same number of points as the ray
            pts_index = torch.randint(
                0, len(self.dataset.all_points), size=(self.train_num_rays,)
            )
            pts = self.dataset.all_points[pts_index]
            pts_weights = self.dataset.all_points_confidence[pts_index]
            if self.dataset.pts3d_normal is not None:
                pts_normal = self.dataset.pts3d_normal[pts_index]
            else:
                pts_normal = torch.tensor([])

            if self.dataset.directions.ndim == 3:  # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4:  # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index, y, x].view(
                -1, self.dataset.all_images.shape[-1]
            )
        else:
            c2w = self.dataset.all_c2w[index][0]
            pts = torch.tensor([])
            pts_weights = torch.tensor([])
            pts_normal = torch.tensor([])
            if self.dataset.directions.ndim == 3:  # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4:  # (N, H, W, 3)
                directions = self.dataset.directions[index][0]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(
                -1, self.dataset.all_images.shape[-1]
            )

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        batch.update(
            {
                "rays": rays,
                "rgb": rgb,
                "pts": pts,
                "pts_normal": pts_normal,
                "pts_weights": pts_weights,
            }
        )

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)

        loss = 0.0

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(
                self.train_num_rays
                * (self.train_num_samples / out["num_samples_full"].sum().item())
            )
            self.train_num_rays = min(
                int(self.train_num_rays * 0.9 + train_num_rays * 0.1),
                self.config.model.max_train_num_rays,
            )

        loss_rgb_mse = F.mse_loss(
            out["comp_rgb_full"][out["rays_valid_full"][..., 0]],
            batch["rgb"][out["rays_valid_full"][..., 0]],
        )
        self.add_scalar("train/loss_rgb_mse", loss_rgb_mse)
        loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

        loss_rgb_l1 = F.l1_loss(
            out["comp_rgb_full"][out["rays_valid_full"][..., 0]],
            batch["rgb"][out["rays_valid_full"][..., 0]],
        )
        self.add_scalar("train/loss_rgb", loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)

        loss_eikonal = (
            (torch.linalg.norm(out["sdf_grad_samples"], ord=2, dim=-1) - 1.0) ** 2
        ).mean()
        self.add_scalar("train/loss_eikonal", loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)

        opacity = torch.clamp(out["opacity"].squeeze(-1), 1.0e-3, 1.0 - 1.0e-3)

        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.add_scalar("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)

        loss_sparsity = torch.exp(
            -self.config.system.loss.sparsity_scale * out["sdf_samples"].abs()
        ).mean()
        self.add_scalar("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)

        if self.C(self.config.system.loss.lambda_curvature) > 0:
            assert (
                "sdf_laplace_samples" in out
            ), "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
            loss_curvature = out["sdf_laplace_samples"].abs().mean()
            self.add_scalar("train/loss_curvature", loss_curvature)
            loss += loss_curvature * self.C(self.config.system.loss.lambda_curvature)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(
                out["weights"], out["points"], out["intervals"], out["ray_indices"]
            )
            self.add_scalar("train/loss_distortion", loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)
        # backgound
        if self.C(self.config.system.loss.lambda_distortion_bg) > 0:
            loss_distortion_bg = flatten_eff_distloss(
                out["weights_bg"],
                out["points_bg"],
                out["intervals_bg"],
                out["ray_indices_bg"],
            )
            self.add_scalar("train/loss_distortion_bg", loss_distortion_bg)
            loss += loss_distortion_bg * self.C(
                self.config.system.loss.lambda_distortion_bg
            )

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.add_scalar(f"train/loss_{name}", value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_

        # sdf loss proposed in Geo-Neus and normal loss proposed in regsdf
        if self.C(self.config.system.loss.lambda_sdf_l1) > 0:
            pts = batch["pts"]
            pts_normal = batch["pts_normal"]
            pts_weights = batch["pts_weights"]
            pts2sdf, pts2sdf_grad = self.model.geometry(
                pts, with_grad=True, with_feature=False
            )
            loss_sdf = F.l1_loss(pts2sdf, torch.zeros_like(pts2sdf)) * pts_weights
            loss_sdf = loss_sdf.mean(dim=0)

            normal_gt = torch.nn.functional.normalize(pts_normal, p=2, dim=-1)
            normal_pred = torch.nn.functional.normalize(pts2sdf_grad, p=2, dim=-1)
            loss_normal = (1.0 - torch.sum(normal_pred * normal_gt, dim=-1)).mean()
            self.add_scalar("train/loss_sdf_l1", loss_sdf)
            self.add_scalar("train/loss_normal_cos", loss_normal)
            loss += loss_sdf * self.C(self.config.system.loss.lambda_sdf_l1)
            loss += loss_normal * self.C(
                self.config.system.loss.get(
                    "lambda_normal", self.config.system.loss.lambda_sdf_l1
                )
            )

        self.add_scalar("train/inv_s", out["inv_s"])

        for name, value in self.config.system.loss.items():
            if name.startswith("lambda"):
                self.add_scalar(f"train_params/{name}", self.C(value))

        self.add_scalar("train/num_rays", float(self.train_num_rays))

        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        out = self.forward(batch)
        psnr = self.criterions["psnr"](
            out["comp_rgb_full"].to(batch["rgb"]), batch["rgb"]
        )
        W, H = self.dataset.img_wh
        self.save_image_grid(
            f"it{self.global_step}-test/{batch['index'][0].item()}.png",
            [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC"},
                }
            ]
            + [
                {"type": "grayscale", "img": out["depth"].view(H, W), "kwargs": {}},
                {
                    "type": "rgb",
                    "img": out["comp_normal"].view(H, W, 3),
                    "kwargs": {"data_format": "HWC", "data_range": (-1, 1)},
                },
            ],
        )

        self.add_scalar("test/psnr", psnr)
        return {"psnr": psnr, "index": batch["index"]}

    def export(self):
        mesh = self.model.export(self.config.export, device=self.device)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh,
        )
