import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss
from math import exp, ceil
from torch.autograd import Variable

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.utils import cleanup
from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy

class S3IM(torch.nn.Module):
    def __init__(self, s3im_kernel_size = 4, s3im_stride=4, s3im_repeat_time=10, s3im_patch_height=64, size_average = True):
        super(S3IM, self).__init__()
        self.s3im_kernel_size = s3im_kernel_size
        self.s3im_stride = s3im_stride
        self.s3im_repeat_time = s3im_repeat_time
        self.s3im_patch_height = s3im_patch_height
        self.size_average = size_average
        self.channel = 1
        self.s3im_kernel = self.create_kernel(s3im_kernel_size, self.channel)


    def gaussian(self, s3im_kernel_size, sigma):
        gauss = torch.Tensor([exp(-(x - s3im_kernel_size//2)**2/float(2*sigma**2)) for x in range(s3im_kernel_size)])
        return gauss/gauss.sum()

    def create_kernel(self, s3im_kernel_size, channel):
        _1D_window = self.gaussian(s3im_kernel_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        s3im_kernel = Variable(_2D_window.expand(channel, 1, s3im_kernel_size, s3im_kernel_size).contiguous())
        return s3im_kernel

    def _ssim(self, img1, img2, s3im_kernel, s3im_kernel_size, channel, size_average = True, s3im_stride=None):
        mu1 = F.conv2d(img1, s3im_kernel, padding = (s3im_kernel_size-1)//2, groups = channel, stride=s3im_stride)
        mu2 = F.conv2d(img2, s3im_kernel, padding = (s3im_kernel_size-1)//2, groups = channel, stride=s3im_stride)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, s3im_kernel, padding = (s3im_kernel_size-1)//2, groups = channel, stride=s3im_stride) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, s3im_kernel, padding = (s3im_kernel_size-1)//2, groups = channel, stride=s3im_stride) - mu2_sq
        sigma12 = F.conv2d(img1*img2, s3im_kernel, padding = (s3im_kernel_size-1)//2, groups = channel, stride=s3im_stride) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def ssim_loss(self, img1, img2):
        """
        img1, img2: torch.Tensor([b,c,h,w])
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.s3im_kernel.data.type() == img1.data.type():
            s3im_kernel = self.s3im_kernel
        else:
            s3im_kernel = self.create_kernel(self.s3im_kernel_size, channel)

            if img1.is_cuda:
                s3im_kernel = s3im_kernel.cuda(img1.get_device())
            s3im_kernel = s3im_kernel.type_as(img1)

            self.s3im_kernel = s3im_kernel
            self.channel = channel


        return self._ssim(img1, img2, s3im_kernel, self.s3im_kernel_size, channel, self.size_average, s3im_stride=self.s3im_stride)

    def forward(self, src_vec, tar_vec):
        loss = 0.0
        index_list = []
        for i in range(self.s3im_repeat_time):
            if i == 0:
                tmp_index = torch.arange(len(tar_vec))
                index_list.append(tmp_index)
            else:
                ran_idx = torch.randperm(len(tar_vec))
                index_list.append(ran_idx)
        res_index = torch.cat(index_list)
        tar_all = tar_vec[res_index]
        src_all = src_vec[res_index]
        tar_patch = tar_all.permute(1, 0).reshape(1, 3, self.s3im_patch_height, -1)
        src_patch = src_all.permute(1, 0).reshape(1, 3, self.s3im_patch_height, -1)
        loss = (1 - self.ssim_loss(src_patch, tar_patch))
        return loss
    
@systems.register('neus-system')
class NeuSSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
        }
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays
        self.s3im_loss = S3IM()

    def forward(self, batch):
        return self.model(batch['rays'])
    
    def preprocess_data(self, batch, stage):
        if 'index' in batch: # validation / testing
            index = batch['index'].cpu()
        else:
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,))
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,))
        if stage in ['train']:
            c2w = self.dataset.all_c2w[index]
            
            # sample the same number of points as the ray
            pts_index = torch.randint(0, len(self.dataset.all_points), size=(self.train_num_rays,))
            pts = self.dataset.all_points[pts_index]
            pts_weights = self.dataset.all_points_confidence[pts_index]
            if self.dataset.pts3d_normal is not None:
                pts_normal = self.dataset.pts3d_normal[pts_index]
            else:
                pts_normal = torch.tensor([])
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,)
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,)
            )
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)
        else:
            c2w = self.dataset.all_c2w[index][0]
            pts = torch.tensor([])
            pts_weights = torch.tensor([])
            pts_normal = torch.tensor([])
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0] 
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32)
        
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])
        
        self.model.background_color = self.model.background_color.to(self.device)
        batch.update({
            'rays': rays.to(self.device),
            'rgb': rgb.to(self.device),
            'fg_mask': fg_mask.to(self.device),
            'pts': pts.to(self.device),
            'pts_normal': pts_normal.to(self.device),
            'pts_weights': pts_weights.to(self.device)
        })      
    
    def training_step(self, batch, batch_idx):
        out = self(batch)

        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples_full'].sum().item()))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)
            
        if self.C(self.config.system.loss.lambda_rgb_mse) > 0:
            loss_rgb_mse = F.mse_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
            self.log('train/loss_rgb_mse', loss_rgb_mse)
            loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

        if self.C(self.config.system.loss.lambda_rgb_l1) > 0:
            loss_rgb_l1 = F.l1_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
            self.log('train/loss_rgb', loss_rgb_l1)
            loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)        

        if self.C(self.config.system.loss.get('lambda_rgb_s3im', 0))   > 0:
            loss_rgb_s3im = self.s3im_loss(out['comp_rgb_full'], batch['rgb'])
            self.log('train/loss_rgb_s3im', loss_rgb_s3im)
            loss += loss_rgb_s3im *   self.C(self.config.system.loss.get('lambda_rgb_s3im', 0))  
        
        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
        
        opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
        loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
        self.log('train/loss_mask', loss_mask)
        loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.has_mask else 0.0)

        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.log('train/loss_opaque', loss_opaque)
        loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)

        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
        self.log('train/loss_sparsity', loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)

        if self.C(self.config.system.loss.lambda_curvature) > 0:
            assert 'sdf_laplace_samples' in out, "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
            loss_curvature = out['sdf_laplace_samples'].abs().mean()
            self.log('train/loss_curvature', loss_curvature)
            loss += loss_curvature * self.C(self.config.system.loss.lambda_curvature)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)    

        if self.config.model.learned_background and self.C(self.config.system.loss.lambda_distortion_bg) > 0:
            loss_distortion_bg = flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])
            self.log('train/loss_distortion_bg', loss_distortion_bg)
            loss += loss_distortion_bg * self.C(self.config.system.loss.lambda_distortion_bg)        

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        # sdf loss proposed in Geo-Neus and normal loss proposed in regsdf
        if self.C(self.config.system.loss.lambda_sdf_l1) > 0:
            pts = batch['pts']
            pts_normal = batch['pts_normal']
            pts_weights = batch['pts_weights']
            pts2sdf, pts2sdf_grad = self.model.geometry(pts, with_grad=True, with_feature=False)
            loss_sdf = F.l1_loss(pts2sdf, torch.zeros_like(pts2sdf)) * pts_weights
            loss_sdf = loss_sdf.mean(dim=0)
            
            normal_gt = torch.nn.functional.normalize(pts_normal, p=2, dim=-1)
            normal_pred = torch.nn.functional.normalize(pts2sdf_grad, p=2, dim=-1)
            loss_normal = (1.0 - torch.sum(normal_pred * normal_gt, dim=-1)).mean()
            self.log('train/loss_sdf_l1', loss_sdf)
            self.log('train/loss_normal_cos', loss_normal)
            loss += loss_sdf * self.C(self.config.system.loss.lambda_sdf_l1)
            loss += loss_normal * self.C(self.config.system.loss.get('lambda_normal', self.config.system.loss.lambda_sdf_l1))

        self.log('train/inv_s', out['inv_s'], prog_bar=True)

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))

        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)         

    def test_step(self, batch, batch_idx):
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }      
    
    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)    

            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                '(\d+)\.png',
                save_format='mp4',
                fps=30
            )
            
            self.export()
    
    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh
        )        
