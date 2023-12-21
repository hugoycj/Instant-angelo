import torch
import torch.nn as nn

import models
from models.utils import get_activation, reflect, generate_ide_fn
from models.network_utils import get_encoding, get_mlp
from systems.utils import update_module_step
from pytorch_lightning.utilities.rank_zero import rank_zero_info

@models.register('volume-radiance')
class VolumeRadiance(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3
        encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + encoding.n_output_dims
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)    
        self.encoding = encoding
        self.network = network
    
    def forward(self, features, dirs, *args):
        dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)

    def regularizations(self, out):
        return {}
    
@models.register('volume-dual-color')
class VolumeDualColor(nn.Module):
    def __init__(self, config):
        super(VolumeDualColor, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3
        encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + encoding.n_output_dims
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)    
        self.encoding = encoding
        self.network = network
    def forward(self, features, dirs, *args):
        dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            basecolor = get_activation(self.config.color_activation)(features[..., 1:4])
            color = get_activation(self.config.color_activation)(color) + basecolor
        return color

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)

    def regularizations(self, out):
        return {}


@models.register('volume-dual-colorV2')
class VolumeDualColorV2(nn.Module):
    def __init__(self, config):
        super(VolumeDualColorV2, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3
        
        self.use_ide = False
        if self.use_ide:
            import numpy as np
            self.encoding = generate_ide_fn(5)
            num_sh = (2 ** np.arange(5) + 1).sum() * 2
            self.n_input_dims = self.config.input_feature_dim + num_sh
        else:
            self.encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
            self.n_input_dims = self.config.input_feature_dim + self.encoding.n_output_dims
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)    
        self.network = network

    def forward(self, features, viewdirs, normals):
        
        VdotN = (-viewdirs * normals).sum(-1, keepdim=True)
        refdirs = 2 * VdotN * normals + viewdirs
        
        if self.use_ide:
            tint = get_activation(self.config.color_activation)(features[..., 4:5])
            roughness = get_activation(self.config.color_activation)(features[..., 5:6])
            
            refdirs = (refdirs + 1.) / 2. # (-1, 1) => (0, 1)
            refdirs_embd = self.encoding(refdirs, roughness)
        else:
            refdirs = (refdirs + 1.) / 2. # (-1, 1) => (0, 1)
            refdirs_embd = self.encoding(refdirs.view(-1, self.n_dir_dims))
            
        network_inp = torch.cat([features.view(-1, features.shape[-1]), refdirs_embd] + [normals.view(-1, normals.shape[-1])] , dim=-1)
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            basecolor = get_activation(self.config.color_activation)(features[..., 1:4])
            color = get_activation(self.config.color_activation)(color) + basecolor
        return color


    def regularizations(self, out):
        return {}

@models.register('volume-dual-colorV3')
class VolumeDualColorV3(nn.Module):
    def __init__(self, config):
        super(VolumeDualColorV3, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3

        self.encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + self.encoding.n_output_dims
        self.cam_network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.ref_network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.weight_network = get_mlp(self.config.input_feature_dim, 1, self.config.weitht_network_config)
        
    def forward(self, features, viewdirs, normals):
        dirs = (viewdirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
        
        VdotN = (-viewdirs * normals).sum(-1, keepdim=True)
        refdirs = 2 * VdotN * normals + viewdirs
        refdirs = (refdirs + 1.) / 2. # (-1, 1) => (0, 1)
        refdirs_embd = self.encoding(refdirs.view(-1, self.n_dir_dims))

        network_inp = torch.cat([features.view(-1, features.shape[-1]), normals.view(-1, normals.shape[-1])], dim=-1)
        ref_weight = self.weight_network(network_inp)

        cam_network_inp = torch.cat([network_inp, dirs_embd], dim=-1)
        ref_network_inp = torch.cat([network_inp, refdirs_embd], dim=-1)
        cam_color = self.cam_network(cam_network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        ref_color = self.ref_network(ref_network_inp).view(*features.shape[:-1], self.n_output_dims).float()

        color = ref_weight * get_activation(self.config.color_activation)(ref_color) + \
                (1-ref_weight) * get_activation(self.config.color_activation)(cam_color)
        return color

    def regularizations(self, out):
        return {}

@models.register('volume-color')
class VolumeColor(nn.Module):
    def __init__(self, config):
        super(VolumeColor, self).__init__()
        self.config = config
        self.n_output_dims = 3
        self.n_input_dims = self.config.input_feature_dim
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.network = network
    
    def forward(self, features, *args):
        network_inp = features.view(-1, features.shape[-1])
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def regularizations(self, out):
        return {}



####################################################################################################################
# Code from svox2


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def svox2_eval_sh_bases(basis_dim : int, dirs : torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.
    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions
    :return: torch.Tensor (..., basis_dim)
    """
    result = torch.empty((*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y;
        result[..., 2] = SH_C1 * z;
        result[..., 3] = -SH_C1 * x;
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy;
            result[..., 5] = SH_C2[1] * yz;
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy);
            result[..., 7] = SH_C2[3] * xz;
            result[..., 8] = SH_C2[4] * (xx - yy);

            if basis_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy);
                result[..., 10] = SH_C3[1] * xy * z;
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy);
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy);
                result[..., 14] = SH_C3[5] * z * (xx - yy);
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy);

                if basis_dim > 16:
                    result[..., 16] = SH_C4[0] * xy * (xx - yy);
                    result[..., 17] = SH_C4[1] * yz * (3 * xx - yy);
                    result[..., 18] = SH_C4[2] * xy * (7 * zz - 1);
                    result[..., 19] = SH_C4[3] * yz * (7 * zz - 3);
                    result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3);
                    result[..., 21] = SH_C4[5] * xz * (7 * zz - 3);
                    result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1);
                    result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy);
                    result[..., 24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
    return result

def svox2_sh_eval(degree, sh_param, dirs):
    sh_bases = svox2_eval_sh_bases((degree+1)**2, dirs)
    if sh_param.dim == 2:
        sh_param = sh_param[:, None]  # SH, 3
    result = (sh_param[:, :(degree+1)**2] * sh_bases[..., None]).sum(dim=1)
    return result

@models.register('volume-SH')
class VolumeSphericalHarmonic(nn.Module):
    def __init__(self, config):
        super(VolumeSphericalHarmonic, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.sh_level = config['sh_level']
        self.sh_dc_coeff = 1
        self.sh_extra_coeff = (self.sh_level + 1) ** 2 - 1
        self.n_output_dims = 3 * (self.sh_level + 1) ** 2 # 45 extra_param
        self.n_input_dims = self.config.input_feature_dim
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)    
        self.network = network
        
    def forward(self, features, dirs, *args):
        network_inp = torch.cat([features.view(-1, features.shape[-1])] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        sh_coeff = self.network(network_inp).view(*features.shape[:-1], (self.sh_dc_coeff + self.sh_extra_coeff), 3).float()
        color = svox2_sh_eval(self.sh_level, sh_coeff, dirs)
        return color, sh_coeff

    def get_sh_coeff(self, features, *args):
        network_inp = torch.cat([features.view(-1, features.shape[-1])] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        sh_coeff = self.network(network_inp).view(*features.shape[:-1], (self.sh_dc_coeff + self.sh_extra_coeff), 3).float()
        return sh_coeff

    def update_step(self, epoch, global_step):
        pass
    
    def regularizations(self, out):
        return {}


def svox2_sh_eval_progressive(current_degree, sh_param, dirs):
    max_sh_param = (current_degree+1)**2
    sh_bases = svox2_eval_sh_bases(max_sh_param, dirs)
    if sh_param.dim == 2:
        sh_param = sh_param[:, None]  # SH, 3
    result = (sh_param[:, :max_sh_param] * sh_bases[..., None]).sum(dim=1)
    return result

@models.register('volume-progressive-SH')
class VolumeProgressiveSphericalHarmonic(nn.Module):
    def __init__(self, config):
        super(VolumeProgressiveSphericalHarmonic, self).__init__()
        self.config = config
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.current_level = 0
        self.sh_level = config['sh_level']
        self.sh_dc_coeff = 1
        self.sh_extra_coeff = (self.sh_level + 1) ** 2 - 1
        self.n_output_dims = 3 * (self.sh_level + 1) ** 2 # 45 extra_param
        self.n_input_dims = self.config.input_feature_dim
        self.start_level, self.start_step, self.update_steps = config['start_level'], config['start_step'], config['update_steps']
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)    
        self.network = network
        
    def forward(self, features, dirs, *args):
        network_inp = torch.cat([features.view(-1, features.shape[-1])] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        sh_coeff = self.network(network_inp).view(*features.shape[:-1], (self.sh_dc_coeff + self.sh_extra_coeff), 3).float()
        color = svox2_sh_eval_progressive(self.current_level, sh_coeff, dirs)
        return color, sh_coeff

    def get_sh_coeff(self, features, *args):
        network_inp = torch.cat([features.view(-1, features.shape[-1])] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
        sh_coeff = self.network(network_inp).view(*features.shape[:-1], (self.sh_dc_coeff + self.sh_extra_coeff), 3).float()
        return sh_coeff

    def update_step(self, epoch, global_step):
        current_level = min(self.start_level + max(global_step - self.start_step, 0) // self.update_steps, self.sh_level)
        if current_level > self.current_level:
            rank_zero_info(f'Update SH level to {current_level}')
        self.current_level = current_level
    
    def regularizations(self, out):
        return {}
