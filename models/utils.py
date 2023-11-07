import gc
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import numpy as np

import tinycudann as tcnn


def chunk_batch(func, chunk_size, move_to_cpu, *args, **kwargs):
    B = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    out = defaultdict(list)
    out_type = None
    for i in range(0, B, chunk_size):
        out_chunk = func(*[arg[i:i+chunk_size] if isinstance(arg, torch.Tensor) else arg for arg in args], **kwargs)
        if out_chunk is None:
            continue
        out_type = type(out_chunk)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
        elif isinstance(out_chunk, tuple) or isinstance(out_chunk, list):
            chunk_length = len(out_chunk)
            out_chunk = {i: chunk for i, chunk in enumerate(out_chunk)}
        elif isinstance(out_chunk, dict):
            pass
        else:
            print(f'Return value of func must be in type [torch.Tensor, list, tuple, dict], get {type(out_chunk)}.')
            exit(1)
        for k, v in out_chunk.items():
            v = v if torch.is_grad_enabled() else v.detach()
            v = v.cpu() if move_to_cpu else v
            out[k].append(v)
    
    if out_type is None:
        return

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    if out_type is torch.Tensor:
        return out[0]
    elif out_type in [tuple, list]:
        return out_type([out[i] for i in range(chunk_length)])
    elif out_type is dict:
        return out


class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))

trunc_exp = _TruncExp.apply


def get_activation(name):
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == 'none':
        return lambda x: x
    elif name.startswith('scale'):
        scale_factor = float(name[5:])
        return lambda x: x.clamp(0., scale_factor) / scale_factor
    elif name.startswith('clamp'):
        clamp_max = float(name[5:])
        return lambda x: x.clamp(0., clamp_max)
    elif name.startswith('mul'):
        mul_factor = float(name[3:])
        return lambda x: x * mul_factor
    elif name == 'lin2srgb':
        return lambda x: torch.where(x > 0.0031308, torch.pow(torch.clamp(x, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*x).clamp(0., 1.)
    elif name == 'trunc_exp':
        return trunc_exp
    elif name.startswith('+') or name.startswith('-'):
        return lambda x: x + float(name)
    elif name == 'sigmoid':
        return lambda x: torch.sigmoid(x)
    elif name == 'tanh':
        return lambda x: torch.tanh(x)
    else:
        return getattr(F, name)


def dot(x, y):
    return torch.sum(x*y, -1, keepdim=True)


def reflect(x, n):
    return 2 * dot(x, n) * n - x


def scale_anything(dat, inp_scale, tgt_scale):
    if inp_scale is None:
        inp_scale = [dat.min(), dat.max()]
    dat = (dat  - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()


# Copyright 2022 Google LLC
def reflect(viewdirs, normals):
    """Reflect view directions about normals.

    The reflection of a vector v about a unit vector n is a vector u such that
    dot(v, n) = dot(u, n), and dot(u, u) = dot(v, v). The solution to these two
    equations is u = 2 dot(n, v) n - v.

    Args:
      viewdirs: [..., 3] array of view directions.
      normals: [..., 3] array of normal directions (assumed to be unit vectors).

    Returns:
      [..., 3] array of reflection directions.
    """
    return 2.0 * torch.sum(
        normals * viewdirs, dim=-1, keepdims=True) * normals - viewdirs


def l2_normalize(x, eps=torch.finfo(torch.float32).eps):
    """Normalize x to unit length along last axis."""
    eps = torch.tensor(eps, device=x.device)
    return x / torch.sqrt(torch.maximum(torch.sum(x**2, dim=-1, keepdims=True), eps))


def compute_weighted_mae(weights, normals, normals_gt):
    """Compute weighted mean angular error, assuming normals are unit length."""
    one_eps = torch.tensor(1 - torch.finfo(torch.float32).eps,
                           device=weights.device)
    return (weights * torch.arccos(
        torch.clip((normals * normals_gt).sum(-1), -one_eps,
                   one_eps))).sum() / weights.sum() * 180.0 / torch.pi


def compute_weighted_normal_loss(weights, normals, normals_gt):
    N = len(weights)
    n1 = torch.abs(normals - normals_gt).sum(axis=-1)
    n2 = torch.abs(1 - (normals * normals_gt).sum(axis=-1))
    return (weights * (n1 + n2)).sum() / weights.sum() / N


def generalized_binomial_coeff(a, k):
    """Compute generalized binomial coefficients."""
    return np.prod(a - np.arange(k)) / np.math.factorial(k)


def assoc_legendre_coeff(l, m, k):
    """Compute associated Legendre polynomial coefficients.

    Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
    (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

    Args:
      l: associated Legendre polynomial degree.
      m: associated Legendre polynomial order.
      k: power of cos(theta).

    Returns:
      A float, the coefficient of the term corresponding to the inputs.
    """
    return ((-1)**m * 2**l * np.math.factorial(l) / np.math.factorial(k) /
            np.math.factorial(l - k - m) *
            generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))


def sph_harm_coeff(l, m, k):
    """Compute spherical harmonic coefficients."""
    return (np.sqrt(
        (2.0 * l + 1.0) * np.math.factorial(l - m) /
        (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))


def get_ml_array(deg_view):
    """Create a list with all pairs of (l, m) values to use in the encoding."""
    ml_list = []
    for i in range(deg_view):
        l = 2**i
        # Only use nonnegative m values, later splitting real and imaginary parts.
        for m in range(l + 1):
            ml_list.append((m, l))

    # Convert list into a numpy array.
    ml_array = np.array(ml_list).T
    return ml_array


def generate_ide_fn(deg_view):
    """Generate integrated directional encoding (IDE) function.

    This function returns a function that computes the integrated directional
    encoding from Equations 6-8 of arxiv.org/abs/2112.03907.

    Args:
      deg_view: number of spherical harmonics degrees to use.

    Returns:
      A function for evaluating integrated directional encoding.

    Raises:
      ValueError: if deg_view is larger than 5.
    """
    if deg_view > 5:
        print('WARNING: Only deg_view of at most 5 is numerically stable.')
    #   raise ValueError('Only deg_view of at most 5 is numerically stable.')

    ml_array = get_ml_array(deg_view)
    l_max = 2**(deg_view - 1)

    # Create a matrix corresponding to ml_array holding all coefficients, which,
    # when multiplied (from the right) by the z coordinate Vandermonde matrix,
    # results in the z component of the encoding.
    mat = torch.zeros((l_max + 1, ml_array.shape[1]))
    for i, (m, l) in enumerate(ml_array.T):
        for k in range(l - m + 1):
            mat[k, i] = sph_harm_coeff(l, m, k)

    def integrated_dir_enc_fn(xyz, kappa_inv):
        """Function returning integrated directional encoding (IDE).

        Args:
          xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
          kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
            Mises-Fisher distribution.

        Returns:
          An array with the resulting IDE.
        """
        x = xyz[..., 0:1]
        y = xyz[..., 1:2]
        z = xyz[..., 2:3]

        # Compute z Vandermonde matrix.
        vmz = torch.cat([z**i for i in range(mat.shape[0])], dim=-1)

        # Compute x+iy Vandermonde matrix.
        vmxy = torch.cat(
            [(x + 1j * y)**m for m in ml_array[0, :]], dim=-1)

        # Get spherical harmonics.
        sph_harms = vmxy * torch.matmul(vmz, mat.to(vmz.device))

        # Apply attenuation function using the von Mises-Fisher distribution
        # concentration parameter, kappa.
        sigma = torch.tensor(
            0.5 * ml_array[1, :] * (ml_array[1, :] + 1), dtype=torch.float32)
        ide = sph_harms * torch.exp(-sigma.to(kappa_inv.device) * kappa_inv)

        # Split into real and imaginary parts and return
        return torch.cat([torch.real(ide), torch.imag(ide)], dim=-1)

    return integrated_dir_enc_fn
