import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
from training.training_loop import save_image_grid
from torch_utils.misc import copy_params_and_buffers

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def generate_images(G, seeds, noise_mode, truncation_psi):
    device = torch.device('cuda')
    result = []
    for seed_idx, seed in enumerate(seeds):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        img = G(z, c=torch.zeros([1, G.c_dim], device=device), truncation_psi=truncation_psi, noise_mode=noise_mode)
        result.append(img.cpu())
    return result


#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--src_network', 'src_network_pkl', help='Network in source domain pickle filename', required=True)
@click.option('--dst_network', 'dst_network_pkl', help='Network in destinaltion domain pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--max_blocks', 'max_blocks', type=int, help='maximum number of blocks copied from src_network')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--output', help='Where to save the output images grid', type=str, default='grid.png', metavar='DIR')
def generate_comparision(
    ctx: click.Context,
    src_network_pkl: str,
    dst_network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    max_blocks: Optional[int],
    noise_mode: str,
    output: str,
):

    device = torch.device('cuda')
    with dnnlib.util.open_url(src_network_pkl) as src_f, dnnlib.util.open_url(dst_network_pkl) as dst_f:
        src_G = legacy.load_network_pkl(src_f)['G_ema'].to(device) # type: ignore
        dst_G = legacy.load_network_pkl(dst_f)['G_ema'].to(device) # type: ignore


    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    assert src_G.c_dim == 0, 'labels are not supported'

    if max_blocks is None:
        max_blocks = len(src_G.synthesis.block_resolutions)

    images = generate_images(dst_G, seeds, noise_mode, truncation_psi)
    # Generate images.
    for ind, res in list(enumerate(src_G.synthesis.block_resolutions))[:max_blocks]:
        src_block = getattr(src_G.synthesis, f'b{res}')
        dst_block = getattr(dst_G.synthesis, f'b{res}')
        copy_params_and_buffers(src_block, dst_block, require_all=True)
        images.extend(generate_images(dst_G, seeds, noise_mode, truncation_psi))
    gw = len(seeds)
    gh = len(images) // len(seeds)
    save_image_grid(torch.cat(images).numpy(), output, drange=[-1, 1], grid_size=(gw, gh))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_comparision() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
