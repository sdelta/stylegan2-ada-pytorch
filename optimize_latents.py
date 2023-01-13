# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy

def transfer(
    G,
    CLIP_loss,
    source: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    num_steps                  = 200,
    alpha                      = 0.005,
    lr                         = 0.05,
    verbose                    = False,
    device: torch.device
):
    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)

    result = copy.deepcopy(source).requires_grad_(True).to(device)
    source = copy.deepcopy(source).requires_grad_(False).to(device)

    optimizer = torch.optim.Adam([result], betas=(0.9, 0.999), lr=lr)

    for step in range(num_steps)
        optimizer.zero_grad(set_to_none=True)
        gen_img = G.synthesis(result)
        gen_clip = -self.clip_subloss.get_similarities(gen_img).mean()
        loss = gen_clip + alpha * torch.linalg.norm(result - source)
        loss.backward()
        optimizer.step()
    return result


def generate_and_save(w, fname):
    img = G.synthesis(w.unsqueeze(0))
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(fname)


#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--source_prefix', 'source_prefix', help='where to save unmodified image', metavar='FILE')
@click.option('--projected-w', help='file with projection of the source image', type=str, metavar='FILE', required=True)
@click.option('--num-steps', help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--alpha', help='L2 reg coefficient', type=float, default=0.005, show_default=True)
@click.option('--lr', help='learning rate', type=float, default=0.05, show_default=True)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR')
def run_projection(
    network_pkl: str,
    source_prefix: Optional[str],
    projected_w: str,
    num_steps: int,
    alpha: float,
    lr: float,
    outdir: str,
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    device = torch.device('cuda')
    clip_loss = ClipSubloss(device, phrase)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    ws = np.load(projected_w)['w']
    ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
    assert ws.shape[1:] == (G.num_ws, G.w_dim)
    for idx, w in enumerate(ws):
        print('Modifying image #{}'.format(idx))
        new_w = transfer(
            G, clip_loss, w.unsqueeze(0),
            num_steps, alpha, lr,
            device
        )
        np.savez(f'{outdir}/projected_w_{idx:02d}.npz', w=new_w.cpu().numpy())
        generate_img(new_w, f'{outdir}/result_{idx:02d}.png')
        if source_prefix is not None:
            generate_img(w, f'{outdir}/{source_prefix}_{idx:02d}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
