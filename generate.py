"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""
import hydra
from omegaconf import DictConfig, OmegaConf

import os
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import utils.distributed as dist
from utils.utils import parse_int_list, open_url
# Bring tomography_utils packages onto the path
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'tomography_utils')))
from tomography_utils.mlem import mlem_precomputed

#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
    net, latents, cond_images, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, cond_images, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, cond_images, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])
    
    def randpoisson(self, input):
        return torch.stack([torch.poisson(input[i], generator=gen) for i, gen in enumerate(self.generators)])


#----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="config1")
def main(cfg : DictConfig) -> None:
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """
    # Launch multiprocessing
    dist.init()
    assert torch.cuda.is_available()
    device=torch.device('cuda')
    # select sampling dict config
    cfg = cfg.sampling
    assert isinstance(cfg, DictConfig)
    # The input seeds selected will be used to generate a random number generator 
    # The total number of seeds selected equals the total number of generated images
    seeds = parse_int_list(cfg.seeds)

    # batch seeds and dispatch them to every process 
    num_batches = ((len(seeds) - 1) // (cfg.max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # dataset instantiation for conditional low dose image
    dataset = hydra.utils.instantiate(cfg.dataset)
    if cfg.conditional :
        # Projection matrix 
        projector_path = os.path.join('training', 'training_data', f'projector_{64}x{64}.npz')
        assert os.path.isfile(projector_path) 
        projector = torch.from_numpy(np.load(projector_path)['projector']).to(device).to(torch.float32)
        mlem_max_iterations = 50

    all_images = []
    all_cond_images = []

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{cfg.network_pkl}"...')
    with open_url(cfg.network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{cfg.outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and conditionals
        rnd = StackedRandomGenerator(device, batch_seeds)
        if cfg.conditional :
            # generate every gaussian with a new a random number generator
            # latents channel dimension equal to half the nn input channel dimension when using conditional diffusion model 
            # since we concatenate (low count, high count) photons images along channel dimension
            latents = rnd.randn([batch_size, net.img_channels // 2] + net.img_resolution, device=device)
            # generate low dose sinogram indices with a new a random number generator, the same as its corresponding Gaussian sampled on the same "reality"
            cond_indices = rnd.randint(len(dataset), size=[batch_size,] , device=device).cpu()
            # select sinograms from the generated indices, sinograms are automatically at the right photons intensity (the low count setting) 
            # as the renormalisation is given in the config to the dataset object
            cond_images = torch.from_numpy(dataset[cond_indices]).to(device).to(torch.float32)
            # sample poisson counts with corresponding intensity and corresponding random number generator
            cond_images = rnd.randpoisson(cond_images)
            # reconstruct images and renormalize
            cond_images = mlem_precomputed(cond_images, projector, max_iter=mlem_max_iterations, device=device)
            cond_images = (cond_images / (torch.max(cond_images, dim=-1)[0][:, None] / 2) - 1).to(torch.float64).reshape(-1, 1, 64, 64)
            assert latents.shape == cond_images.shape, f"latents.shape {latents.shape}, cond_images.shape {cond_images.shape}"
        else : 
            latents = rnd.randn([batch_size, net.img_channels] + net.img_resolution, device=device)
            cond_images = None
        
        # Pick labels
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if cfg.class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, cfg.class_idx] = 1

        # Generate images.
        images = edm_sampler(net, latents, cond_images, class_labels, randn_like=rnd.randn_like,**OmegaConf.to_container(cfg.sampler_kwargs))
        
        # If the option is selected save the images in pixel format [0, 255]
        if cfg.is_image :
            images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for seed, image_np in zip(batch_seeds, images_np):
                image_dir = os.path.join(cfg.outdir, f'{seed-seed%1000:06d}') if cfg.subdirs else cfg.outdir
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:06d}.png')
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)
        # else it appends the results to list
        else:
            all_images.append(images)
            if cfg.conditional :
                all_cond_images.append(cond_images)

    # save images as a npz file of fomat N x H x W (assuming here C = 1)
    # concat the list of generetaed images into a unique tensor
    if not cfg.is_image :
        all_images = torch.cat(all_images, dim=0)[:, 0, ...]
        if cfg.conditional :
            all_cond_images = torch.cat(all_cond_images, dim=0)[:, 0, ...]
        # Create a destination list of tensor to gather the generated images from all the processes
        if dist.get_rank() == 0:
            all_images_dst = [torch.zeros_like(all_images).to(torch.float64) for _ in range(dist.get_world_size())]
            if cfg.conditional :
                all_cond_images_dst = [torch.zeros_like(all_images).to(torch.float64) for _ in range(dist.get_world_size())]
            torch.distributed.gather(all_images, all_images_dst , dst=0)
        else : 
            torch.distributed.gather(all_images, dst=0)
        if cfg.conditional:
            if dist.get_rank() == 0:
                torch.distributed.gather(all_cond_images, all_cond_images_dst , dst=0)
            else :
                torch.distributed.gather(all_cond_images , dst=0)
        # save into a numpy array
        if dist.get_rank() == 0:
            all_images_dst = torch.cat(all_images_dst, dim=0).cpu().numpy()
            os.makedirs(cfg.outdir, exist_ok=True)
            if cfg.conditional:
                all_cond_images_dst = torch.cat(all_cond_images_dst, dim=0).cpu().numpy()
                np.savez(cfg.outdir + f"/{dataset.name}_steps_{cfg.sampler_kwargs.num_steps}_stochasticity_{cfg.sampler_kwargs.S_churn}",\
                    images=all_images_dst, cond_images = all_cond_images_dst)  
            else :
                np.savez(cfg.outdir + f"/{dataset.name}_steps_{cfg.sampler_kwargs.num_steps}_stochasticity_{cfg.sampler_kwargs.S_churn}",\
                    images=all_images_dst)  

    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
