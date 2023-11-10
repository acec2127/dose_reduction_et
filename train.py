import hydra
from omegaconf import DictConfig, OmegaConf

import os
import re
import json

import torch

from utils.utils import EasyDict, Logger
import utils.distributed as dist

@hydra.main(version_base=None, config_path="configs", config_name="config1")
def main(cfg : DictConfig) -> None:
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".
    """
    # Load training config 
    cfg = cfg.training 

    # Program type of training
    assert cfg.secondary_options.training_type in ['images', 'pet']
    # Classical diffusion with images 
    if cfg.secondary_options.training_type == 'images' :
        from training.training_loop_images import training_loop
    # Specific diffusion model tailored for PET dose-enhancement 
    elif cfg.secondary_options.training_type == 'pet' :
        from training.training_loop_pet import training_loop
        
    # Make sure there is a GPU on the machine and launch processes
    assert torch.cuda.is_available()
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Training options to be passed to training loop
    # We first convert the DictConfig object to a Dict with the OmegaConf.to_container method
    # EasyDict class is a convenience class which enables access to values of a dict as class attributes
    opts = EasyDict(OmegaConf.to_container(cfg.options))

    # Dataloader arguments
    data_loader_args = OmegaConf.to_container(cfg.dataloader)

    # Make sure training at least one image
    opts.total_kimg = max(int(opts.total_kimg), 1)

    # Make sure selected learning rates options are valid 
    assert (opts.lr_max >= opts.lr_start) and (opts.lr_max >= opts.lr_end)
    
    # Random seed
    if opts.seed is None: 
        # If random seed not specified we randomly generate one
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        # We broadcast the seed value of process 0 to the other process in order to have the same random seed accross all processes
        torch.distributed.broadcast(seed, src=0)
    opts.seed = int(seed)

    # Transfer learning from network pikled state
    # Here the resume_kimg is kept to 0 since we do want to start a new training procedure from that checkpoint
    # However in that case we do not want to apply a warmup phase on the ema as we already done it before
    if opts.resume_pkl is not None:
        if opts.resume_state_dump is not None:
            raise Exception('resume_pkl and resume_state_dump cannot be specified at the same time')
        opts.ema_rampup_ratio = None
    # Resume from previous training state 
    # Here we want to resume the training from previous training state
    elif opts.resume_state_dump is not None:
        # Make sure the filepath leads to appropriate training state pt file
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume_state_dump))
        if not match or not os.path.isfile(opts.resume_state_dump):
            raise Exception('resume_state_dump must point to training-state-*.pt from a previous training run')
        # Link the pt file to its pkl counterpart
        opts.resume_pkl = os.path.join(os.path.dirname(opts.resume_state_dump), f'network-snapshot-{match.group(1)}.pkl')
        # Kimg at which training alted
        opts.resume_kimg = int(match.group(1))
    
    
    # Description string.
    cond_str = 'guided_cond' if cfg.secondary_options.guided_cond else None 
    desc = f'{cfg.secondary_options.dataset_name:s}-gpus{dist.get_world_size():d}-batch{opts.batch_size:d}'\
    +  (f'--{cond_str:s}' if cond_str is not None else '')
    # Optional description string to append to the running directory  
    if cfg.secondary_options.desc is not None:
        desc += f'-{cfg.secondary_options.desc}'


    # Pick output directory.
    """Need to be done only once so only rank 0 process proceed.
    Previous run in the running directory have a folder name which beggins by some id number.
    We observe the id number of previous run in the running directory and pick a new folder name for for the actual run 
    that begins with an id number equal to the max id number of previous run in the folder + 1, followed by the description string"""
    if dist.get_rank() != 0:
        opts.run_dir = None
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.run_dir):
            prev_run_dirs = [x for x in os.listdir(opts.run_dir) if os.path.isdir(os.path.join(opts.run_dir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        opts.run_dir = os.path.join(opts.run_dir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(opts.run_dir)

    # Print options.
    # print0 is a function where only rank 0 process print the output
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(opts, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {opts.run_dir}')
    dist.print0(f'Dataset path:            {cfg.dataset.path}')
    if cfg.secondary_options.guided_cond :
        dist.print0(f'Class-conditional:       {cfg.secondary_options.guided_cond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {opts.batch_size}')
    dist.print0()


    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(opts.run_dir, exist_ok=True)
        # Json file with all training options
        with open(os.path.join(opts.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(opts, f, indent=2)
        # Log text file with all printed outputs from sysout and errors from syserr
        Logger(file_name=os.path.join(opts.run_dir, 'log.txt'), file_mode='a', should_flush=True)
    
    # Train.
    training_loop(cfg, **opts, data_loader_args=data_loader_args)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
