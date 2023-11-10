"""Main training loop."""
import hydra
from omegaconf import OmegaConf

import os
import time
import copy
import json
import pickle
import psutil

import numpy as np
import torch
from torch.optim import Adam

from .sampler import InfiniteSampler

# Bring utils and tomography_utils packages onto the path
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'utils')))
sys.path.append(os.path.abspath(os.path.join('..', 'tomography_utils')))

from tomography_utils.mlem import mlem_precomputed
                
from utils.torch_utils import print_module_summary, copy_params_and_buffers, ddp_sync, check_ddp_consistency
from utils.utils import open_url, format_time
import utils.training_stats as training_stats
import utils.distributed as dist

#----------------------------------------------------------------------------
def training_loop(
    cfg,                                        # Training configurations with dataset, network, loss, optmizer, augmentation pipeline classes instations parameters 
    run_dir             = '.',                  # Output directory.
    seed                = 0,                    # Global random seed.
    batch_size          = 64,                   # Total batch size for one training iteration.
    batch_gpu           = None,                 # Limit batch size per GPU, None = no limit.
    total_kimg          = 10,                   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 0.1,                  # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,                 # EMA ramp-up ratio, None = no rampup.
    lr_max              = 2e-3,                 # Maximum learning rate 
    lr_rampup_ratio     = 0.1,                  # Learning rate ramp-up ratio of total amount of training, need to be comprised between 0 and 1.
    lr_start            = 1e-4,                 # Sarting learning rate
    lr_end              = 1e-5,                 # End Learning rate 
    kimg_per_tick       = 50,                   # Interval of progress prints.
    snapshot_ticks      = 50,                   # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,                  # How often to dump training state, None = disable.
    resume_pkl          = None,                 # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,                 # Start from the given training state, None = reset training state.
    resume_kimg         = 0,                    # Start from the given training progress.
    cudnn_benchmark     = True,                 # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'), # Assign the default CUDA device for each process
    data_loader_args  = {},                     # Options for torch.utils.data.DataLoader.
):
    # Initialize.
    start_time = time.time()
    # Making sure every process has a different numpy and torch random seed 
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
    torch.backends.cudnn.benchmark = cudnn_benchmark
    """TF32 tensor cores are designed to achieve better performance on matmul and convolutions on torch.float32 tensors by rounding 
    input data to have 10 bits of mantissa, and accumulating results with FP32 precision, maintaining FP32 dynamic range."""
    # A bool that controls where TensorFloat-32 tensor cores may be used in cuDNN convolutions on Ampere or newer GPUs
    torch.backends.cudnn.allow_tf32 = False
    # A bool that controls whether TensorFloat-32 tensor cores may be used in matrix multiplications on Ampere or newer GPUs
    torch.backends.cuda.matmul.allow_tf32 = False
    # A bool that controls whether reduced precision reductions (e.g., with fp16 accumulation type) are allowed with fp16 GEMMs.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    """batch_gpu is the max batch size per gpu. If this value is bigger than bach_size divided by the number of processes,
    denoted by batch_gpu_total, then all good, we just lower that value, otherwise we need to set the total number 
    of accumulation rounds such that batch_gpu multiplied by that value is superior to batch_gpu_total """
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    # This assertion means that we need to choose batch sizes such that : batch_gpu | batch_gpu_total if batch_gpu < batch_gpu_total
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset, sampler and dataloader
    dist.print0('Loading dataset...')
    dataset_obj = hydra.utils.instantiate(cfg.dataset)

    # ONLY FOR SINOGRAMS : Ensure images height number of pixels is equal to images width number of pixels
    # ONLY FOR SINOGRAMS : 
    assert dataset_obj.resolution[0] == dataset_obj.resolution[1]
    resolution = dataset_obj.resolution[0]

    # Ensure dataset images are divisible by th total number of stages in the network
    assert dataset_obj.resolution[0] % 2 ** (len(cfg.network.channel_mult) - 1) == 0

    # Here we use inifinite sampler, which loops over the dataset indefinitely, shuffling items as it goes 
    # and makes sure that each process sample a different image. The sampler is distributed, meaning that each process output a different image.
    # This is done by specifying a common seed for each process tot he sampler and then selecting disjoint generated images for each process rank
    dataset_sampler = InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    # Since we have an infinte sampler, we can just iterate image after image by first transforming 
    # the dataloader into a generator with iter() and then calling the next item with next() 
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_args))
    
    # Construct network.
    dist.print0('Constructing network...')
    net = hydra.utils.instantiate(cfg.network) 
    # Making sure network is in training mode and that gradient accumulation is enabled
    net.train().requires_grad_(True).to(device)
    # Apply a first forward pass on the model and print the module with the total number of parameters in it
    if dist.get_rank() == 0:
        with torch.no_grad():
            # We concatenate images and conditional images (low-count and high count reconstructed images)
            # along dimension channel dimension, so the channel dimension of a single reconstructed image is net.img_channels // 2
            images = torch.zeros([batch_gpu, net.img_channels // 2] + net.img_resolution, device=device)
            sigma = torch.ones([batch_gpu], device=device)
            cond_images = torch.zeros([batch_gpu, net.img_channels // 2] + net.img_resolution, device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            print_module_summary(net, [images, sigma, cond_images, labels], max_nesting=2)
    
    # Setup loss function, optimizer and augmentation pipeline
    dist.print0('Setting up optimizer...')
    loss_fn = hydra.utils.instantiate(cfg.loss)
    optimizer = Adam(params=net.parameters(), **OmegaConf.to_container(cfg.optimizer)) # subclass of torch.optim.Optimizer
    augment_pipe = hydra.utils.instantiate(cfg.augment)

    # Setup parallelization and coordination of forward and backward operations accross processes
    # Argument broadcast_buffers is set to false so that each process maintains its own copy of the buffers
    # as we do not need to synchronize buffers during forward pass in our model - there are no modules using statitics of the batch 
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    # Setup Exponential Moving Average
    # We put it into eval mode and do not require gradient accumulation as we proceed backward pass on the actual net
    # However only the ema network is retained in the pkl snapshot 
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Transfer learning or resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        # The actual network we are using is the resulting ema one 
        copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    # If resuming training then we need to load back the network parameters and optmizer states 
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Projection matrix 
    projector_path = os.path.join('training', 'training_data', f'projector_{resolution}x{resolution}.npz')
    assert os.path.isfile(projector_path) 
    projector = torch.from_numpy(np.load(projector_path)['projector']).to(device).to(torch.float32)

    # Sinograms parameters
    low_dose_reduce_factor = 0.25
    mlem_max_iterations = 50
    
    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    # Initial maintenance time account for the time to run the portion of code in traning_loop located above this line
    maintenance_time = tick_start_time - start_time
    total_nimg = total_kimg * 1000
    # Round to nearest multiple of batch_size
    warmup_phase_nimg = batch_size * round( (lr_rampup_ratio * total_nimg) / batch_size) 
    annealing_phase_nimg = total_nimg - warmup_phase_nimg
    stats_jsonl = None
    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            # Acculmulates gradient accross all accumulation rounds and synchronize only once the process reached the last round 
            with ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                sinograms_high_dose = next(dataset_iterator).to(device).to(torch.float32) # This line is normally written as : images, labels = next(dataset_iterator) (and then don't forget .to(device)!)
                # However since our model is unconditional with respect to the classifier free guidance option, in order to save GPU memory our Dataset object output only images without labels.
                # Here label is the the tensor we defined earlier with only zeros.  
                # Poisson simulation of high dose sinograms
                sinograms_high_dose = torch.distributions.poisson.Poisson(sinograms_high_dose).sample() 
                # simulation of low dose sinograms on the same probability event through binomial
                sinograms_low_dose = torch.distributions.binomial.Binomial(sinograms_high_dose, low_dose_reduce_factor).sample()  
                # reconstruction through mlem
                recon_high_dose = mlem_precomputed(sinograms_high_dose, projector, max_iter=mlem_max_iterations, device=device)
                recon_low_dose = mlem_precomputed(sinograms_low_dose, projector, max_iter=mlem_max_iterations, device=device)
                # normalize reconstructions such that they lie in [-1, 1]
                recon_high_dose = recon_high_dose / (torch.max(recon_high_dose, dim=-1)[0][:, None] / 2) - 1
                recon_low_dose = recon_low_dose / (torch.max(recon_low_dose, dim=-1)[0][:, None] / 2) - 1
                # feed concatenated images along channel dimension to loss function
                # feed the zero vector label and the augment pipeline
                loss = loss_fn(net=ddp, images=torch.stack((recon_high_dose.reshape(-1, 64, 64), recon_low_dose.reshape(-1, 64, 64)), dim=1), 
                               labels=labels, augment_pipe=augment_pipe)
                # We report loss for later statistics updated every tick
                training_stats.report('Loss/loss', loss)
                # Backward pass 
                loss.sum().mul(1. / batch_gpu_total).backward()

        # Update weights.
        # We use linear warm-up and then a cosine annealing 
        for g in optimizer.param_groups:
            if cur_nimg <= warmup_phase_nimg :
                # the max is taken in case warmup_phase_nimg is equal to 0, in that case we want to start from lr_max
                g['lr'] = lr_start + ( lr_max - lr_start ) * min( cur_nimg / max(warmup_phase_nimg, 1e-8), 1 )
            else  :
                g['lr'] = lr_end + ( lr_max - lr_end ) * 0.5 * ( 1 + np.cos( np.pi * (cur_nimg - warmup_phase_nimg) / annealing_phase_nimg ) )
        for param in net.parameters():
            # Step to make sure Nan, +infty, -infty values in parameters gradients aren't present
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        # Apply a gradient step
        optimizer.step()

        # Update EMA.
        # We take the half-life parametrization and then transform it into the smoothing factor parameterization. 
        # We optionally add an initial warm-up period
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats solely of process 0
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        # Time between end of the last tick and the begining of the next tick - basically the time to run ther code in the loop beneath this printing block
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        # Amount of physical RAM that the process currently occupies in the system's memory.
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        # Returns the maximum GPU memory occupied by tensors in bytes for a given device.
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        # Returns the maximum GPU memory managed by the caching allocator, which is the allocated memory plus pre-cached memory.
        # So max_memory_reserved >= max_memory_allocated
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        # Reset the starting point in tracking the gpu permances metrics
        torch.cuda.reset_peak_memory_stats()
        # Only printing cpu / gpu perfomrances of process 0
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            value = copy.deepcopy(ema).eval().requires_grad_(False)
            check_ddp_consistency(value)
            data = dict(ema = value.cpu())
            del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
