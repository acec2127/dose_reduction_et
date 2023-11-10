"""Collection of functions to manage distributed data parallelism. """

import os
import torch
from utils.training_stats import init_multiprocessing

#----------------------------------------------------------------------------

def init():
    # IP adress of the the rank 0 process
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    # Free port fro the rank 0 process
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    # Rank of the process but if program is
    # initialized with torchrun this is handled automatically
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    # Relative rank of the process within a machine. 
    # Since we are using a single machine LOCAL_RANK = RANK.
    # If initialized with torchrun this is handled automatically
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    # Total number of processes but if program is
    # initialized with torchrun this is handled automatically
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'

    # Backend communication protocol between gpus
    backend = 'gloo' if os.name == 'nt' else 'nccl'
    # Initialisation of the processes
    torch.distributed.init_process_group(backend=backend, init_method='env://')
    # Attribute different GPU for each process
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

    # Device to use for multiprocess communication. None = single-process.
    sync_device = torch.device('cuda') if get_world_size() > 1 else None
    # Initialization `torch_utils.training_stats` for collecting statistics across multiple processes.
    init_multiprocessing(rank=get_rank(), sync_device=sync_device)

#----------------------------------------------------------------------------

def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

#----------------------------------------------------------------------------

def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

#----------------------------------------------------------------------------

def should_stop():
    return False

#----------------------------------------------------------------------------

def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)

#----------------------------------------------------------------------------
