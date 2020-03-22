import os

import torch
import torch.distributed as dist

from mega_core.utils import gpu_indices, ompi_size, ompi_rank, get_master_ip


def init_dist(launcher, args, backend='nccl'):
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, args)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, args)
    else:
        raise ValueError('Invalid launcher type: {}'.format(launcher))


def _init_dist_pytorch(backend, args):
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend=backend, init_method="env://"
    )


def _init_dist_mpi(backend, args):
    gpus = list(gpu_indices())
    gpu_num = len(gpus)
    world_size = ompi_size()
    rank = ompi_rank()
    dist_url = 'tcp://' + get_master_ip() + ':23456'
    torch.cuda.set_device(int(gpus[0]))  # Set current GPU to the first
    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
        group_name='mtorch')
    print(
        "World Size is {}, Backend is {}, Init Method is {}, rank is {}, gpu num is{}"
        .format(world_size, backend, dist_url, ompi_rank(), gpu_num))
