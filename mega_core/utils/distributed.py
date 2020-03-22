import os
import numpy as np
import subprocess
from contextlib import contextmanager
import logging


def ompi_rank():
    """Find OMPI world rank without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_RANK') or 0)


def ompi_size():
    """Find OMPI world size without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_SIZE') or 1)


def ompi_local_rank():
    """Find OMPI local rank without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK') or 0)


def ompi_local_size():
    """Find OMPI local size without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE') or 1)


def ompi_universe_size():
    """Find OMPI universe size without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_UNIVERSE_SIZE') or 1)


@contextmanager
def run_and_terminate_process(*args, **kwargs):
    """Run a process and terminate it at the end
    """
    p = None
    try:
        p = subprocess.Popen(*args, **kwargs)
        yield p
    finally:
        if not p:
            return
        try:
            p.terminate()  # send sigterm
        except OSError:
            pass
        try:
            p.kill()  # send sigkill
        except OSError:
            pass


def get_gpus_nocache():
    """List of NVIDIA GPUs
    """
    cmds = 'nvidia-smi --query-gpu=name --format=csv,noheader'.split(' ')
    with run_and_terminate_process(
            cmds, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1) as process:
        return [
            str(line).strip() for line in iter(process.stdout.readline, b'')
        ]


_GPUS = get_gpus_nocache()


def get_gpus():
    """List of NVIDIA GPUs
    """
    return _GPUS


def gpu_indices(divisible=True):
    """Get the GPU device indices for this process/rank
    :param divisible: if GPU count of all ranks must be the same
    :rtype: list[int]
    """
    local_size = ompi_local_size()
    local_rank = ompi_local_rank()
    assert 0 <= local_rank < local_size, \
        "Invalid local_rank: {} local_size: {}".format(local_rank, local_size)
    gpu_count = len(get_gpus())
    assert gpu_count >= local_size > 0, \
        "GPU count: {} must be >= LOCAL_SIZE: {} > 0".format(gpu_count, local_size)
    if divisible:
        ngpu = int(gpu_count / local_size)
        gpus = np.arange(local_rank * ngpu, (local_rank + 1) * ngpu)
        if gpu_count % local_size != 0:
            logging.warning(
                "gpu_count: {} not divisible by local_size: {}; " + "some GPUs may be unused"
                .format(gpu_count, local_size))
    else:
        gpus = np.array_split(range(gpu_count), local_size)[local_rank]
    return gpus.astype(int)
