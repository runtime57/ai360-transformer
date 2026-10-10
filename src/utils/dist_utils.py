import os

import torch
import torch.distributed as dist


def init_distributed():
    """
    Initialize torch.distributed if the script is launched via torchrun
    (WORLD_SIZE > 1). Otherwise, does nothing.

    Returns:
        rank (int): global rank of the current process.
        local_rank (int): rank of the process on the current node.
        world_size (int): total number of processes.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size == 1:
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        dist.init_process_group(backend="gloo")
    return rank, local_rank, world_size


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main_process():
    return get_rank() == 0


def barrier():
    if is_distributed():
        dist.barrier()


def all_reduce_mean(values: dict, device):
    """
    Average a dict of python floats over all processes.

    Args:
        values (dict[str, float]): values of the current process.
        device (str): device for the communication tensor.
    Returns:
        values (dict[str, float]): values averaged over all processes.
    """
    if not is_distributed() or len(values) == 0:
        return values
    keys = list(values.keys())
    tensor = torch.tensor([float(values[k]) for k in keys], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return dict(zip(keys, tensor.tolist()))


def any_process(flag: bool, device):
    """
    Returns True on every process if the flag is True on at least one.
    """
    if not is_distributed():
        return flag
    tensor = torch.tensor([int(flag)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(tensor.item())


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()
