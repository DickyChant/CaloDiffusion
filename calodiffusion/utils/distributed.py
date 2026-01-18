"""Utilities for PyTorch Distributed Data Parallel (DDP) training."""

import os
import torch
import torch.distributed as dist


def setup_distributed(backend, master_addr, master_port, rank, world_size):
    """Initialize the distributed process group.
    
    Args:
        backend: Distributed backend ('nccl' or 'gloo')
        master_addr: Master node address
        master_port: Master node port
        rank: Global rank of the current process
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    
    dist.init_process_group(
        backend=backend,
        init_method=f'env://',
        world_size=world_size,
        rank=rank
    )
    
    print(f"Initialized DDP: rank {rank}/{world_size}, backend={backend}")


def cleanup_distributed():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank=None):
    """Check if current process is the main process (rank 0).
    
    Args:
        rank: Optional rank to check. If None, uses current process rank.
        
    Returns:
        bool: True if this is the main process (rank 0)
    """
    if not dist.is_initialized():
        return True
    
    if rank is None:
        rank = dist.get_rank()
    
    return rank == 0


def get_rank():
    """Get the current process rank.
    
    Returns:
        int: Current process rank, or 0 if not in distributed mode
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """Get the total number of processes.
    
    Returns:
        int: Total number of processes, or 1 if not in distributed mode
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_tensor(tensor, world_size):
    """All-reduce a tensor and average across processes.
    
    Args:
        tensor: Tensor to reduce
        world_size: Total number of processes
        
    Returns:
        Reduced tensor averaged across all processes
    """
    if not dist.is_initialized():
        return tensor
    
    # Clone to avoid modifying the original
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt = rt / world_size
    return rt


def get_distributed_info_from_slurm():
    """Extract distributed training info from SLURM environment variables.
    
    Returns:
        dict: Dictionary with keys:
            - n_nodes: Number of nodes
            - gpus_per_node: Number of GPUs per node
            - node_id: Current node ID
            - local_rank: Local rank on current node
            - global_rank: Global rank across all nodes
            - world_size: Total number of processes
            - master_addr: Master node address
            - master_port: Master node port (default: 29500)
        Returns None if not running in SLURM environment
    """
    if 'SLURM_JOB_ID' not in os.environ:
        return None
    
    info = {}
    
    # Number of nodes
    info['n_nodes'] = int(os.environ.get('SLURM_NNODES', 1))
    
    # GPUs per node (tasks per node in SLURM)
    info['gpus_per_node'] = int(os.environ.get('SLURM_NTASKS_PER_NODE', 1))
    
    # Current node ID
    info['node_id'] = int(os.environ.get('SLURM_NODEID', 0))
    
    # Local rank (GPU ID on current node)
    info['local_rank'] = int(os.environ.get('SLURM_LOCALID', 0))
    
    # Global rank (unique ID across all nodes)
    info['global_rank'] = int(os.environ.get('SLURM_PROCID', 0))
    
    # Total world size
    info['world_size'] = info['n_nodes'] * info['gpus_per_node']
    
    # Master node address
    # Get the first node from the node list
    nodelist = os.environ.get('SLURM_NODELIST', 'localhost')
    # Simple parsing for single node or node range
    if '[' in nodelist:
        # Format like "node[001-002]" -> "node001"
        master_node = nodelist.split('[')[0] + nodelist.split('[')[1].split('-')[0].split(',')[0]
    else:
        master_node = nodelist.split(',')[0]
    info['master_addr'] = master_node
    
    # Master port (default or from env)
    info['master_port'] = os.environ.get('MASTER_PORT', '29500')
    
    return info
