import numpy as np
import torch
from functools import reduce
from numpy.lib import recfunctions
from torch.utils.data import IterableDataset


class Dataset(IterableDataset):
    def __init__(
        self,
        files,
        data_type="training",
        weighted_sampling=False,
        device="cpu",
        max_length=1,
        verbose=0,
    ):
        self.verbose = verbose
        self.files = files
        self.data_type = data_type
        if data_type == "validation":
            self.data_type = "test"

        if len(files):
            f = np.load(files[0])
            self.all_number_of_samples = len(f[f.files[0]]) * len(files)
        else:
            self.all_number_of_samples = 0
        self.weighted_sampling = weighted_sampling

        self.device = device

    def __len__(self):
        return int(self.all_number_of_samples)

    def __getitem__(self, index):
        raise NotImplementedError

    def shuffleFileList(self):
        np.random.shuffle(self.files)

    def __iter__(self):
        # Distributed training support: each rank gets a separate set of files
        # to iterate over to avoid duplicate data across processes
        import torch.distributed as dist
        
        files_to_read = list(self.files)
        
        # First, split files by DDP rank if running distributed
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            # Split files across ranks
            files_to_read = [f for i, f in enumerate(files_to_read) if i % world_size == rank]
        
        # Multi-worker support: each worker gets a separate set of files
        # to iterate over to avoid double iterations within a single process
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            files_to_read = np.array_split(files_to_read, worker_info.num_workers)[
                worker_info.id
            ]

        for file in files_to_read:
            if self.verbose:
                print(f"Loading {file}")
            with np.load(file) as data:
                Es = data["E"]
                layers = data["layers"]
                showers = data["showers"]
                for E, layer, shower in zip(Es, layers, showers):
                    yield E, layer, shower
        return None
