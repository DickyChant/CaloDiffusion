import os
import numpy as np
import h5py as h5
import torch
from torch.utils.data import IterableDataset, get_worker_info

from calodiffusion.utils import HGCal_utils


class HGCalH5IterableDataset(IterableDataset):
    """Stream HGCal H5 files in chunks to avoid loading everything in RAM."""

    def __init__(
        self,
        files,
        data_folder,
        config,
        shape_pad,
        emax,
        emin,
        max_deposit,
        shower_map,
        dataset_num,
        orig_shape,
        ecut=0.0,
        nevts=-1,
        max_cells=None,
        shower_scale=200.0,
        pre_embed=False,
        NN_embed=None,
        use_pidm=True,
        chunk_size=256,
        embed_batch_size=256,
        shuffle_files=False,
        shuffle_samples=False,
        seed=1234,
        rank=0,
        world_size=1,
        verbose=False,
    ):
        super().__init__()
        self.verbose = verbose
        self.files = [
            f if os.path.isabs(f) else os.path.join(data_folder, f) for f in files
        ]
        self.shape_pad = (
            shape_pad.copy() if isinstance(shape_pad, list) else list(shape_pad)
        )
        self.emax = np.array(emax, dtype=np.float32)
        self.emin = np.array(emin, dtype=np.float32)
        self.max_deposit = max_deposit
        self.shower_map = shower_map
        self.dataset_num = dataset_num
        self.orig_shape = orig_shape
        self.ecut = ecut
        self.nevts = nevts
        self.max_cells = max_cells
        self.shower_scale = shower_scale
        self.pre_embed = pre_embed
        self.NN_embed = NN_embed
        self.use_pidm = use_pidm
        self.chunk_size = int(chunk_size)
        self.embed_batch_size = int(embed_batch_size)
        self.shuffle_files = shuffle_files
        self.shuffle_samples = shuffle_samples
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

        self._file_counts = []
        for f in self.files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"HGCal H5 file not found: {f}")
            with h5.File(f, "r") as h5f:
                n = int(h5f["showers"].shape[0])
            if self.nevts > 0:
                n = min(n, self.nevts)
            self._file_counts.append(n)

        self._rank_event_counts = []
        for r in range(self.world_size):
            self._rank_event_counts.append(
                int(np.sum(self._file_counts[r :: self.world_size]))
            )
        if self._rank_event_counts:
            self._min_rank_events = int(min(self._rank_event_counts))
            self._rank_total_events = int(self._rank_event_counts[self.rank])
        else:
            self._min_rank_events = 0
            self._rank_total_events = 0

        self._rank_files = self.files[self.rank :: self.world_size]
        self._rank_counts = self._file_counts[self.rank :: self.world_size]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        if self.world_size > 1:
            return self._min_rank_events
        return self._rank_total_events

    def _reshape_data(self, data):
        if not self.orig_shape:
            dshape = list(self.shape_pad)
            expected_per_sample = int(np.prod(dshape[1:]))
            if data.size % expected_per_sample != 0:
                raise ValueError(
                    f"Cannot reshape data: size {data.size} not divisible by "
                    f"{expected_per_sample} (target per-sample shape {tuple(dshape[1:])})"
                )
            if dshape[0] == -1:
                dshape[0] = data.size // expected_per_sample
            if data.size != int(np.prod(dshape)):
                dshape_auto = [-1] + dshape[1:]
                data = np.reshape(data, tuple(dshape_auto))
            else:
                data = np.reshape(data, tuple(dshape))
        else:
            data = np.reshape(data, (data.shape[0], -1))
        return data

    def __iter__(self):
        worker_info = get_worker_info()
        files = list(self._rank_files)
        counts = list(self._rank_counts)

        if self.shuffle_files and len(files) > 1:
            rng = np.random.RandomState(self.seed + 1000 * self.epoch + self.rank)
            order = rng.permutation(len(files))
            files = [files[i] for i in order]
            counts = [counts[i] for i in order]

        max_events = (
            self._min_rank_events if self.world_size > 1 else self._rank_total_events
        )
        if worker_info is not None and worker_info.num_workers > 1:
            per_worker = max_events // worker_info.num_workers
            extra = max_events % worker_info.num_workers
            max_events = per_worker + (1 if worker_info.id < extra else 0)

        if worker_info is not None and len(files) > 0:
            splits = np.array_split(np.arange(len(files)), worker_info.num_workers)
            idx = splits[worker_info.id]
            files = [files[i] for i in idx]
            counts = [counts[i] for i in idx]

        yielded = 0
        for fpath, n_events in zip(files, counts):
            if self.verbose:
                print(f"[STREAM] Loading {fpath}", flush=True)
            with h5.File(fpath, "r") as h5f:
                total = n_events
                for start in range(0, total, self.chunk_size):
                    if yielded >= max_events:
                        return
                    end = min(start + self.chunk_size, total)
                    gen_info = h5f["gen_info"][start:end].astype(np.float32)
                    showers = h5f["showers"][start:end]
                    if self.max_cells is not None:
                        showers = showers[:, :, : self.max_cells]
                    showers = showers.astype(np.float32) * self.shower_scale

                    if self.pre_embed:
                        if self.NN_embed is None:
                            raise RuntimeError("pre_embed=True but NN_embed is None")
                        showers = self.NN_embed.enc_batches(
                            torch.Tensor(showers), batch_size=self.embed_batch_size
                        )

                    e_raw = gen_info[:, 0]
                    showers_pp, layerE_pp = HGCal_utils.preprocess_hgcal_shower(
                        showers,
                        e_raw,
                        self.shape_pad,
                        self.shower_map,
                        dataset_num=self.dataset_num,
                        orig_shape=self.orig_shape,
                        ecut=self.ecut,
                        max_deposit=self.max_deposit,
                        verbose=False,
                    )

                    gen_pp = (gen_info - self.emin) / (self.emax - self.emin)
                    E = gen_pp[:, 0].astype(np.float32)
                    showers_pp = showers_pp.astype(np.float32)
                    if layerE_pp is not None:
                        layerE_pp = layerE_pp.astype(np.float32)

                    data = self._reshape_data(showers_pp)

                    if self.use_pidm and layerE_pp is None:
                        raise ValueError(
                            "PIDM requested but layerE is None. "
                            "Ensure SHOWERMAP includes 'layer'."
                        )

                    if self.shuffle_samples and data.shape[0] > 1:
                        rng = np.random.RandomState(
                            self.seed + 1000 * self.epoch + self.rank + start
                        )
                        order = rng.permutation(data.shape[0])
                    else:
                        order = range(data.shape[0])

                    for i in order:
                        if yielded >= max_events:
                            return
                        if self.use_pidm:
                            yield E[i], data[i], layerE_pp[i]
                        else:
                            yield E[i], data[i]
                        yielded += 1
