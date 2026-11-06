import os
import resource
import sys

import torch
import torch.distributed as dist


def dist_env_vars() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def init_process_group(backend: str = "gloo", device: torch.device | None = None) -> None:
    # One thread per process, otherwise each process grabs every core.
    torch.set_num_threads(1)
    if backend == "gloo":
        # This machine's hostname does not resolve. Ranks talk over loopback.
        os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
        dist.init_process_group(backend=backend)
        return
    dist.init_process_group(backend=backend, device_id=device)


def shutdown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def rank() -> int:
    return dist.get_rank()


def world_size() -> int:
    return dist.get_world_size()


def print_peak_memory() -> None:
    who = f"rank {rank()}" if dist.is_initialized() else "process"
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
        print(f"{who} max CUDA memory {peak / 1024**2:.1f} MB", flush=True)
        return
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    # print(f"{who} peak RSS {peak / 1024**2:.1f} MB", flush=True)
