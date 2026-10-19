"""Wall-clock timing that waits for queued accelerator operations."""

from contextlib import contextmanager
from time import perf_counter

import torch


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@contextmanager
def elapsed_seconds(device):
    synchronize(device)
    started = perf_counter()
    measurement = {}
    try:
        yield measurement
    finally:
        synchronize(device)
        measurement["seconds"] = perf_counter() - started
