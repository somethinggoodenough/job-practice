from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass(frozen=True)
class Case:
    name: str
    shapes: tuple[tuple[int, ...], ...]


@dataclass
class Spec:
    name: str
    description: str
    cases: list[Case]
    make_inputs: Callable
    reference: Callable
    implementations: dict[str, Callable]
    logical_bytes: Callable
    patterns: tuple[str, ...] = ("random", "zeros", "tiny", "large")
    tolerances: dict = field(default_factory=lambda: {
        "float32": (2e-4, 2e-5),
        "float16": (5e-3, 5e-3),
        "bfloat16": (3e-2, 3e-2),
    })


def tensor(shape, dtype, device, generator, pattern):
    x = torch.randn(shape, dtype=torch.float32, device=device, generator=generator)
    if pattern == "zeros":
        x.zero_()
    elif pattern == "tiny":
        x.mul_(1e-4)
    elif pattern == "large":
        x.mul_(100)
    return x.to(dtype)


def io_bytes(inputs, output):
    """Logical I/O, not measured DRAM traffic (caches/reloads are excluded)."""
    return sum(x.numel() * x.element_size() for x in inputs) + output.numel() * output.element_size()
