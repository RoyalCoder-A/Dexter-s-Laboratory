from dataclasses import dataclass

import numpy as np
import torch

from drl_in_action.genetic_algo.model import LayersType


@dataclass
class Population:
    params: torch.Tensor
    fit: int = 0


def spawn_population(N: int, layers: LayersType) -> list[Population]:
    size = 0
    for layer in layers:
        size += np.prod(layer) + layer[0]
    result: list[Population] = []
    for _ in range(N):
        vec = torch.randn(size) / 2.0
        result.append(Population(vec))
    return result


def recombine(x1: Population, x2: Population) -> tuple[Population, Population]:
    l = x1.params.shape[0]
    split_point = np.random.randint(l)
    child_1 = torch.zeros(l)
    child_2 = torch.zeros(l)
    child_1[0:split_point] = x1.params[0:split_point]
    child_1[split_point:] = x2.params[split_point:]
    child_2[0:split_point] = x2.params[0:split_point]
    child_2[split_point:] = x1.params[split_point:]
    return Population(child_1), Population(child_2)


def mutate(x: Population, rate: float) -> Population:
    num_to_change = int(rate * x.params.shape[0])
    idx = np.random.randint(low=0, high=x.params.shape[0], size=num_to_change)
    x.params[idx] = torch.rand(num_to_change) / 10.0
    return x
