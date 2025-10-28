from typing import TypedDict, cast

import numpy as np
import torch


LayersType = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


class UnpackedParams(TypedDict):
    l1: tuple[torch.Tensor, torch.Tensor]
    l2: tuple[torch.Tensor, torch.Tensor]
    l3: tuple[torch.Tensor, torch.Tensor]


def model(state: torch.Tensor, params: UnpackedParams) -> torch.Tensor:
    y = torch.nn.functional.linear(state, params["l1"][0], params["l1"][1])
    y = torch.nn.functional.relu(y)
    y = torch.nn.functional.linear(y, params["l2"][0], params["l2"][1])
    y = torch.nn.functional.relu(y)
    y = torch.nn.functional.linear(y, params["l3"][0], params["l3"][1])
    y = torch.nn.functional.softmax(y)
    return y


def unpack_params(
    params: torch.Tensor,
    layers: LayersType,
) -> UnpackedParams:
    result = {}
    e = 0
    for i, l in enumerate(layers):
        s, e = e, e + np.prod(l)
        weights = params[s:e].view(l)
        s, e = e, e + l[0]
        biases = params[s:e]
        result[f"l{i + 1}"] = (weights, biases)
    return cast(UnpackedParams, result)
