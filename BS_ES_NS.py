from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from utils5 import evaluat_metrics


def BS_ES_NS(ppi_list: Iterable, train_index: Iterable[int], val_index: Iterable[int], test_index: Iterable[int]):
    pairs = np.asarray(ppi_list, dtype=np.int64)
    train_nodes = set(pairs[np.asarray(list(train_index), dtype=np.int64)].reshape(-1).tolist())
    groups = ([], [], [])
    for edge_id in test_index:
        u, v = pairs[int(edge_id)]
        seen = int(u in train_nodes) + int(v in train_nodes)
        groups[2 - seen].append(int(edge_id))  # 2 -> BS, 1 -> ES, 0 -> NS
    return tuple(groups)


def BS_ES_NS_metrics(logits: torch.Tensor, labels: torch.Tensor, bs_index, es_index, ns_index):
    scores = []
    for index in (bs_index, es_index, ns_index):
        if not index:
            scores.append(0.0)
        else:
            idx = torch.as_tensor(index, dtype=torch.long, device=logits.device)
            scores.append(evaluat_metrics(logits[idx], labels[idx]))
    return tuple(scores)
