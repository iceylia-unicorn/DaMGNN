from __future__ import annotations

import numpy as np


def c_degree(graph) -> np.ndarray:
    return (graph.in_degrees().detach().cpu().numpy() + graph.out_degrees().detach().cpu().numpy())
