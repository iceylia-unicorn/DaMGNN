from __future__ import annotations

import os
from collections import defaultdict, deque
from pathlib import Path

import dgl
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
REQUIRED_RELATIONS = {"SEQ", "STR_KNN", "STR_DIS"}


def _data_root() -> Path:
    return Path(os.environ.get("DAMGNN_DATA_ROOT", ROOT / "data"))


def _dataset_path(dataset: str) -> Path:
    path = _data_root() / dataset / "processed.pt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing preprocessed dataset: {path}. See DATA_FORMAT.md for the required "
            "MAPE-PPI graph and interaction bundle format."
        )
    return path


def _load_bundle(dataset: str) -> dict:
    bundle = torch.load(_dataset_path(dataset), map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError("processed.pt must contain a dictionary; see DATA_FORMAT.md")
    required = {"protein_graphs", "protein_ids", "pairs", "labels"}
    missing = required.difference(bundle)
    if missing:
        raise KeyError(f"processed.pt is missing required keys: {sorted(missing)}")
    return bundle


def _validate_protein_graphs(graphs, protein_ids) -> None:
    if len(graphs) != len(protein_ids):
        raise ValueError("protein_graphs and protein_ids must have the same length")
    for position, graph in enumerate(graphs):
        if not isinstance(graph, dgl.DGLHeteroGraph):
            raise TypeError(f"protein_graphs[{position}] is not a DGL heterograph")
        if "amino_acid" not in graph.ntypes or "x" not in graph.nodes["amino_acid"].data:
            raise ValueError(f"protein_graphs[{position}] needs amino_acid node features named 'x'")
        present = {etype[1] for etype in graph.canonical_etypes}
        missing = REQUIRED_RELATIONS.difference(present)
        if missing:
            raise ValueError(f"protein_graphs[{position}] is missing relations: {sorted(missing)}")


def _normalise_pairs_and_labels(bundle: dict, num_proteins: int):
    pairs = np.asarray(bundle["pairs"], dtype=np.int64)
    labels = torch.as_tensor(bundle["labels"], dtype=torch.float32)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape [num_interactions, 2]")
    if labels.ndim != 2 or labels.shape[0] != pairs.shape[0]:
        raise ValueError("labels must have shape [num_interactions, num_classes]")
    if pairs.size and (pairs.min() < 0 or pairs.max() >= num_proteins):
        raise ValueError("pairs contain protein indices outside protein_ids")
    return pairs, labels


def _edge_order(pairs: np.ndarray, mode: str, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    count = len(pairs)
    if mode == "random":
        return rng.permutation(count).tolist()
    if mode not in {"bfs", "dfs"}:
        raise ValueError("split_mode must be one of: random, bfs, dfs")
    adjacency = defaultdict(list)
    for edge_id, (u, v) in enumerate(pairs.tolist()):
        adjacency[u].append((v, edge_id))
        adjacency[v].append((u, edge_id))
    for neighbours in adjacency.values():
        rng.shuffle(neighbours)
    unseen_edges = set(range(count))
    unseen_nodes = set(adjacency)
    order = []
    while unseen_nodes:
        start = int(rng.choice(np.fromiter(unseen_nodes, dtype=np.int64)))
        frontier = deque([start])
        unseen_nodes.remove(start)
        while frontier:
            node = frontier.popleft() if mode == "bfs" else frontier.pop()
            for neighbour, edge_id in adjacency[node]:
                if edge_id in unseen_edges:
                    order.append(edge_id)
                    unseen_edges.remove(edge_id)
                if neighbour in unseen_nodes:
                    unseen_nodes.remove(neighbour)
                    frontier.append(neighbour)
    order.extend(sorted(unseen_edges))
    return order


def _split_indices(bundle: dict, pairs: np.ndarray, mode: str, seed: int) -> dict:
    splits = bundle.get("splits", {})
    candidate = splits.get(mode) if isinstance(splits, dict) else None
    if candidate is None and isinstance(splits, dict) and {"train_index", "val_index", "test_index"}.issubset(splits):
        candidate = splits
    if candidate is not None:
        required = {"train_index", "val_index", "test_index"}
        if not required.issubset(candidate):
            raise KeyError(f"Stored {mode} split must contain {sorted(required)}")
        return {key: [int(i) for i in candidate[key]] for key in required}
    order = _edge_order(pairs, mode, seed)
    train_end, val_end = int(.6 * len(order)), int(.8 * len(order))
    return {"train_index": order[:train_end], "val_index": order[train_end:val_end], "test_index": order[val_end:]}


def _training_graph(pairs: np.ndarray, train_index: list[int], num_nodes: int):
    train_pairs = pairs[np.asarray(train_index, dtype=np.int64)]
    if len(train_pairs):
        source = np.concatenate([train_pairs[:, 0], train_pairs[:, 1]])
        target = np.concatenate([train_pairs[:, 1], train_pairs[:, 0]])
    else:
        source = target = np.empty(0, dtype=np.int64)
    return dgl.graph((torch.from_numpy(source), torch.from_numpy(target)), num_nodes=num_nodes).to(DEVICE)


def load_data(dataset: str, split_mode: str, seed: int):
    """Load one DaMGNN dataset and construct a message-passing graph from train edges only."""
    bundle = _load_bundle(dataset)
    graphs, protein_ids = bundle["protein_graphs"], bundle["protein_ids"]
    _validate_protein_graphs(graphs, protein_ids)
    pairs, labels = _normalise_pairs_and_labels(bundle, len(protein_ids))
    split = _split_indices(bundle, pairs, split_mode, seed)
    ppi_graph = _training_graph(pairs, split["train_index"], len(protein_ids))
    return graphs, ppi_graph, pairs, labels.to(DEVICE), split


def load_pretrain_data(dataset: str):
    """Load only protein graphs for VQ-VAE pre-training."""
    bundle = _load_bundle(dataset)
    graphs, protein_ids = bundle["protein_graphs"], bundle["protein_ids"]
    _validate_protein_graphs(graphs, protein_ids)
    return graphs


def collate(graphs):
    """Batch protein heterographs for the VQ-VAE encoder."""
    return dgl.batch(graphs)
