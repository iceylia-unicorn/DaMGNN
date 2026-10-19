from __future__ import annotations

from pathlib import Path

import numpy as np

from degree import c_degree


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib. Install it with `pip install matplotlib`.") from exc
    return plt


def _save_or_show(plt, output_path=None):
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_G_histogram(graph, output_path=None):
    plt = _plt()
    plt.hist(c_degree(graph), bins="auto")
    plt.xlabel("Total node degree")
    plt.ylabel("Number of proteins")
    _save_or_show(plt, output_path)


def plot_G_bar(graph, output_path=None):
    plt = _plt()
    degree, count = np.unique(c_degree(graph), return_counts=True)
    plt.bar(degree, count)
    plt.xlabel("Total node degree")
    plt.ylabel("Number of proteins")
    _save_or_show(plt, output_path)


def plot_G_sample(graph, output_path=None):
    return plot_G_histogram(graph, output_path)
