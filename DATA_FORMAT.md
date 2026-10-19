# Data format

Place one file at:

```text
data/<SHS27k|SHS148k|STRING>/processed.pt
```

Save a dictionary with `torch.save(bundle, path)`:

| Key | Type | Meaning |
| --- | --- | --- |
| `protein_graphs` | `list[dgl.DGLHeteroGraph]` | One graph per protein, in `protein_ids` order. Each graph has node type `amino_acid`, node feature `x` of shape `[residues, 7]`, and relations `SEQ`, `STR_KNN`, `STR_DIS`. |
| `protein_ids` | list of strings | Stable protein identifiers. |
| `pairs` | array-like `[N, 2]` | Zero-based indices into `protein_ids`. |
| `labels` | array-like `[N, 7]` | Multi-hot float/binary labels in this order: reaction, binding, ptmod, activation, inhibition, catalysis, expression. |
| `splits` (optional) | dict | Fixed paper splits, e.g. `{"random": {"train_index": [...], "val_index": [...], "test_index": [...]}}`. |

If `splits` is omitted, the loader creates deterministic 60:20:20 Random/BFS/DFS splits using the provided seed.

Set `DAMGNN_DATA_ROOT` to use another data directory.
