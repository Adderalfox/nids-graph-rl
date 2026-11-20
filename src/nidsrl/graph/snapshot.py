"""Window graph snapshots: hosts are nodes, active host pairs (the env's decision units) are edges.

Identity-free by construction: nodes are re-indexed locally per window, and node features are
degree/volume statistics only. Host IDs are never inputs (ADR 0003). Edge features are the
environment's pair-window observation (optionally masked by the feature policy), which already
carries each pair's own history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nidsrl.rl.env import N_CONT
from nidsrl.rl.rundata import N_FEATURES

LOGN_COL = N_FEATURES + N_CONT  # obs column holding log1p(flows) / 10
NODE_DIM = 4  # log out-degree, log in-degree, log out-flows, log in-flows
_NODE_SCALE = np.array([5.0, 5.0, 12.0, 12.0], dtype=np.float32)


@dataclass
class Snapshot:
    x: np.ndarray  # (E, d) edge features
    src: np.ndarray  # (E,) local node index
    dst: np.ndarray  # (E,)
    node_x: np.ndarray  # (N, NODE_DIM)

    @property
    def n_edges(self) -> int:
        return len(self.src)

    @property
    def n_nodes(self) -> int:
        return len(self.node_x)

    def nbytes(self) -> int:
        return int(self.x.nbytes + self.src.nbytes + self.dst.nbytes + self.node_x.nbytes)


def build_snapshot(
    obs: np.ndarray, src_ids: np.ndarray, dst_ids: np.ndarray, keep: np.ndarray | None = None
) -> Snapshot:
    """`obs` must be the FULL observation (the flow count is read from it before masking)."""
    e = len(obs)
    d = len(keep) if keep is not None else obs.shape[1]
    if e == 0:
        z = np.zeros(0, dtype=np.int64)
        return Snapshot(np.zeros((0, d), np.float32), z, z, np.zeros((0, NODE_DIM), np.float32))
    nodes, inv = np.unique(np.concatenate([src_ids, dst_ids]), return_inverse=True)
    s, t = inv[:e].astype(np.int64), inv[e:].astype(np.int64)
    n = len(nodes)
    flows = np.expm1(obs[:, LOGN_COL].astype(np.float64) * 10.0).clip(min=0.0)
    node_x = np.stack(
        [
            np.log1p(np.bincount(s, minlength=n)),
            np.log1p(np.bincount(t, minlength=n)),
            np.log1p(np.bincount(s, weights=flows, minlength=n)),
            np.log1p(np.bincount(t, weights=flows, minlength=n)),
        ],
        axis=1,
    ).astype(np.float32) / _NODE_SCALE
    x = (obs[:, keep] if keep is not None else obs).astype(np.float32)
    return Snapshot(x, s, t, node_x)


@dataclass
class GraphBatch:
    """Disjoint union of snapshots. edge_ptr[i]:edge_ptr[i+1] are snapshot i's edges."""

    x: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    node_x: np.ndarray
    edge_ptr: np.ndarray
    node_ptr: np.ndarray


def collate(snaps: list[Snapshot]) -> GraphBatch:
    node_ptr = np.concatenate([[0], np.cumsum([s.n_nodes for s in snaps])]).astype(np.int64)
    edge_ptr = np.concatenate([[0], np.cumsum([s.n_edges for s in snaps])]).astype(np.int64)
    return GraphBatch(
        x=np.concatenate([s.x for s in snaps]),
        src=np.concatenate([s.src + node_ptr[i] for i, s in enumerate(snaps)]),
        dst=np.concatenate([s.dst + node_ptr[i] for i, s in enumerate(snaps)]),
        node_x=np.concatenate([s.node_x for s in snaps]),
        edge_ptr=edge_ptr,
        node_ptr=node_ptr,
    )
