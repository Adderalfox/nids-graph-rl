"""Window-snapshot replay for graph agents (numpy, unit-tested without torch).

A transition's state is one edge of a window graph, so replay stores whole window snapshots per
run plus transitions that point into them (window index, edge row). A training batch takes a few
random windows with ALL their transitions (one graph forward serves many transitions) and the
snapshots where those pairs next appeared (for the TD targets). Oldest runs are evicted first.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from nidsrl.graph.snapshot import GraphBatch, Snapshot, collate


@dataclass
class GraphRun:
    snaps: list[Snapshot]
    s0: np.ndarray  # snapshot index of the state
    i0: np.ndarray  # edge row inside that snapshot
    s1: np.ndarray  # snapshot index of the next state (ignored when done)
    i1: np.ndarray
    a: np.ndarray
    r: np.ndarray
    done: np.ndarray
    dt: np.ndarray

    @property
    def n(self) -> int:
        return len(self.a)

    def nbytes(self) -> int:
        return sum(s.nbytes() for s in self.snaps) + 8 * 8 * self.n


class GraphRunBuilder:
    """Collects one episode's snapshots and transitions while the env is being rolled out."""

    def __init__(self) -> None:
        self.snaps: list[Snapshot] = []
        self._win: dict[int, int] = {}
        self._parts: list[dict[str, np.ndarray]] = []

    def add_snapshot(self, t: int, snap: Snapshot) -> int:
        self._win[int(t)] = len(self.snaps)
        self.snaps.append(snap)
        return len(self.snaps) - 1

    def add_transitions(self, tr: dict[str, Any], final: bool = False) -> None:
        if len(tr["a"]) == 0:
            return
        s0 = np.array([self._win[int(t)] for t in tr["t0"]], dtype=np.int64)
        s1 = s0 if final else np.array([self._win[int(t)] for t in tr["t1"]], dtype=np.int64)
        self._parts.append(
            {"s0": s0, "i0": tr["i0"], "s1": s1, "i1": tr["i1"], "a": tr["a"], "r": tr["r"],
             "done": tr["done"], "dt": tr["dt"]}  # fmt: skip
        )

    def build(self) -> GraphRun:
        def cat(k: str, dtype: Any) -> np.ndarray:
            if not self._parts:
                return np.zeros(0, dtype=dtype)
            return np.concatenate([p[k] for p in self._parts]).astype(dtype)

        return GraphRun(
            self.snaps, cat("s0", np.int64), cat("i0", np.int64), cat("s1", np.int64),
            cat("i1", np.int64), cat("a", np.int64), cat("r", np.float32),
            cat("done", bool), cat("dt", np.int64),
        )  # fmt: skip


@dataclass
class UpdateBatch:
    graph: GraphBatch  # windows of the states
    rows: np.ndarray  # row of each transition's state in graph edges
    a: np.ndarray
    r: np.ndarray
    done: np.ndarray
    dt: np.ndarray
    next_graph: GraphBatch | None  # snapshots of the next states (None if all done)
    next_rows: np.ndarray  # row of each next state in next_graph (0 where done)


class GraphReplay:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.runs: deque[GraphRun] = deque()
        self.total = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.total

    def add(self, run: GraphRun) -> None:
        if run.n == 0:
            return
        self.runs.append(run)
        self.total += run.n
        while self.total > self.capacity and len(self.runs) > 1:
            self.total -= self.runs.popleft().n

    def sample(self, n_windows: int) -> UpdateBatch:
        if not self.runs:
            raise ValueError("cannot sample from an empty replay")
        w = np.array([r.n for r in self.runs], dtype=np.float64)
        picks = self._rng.choice(len(self.runs), size=n_windows, p=w / w.sum())
        s_snaps: list[Snapshot] = []
        a, r, done, dt = [], [], [], []
        next_keys: dict[tuple[int, int], int] = {}
        next_snaps: list[Snapshot] = []
        pending_next: list[tuple[int, int, int]] = []  # (transition offset, key index, i1)
        offset = 0
        base_rows: list[tuple[int, np.ndarray]] = []
        for k in picks:
            run = self.runs[int(k)]
            t = int(self._rng.integers(0, run.n))
            sel = np.flatnonzero(run.s0 == run.s0[t])
            j = len(s_snaps)
            s_snaps.append(run.snaps[int(run.s0[t])])
            base_rows.append((j, run.i0[sel]))
            a.append(run.a[sel])
            r.append(run.r[sel])
            done.append(run.done[sel])
            dt.append(run.dt[sel])
            for q, idx in enumerate(sel):
                if run.done[idx]:
                    continue
                key = (int(k), int(run.s1[idx]))
                if key not in next_keys:
                    next_keys[key] = len(next_snaps)
                    next_snaps.append(run.snaps[key[1]])
                pending_next.append((offset + q, next_keys[key], int(run.i1[idx])))
            offset += len(sel)
        graph = collate(s_snaps)
        rows = np.concatenate([graph.edge_ptr[j] + i0 for j, i0 in base_rows]).astype(np.int64)
        n_tr = offset
        next_rows = np.zeros(n_tr, dtype=np.int64)
        next_graph = collate(next_snaps) if next_snaps else None
        for pos, kidx, i1 in pending_next:
            assert next_graph is not None
            next_rows[pos] = next_graph.edge_ptr[kidx] + i1
        return UpdateBatch(
            graph, rows, np.concatenate(a), np.concatenate(r), np.concatenate(done),
            np.concatenate(dt), next_graph, next_rows,
        )  # fmt: skip
