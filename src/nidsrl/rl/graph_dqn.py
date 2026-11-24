"""Torch backend for the graph DQN: EdgeGAT (3 dueling outputs) + Polyak target network."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn

from nidsrl.graph.snapshot import NODE_DIM, GraphBatch, Snapshot, collate
from nidsrl.models.gat import EdgeGAT


class TorchGraphLearner:
    def __init__(
        self,
        edge_dim: int,
        hidden: int = 64,
        heads: int = 4,
        layers: int = 2,
        dueling: bool = True,
        lr: float = 3e-4,
        tau: float = 0.005,
        huber_delta: float = 1.0,
        grad_clip: float = 10.0,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.online = EdgeGAT(
            edge_dim, NODE_DIM, hidden, heads, layers, n_out=3, dueling=dueling
        ).to(self.device)
        self.target = copy.deepcopy(self.online).requires_grad_(False)
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.tau, self.delta, self.clip = tau, huber_delta, grad_clip
        self.meta = {"edge_dim": edge_dim, "hidden": hidden, "heads": heads,
                     "layers": layers, "dueling": dueling}  # fmt: skip

    def _t(self, a: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(a, dtype=dtype, device=self.device)

    def _inputs(self, g: GraphBatch) -> tuple[torch.Tensor, ...]:
        return (self._t(g.x), self._t(g.node_x), self._t(g.src, torch.long),
                self._t(g.dst, torch.long))  # fmt: skip

    @torch.no_grad()
    def q_snapshot(self, snap: Snapshot) -> np.ndarray:
        if snap.n_edges == 0:
            return np.zeros((0, 3), dtype=np.float32)
        self.online.eval()  # acting never runs in train mode
        return self.online(*self._inputs(collate([snap]))).cpu().numpy()

    @torch.no_grad()
    def next_q(self, g: GraphBatch) -> tuple[np.ndarray, np.ndarray]:
        self.online.eval()
        inp = self._inputs(g)
        return self.online(*inp).cpu().numpy(), self.target(*inp).cpu().numpy()

    def fit(
        self, g: GraphBatch, rows: np.ndarray, a: np.ndarray, y: np.ndarray,
        w: np.ndarray | None = None,
    ) -> dict[str, Any]:  # fmt: skip
        self.online.train()
        q = self.online(*self._inputs(g))
        q_sa = q[self._t(rows, torch.long), self._t(a, torch.long)]
        yt = self._t(y)
        per = nn.functional.huber_loss(q_sa, yt, delta=self.delta, reduction="none")
        loss = per.mean() if w is None else (per * self._t(w)).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.clip)
        self.opt.step()
        with torch.no_grad():
            for p_t, p in zip(self.target.parameters(), self.online.parameters(), strict=True):
                p_t.mul_(1.0 - self.tau).add_(p.detach(), alpha=self.tau)
        return {"loss": float(loss.item()), "td_error": (q_sa.detach() - yt).abs().cpu().numpy()}

    def state(self) -> dict[str, Any]:
        return {"online": copy.deepcopy(self.online.state_dict()), "meta": self.meta}

    def load(self, state: dict[str, Any]) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["online"])
