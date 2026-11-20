"""Edge-classifying graph attention network (pure torch, no torch_geometric).

Each layer is a GATv2-style attention over incoming messages in BOTH directions (a host
aggregates what it sends and receives; a learned direction embedding keeps them apart), with
edge features inside the attention score. The output head sees [h_src, h_dst, edge embedding]
and produces n_out values per edge (1 logit for supervised, 3 Q-values for the RL agent).
`layers=0` removes all message passing (edge-MLP ablation: same parameters elsewhere).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn

from nidsrl.graph.snapshot import GraphBatch, Snapshot, collate


def scatter_softmax(score: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax of score (M, H) within groups idx (M,) over n groups."""
    h = score.shape[1]
    mx = torch.full((n, h), float("-inf"), device=score.device, dtype=score.dtype)
    mx = mx.scatter_reduce(0, idx[:, None].expand(-1, h), score, reduce="amax", include_self=True)
    ex = (score - mx[idx]).exp()
    den = torch.zeros((n, h), device=score.device, dtype=score.dtype).index_add_(0, idx, ex)
    return ex / (den[idx] + 1e-9)


class GATLayer(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.h, self.dk = heads, dim // heads
        self.ws = nn.Linear(dim, dim, bias=False)
        self.wd = nn.Linear(dim, dim, bias=False)
        self.we = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.att = nn.Parameter(torch.empty(heads, self.dk))
        nn.init.xavier_uniform_(self.att)
        self.direction = nn.Parameter(torch.zeros(2, dim))
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, h: torch.Tensor, e: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
    ) -> torch.Tensor:
        n, n_e = h.shape[0], src.shape[0]
        m_src, m_dst = torch.cat([src, dst]), torch.cat([dst, src])
        d_emb = torch.cat(
            [self.direction[0].expand(n_e, -1), self.direction[1].expand(n_e, -1)]
        )
        t = nn.functional.leaky_relu(
            self.ws(h)[m_src] + self.wd(h)[m_dst] + self.we(torch.cat([e, e])) + d_emb, 0.2
        )
        score = (t.view(-1, self.h, self.dk) * self.att).sum(-1)
        alpha = scatter_softmax(score, m_dst, n)
        msg = self.wv(h)[m_src].view(-1, self.h, self.dk) * alpha.unsqueeze(-1)
        agg = torch.zeros((n, self.h, self.dk), device=h.device, dtype=h.dtype)
        agg = agg.index_add_(0, m_dst, msg).reshape(n, -1)
        return self.norm(h + nn.functional.elu(self.out(agg)))


class EdgeGAT(nn.Module):
    def __init__(
        self,
        edge_dim: int,
        node_dim: int,
        hidden: int = 64,
        heads: int = 4,
        layers: int = 2,
        n_out: int = 1,
        dueling: bool = False,
        head_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.node_enc = nn.Sequential(nn.Linear(node_dim, hidden), nn.ReLU())
        self.layers = nn.ModuleList(GATLayer(hidden, heads) for _ in range(layers))
        self.body = nn.Sequential(
            nn.Linear(3 * hidden, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, head_hidden), nn.ReLU(),
        )  # fmt: skip
        self.dueling = dueling and n_out > 1
        if self.dueling:
            self.value = nn.Linear(head_hidden, 1)
            self.adv = nn.Linear(head_hidden, n_out)
        else:
            self.head = nn.Linear(head_hidden, n_out)

    def forward(
        self, x: torch.Tensor, node_x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor
    ) -> torch.Tensor:
        e = self.edge_enc(x)
        h = self.node_enc(node_x)
        for layer in self.layers:
            h = layer(h, e, src, dst)
        z = self.body(torch.cat([h[src], h[dst], e], dim=1))
        if not self.dueling:
            return self.head(z)
        adv = self.adv(z)
        return self.value(z) + adv - adv.mean(dim=1, keepdim=True)


class GraphScorer:
    """Supervised wrapper: weighted BCE on edge labels, scores = P(attack)."""

    def __init__(
        self,
        edge_dim: int,
        node_dim: int,
        hidden: int = 64,
        heads: int = 4,
        layers: int = 2,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.net = EdgeGAT(edge_dim, node_dim, hidden, heads, layers, n_out=1).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.meta = {"edge_dim": edge_dim, "node_dim": node_dim, "hidden": hidden,
                     "heads": heads, "layers": layers}  # fmt: skip

    def _t(self, a: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(a, dtype=dtype, device=self.device)

    def _logits(self, b: GraphBatch) -> torch.Tensor:
        return self.net(
            self._t(b.x), self._t(b.node_x), self._t(b.src, torch.long), self._t(b.dst, torch.long)
        ).squeeze(1)

    def _loss(self, b: GraphBatch, y: np.ndarray, w: np.ndarray) -> torch.Tensor:
        per = nn.functional.binary_cross_entropy_with_logits(
            self._logits(b), self._t(y), reduction="none"
        )
        wt = self._t(w)
        return (per * wt).sum() / wt.sum().clamp_min(1e-9)

    def fit_batch(self, b: GraphBatch, y: np.ndarray, w: np.ndarray) -> float:
        self.net.train()
        loss = self._loss(b, y, w)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.opt.step()
        return float(loss.item())

    @torch.no_grad()
    def eval_loss(self, b: GraphBatch, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
        self.net.eval()
        return float(self._loss(b, y, w).item()), float(self._t(w).sum().item())

    @torch.no_grad()
    def score(self, snap: Snapshot) -> np.ndarray:
        if snap.n_edges == 0:
            return np.zeros(0, dtype=np.float32)
        self.net.eval()
        return torch.sigmoid(self._logits(collate([snap]))).cpu().numpy().astype(np.float32)

    def state(self) -> dict[str, Any]:
        return {"net": copy.deepcopy(self.net.state_dict()), "meta": self.meta}

    def load(self, state: dict[str, Any]) -> None:
        self.net.load_state_dict(state["net"])
