"""Torch backend for the DQN trainer: MLP Q-network (optionally dueling) + Polyak target.

Kept deliberately thin: all RL logic (targets, replay, schedules, loop) lives in numpy
modules that are unit-tested without torch. This file only does forward, loss and update.
Double DQN is selected in `TrainConfig.double` (the trainer builds the targets).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn

N_ACTIONS = 3


class QNet(nn.Module):
    def __init__(
        self, obs_dim: int, hidden: tuple[int, ...] = (256, 256), dueling: bool = True
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers)
        self.dueling = dueling
        if dueling:
            self.value = nn.Linear(d, 1)
            self.adv = nn.Linear(d, N_ACTIONS)
        else:
            self.head = nn.Linear(d, N_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.body(x)
        if not self.dueling:
            return self.head(z)
        adv = self.adv(z)
        return self.value(z) + adv - adv.mean(dim=1, keepdim=True)


class TorchLearner:
    def __init__(
        self,
        obs_dim: int,
        hidden: tuple[int, ...] = (256, 256),
        dueling: bool = True,
        lr: float = 3e-4,
        tau: float = 0.005,
        huber_delta: float = 1.0,
        grad_clip: float = 10.0,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.online = QNet(obs_dim, hidden, dueling).to(self.device)
        self.target = copy.deepcopy(self.online).requires_grad_(False)
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.tau, self.delta, self.clip = tau, huber_delta, grad_clip
        self.meta = {"obs_dim": obs_dim, "hidden": list(hidden), "dueling": dueling}

    def _t(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def q_values(self, x: np.ndarray) -> np.ndarray:
        self.online.eval()  # no dropout is used, but acting must never run in train mode
        out = [self.online(self._t(x[i : i + 8192])).cpu().numpy() for i in range(0, len(x), 8192)]
        return np.concatenate(out) if out else np.zeros((0, N_ACTIONS), dtype=np.float32)

    @torch.no_grad()
    def next_q(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xt = self._t(x)
        self.online.eval()
        return self.online(xt).cpu().numpy(), self.target(xt).cpu().numpy()

    def fit(
        self, s: np.ndarray, a: np.ndarray, y: np.ndarray, w: np.ndarray | None = None
    ) -> dict[str, Any]:
        self.online.train()
        q = self.online(self._t(s)).gather(1, torch.as_tensor(a, device=self.device)[:, None])
        q = q.squeeze(1)
        yt = self._t(y)
        per = nn.functional.huber_loss(q, yt, delta=self.delta, reduction="none")
        loss = per.mean() if w is None else (per * self._t(w)).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.clip)
        self.opt.step()
        with torch.no_grad():  # Polyak averaging of the target network
            for p_t, p in zip(self.target.parameters(), self.online.parameters(), strict=True):
                p_t.mul_(1.0 - self.tau).add_(p.detach(), alpha=self.tau)
        return {"loss": float(loss.item()), "td_error": (q.detach() - yt).abs().cpu().numpy()}

    def state(self) -> dict[str, Any]:
        return {"online": copy.deepcopy(self.online.state_dict()), "meta": self.meta}

    def load(self, state: dict[str, Any]) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["online"])
