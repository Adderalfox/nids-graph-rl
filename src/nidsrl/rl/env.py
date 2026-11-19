"""Host-pair response environment (simulated; see docs/host_window_mdp.md).

Each 10 s window the agent sees every ACTIVE host pair (src -> dst) that has flows in the
window and chooses one action per pair:
    0 monitor   nothing happens
    1 inspect   the pair's flows are held (removed) for `inspect_windows` windows
    2 block     the pair's flows are held (removed) for `block_windows` windows
Held flows never reach the agent: the pair simply does not appear while held, which is
how an action changes the next state. The effects use the recorded flows, so they are a
replay-based simulation, not real counterfactual traffic.

Ground-truth counts (attack/benign flows per group) drive rewards and evaluation only;
they are never part of the observation.

Reward for an acted group (class-balanced, then compressed):
    flow term   alert: det_gain*u_a*A - fp_cost*u_b*B        no alert: -miss_cost*u_a*A
    response    alert: prevent_gain*u_a*A_held - collateral_cost*u_b*B_held
    r = sign(x)*log1p(|x|) - action_cost[a]
with u_a, u_b the balanced class weights (0.5 / class share, from SOURCE TRAIN only).
A_held/B_held are the pair's recorded attack/benign flows in the held windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nidsrl.data.taxonomy import GROUPS
from nidsrl.rl.rundata import N_FEATURES, RunData

MONITOR, INSPECT, BLOCK = 0, 1, 2
N_CONT = 35
N_HISTORY = 6
OBS_DIM = N_FEATURES + N_CONT + 1 + N_HISTORY
_HOLD_LOOKAHEAD_MAX = 64


@dataclass
class RewardConfig:
    u_a: float = 1.0
    u_b: float = 1.0
    det_gain: float = 1.0
    miss_cost: float = 1.0
    fp_cost: float = 1.0
    prevent_gain: float = 1.0
    collateral_cost: float = 1.0
    action_cost: tuple[float, float, float] = (0.0, 0.05, 0.1)


@dataclass
class EnvConfig:
    inspect_windows: int = 1
    block_windows: int = 6
    reward: RewardConfig = field(default_factory=RewardConfig)


def class_weights(plan: dict[str, Any], split: str = "train") -> tuple[float, float]:
    """Balanced per-flow class weights u_a, u_b = 0.5 / share (from the source train split)."""
    st = plan["stats"][split]
    p_a = st["attacks"] / max(st["flows"], 1)
    p_a = min(max(p_a, 1e-6), 1 - 1e-6)
    return 0.5 / p_a, 0.5 / (1 - p_a)


@dataclass
class RunResult:
    """Per-group outcome of one episode (everything needed for flow-level evaluation)."""

    alert: np.ndarray
    score: np.ndarray
    held: np.ndarray
    A: np.ndarray
    B: np.ndarray
    cats: np.ndarray
    n_inspect: int
    n_block: int
    n_decisions: int
    final_transitions: dict[str, np.ndarray]


def _slog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class HostPairEnv:
    def __init__(self, cfg: EnvConfig | None = None) -> None:
        self.cfg = cfg or EnvConfig()
        self.run: RunData | None = None

    # ------------------------------------------------------------------ setup
    def reset(self, run: RunData) -> None:
        self.run = run
        G, P = run.n_groups, run.n_pairs
        self.t = 0
        self.hold_until = np.full(P, -1, dtype=np.int64)
        self.hold_score = np.zeros(P, dtype=np.float32)
        self.last_logn = np.zeros(P, dtype=np.float32)
        self.n_active = np.zeros(P, dtype=np.int64)
        self.last_act = np.zeros(P, dtype=np.int8)
        self.pend_has = np.zeros(P, dtype=bool)
        self.pend_obs = np.zeros((P, OBS_DIM), dtype=np.float32)
        self.pend_act = np.zeros(P, dtype=np.int8)
        self.pend_rew = np.zeros(P, dtype=np.float32)
        self.pend_t = np.zeros(P, dtype=np.int64)
        self.g_alert = np.zeros(G, dtype=bool)
        self.g_held = np.zeros(G, dtype=bool)
        self.g_score = np.full(G, np.nan, dtype=np.float32)
        self._gids = np.zeros(0, dtype=np.int64)
        self._obs = np.zeros((0, OBS_DIM), dtype=np.float32)
        self.n_inspect = self.n_block = self.n_decisions = 0

    @property
    def done(self) -> bool:
        assert self.run is not None
        return self.t >= self.run.n_windows

    # ------------------------------------------------------------ observation
    def _build_obs(self, gids: np.ndarray) -> np.ndarray:
        r = self.run
        assert r is not None
        p = r.g_pair[gids]
        logn = np.log1p(r.g_n[gids]).astype(np.float32)
        delta = logn - self.last_logn[p]
        since = np.where(
            self.hold_until[p] >= 0,
            np.clip(self.t - self.hold_until[p], 0, 6) / 6.0,
            1.0,
        ).astype(np.float32)
        history = np.stack(
            [
                self.last_logn[p] / 10.0,
                delta / 10.0,
                np.log1p(self.n_active[p]).astype(np.float32) / np.log(181.0),
                (self.last_act[p] == INSPECT).astype(np.float32),
                (self.last_act[p] == BLOCK).astype(np.float32),
                since,
            ],
            axis=1,
        )
        return np.concatenate(
            [r.g_mean[gids], r.g_max[gids], (logn / 10.0)[:, None], history], axis=1
        ).astype(np.float32)

    def observe(self) -> dict[str, Any]:
        """Observation for the current window: only pairs that are not currently held."""
        r = self.run
        assert r is not None and not self.done
        s0, s1 = r.win_ptr[self.t], r.win_ptr[self.t + 1]
        gids = np.arange(s0, s1, dtype=np.int64)
        pairs = r.g_pair[gids]
        held = self.hold_until[pairs] >= self.t
        hg = gids[held]
        if len(hg):
            self.g_held[hg] = True
            self.g_alert[hg] = True
            self.g_score[hg] = self.hold_score[pairs[held]]
        active = gids[~held]
        self._gids = active
        self._obs = self._build_obs(active)

        trans = None
        if len(active):
            p = r.g_pair[active]
            done_mask = self.pend_has[p]
            if done_mask.any():
                pp = p[done_mask]
                trans = {
                    "s": self.pend_obs[pp].copy(),
                    "a": self.pend_act[pp].copy(),
                    "r": self.pend_rew[pp].copy(),
                    "s2": self._obs[done_mask].copy(),
                    "done": np.zeros(len(pp), dtype=bool),
                    # windows between the decision and the pair's next decision (>= 1):
                    # semi-MDP discounting uses gamma ** dt
                    "dt": (self.t - self.pend_t[pp]).astype(np.int64),
                }
                self.pend_has[pp] = False
        return {
            "t": self.t,
            "gids": active,
            "obs": self._obs,
            "src": r.pair_src[r.g_pair[active]],
            "dst": r.pair_dst[r.g_pair[active]],
            "transitions": trans,
        }

    def current_labels(self) -> tuple[np.ndarray, np.ndarray]:
        """Ground-truth (attack, benign) flow counts of the active groups.

        For oracle diagnostics only; learned policies must never call this.
        """
        r = self.run
        assert r is not None
        return r.g_A[self._gids], r.g_B[self._gids]

    # ------------------------------------------------------------------- step
    def _lookahead(self, gids: np.ndarray, hold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = self.run
        assert r is not None
        a_h = np.zeros(len(gids), dtype=np.float64)
        b_h = np.zeros(len(gids), dtype=np.float64)
        if len(gids) == 0:
            return a_h, b_h
        pos = r.pos_in_pw[gids]
        pair = r.g_pair[gids]
        G = r.n_groups
        for j in range(1, int(min(hold.max(), _HOLD_LOOKAHEAD_MAX)) + 1):
            p2 = np.minimum(pos + j, G - 1)
            ok = (
                (pos + j < G)
                & (r.g_pair[r.order_pw[p2]] == pair)
                & (r.g_win[r.order_pw[p2]] <= self.t + hold)
            )
            g2 = r.order_pw[p2]
            a_h += np.where(ok, r.g_A[g2], 0)
            b_h += np.where(ok, r.g_B[g2], 0)
        return a_h, b_h

    def step(self, actions: np.ndarray, scores: np.ndarray | None = None) -> np.ndarray:
        """Apply one action per active pair. Returns the per-pair rewards."""
        r = self.run
        assert r is not None
        gids = self._gids
        a = np.asarray(actions, dtype=np.int64)
        if a.shape != gids.shape:
            raise ValueError(f"expected {len(gids)} actions, got {a.shape}")
        sc = a.astype(np.float32) if scores is None else np.asarray(scores, dtype=np.float32)
        cfg = self.cfg.reward

        A = r.g_A[gids].astype(np.float64)
        B = r.g_B[gids].astype(np.float64)
        alert = a != MONITOR
        hold = np.where(
            a == INSPECT, self.cfg.inspect_windows, np.where(a == BLOCK, self.cfg.block_windows, 0)
        )

        flow = np.where(
            alert,
            cfg.det_gain * cfg.u_a * A - cfg.fp_cost * cfg.u_b * B,
            -cfg.miss_cost * cfg.u_a * A,
        )
        resp = np.zeros(len(gids))
        if alert.any():
            a_h, b_h = self._lookahead(gids[alert], hold[alert])
            resp[alert] = cfg.prevent_gain * cfg.u_a * a_h - cfg.collateral_cost * cfg.u_b * b_h
        costs = np.asarray(cfg.action_cost)[a]
        reward = (_slog(flow + resp) - costs).astype(np.float32)

        p = r.g_pair[gids]
        self.g_score[gids] = sc
        self.g_alert[gids[alert]] = True
        self.hold_until[p[alert]] = self.t + hold[alert]
        self.hold_score[p[alert]] = sc[alert]
        self.n_inspect += int((a == INSPECT).sum())
        self.n_block += int((a == BLOCK).sum())
        self.n_decisions += len(a)

        self.pend_obs[p] = self._obs
        self.pend_act[p] = a
        self.pend_rew[p] = reward
        self.pend_t[p] = self.t
        self.pend_has[p] = True
        self.last_logn[p] = np.log1p(r.g_n[gids]).astype(np.float32)
        self.n_active[p] += 1
        self.last_act[p] = a
        self.t += 1
        return reward

    # ----------------------------------------------------------------- finish
    def finish(self) -> RunResult:
        """Close the episode: flush unfinished transitions as terminal and return results."""
        r = self.run
        assert r is not None
        # groups in windows never reached (episode ended early) count as unalerted
        left = np.flatnonzero(self.pend_has)
        final = {
            "s": self.pend_obs[left].copy(),
            "a": self.pend_act[left].copy(),
            "r": self.pend_rew[left].copy(),
            "s2": np.zeros((len(left), OBS_DIM), dtype=np.float32),
            "done": np.ones(len(left), dtype=bool),
            "dt": np.ones(len(left), dtype=np.int64),
        }
        self.pend_has[left] = False
        score = np.where(np.isnan(self.g_score), -np.inf, self.g_score).astype(np.float32)
        return RunResult(
            alert=self.g_alert.copy(),
            score=score,
            held=self.g_held.copy(),
            A=r.g_A.copy(),
            B=r.g_B.copy(),
            cats=r.g_cats.copy(),
            n_inspect=self.n_inspect,
            n_block=self.n_block,
            n_decisions=self.n_decisions,
            final_transitions=final,
        )


assert len(GROUPS) == 8
