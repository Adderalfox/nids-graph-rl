"""Experiment A: MLP + DQN family agent in the host-pair environment.

Train on SOURCE train runs, select the epoch on SOURCE val runs (macro-F1 of the greedy
policy), then evaluate zero-shot on every protocol pair of this source. Nothing from a
target dataset is used for training, thresholds or model selection.

Variants (same network/RL machinery apart from the named switch, so later experiments
isolate the graph encoder and the replay strategy):
    dqn   plain MLP DQN            ddqn  + Double DQN            d3qn  + dueling head
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from nidsrl.experiments.rl_common import SourceData, ablation_tag, source_env_cfg, source_pairs
from nidsrl.rl.env import HostPairEnv
from nidsrl.rl.obs import obs_keep_indices
from nidsrl.rl.policy import QPolicy
from nidsrl.rl.rollout import evaluate_loaded_runs, evaluate_pair_rl
from nidsrl.rl.trainer import TrainConfig, train_dqn
from nidsrl.utils.device import resolve_device
from nidsrl.utils.seed import seed_everything

VARIANTS = {
    "dqn": {"dueling": False, "double": False},
    "ddqn": {"dueling": False, "double": True},
    "d3qn": {"dueling": True, "double": True},
}


def run_dqn(
    root: str | Path,
    source: str,
    seed: int,
    *,
    variant: str = "d3qn",
    drop_groups: tuple[str, ...] = (),
    keep_groups: tuple[str, ...] = (),
    train_cfg: TrainConfig | None = None,
    hidden: tuple[int, ...] = (256, 256),
    lr: float = 3e-4,
    tau: float = 0.005,
    inspect_windows: int = 1,
    block_windows: int = 6,
    val_stride: int = 1,
    eval_stride: int = 1,
    targets: list[str] | None = None,
    min_group_flows: int = 500,
    tag: str = "",
    device: str = "auto",
    ckpt_path: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    from nidsrl.rl.dqn import TorchLearner  # needs torch; imported lazily

    root = Path(root)
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}")
    seed_everything(seed)
    tc = train_cfg or TrainConfig()
    tc = TrainConfig(**{**tc.to_dict(), "double": VARIANTS[variant]["double"]})
    dev = resolve_device(device)
    keep = obs_keep_indices(drop_groups, keep_groups)
    data = SourceData(root, source)
    env_cfg = source_env_cfg(data.plan, inspect_windows, block_windows)
    pairs = source_pairs(root, source, targets)
    learner = TorchLearner(
        len(keep), hidden, VARIANTS[variant]["dueling"], lr=lr, tau=tau, device=dev
    )
    val_runs = data.runs("val", val_stride)
    best: dict[str, Any] = {}

    def validate(policy):
        return evaluate_loaded_runs(
            policy, val_runs, env_cfg=env_cfg, min_group_flows=min_group_flows
        )

    def on_best(rec):
        best["state"], best["epoch"] = learner.state(), rec["epoch"]
        if ckpt_path is not None:
            import torch

            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({**best["state"], "keep": keep.tolist(), "source": source}, ckpt_path)

    log(f"[{variant}/{source}/s{seed}] device={dev} obs_dim={len(keep)} train_runs="
        f"{len(data.ids('train'))} val_runs={len(val_runs)}")  # fmt: skip
    history = train_dqn(
        learner, HostPairEnv(env_cfg), data.ids("train"), data.get("train"), tc,
        obs_dim=len(keep), keep=keep, seed=seed, validate=validate, on_best=on_best, log=log,
    )  # fmt: skip
    if "state" in best:
        learner.load(best["state"])
    else:
        log("  WARNING: no validation improvement recorded; using the last weights")

    policy = QPolicy(learner.q_values, keep, eps=0.0)
    results = []
    for pair in pairs:
        log(f"  evaluating {source} -> {pair['target']} ({pair['eval_set']})")
        results.append(
            evaluate_pair_rl(
                policy, root, pair, env_cfg=env_cfg, min_group_flows=min_group_flows,
                run_stride=eval_stride,
            )  # fmt: skip
        )
    return {
        "meta": {
            "kind": "rl_env",
            "model": f"{variant}-mlp",
            "variant": ablation_tag(drop_groups, keep_groups, tag),
            "source": source,
            "seed": seed,
            "best_epoch": best.get("epoch"),
            "n_obs_features": len(keep),
            "hidden": list(hidden),
            "lr": lr,
            "tau": tau,
            "train": tc.to_dict(),
            "env": {"inspect_windows": inspect_windows, "block_windows": block_windows},
            "reward": vars(env_cfg.reward) | {"action_cost": list(env_cfg.reward.action_cost)},
            "eval_stride": eval_stride,
        },
        "history": history,
        "pairs": results,
    }
