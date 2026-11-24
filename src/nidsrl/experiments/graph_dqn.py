"""Experiment B: EdgeGAT as the Q-network of the dueling Double DQN (graph + RL combined).

Same environment, reward, TD targets, feature policy and evaluation as experiment A; only the
Q-network (graph attention over the window graph) and the replay (window snapshots) change.
`layers=0` is the no-message-passing ablation under the identical RL machinery.
Reports the raw argmax policy and the source-val-calibrated operating point.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from nidsrl.experiments.calibration import q_threshold_graph
from nidsrl.experiments.gat_supervised import collect_snap_run
from nidsrl.experiments.rl_common import SourceData, ablation_tag, source_env_cfg, source_pairs
from nidsrl.rl.env import HostPairEnv
from nidsrl.rl.graph_trainer import train_graph_dqn
from nidsrl.rl.obs import obs_keep_indices
from nidsrl.rl.policy import GraphQPolicy, GraphQThresholdPolicy
from nidsrl.rl.rollout import evaluate_loaded_runs, evaluate_pair_rl
from nidsrl.rl.trainer import TrainConfig
from nidsrl.utils.device import resolve_device
from nidsrl.utils.seed import seed_everything


def run_graph_dqn(
    root: str | Path,
    source: str,
    seed: int,
    *,
    layers: int = 2,
    hidden: int = 64,
    heads: int = 4,
    lr: float = 3e-4,
    tau: float = 0.005,
    windows_per_update: int = 8,
    train_cfg: TrainConfig | None = None,
    drop_groups: tuple[str, ...] = (),
    keep_groups: tuple[str, ...] = (),
    inspect_windows: int = 1,
    block_windows: int = 6,
    fp_cost: float = 1.0,
    fpr_target: float = 0.01,
    val_stride: int = 1,
    eval_stride: int = 1,
    targets: list[str] | None = None,
    min_group_flows: int = 500,
    tag: str = "",
    device: str = "auto",
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, Any]]:
    from nidsrl.rl.graph_dqn import TorchGraphLearner  # needs torch; imported lazily

    root = Path(root)
    seed_everything(seed)
    tc = train_cfg or TrainConfig()
    keep = obs_keep_indices(drop_groups, keep_groups)
    data = SourceData(root, source)
    env_cfg = source_env_cfg(data.plan, inspect_windows, block_windows, fp_cost)
    pairs = source_pairs(root, source, targets)
    dev = resolve_device(device)
    learner = TorchGraphLearner(len(keep), hidden, heads, layers, True, lr, tau, device=dev)
    val_runs = data.runs("val", val_stride)
    best: dict[str, Any] = {}
    name = "edgemlp" if layers == 0 else f"gat{layers}"

    def validate(policy: GraphQPolicy) -> dict[str, Any]:
        return evaluate_loaded_runs(
            policy, val_runs, env_cfg=env_cfg, min_group_flows=min_group_flows
        )

    def on_best(rec: dict[str, Any]) -> None:
        best["state"], best["epoch"] = learner.state(), rec["epoch"]

    log(f"[{name}-d3qn/{source}/s{seed}] device={dev} edge_dim={len(keep)} "
        f"train_runs={len(data.ids('train'))} val_runs={len(val_runs)}")  # fmt: skip
    history = train_graph_dqn(
        learner, HostPairEnv(env_cfg), data.ids("train"), data.get("train"), tc,
        keep=keep, windows_per_update=windows_per_update, seed=seed, validate=validate,
        on_best=on_best, log=log,
    )  # fmt: skip
    if "state" in best:
        learner.load(best["state"])
    else:
        log("  WARNING: no validation improvement recorded; using the last weights")

    val_snaps = [
        collect_snap_run(HostPairEnv(env_cfg), r, keep) for r in data.runs("val", val_stride)
    ]
    thr = q_threshold_graph(learner.q_snapshot, val_snaps, fpr_target)
    log(f"  calibrated threshold on attack_score(Q) = {thr:.4f} (source-val FPR {fpr_target})")
    policies = {
        "argmax": (GraphQPolicy(learner.q_snapshot, keep, eps=0.0), f"{name}-d3qn"),
        "cal": (GraphQThresholdPolicy(learner.q_snapshot, thr, keep), f"{name}-d3qn-cal"),
    }
    out = {}
    for key, (policy, model) in policies.items():
        results = []
        for pair in pairs:
            log(f"  [{key}] evaluating {source} -> {pair['target']} ({pair['eval_set']})")
            results.append(
                evaluate_pair_rl(
                    policy, root, pair, env_cfg=env_cfg, min_group_flows=min_group_flows,
                    run_stride=eval_stride,
                )  # fmt: skip
            )
        out[key] = {
            "meta": {
                "kind": "rl_env", "model": model,
                "variant": ablation_tag(drop_groups, keep_groups, tag),
                "source": source, "seed": seed, "best_epoch": best.get("epoch"),
                "threshold": thr if key == "cal" else None, "fpr_target": fpr_target,
                "layers": layers, "hidden": hidden, "heads": heads, "lr": lr, "tau": tau,
                "windows_per_update": windows_per_update, "train": tc.to_dict(),
                "env": {"inspect_windows": inspect_windows, "block_windows": block_windows,
                        "fp_cost": fp_cost},
                "n_obs_features": len(keep), "eval_stride": eval_stride,
            },
            "history": history,
            "pairs": results,
        }  # fmt: skip
    return out
