from nidsrl.experiments.feature_selection import nested_picks, run_selection


def test_nested_pick_never_uses_heldout_dataset():
    srcs = ["x", "y", "z"]
    cands = ["c1", "c2"]
    auc = {}
    for c in cands:
        for s in srcs:
            for t in srcs:
                if s != t:
                    auc[(c, s, t)] = 0.5
    # c2 is great only on pairs involving z; for held-out z it must be ignored
    for s, t in [("x", "z"), ("z", "x"), ("y", "z"), ("z", "y")]:
        auc[("c2", s, t)] = 0.9
    auc[("c1", "x", "y")] = auc[("c1", "y", "x")] = 0.6
    out = nested_picks(auc, srcs, cands)
    assert out["per_heldout"]["z"]["pick"] == "c1"  # only the x<->y pairs count
    assert out["per_heldout"]["x"]["pick"] == "c2"
    assert out["per_heldout"]["z"]["selected_on"] == ["x", "y"]
    assert out["global"]["pick"] == "c2"


def test_run_selection_end_to_end_and_resumable(world):
    root, _, _ = world
    cands = {"base": (), "no_volume": ("volume",)}
    saved = {}
    res = run_selection(root, ["a", "b"], 0, candidates=cands, eval_stride=1,
                        min_group_flows=50, max_iter=15, on_result=saved.__setitem__)  # fmt: skip
    assert set(saved) == {"base|a", "base|b", "no_volume|a", "no_volume|b"}
    assert res["global"]["pick"] in cands and set(res["per_heldout"]) == {"a", "b"}
    again = run_selection(root, ["a", "b"], 0, candidates=cands, eval_stride=1,
                          min_group_flows=50, max_iter=15, done=saved)  # fmt: skip
    assert again["raw"] == res["raw"]
