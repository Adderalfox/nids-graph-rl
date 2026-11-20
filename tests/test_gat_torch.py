"""Torch tests for the GAT (skipped without torch). Run on your machine first."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nidsrl.graph.snapshot import NODE_DIM, Snapshot, collate  # noqa: E402
from nidsrl.models.gat import EdgeGAT, GraphScorer, scatter_softmax  # noqa: E402


def _snap(rng, n=6, e=10, d=5):
    src, dst = rng.integers(0, n, e), rng.integers(0, n, e)
    return Snapshot(
        rng.normal(size=(e, d)).astype(np.float32), src.astype(np.int64), dst.astype(np.int64),
        rng.normal(size=(n, NODE_DIM)).astype(np.float32),
    )  # fmt: skip


def test_scatter_softmax_sums_to_one_per_group():
    idx = torch.tensor([0, 0, 1, 2, 2, 2])
    out = scatter_softmax(torch.randn(6, 3), idx, 4)
    sums = torch.zeros(4, 3).index_add_(0, idx, out)
    assert torch.allclose(sums[[0, 1, 2]], torch.ones(3, 3), atol=1e-5)
    assert torch.all(sums[3] == 0)  # empty group


def test_edge_outputs_are_invariant_to_node_relabelling():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    s = _snap(rng)
    perm = rng.permutation(s.n_nodes)  # new index of old node i is perm[i]
    inv = np.argsort(perm)
    s2 = Snapshot(s.x, perm[s.src], perm[s.dst], s.node_x[inv])
    net = EdgeGAT(5, NODE_DIM, hidden=16, heads=2, layers=2, n_out=3, dueling=True).eval()

    def run(sn):
        g = collate([sn])
        return net(
            torch.tensor(g.x), torch.tensor(g.node_x), torch.tensor(g.src), torch.tensor(g.dst)
        )

    assert torch.allclose(run(s), run(s2), atol=1e-5)


def test_message_passing_changes_outputs_but_layers0_ignores_neighbours():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    s = _snap(rng)
    g = collate([s])
    t = lambda a: torch.tensor(a)  # noqa: E731
    gat = EdgeGAT(5, NODE_DIM, hidden=16, heads=2, layers=2).eval()
    mlp = EdgeGAT(5, NODE_DIM, hidden=16, heads=2, layers=0).eval()
    x2 = g.x.copy()
    x2[0] += 5.0  # perturb ONE edge; look at another edge sharing no change of its own
    other = next(i for i in range(1, s.n_edges) if {s.src[i], s.dst[i]} & {s.src[0], s.dst[0]})
    a = gat(t(g.x), t(g.node_x), t(g.src), t(g.dst))[other]
    b = gat(t(x2), t(g.node_x), t(g.src), t(g.dst))[other]
    assert not torch.allclose(a, b)  # neighbour's features reach it
    a = mlp(t(g.x), t(g.node_x), t(g.src), t(g.dst))[other]
    b = mlp(t(x2), t(g.node_x), t(g.src), t(g.dst))[other]
    assert torch.allclose(a, b)


def test_scorer_overfits_a_tiny_batch_and_roundtrips_state():
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    s = _snap(rng, e=40)
    y = (s.x[:, 0] > 0).astype(np.float32)
    w = np.ones(40, dtype=np.float32)
    sc = GraphScorer(5, NODE_DIM, hidden=16, heads=2, layers=1, lr=1e-2)
    g = collate([s])
    first = sc.fit_batch(g, y, w)
    for _ in range(150):
        last = sc.fit_batch(g, y, w)
    assert last < 0.5 * first
    p = sc.score(s)
    assert p.shape == (40,) and ((p > 0.5) == (y > 0.5)).mean() > 0.9
    sc2 = GraphScorer(5, NODE_DIM, hidden=16, heads=2, layers=1)
    sc2.load(sc.state())
    assert np.allclose(sc2.score(s), p, atol=1e-5)
