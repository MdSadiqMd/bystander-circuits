"""L4 mechanistic collateral on the TransformerLens model with the intervention applied.

Cross-circuit CAD (novel contribution 1): Lexsi's Circuit Attribution Divergence
    CAD(C) = sum_{e in C} |s_e(θ) - s_e(θ')| / sum_{e in C} |s_e(θ)|
(Sadhu, Seth, Sankarapu 2026, computed there on the forget circuit only) evaluated here on every
*bystander* circuit as well as on the target. s_e are signed edge EAP-IG scores on the circuit's own
task data; base and intervened scores use the *same* example subset and IG steps (audit/run_audit.py
`base_edges`), a same-data base recomputation gives the measurement floor (`cad_repeat_*`) and a
second-seed base recomputation the sampling floor (`cad_resample_*`). C is the
edge set induced by the circuit's top-k nodes (from discovery), and a second variant uses the top-1
% edges by |s_e|. 0 = intact, ~1 = dismantled, > 1 = sign-flipped.

AS-NC: activation shift at nodes outside the target circuit, mean over prompts of
||a'_n - a_n|| / ||a_n|| at the last prompt position (relative L2), averaged over non-circuit nodes.

Node turnover (`circuit_distance`): 1 - Jaccard(top-k nodes before, top-k nodes after) on |node|
scores of the target task, plus Spearman of the full node-score vectors. Chen et al. 2026's Circuit
Distance is an edge-level L1 of causal effects; its analogue here is the whole-graph edge L1 in
audit/analyze.py (`target_edge_l1`), not this node set measure.

The intervention is applied with interventions.cell.apply_tl (weights in place or permanent hooks),
and undone afterwards, so one loaded model serves every cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from aya_circuits.discover import node_names
from aya_circuits.overlap import jaccard
from circuitkit.applications.steering import ActivationSteering
from circuitkit.backends.eap.attribute import attribute
from circuitkit.backends.eap.graph import Graph
from safetensors.torch import load_file
from scipy.stats import spearmanr


def edge_scores(
    model, spec, discovery_cfg: dict, device: str, ig_steps: int
) -> tuple[torch.Tensor, Graph]:
    graph = Graph.from_model(model, node_scores=True)
    dl = spec.build_dataloader(model, discovery_cfg, device)
    attribute(
        model, graph, dl, spec.metric_fn(), method="EAP-IG-inputs", ig_steps=ig_steps, quiet=True
    )
    return graph.scores.detach().float().cpu(), graph


def node_slot_masks(graph: Graph, nodes: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """(source slots, destination slots) that belong to `nodes`: [n_forward], [n_backward];
    a head owns its q, k and v destination slots, logits is in neither."""
    fwd_in = torch.tensor([n in nodes for n in node_names(graph)])
    bwd_in = torch.zeros(graph.n_backward, dtype=torch.bool)
    for n in nodes:
        if n.startswith("MLP "):
            bwd_in[graph._backward_index(graph.cfg, f"m{n[4:]}")] = True
        else:
            layer, h = n[1:].split(".")
            for qkv in "qkv":
                bwd_in[graph._backward_index(graph.cfg, f"a{layer}.h{h}", qkv=qkv)] = True
    return fwd_in, bwd_in


def induced_edge_mask(graph: Graph, circuit_nodes: list[str]) -> torch.Tensor:
    """Boolean [n_forward, n_backward]: real edges whose source and destination are circuit nodes
    (logits is always a destination; input is never a source in the circuit)."""
    fwd_in, bwd_in = node_slot_masks(graph, circuit_nodes)
    bwd_in = bwd_in.clone()
    bwd_in[-1] = True
    return graph.real_edge_mask.cpu() & fwd_in[:, None] & bwd_in[None, :]


def touching_edge_mask(graph: Graph, nodes: list[str]) -> torch.Tensor:
    """Boolean [n_forward, n_backward]: real edges with the source or the destination in `nodes`
    (edges into logits count only through their source)."""
    fwd_in, bwd_in = node_slot_masks(graph, nodes)
    return graph.real_edge_mask.cpu() & (fwd_in[:, None] | bwd_in[None, :])


def cad(base: torch.Tensor, new: torch.Tensor, mask: torch.Tensor) -> dict:
    b, n = base[mask], new[mask]
    den = b.abs().sum().clamp_min(1e-12)
    return {
        "cad": float((b - n).abs().sum() / den),
        "n_edges": int(mask.sum()),
        "sign_flip_frac": float(((b * n) < 0).float().mean()) if mask.any() else 0.0,
        "spearman": float(spearmanr(b.numpy(), n.numpy()).statistic) if mask.sum() > 2 else None,
    }


def cad_triple(base: dict[str, torch.Tensor], new: torch.Tensor, mask: torch.Tensor) -> dict:
    """CAD of `new` against the base edges plus the two floors on the same mask: same-data repeat
    (`edges_repeat`) and other-subset resample (`edges_alt_seed`), as `base_edges` stores them."""
    return {
        "cad_induced": cad(base["edges"], new, mask),
        "cad_repeat_induced": cad(base["edges"], base["edges_repeat"], mask),
        "cad_resample_induced": cad(base["edges"], base["edges_alt_seed"], mask),
    }


def top_edges_mask(base: torch.Tensor, graph: Graph, frac: float = 0.01) -> torch.Tensor:
    real = graph.real_edge_mask.cpu()
    k = max(1, int(real.sum().item() * frac))
    scores = base.abs().clone()
    scores[~real] = -1
    idx = torch.topk(scores.view(-1), k).indices
    m = torch.zeros_like(real).view(-1)
    m[idx] = True
    return m.view_as(real)


@torch.no_grad()
def last_position_acts(
    model, texts: list[str], nodes: list[str], batch_size: int = 8
) -> dict[str, torch.Tensor]:
    helper = ActivationSteering(model, dict.fromkeys(nodes, 1.0))
    hooks = {n: helper._get_hook_point_from_node(n) for n in nodes}
    names = sorted(set(hooks.values()))
    out = {n: [] for n in nodes}
    for i in range(0, len(texts), batch_size):
        toks = model.to_tokens(texts[i : i + batch_size], prepend_bos=False, padding_side="left")
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n in names)
        for n in nodes:
            a = cache[hooks[n]][:, -1]
            if a.dim() == 3:
                a = a[:, helper._node_head_index(n)]
            out[n].append(a.float().cpu())
    return {n: torch.cat(v) for n, v in out.items()}


def as_nc(
    base_acts: dict[str, torch.Tensor], new_acts: dict[str, torch.Tensor], circuit_nodes: list[str]
) -> dict:
    outside = [n for n in base_acts if n not in circuit_nodes]
    inside = [n for n in base_acts if n in circuit_nodes]

    def rel(ns):
        vals = [
            ((new_acts[n] - base_acts[n]).norm(dim=-1) / base_acts[n].norm(dim=-1).clamp_min(1e-6))
            .mean()
            .item()
            for n in ns
        ]
        return sum(vals) / len(vals) if vals else None

    return {
        "as_nc": rel(outside),
        "as_circuit": rel(inside),
        "n_outside": len(outside),
        "n_inside": len(inside),
    }


def circuit_distance(base_nodes: dict[str, float], new_nodes: dict[str, float], k: int) -> dict:
    def top(d: dict[str, float]) -> list[str]:
        return sorted(d, key=lambda n: abs(d[n]), reverse=True)[:k]

    keys = list(base_nodes)
    return {
        "circuit_distance": 1 - jaccard(top(base_nodes), top(new_nodes)),
        "node_spearman": float(
            spearmanr(
                [base_nodes[n] for n in keys], [new_nodes.get(n, 0.0) for n in keys]
            ).statistic
        ),
    }


class BaseCircuits:
    """Signed base edge scores + node sets for every circuit, loaded once per audit process."""

    def __init__(self, run_root: Path, circuit_names: list[str], seed: int, top_fraction: float):
        self.data = {}
        for c in circuit_names:
            d = run_root / "circuits" / c / f"seed{seed}"
            edges = load_file(str(d / "edges.safetensors"))["edges"]
            nodes = json.loads((d / "nodes.json").read_text())
            signed = dict(zip(nodes["names"], nodes["signed_rowsum"], strict=True))
            signed.pop("input", None)
            k = max(1, round(len(signed) * top_fraction))
            top = sorted(signed, key=lambda n: abs(signed[n]), reverse=True)[:k]
            self.data[c] = {"edges": edges, "signed_nodes": signed, "top_nodes": top, "k": k}
