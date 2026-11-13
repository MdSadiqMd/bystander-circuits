"""EAP-IG on inputs with a chosen quadrature rule and per-example edge scores (amendment A19).

CircuitKIT's `get_scores_eap_ig_inputs` integrates the input-interpolation path with a
right-endpoint Riemann sum (α = k/m, k = 1..m), accumulates the edge scores in the model dtype
(bf16) and, with a batch-mean metric, returns the per-example mean divided once more by the batch
size (docs/audit-2026-11-06.md A13). This module keeps its forward hooks and index layout
(`eap_utils.make_hooks_and_matrices`) and changes three things:
  rule        the nodes α_i and weights w_i are arguments (`RULES`; Gauss–Legendre by default)
  precision   every edge score is accumulated in fp32
  resolution  scores are kept per example ([N, n_forward, n_backward]), so a leave-one-out
              jackknife over examples is possible; the loss is the per-example vector, so the mean
              over examples needs no batch correction
With `RULES["riemann3"]` the per-example mean equals CircuitKIT's score times the batch size, up to
bf16 accumulation (checks/check_quadrature.py).
"""

from __future__ import annotations

from functools import partial

import numpy as np
import torch
from circuitkit.backends.eap.eap_utils import make_hooks_and_matrices, tokenize_batch_pair
from circuitkit.backends.eap.graph import Graph
from einops import einsum


def gauss_legendre(n: int) -> tuple[list[float], list[float]]:
    """n-node Gauss–Legendre rule on [0, 1]: exact for polynomial integrands up to degree 2n − 1."""
    x, w = np.polynomial.legendre.leggauss(n)
    return ((x + 1) / 2).tolist(), (w / 2).tolist()


def riemann(m: int) -> tuple[list[float], list[float]]:
    """Right-endpoint Riemann sum with m steps (CircuitKIT's rule): O(1/m) error."""
    return [k / m for k in range(1, m + 1)], [1 / m] * m


def midpoint(m: int) -> tuple[list[float], list[float]]:
    """Composite midpoint rule with m steps: O(1/m²) error on piecewise-smooth integrands."""
    return [(k - 0.5) / m for k in range(1, m + 1)], [1 / m] * m


RULES = {
    "riemann3": riemann(3),
    "midpoint3": midpoint(3),
    "gl3": gauss_legendre(3),
    "riemann6": riemann(6),
    "midpoint6": midpoint(6),
    "gl6": gauss_legendre(6),
    "gl15": gauss_legendre(15),
    "riemann48": riemann(48),
    "midpoint48": midpoint(48),
}
THREE_PASS = ("riemann3", "midpoint3", "gl3")
SIX_PASS = ("riemann6", "midpoint6", "gl6")


def _backward_hooks(graph: Graph, act_diff: torch.Tensor, scores: torch.Tensor) -> list:
    """Per-example variant of CircuitKIT's gradient hooks: scores [batch, n_forward, n_backward]
    += Σ_pos (corrupt − clean activation of every upstream source) · ∂loss/∂(destination input),
    accumulated in fp32."""

    def grad_hook(prev: int, bwd, grads: torch.Tensor, hook):
        g = grads.detach()
        if g.ndim == 3:
            g = g.unsqueeze(2)
        s = einsum(
            act_diff[:, :, :prev, : g.shape[-1]],
            g,
            "batch pos forward hidden, batch pos backward hidden -> batch forward backward",
        )
        scores[:, :prev, bwd] += s.squeeze(2).float() if isinstance(bwd, int) else s.float()

    hooks = []
    for layer in range(graph.cfg["n_layers"]):
        node = graph.nodes[f"a{layer}.h0"]
        prev = graph.prev_index(node)
        for i, letter in enumerate("qkv"):
            hooks.append(
                (
                    node.qkv_inputs[i],
                    partial(grad_hook, prev, graph.backward_index(node, qkv=letter)),
                )
            )
        node = graph.nodes[f"m{layer}"]
        hooks.append(
            (node.in_hook, partial(grad_hook, graph.prev_index(node), graph.backward_index(node)))
        )
    node = graph.nodes["logits"]
    hooks.append(
        (node.in_hook, partial(grad_hook, graph.prev_index(node), graph.backward_index(node)))
    )
    return hooks


def eap_ig_inputs(
    model, graph: Graph, dataloader, loss_vec, nodes: list[float], weights: list[float]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-example EAP-IG-inputs edge scores with the quadrature rule (nodes, weights).

    loss_vec(logits, clean_logits, input_lengths, labels) must return one loss per example (no
    mean). Returns (scores [N, n_forward, n_backward] fp32 on CPU in dataloader order, clean loss
    [N], corrupt loss [N]); the last two are the attribution-free behavioural measure of the task
    under the current model."""
    side = getattr(dataloader, "pair_padding_side", None)
    templated = getattr(dataloader, "templated", False)
    input_node = graph.nodes["input"]
    input_idx = graph.forward_index(input_node)
    all_scores, clean_losses, corrupt_losses = [], [], []
    for clean, corrupted, label in dataloader:
        ct, kt, cm, km, lengths, n_pos = tokenize_batch_pair(
            model, clean, corrupted, pair_padding_side=side, templated=templated
        )
        n = len(clean)
        (fwd_corrupt, fwd_clean, _), act_diff = make_hooks_and_matrices(
            model, graph, n, n_pos, None
        )
        scores = torch.zeros((n, graph.n_forward, graph.n_backward), device=model.cfg.device)
        bwd = _backward_hooks(graph, act_diff, scores)
        with torch.inference_mode():
            with model.hooks(fwd_hooks=fwd_corrupt):
                corrupt_logits = model(kt, attention_mask=km)
            x_corrupt = act_diff[:, :, input_idx].clone()
            with model.hooks(fwd_hooks=fwd_clean):
                clean_logits = model(ct, attention_mask=cm)
            x_clean = x_corrupt - act_diff[:, :, input_idx]
            clean_losses.append(loss_vec(clean_logits, clean_logits, lengths, label).float().cpu())
            corrupt_losses.append(
                loss_vec(corrupt_logits, clean_logits, lengths, label).float().cpu()
            )

        def interpolate(alpha: float, x0=x_corrupt, x1=x_clean):
            def hook(act, hook):
                x = x0 + alpha * (x1 - x0)
                x.requires_grad = True
                return x

            return hook

        for alpha, w in zip(nodes, weights, strict=True):
            with model.hooks(fwd_hooks=[(input_node.out_hook, interpolate(alpha))], bwd_hooks=bwd):
                logits = model(ct, attention_mask=cm)
                (w * loss_vec(logits, clean_logits, lengths, label)).sum().backward()
        all_scores.append(scores.cpu())
        del act_diff, scores, bwd, clean_logits, corrupt_logits, logits
    model.zero_grad(set_to_none=True)
    return torch.cat(all_scores), torch.cat(clean_losses), torch.cat(corrupt_losses)


def jackknife_se(loo: torch.Tensor | list[float]) -> float:
    """Leave-one-out jackknife standard error from the n leave-one-out statistics."""
    t = torch.as_tensor(loo, dtype=torch.float64)
    n = t.numel()
    return float(((n - 1) / n * ((t - t.mean()) ** 2).sum()).sqrt())


def loo_means(scores: torch.Tensor) -> torch.Tensor:
    """[N, ...] per-example scores -> [N, ...] leave-one-out means (row j leaves example j out)."""
    n = scores.shape[0]
    return (scores.sum(0, keepdim=True) - scores) / (n - 1)
