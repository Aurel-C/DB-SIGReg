"""Diagnostics for comparing SIGReg gradient estimators.

The central question of DB-SIGReg is how close its gradient is to the exact
large-batch SIGReg gradient. Because the SIGReg loss is a quadratic function of
batch means, the exact gradient over a window of K micro-batches can be computed
with micro-batch memory in two passes: first accumulate the window's ECF means
under no_grad, then backpropagate the per-sample pseudo-loss with those exact
constants. This module computes, in one fixed projection frame:

* ``exact``: the exact window gradient (two-pass, N = K * B samples).
* ``db``: DB-SIGReg's gradient, using the active buffer's (stale) statistics.
* ``db_fresh``: the same pseudo-loss with statistics from an independent window
  at the current weights. ``db`` vs ``db_fresh`` isolates the one-step lag.
* ``minibatch``: plain SIGReg with gradient accumulation (mean of per
  micro-batch ECF-loss gradients).

All gradients are of the regularizer term only, with respect to the encoder
and projector parameters.

Two references are reported. ``*_vs_window`` compares against the exact
gradient of the same window; this favours mini-batch SIGReg, which shares the
window's samples, while DB's statistics come from other samples. ``*_vs_pop``
compares against the window's per-sample terms weighted by the ECF error of a
large independent pool at the current weights (a proxy for the population
error). It isolates the quality of each estimator's error signal: sampling
noise, the mini-batch self-correlation bias, and DB's one-step lag.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def ecf_sums(flat: Tensor, axes: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    theta = (flat @ axes).unsqueeze(-1) * t
    return theta.cos().sum(dim=0), theta.sin().sum(dim=0)


def ecf_error(cos_mean: Tensor, sin_mean: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    err = (cos_mean - target).square() + sin_mean.square()
    return (err * weights).sum(dim=-1).mean()


@torch.no_grad()
def embedding_metrics(proj: Tensor, axes: Tensor, t: Tensor, target: Tensor, weights: Tensor) -> dict[str, float]:
    """Full-set ECF error and spread of a set of projector outputs (N, D)."""

    flat = proj.float()
    cos_sum, sin_sum = ecf_sums(flat, axes, t)
    n = flat.shape[0]
    projected = flat @ axes
    return {
        "ecf_error": float(ecf_error(cos_sum / n, sin_sum / n, target, weights)),
        "axis_std": float(projected.std(dim=0).mean()),
        "dim_std": float(flat.std(dim=0).mean()),
    }


def _flat_grad(loss: Tensor, params: Sequence[nn.Parameter]) -> Tensor:
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)])


def _proj(model: nn.Module, views: Tensor | tuple[Tensor, Tensor]) -> Tensor:
    groups = views if isinstance(views, (tuple, list)) else (views,)
    projs = []
    for v in groups:
        _, proj = model(v)
        projs.append(proj.reshape(-1, proj.shape[-1]).float())
    return torch.cat(projs)


def _window_means(
    model: nn.Module, batches: Sequence[Tensor], axes: Tensor, t: Tensor
) -> tuple[Tensor, Tensor]:
    cos_sum = torch.zeros(axes.shape[1], t.numel(), device=axes.device)
    sin_sum = torch.zeros_like(cos_sum)
    count = 0
    with torch.no_grad():
        for views in batches:
            flat = _proj(model, views)
            c, s = ecf_sums(flat, axes, t)
            cos_sum += c
            sin_sum += s
            count += flat.shape[0]
    return cos_sum / count, sin_sum / count


def _pseudo_grad(
    model: nn.Module,
    params: Sequence[nn.Parameter],
    batches: Sequence[Tensor],
    axes: Tensor,
    t: Tensor,
    weights: Tensor,
    cos_err: Tensor,
    sin_err: Tensor,
) -> Tensor:
    """Gradient of sum_k w_k |mu_hat - phi|^2 with the error frozen to the given constants.

    With the window's own means this is the exact window gradient.
    """

    total = sum(
        sum(v.shape[0] * v.shape[1] for v in (views if isinstance(views, (tuple, list)) else (views,)))
        for views in batches
    )
    grad = None
    for views in batches:
        flat = _proj(model, views)
        theta = (flat @ axes).unsqueeze(-1) * t
        pseudo = cos_err * theta.cos() + sin_err * theta.sin()
        loss = 2.0 * (pseudo * weights).sum(dim=-1).mean(dim=-1).sum() / total
        g = _flat_grad(loss, params)
        grad = g if grad is None else grad + g
    assert grad is not None
    return grad


def _minibatch_grad(
    model: nn.Module,
    params: Sequence[nn.Parameter],
    batches: Sequence[Tensor],
    axes: Tensor,
    t: Tensor,
    target: Tensor,
    weights: Tensor,
) -> Tensor:
    grad = None
    for views in batches:
        flat = _proj(model, views)
        c, s = ecf_sums(flat, axes, t)
        n = flat.shape[0]
        loss = ecf_error(c / n, s / n, target, weights) / len(batches)
        g = _flat_grad(loss, params)
        grad = g if grad is None else grad + g
    assert grad is not None
    return grad


def gradient_fidelity(
    model: nn.Module,
    params: Sequence[nn.Parameter],
    window: Sequence[Tensor],
    independent_window: Sequence[Tensor],
    axes: Tensor,
    t: Tensor,
    target: Tensor,
    weights: Tensor,
    active_cos: Tensor,
    active_sin: Tensor,
    pool: Sequence[Tensor] | None = None,
) -> dict[str, float]:
    """Compare gradient estimators on one window, in the frame ``axes``.

    ``active_cos``/``active_sin`` are DB-SIGReg's promoted (lagged) means in that
    frame. Model buffers (e.g. BatchNorm running stats) are restored afterwards.
    """

    buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
    was_training = model.training
    model.train()
    try:
        mean_cos, mean_sin = _window_means(model, window, axes, t)
        exact = _pseudo_grad(model, params, window, axes, t, weights, mean_cos - target, mean_sin)

        db = _pseudo_grad(model, params, window, axes, t, weights, active_cos - target, active_sin)

        ind_cos, ind_sin = _window_means(model, independent_window, axes, t)
        db_fresh = _pseudo_grad(model, params, window, axes, t, weights, ind_cos - target, ind_sin)

        minibatch = _minibatch_grad(model, params, window, axes, t, target, weights)

        reference = None
        if pool:
            pop_cos, pop_sin = _window_means(model, pool, axes, t)
            reference = _pseudo_grad(model, params, window, axes, t, weights, pop_cos - target, pop_sin)
    finally:
        with torch.no_grad():
            for name, b in model.named_buffers():
                b.copy_(buffers[name])
        model.train(was_training)

    def cos(a: Tensor, b: Tensor) -> float:
        return float(F.cosine_similarity(a, b, dim=0))

    def rel(a: Tensor, b: Tensor) -> float:
        return float((a - b).norm() / b.norm().clamp_min(1e-12))

    def gap(c: Tensor, s: Tensor, ref_c: Tensor, ref_s: Tensor) -> float:
        return float((((c - ref_c).square() + (s - ref_s).square()) * weights).sum(-1).mean())

    out = {
        "cos_db": cos(db, exact),
        "cos_db_fresh": cos(db_fresh, exact),
        "cos_minibatch": cos(minibatch, exact),
        "relerr_db": rel(db, exact),
        "relerr_db_fresh": rel(db_fresh, exact),
        "relerr_minibatch": rel(minibatch, exact),
        # Weighted squared distance between the lagged statistics and the window's own.
        "stat_gap_db": float((((active_cos - mean_cos).square() + (active_sin - mean_sin).square()) * weights).sum(-1).mean()),
        "stat_gap_db_fresh": float((((ind_cos - mean_cos).square() + (ind_sin - mean_sin).square()) * weights).sum(-1).mean()),
    }
    if reference is not None:
        for name, grad in (("exact", exact), ("db", db), ("db_fresh", db_fresh), ("minibatch", minibatch)):
            out[f"cos_{name}_vs_pop"] = cos(grad, reference)
            out[f"relerr_{name}_vs_pop"] = rel(grad, reference)
        out["stat_gap_db_vs_pop"] = gap(active_cos, active_sin, pop_cos, pop_sin)
        out["stat_gap_window_vs_pop"] = gap(mean_cos, mean_sin, pop_cos, pop_sin)
        out["stat_gap_db_fresh_vs_pop"] = gap(ind_cos, ind_sin, pop_cos, pop_sin)
    return out
