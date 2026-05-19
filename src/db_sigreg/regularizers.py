from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

DBStatMode = Literal["detached", "include_current"]


@dataclass
class RegularizerStats:
    loss_metric: float
    ecf_error: float
    optimization_loss: float
    loss_count: int
    active_count: int
    shadow_count: int
    swaps: int
    mature: bool


def _make_axes(dim: int, axes: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    basis = torch.randn(dim, axes, device=device, dtype=dtype)
    return basis / basis.norm(dim=0, keepdim=True).clamp_min(1e-12)


class SIGReg(nn.Module):
    """LeJEPA-style sketched isotropic Gaussian regularization.

    This recomputes random projection axes every forward pass and estimates the
    empirical characteristic function from the current mini-batch.
    """

    def __init__(self, axes: int = 256, knots: int = 17, t_max: float = 3.0) -> None:
        super().__init__()
        self.axes = axes
        t = torch.linspace(0.0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        target = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t)
        self.register_buffer("target", target)
        self.register_buffer("weights", weights * target)

    def forward(self, z: Tensor) -> tuple[Tensor, RegularizerStats]:
        flat = z.reshape(-1, z.shape[-1]).float()
        axes = _make_axes(flat.shape[-1], self.axes, flat.device, flat.dtype)
        projected = flat @ axes
        theta = projected.unsqueeze(-1) * self.t.to(projected.device)
        cos_mean = theta.cos().mean(dim=0)
        sin_mean = theta.sin().mean(dim=0)
        err = (cos_mean - self.target.to(projected.device)).square() + sin_mean.square()
        ecf_error = (err * self.weights.to(projected.device)).sum(dim=-1).mean()
        loss = ecf_error * flat.shape[0]
        stats = RegularizerStats(
            loss_metric=float(loss.detach().cpu()),
            ecf_error=float(ecf_error.detach().cpu()),
            optimization_loss=float(loss.detach().cpu()),
            loss_count=flat.shape[0],
            active_count=flat.shape[0],
            shadow_count=0,
            swaps=0,
            mature=True,
        )
        return loss, stats


class DoubleBufferSIGReg(nn.Module):
    """Detached double-buffer SIGReg pseudo-loss.

    The active buffer provides fixed random axes and mature detached ECF error
    constants for gradients. The shadow buffer silently accumulates ECF sums in
    another projection frame and is promoted after an optimizer step.
    """

    def __init__(
        self,
        axes: int = 256,
        knots: int = 17,
        t_max: float = 3.0,
        swap_steps: int = 4,
        stat_mode: DBStatMode = "detached",
        gradient_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if swap_steps < 1:
            raise ValueError("swap_steps must be >= 1")
        if stat_mode not in ("detached", "include_current"):
            raise ValueError(f"unknown DB-SIGReg stat_mode: {stat_mode}")
        self.axes = axes
        self.swap_steps = swap_steps
        self.stat_mode = stat_mode
        self.gradient_scale = gradient_scale
        t = torch.linspace(0.0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        target = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t)
        self.register_buffer("target", target)
        self.register_buffer("weights", weights * target)
        self.register_buffer("active_axes", torch.empty(0))
        self.register_buffer("shadow_axes", torch.empty(0))
        self.register_buffer("active_cos", torch.empty(0))
        self.register_buffer("active_sin", torch.empty(0))
        self.register_buffer("active_cos_sum", torch.empty(0))
        self.register_buffer("active_sin_sum", torch.empty(0))
        self.register_buffer("shadow_cos_sum", torch.empty(0))
        self.register_buffer("shadow_sin_sum", torch.empty(0))
        self.register_buffer("active_count_tensor", torch.tensor(0, dtype=torch.long))
        self.register_buffer("shadow_count_tensor", torch.tensor(0, dtype=torch.long))
        self.register_buffer("step_tensor", torch.tensor(0, dtype=torch.long))
        self.register_buffer("swap_tensor", torch.tensor(0, dtype=torch.long))

    @property
    def active_count(self) -> int:
        return int(self.active_count_tensor.item())

    @property
    def shadow_count(self) -> int:
        return int(self.shadow_count_tensor.item())

    @property
    def swaps(self) -> int:
        return int(self.swap_tensor.item())

    @property
    def mature(self) -> bool:
        return self.active_count > 0

    @property
    def ready_to_swap(self) -> bool:
        return int(self.step_tensor.item()) >= self.swap_steps and self.shadow_count > 0

    def _init_state(self, dim: int, device: torch.device, dtype: torch.dtype) -> None:
        stat_shape = (self.axes, self.t.numel())
        self.active_axes = _make_axes(dim, self.axes, device, dtype)
        self.shadow_axes = _make_axes(dim, self.axes, device, dtype)
        target = self.target.to(device=device, dtype=dtype).expand(stat_shape).clone()
        self.active_cos = target
        self.active_sin = torch.zeros(stat_shape, device=device, dtype=dtype)
        self.active_cos_sum = torch.zeros(stat_shape, device=device, dtype=dtype)
        self.active_sin_sum = torch.zeros(stat_shape, device=device, dtype=dtype)
        self.shadow_cos_sum = torch.zeros(stat_shape, device=device, dtype=dtype)
        self.shadow_sin_sum = torch.zeros(stat_shape, device=device, dtype=dtype)
        self.active_count_tensor = torch.tensor(0, device=device, dtype=torch.long)
        self.shadow_count_tensor = torch.tensor(0, device=device, dtype=torch.long)
        self.step_tensor = torch.tensor(0, device=device, dtype=torch.long)
        self.swap_tensor = torch.tensor(0, device=device, dtype=torch.long)

    def forward(self, z: Tensor) -> tuple[Tensor, RegularizerStats]:
        flat = z.reshape(-1, z.shape[-1]).float()
        if self.active_axes.numel() == 0 or self.active_axes.shape[0] != flat.shape[-1]:
            self._init_state(flat.shape[-1], flat.device, flat.dtype)

        t = self.t.to(flat.device)
        weights = self.weights.to(flat.device)
        target = self.target.to(flat.device)

        active_projected = flat @ self.active_axes
        theta = active_projected.unsqueeze(-1) * t
        current_cos_sum = theta.cos().sum(dim=0)
        current_sin_sum = theta.sin().sum(dim=0)

        if self.stat_mode == "include_current":
            prev_count = self.active_count
            total_count = prev_count + flat.shape[0]
            cos_mean = (self.active_cos_sum.detach() + current_cos_sum) / total_count
            sin_mean = (self.active_sin_sum.detach() + current_sin_sum) / total_count
            err = (cos_mean - target).square() + sin_mean.square()
            pseudo_loss = (err * weights).sum(dim=-1).mean() * total_count
            metric_loss = pseudo_loss.detach()
            ecf_error = float((metric_loss / total_count).cpu())
        else:
            cos_err = (self.active_cos - target).detach()
            sin_err = self.active_sin.detach()
            pseudo = cos_err.unsqueeze(0) * theta.cos() + sin_err.unsqueeze(0) * theta.sin()
            pseudo_loss = self.gradient_scale * flat.shape[0] * (pseudo * weights).sum(dim=-1).mean()
            ecf_error = self._active_error().item()
            total_count = max(self.active_count, 1)
            metric_loss = torch.as_tensor(
                ecf_error * total_count,
                device=flat.device,
                dtype=pseudo_loss.dtype,
            )

        with torch.no_grad():
            shadow_projected = flat.detach() @ self.shadow_axes
            shadow_theta = shadow_projected.unsqueeze(-1) * t
            self.shadow_cos_sum.add_(shadow_theta.cos().sum(dim=0))
            self.shadow_sin_sum.add_(shadow_theta.sin().sum(dim=0))
            self.shadow_count_tensor.add_(flat.shape[0])
            self.step_tensor.add_(1)
            loss_metric = float(metric_loss.cpu())

        loss = metric_loss.detach() + pseudo_loss - pseudo_loss.detach()

        stats = RegularizerStats(
            loss_metric=float(loss_metric),
            ecf_error=float(ecf_error),
            optimization_loss=float(pseudo_loss.detach().cpu()),
            loss_count=int(total_count),
            active_count=self.active_count,
            shadow_count=self.shadow_count,
            swaps=self.swaps,
            mature=self.mature,
        )
        return loss, stats

    def maybe_swap_buffers(self) -> bool:
        """Promote shadow statistics once enough forward passes have warmed it."""

        if not self.ready_to_swap:
            return False
        with torch.no_grad():
            count = self.shadow_count_tensor.clamp_min(1).to(self.shadow_cos_sum.dtype)
            self.active_axes.copy_(self.shadow_axes)
            self.active_cos_sum.copy_(self.shadow_cos_sum)
            self.active_sin_sum.copy_(self.shadow_sin_sum)
            self.active_cos.copy_(self.shadow_cos_sum / count)
            self.active_sin.copy_(self.shadow_sin_sum / count)
            self.active_count_tensor.copy_(self.shadow_count_tensor)
            self.shadow_axes.copy_(
                _make_axes(
                    self.shadow_axes.shape[0],
                    self.shadow_axes.shape[1],
                    self.shadow_axes.device,
                    self.shadow_axes.dtype,
                )
            )
            self.shadow_cos_sum.zero_()
            self.shadow_sin_sum.zero_()
            self.shadow_count_tensor.zero_()
            self.step_tensor.zero_()
            self.swap_tensor.add_(1)
        return True

    def after_optimizer_step(self) -> bool:
        """Backward-compatible alias for older training loops."""

        return self.maybe_swap_buffers()

    def _active_error(self) -> Tensor:
        target = self.target.to(self.active_cos.device)
        weights = self.weights.to(self.active_cos.device)
        err = (self.active_cos - target).square() + self.active_sin.square()
        return (err * weights).sum(dim=-1).mean()
