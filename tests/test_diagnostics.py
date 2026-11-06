import unittest

import torch
from torch import nn

from db_sigreg.diagnostics import _flat_grad, ecf_error, ecf_sums, gradient_fidelity
from db_sigreg.regularizers import SIGReg, _make_axes


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(6, 5)
        self.projector = nn.Linear(5, 4)

    def forward(self, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.encoder(views)
        return emb, self.projector(torch.tanh(emb))


class GradientFidelityTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = TinyModel()
        self.params = list(self.model.parameters())
        self.reg = SIGReg(axes=3, knots=5)
        self.axes = _make_axes(4, 3, torch.device("cpu"), torch.float32)
        self.window = [torch.randn(5, 2, 6) for _ in range(3)]
        self.other = [torch.randn(5, 2, 6) for _ in range(3)]

    def full_batch_grad(self, batches: list[torch.Tensor]) -> torch.Tensor:
        flat = torch.cat([self.model(v)[1].reshape(-1, 4) for v in batches])
        c, s = ecf_sums(flat, self.axes, self.reg.t)
        n = flat.shape[0]
        return _flat_grad(ecf_error(c / n, s / n, self.reg.target, self.reg.weights), self.params)

    def run_fidelity(self, active_cos: torch.Tensor, active_sin: torch.Tensor) -> dict[str, float]:
        return gradient_fidelity(
            self.model,
            self.params,
            self.window,
            self.other,
            self.axes,
            self.reg.t,
            self.reg.target,
            self.reg.weights,
            active_cos,
            active_sin,
        )

    def test_db_with_current_window_statistics_is_exact(self) -> None:
        flat = torch.cat([self.model(v)[1].reshape(-1, 4) for v in self.window]).detach()
        c, s = ecf_sums(flat, self.axes, self.reg.t)
        out = self.run_fidelity(c / flat.shape[0], s / flat.shape[0])
        self.assertAlmostEqual(out["cos_db"], 1.0, places=5)
        self.assertLess(out["relerr_db"], 1e-4)
        self.assertLess(out["stat_gap_db"], 1e-10)

    def test_population_reference_equals_exact_when_pool_is_the_window(self) -> None:
        flat = torch.cat([self.model(v)[1].reshape(-1, 4) for v in self.window]).detach()
        c, s = ecf_sums(flat, self.axes, self.reg.t)
        out = gradient_fidelity(
            self.model, self.params, self.window, self.other, self.axes, self.reg.t, self.reg.target,
            self.reg.weights, c / flat.shape[0], s / flat.shape[0], pool=self.window,
        )
        self.assertAlmostEqual(out["cos_exact_vs_pop"], 1.0, places=5)
        self.assertAlmostEqual(out["cos_db_vs_pop"], 1.0, places=5)
        self.assertLess(out["stat_gap_window_vs_pop"], 1e-10)

    def test_minibatch_is_not_exact(self) -> None:
        out = self.run_fidelity(self.reg.target.expand(3, -1).clone(), torch.zeros(3, 5))
        self.assertLess(out["cos_minibatch"], 0.9999)
        self.assertGreater(out["relerr_minibatch"], 1e-3)

    def test_two_pass_exact_matches_full_batch_autograd(self) -> None:
        # Reconstruct the exact gradient through the public API: db with the
        # window's own means must equal the full-batch autograd gradient.
        expected = self.full_batch_grad(self.window)
        flat = torch.cat([self.model(v)[1].reshape(-1, 4) for v in self.window]).detach()
        c, s = ecf_sums(flat, self.axes, self.reg.t)
        from db_sigreg.diagnostics import _pseudo_grad

        got = _pseudo_grad(
            self.model,
            self.params,
            self.window,
            self.axes,
            self.reg.t,
            self.reg.weights,
            c / flat.shape[0] - self.reg.target,
            s / flat.shape[0],
        )
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-6)


class SIGRegRefreshTest(unittest.TestCase):
    def test_fixed_axes_are_reused_and_default_axes_are_redrawn(self) -> None:
        torch.manual_seed(0)
        z = torch.randn(8, 4)
        fixed = SIGReg(axes=3, knots=5, refresh_every=0)
        fixed(z)
        first = fixed._fixed_axes.clone()
        fixed(z)
        torch.testing.assert_close(fixed._fixed_axes, first)

        fresh = SIGReg(axes=3, knots=5)
        fresh(z)
        first = fresh._fixed_axes.clone()
        fresh(z)
        self.assertFalse(torch.allclose(fresh._fixed_axes, first))

    def test_refresh_every_k_keeps_axes_within_a_window(self) -> None:
        torch.manual_seed(0)
        z = torch.randn(8, 4)
        reg = SIGReg(axes=3, knots=5, refresh_every=2)
        reg(z)
        a = reg._fixed_axes.clone()
        reg(z)
        torch.testing.assert_close(reg._fixed_axes, a)
        reg(z)
        self.assertFalse(torch.allclose(reg._fixed_axes, a))


if __name__ == "__main__":
    unittest.main()
