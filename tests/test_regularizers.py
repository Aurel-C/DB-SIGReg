import unittest

import torch

from db_sigreg.regularizers import DoubleBufferSIGReg


def direct_sigreg_loss(z: torch.Tensor, axes: torch.Tensor, regularizer: DoubleBufferSIGReg) -> torch.Tensor:
    flat = z.reshape(-1, z.shape[-1]).float()
    theta = (flat @ axes).unsqueeze(-1) * regularizer.t.to(flat.device)
    cos_mean = theta.cos().mean(dim=0)
    sin_mean = theta.sin().mean(dim=0)
    err = (cos_mean - regularizer.target.to(flat.device)).square() + sin_mean.square()
    return (err * regularizer.weights.to(flat.device)).sum(dim=-1).mean() * flat.shape[0]


class DoubleBufferSIGRegTest(unittest.TestCase):
    def test_shadow_accumulation_and_promotion_match_batch_statistics(self) -> None:
        torch.manual_seed(0)
        z = torch.randn(6, 4)
        regularizer = DoubleBufferSIGReg(axes=3, knots=5, swap_steps=1)

        _, stats = regularizer(z)
        shadow_axes = regularizer.shadow_axes.detach().clone()
        theta = (z @ shadow_axes).unsqueeze(-1) * regularizer.t

        self.assertFalse(stats.mature)
        self.assertEqual(stats.shadow_count, z.shape[0])
        torch.testing.assert_close(regularizer.shadow_cos_sum, theta.cos().sum(dim=0))
        torch.testing.assert_close(regularizer.shadow_sin_sum, theta.sin().sum(dim=0))

        self.assertTrue(regularizer.after_optimizer_step())
        torch.testing.assert_close(regularizer.active_axes, shadow_axes)
        torch.testing.assert_close(regularizer.active_cos_sum, theta.cos().sum(dim=0))
        torch.testing.assert_close(regularizer.active_sin_sum, theta.sin().sum(dim=0))
        torch.testing.assert_close(regularizer.active_cos, theta.cos().mean(dim=0))
        torch.testing.assert_close(regularizer.active_sin, theta.sin().mean(dim=0))
        self.assertEqual(regularizer.active_count, z.shape[0])
        self.assertEqual(regularizer.shadow_count, 0)

    def test_pseudo_loss_gradient_matches_direct_sigreg_when_statistics_are_current(self) -> None:
        torch.manual_seed(1)
        num_samples, dim, axes, knots = 23, 7, 11, 5
        z = torch.randn(num_samples, dim, requires_grad=True)
        projection_axes = torch.randn(dim, axes)
        projection_axes = projection_axes / projection_axes.norm(dim=0, keepdim=True)
        regularizer = DoubleBufferSIGReg(axes=axes, knots=knots, swap_steps=99)
        regularizer._init_state(dim, z.device, z.dtype)

        with torch.no_grad():
            theta = (z.detach() @ projection_axes).unsqueeze(-1) * regularizer.t
            regularizer.active_axes.copy_(projection_axes)
            regularizer.active_cos.copy_(theta.cos().mean(dim=0))
            regularizer.active_sin.copy_(theta.sin().mean(dim=0))
            regularizer.active_count_tensor.fill_(num_samples)

        db_loss, db_stats = regularizer(z)
        db_loss.backward()
        db_grad = z.grad.detach().clone()

        z_direct = z.detach().clone().requires_grad_(True)
        direct_loss = direct_sigreg_loss(z_direct, projection_axes, regularizer)
        direct_loss.backward()

        self.assertAlmostEqual(db_stats.loss_metric, float(direct_loss.detach()), places=5)
        self.assertAlmostEqual(db_stats.ecf_error, float(direct_loss.detach() / num_samples), places=5)
        torch.testing.assert_close(db_grad, z_direct.grad, rtol=1e-5, atol=1e-6)

    def test_include_current_mode_matches_combined_previous_and_current_objective(self) -> None:
        torch.manual_seed(2)
        prev_samples, current_samples, dim, axes, knots = 13, 5, 7, 11, 5
        previous = torch.randn(prev_samples, dim)
        current = torch.randn(current_samples, dim, requires_grad=True)
        projection_axes = torch.randn(dim, axes)
        projection_axes = projection_axes / projection_axes.norm(dim=0, keepdim=True)
        regularizer = DoubleBufferSIGReg(axes=axes, knots=knots, swap_steps=99, stat_mode="include_current")
        regularizer._init_state(dim, current.device, current.dtype)

        with torch.no_grad():
            previous_theta = (previous @ projection_axes).unsqueeze(-1) * regularizer.t
            regularizer.active_axes.copy_(projection_axes)
            regularizer.active_cos_sum.copy_(previous_theta.cos().sum(dim=0))
            regularizer.active_sin_sum.copy_(previous_theta.sin().sum(dim=0))
            regularizer.active_count_tensor.fill_(prev_samples)
            regularizer.active_cos.copy_(regularizer.active_cos_sum / prev_samples)
            regularizer.active_sin.copy_(regularizer.active_sin_sum / prev_samples)

        db_loss, db_stats = regularizer(current)
        db_loss.backward()
        db_grad = current.grad.detach().clone()

        current_direct = current.detach().clone().requires_grad_(True)
        combined = torch.cat([previous.detach(), current_direct], dim=0)
        direct_loss = direct_sigreg_loss(combined, projection_axes, regularizer)
        direct_loss.backward()

        self.assertEqual(db_stats.loss_count, prev_samples + current_samples)
        self.assertEqual(db_stats.active_count, prev_samples)
        self.assertAlmostEqual(db_stats.loss_metric, float(direct_loss.detach()), places=5)
        self.assertAlmostEqual(db_stats.ecf_error, float(direct_loss.detach() / (prev_samples + current_samples)), places=5)
        torch.testing.assert_close(db_grad, current_direct.grad, rtol=1e-5, atol=1e-6)

    def test_include_current_does_not_append_current_batch_to_active_buffer(self) -> None:
        torch.manual_seed(3)
        prev_samples, current_samples, dim = 13, 5, 7
        regularizer = DoubleBufferSIGReg(axes=11, knots=5, swap_steps=99, stat_mode="include_current")
        regularizer._init_state(dim, torch.device("cpu"), torch.float32)
        active_cos_sum = torch.randn_like(regularizer.active_cos_sum)
        active_sin_sum = torch.randn_like(regularizer.active_sin_sum)

        with torch.no_grad():
            regularizer.active_cos_sum.copy_(active_cos_sum)
            regularizer.active_sin_sum.copy_(active_sin_sum)
            regularizer.active_count_tensor.fill_(prev_samples)

        _, stats = regularizer(torch.randn(current_samples, dim))

        self.assertEqual(stats.loss_count, prev_samples + current_samples)
        self.assertEqual(stats.active_count, prev_samples)
        torch.testing.assert_close(regularizer.active_cos_sum, active_cos_sum)
        torch.testing.assert_close(regularizer.active_sin_sum, active_sin_sum)

    def test_swap_can_happen_independently_of_optimizer_step(self) -> None:
        torch.manual_seed(4)
        regularizer = DoubleBufferSIGReg(axes=3, knots=5, swap_steps=3, stat_mode="include_current")

        for _ in range(2):
            regularizer(torch.randn(4, 6))
            self.assertFalse(regularizer.maybe_swap_buffers())

        regularizer(torch.randn(4, 6))
        self.assertTrue(regularizer.maybe_swap_buffers())
        self.assertEqual(regularizer.active_count, 12)
        self.assertEqual(regularizer.shadow_count, 0)
        self.assertEqual(regularizer.swaps, 1)


if __name__ == "__main__":
    unittest.main()
