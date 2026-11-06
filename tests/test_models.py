import sys
import unittest
from unittest import mock

from torch import nn

from db_sigreg.models import SSLModel
from db_sigreg.train import parse_args


def norm_layers(module: nn.Module) -> list[nn.Module]:
    return [m for m in module.modules() if isinstance(m, (nn.BatchNorm1d, nn.LayerNorm))]


class ProjectorNormTest(unittest.TestCase):
    def make(self, norm: str, proj_norm: str | None) -> SSLModel:
        return SSLModel("cnn", 32, emb_dim=32, proj_dim=8, proj_hidden_dim=16, num_classes=10, norm=norm, proj_norm=proj_norm)

    def test_projector_follows_encoder_norm_by_default(self) -> None:
        self.assertTrue(all(isinstance(m, nn.BatchNorm1d) for m in norm_layers(self.make("bn", None).projector)))
        self.assertTrue(all(isinstance(m, nn.LayerNorm) for m in norm_layers(self.make("gn", None).projector)))
        self.assertEqual(len(norm_layers(self.make("gn", None).projector)), 2)

    def test_proj_norm_none_removes_projector_norms_only(self) -> None:
        model = self.make("gn", "none")
        self.assertEqual(norm_layers(model.projector), [])
        self.assertTrue(any(isinstance(m, nn.GroupNorm) for m in model.encoder.modules()))

    def test_new_flags_default_to_previous_behaviour(self) -> None:
        with mock.patch.object(sys, "argv", ["db-sigreg"]):
            args = parse_args()
        self.assertEqual(args.proj_norm, "same")
        self.assertEqual(args.inv_warmup_steps, 0)


if __name__ == "__main__":
    unittest.main()
