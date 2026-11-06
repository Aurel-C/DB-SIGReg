import unittest

import torch

from db_sigreg.gpu_data import GPUViewLoader


class GPUViewLoaderTest(unittest.TestCase):
    def make(self, train: bool, n: int = 10) -> GPUViewLoader:
        images = torch.randint(0, 256, (n, 32, 32, 3), dtype=torch.uint8)
        labels = torch.arange(n)
        return GPUViewLoader(
            images, labels, batch_size=4, views=2, image_size=32, train=train, device=torch.device("cpu"), seed=0
        )

    def test_train_batches_have_views_and_drop_last(self) -> None:
        loader = self.make(train=True)
        batches = list(loader)
        self.assertEqual(len(batches), 2)
        views, labels = batches[0]
        self.assertEqual(tuple(views.shape), (4, 2, 3, 32, 32))
        self.assertEqual(labels.shape, (4,))
        self.assertFalse(torch.equal(views[:, 0], views[:, 1]))
        self.assertTrue(torch.isfinite(views).all())

    def test_eval_covers_every_sample_in_order(self) -> None:
        loader = self.make(train=False)
        labels = torch.cat([y for _, y in loader])
        torch.testing.assert_close(labels, torch.arange(10))
        views, _ = next(iter(loader))
        self.assertEqual(tuple(views.shape), (4, 1, 3, 32, 32))


if __name__ == "__main__":
    unittest.main()
