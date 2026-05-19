# Double Buffer SIGReg (DB-SIGReg) 🔄
> **Memory-Efficient Distribution Matching for Self-Supervised Learning**

*Status: Theoretical whitepaper available. Minimal PyTorch implementation available.*

DB-SIGReg is a proposed regularization module that brings the stability of big batch distribution matching to small batches. It builds upon the Sketched Isotropic Gaussian Regularization (SIGReg) introduced in LeJEPA, utilizing temporal aggregation and a double-buffered projection mechanism to decouple memory constraints from statistical validity.

📄 **[Read the theoretical whitepaper draft here](db_sigreg.pdf)**

## 🛑 The Bottleneck: The Null Space Trap
Self-supervised architectures utilizing empirical characteristic function tests (like Epps-Pulley) force latent representations to form an isotropic Gaussian. Because computing this in high dimensions is hard, embeddings are projected onto random 1D axes (Cramér-Wold theorem).

This creates a severe optimization contradiction:
1. **To get stable statistics:** You need big batch sizes.
2. **To use an EMA instead of big batches:** The 1D projection axes must remain *fixed* over time. 
3. **The Null Space Trap:** If the projection axes are fixed, the neural network acts adversarially, hiding collapsed representations in the geometric null space between selected axes. The axes *must* be constantly re-sampled, which instantly invalidates the EMA.

## 💡 The Solution: Double Buffering
DB-SIGReg imports the concept of double buffering from systems engineering into representation learning. It maintains two overlapping states:

* **🟢 The Active Buffer:** Uses a fixed projection matrix and an accumulation of historical frequencies. It computes the Epps-Pulley loss and passes clean, stable gradients backward.
* **👻 The Shadow Buffer:** Uses a newly initialized random projection matrix. It accumulates statistics to be used after the swap.

After $K$ steps, the Active Buffer is discarded, the Shadow Buffer is promoted to Active, and a new Shadow Buffer is spun up. **Result: The axes rotate to prevent dimensional collapse, but the network never experiences instability from low batch size.**

## 🛠️ Minimal Implementation

This repository now contains a compact experiment harness:

- `src/db_sigreg/regularizers.py`: LeJEPA-style `SIGReg` baseline and the detached `DoubleBufferSIGReg` pseudo-loss.
- `src/db_sigreg/models.py`: small CNN, ResNet-18, and tiny ViT encoders with an online linear probe.
- `src/db_sigreg/data.py`: CIFAR-10, CIFAR-100, STL-10, and fake-data loaders with multi-view SSL augmentations.
- `src/db_sigreg/train.py`: training loop, gradient accumulation, TensorBoard logging, checkpoints, and throughput/memory metrics.

### Smoke test

```bash
uv run db-sigreg --dataset fake --epochs 1 --limit-train 128 --limit-eval 64 --batch-size 16 --accum-steps 2 --num-workers 0 --device cuda
```

### First real validation on a small GPU

```bash
uv run db-sigreg --dataset cifar10 --backbone cnn --loss dbsigreg --epochs 20 --batch-size 64 --accum-steps 4 --image-size 64
```

Compare against direct SIGReg under the same memory envelope:

```bash
uv run db-sigreg --dataset cifar10 --backbone cnn --loss sigreg --epochs 20 --batch-size 64 --accum-steps 4 --image-size 64
```

Note that `--accum-steps` on direct SIGReg does not make it equivalent to a
single virtual-batch ECF loss: it accumulates gradients from squared mini-batch
ECF errors. DB-SIGReg instead uses fixed projection axes and detached
window-level statistics from the previous buffer, so the optimization dynamics
are intentionally different.

You can also test a less independent but less stale variant that temporarily
combines the active context with the current mini-batch for the loss. The
current batch is not appended to the active buffer; it only warms the shadow
buffer for the next swap.

```bash
uv run db-sigreg --dataset cifar10 --backbone cnn --loss dbsigreg --db-stat-mode include_current --epochs 20 --batch-size 128 --accum-steps 4 --swap-steps 3 --image-size 64
```

Open TensorBoard:

```bash
uv run tensorboard --logdir runs
```

Useful metrics:

- `eval/probe_acc`: quick representation-quality proxy.
- `train/sigreg_metric`: mature active-buffer ECF error for DB-SIGReg, direct ECF statistic for SIGReg.
- `train/sigreg_ecf_error`: count-normalized ECF error; use this when comparing DB-SIGReg virtual batches against mini-batch SIGReg.
- `train/sigreg_optimization_loss`: signed detached pseudo-loss used for DB-SIGReg gradients.
- `buffer/loss_count`: projected sample count used by the current loss, including temporary current-batch stats.
- `perf/samples_per_sec`, `perf/iter_time_sec`, `perf/peak_memory_mb`: speed and memory comparison.
- `buffer/active_count`, `buffer/shadow_count`, `buffer/swaps`, `buffer/mature`: DB-SIGReg buffer health.

## 🚀 Roadmap / Next Steps
- [x] Formulate math and architectural geometry.
- [x] Publish initial whitepaper draft.
- [x] Implement PyTorch module with `.detach()` shadow updates.
- [x] Run empirical baseline on small-scale datasets.

## 📖 Citation
If you find this theoretical framework useful, please consider citing:
```bibtex
@article{db_sigreg_2026,
  title={Double-Buffered SIGReg},
  author={Aurélien Cecille},
  year={2026},
  publisher={GitHub}
}
```
