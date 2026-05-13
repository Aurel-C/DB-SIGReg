# Double Buffer SIGReg (DB-SIGReg) 🔄
> **Memory-Efficient Distribution Matching for Self-Supervised Learning**

*Status: Theoretical whitepaper available. PyTorch implementation in progress.*

DB-SIGReg is a proposed regularization module that brings the stability of massive-batch distribution matching to single-GPU training environments. It builds upon the Sketched Isotropic Gaussian Regularization (SIGReg) introduced in LeJEPA, utilizing temporal aggregation and a double-buffered projection mechanism to decouple memory constraints from statistical validity.

📄 **[Read the theoretical whitepaper draft here](db_sigreg.pdf)**

## 🛑 The Bottleneck: The Null Space Trap
Self-supervised architectures utilizing empirical characteristic function tests (like Epps-Pulley) force latent representations to form an isotropic Gaussian. Because computing this in high dimensions is hard, embeddings are projected onto random 1D axes (Cramér-Wold theorem).

This creates a severe optimization contradiction:
1. **To get stable statistics:** You need massive batch sizes ($B \ge 4096$).
2. **To use an EMA instead of big batches:** The 1D projection axes must remain *fixed* over time. 
3. **The Null Space Trap:** If the projection axes are fixed, the neural network acts adversarially, hiding collapsed representations in the geometric null space between selected axes. The axes *must* be constantly re-sampled, which instantly invalidates the EMA.

## 💡 The Solution: Double Buffering
DB-SIGReg imports the concept of double buffering from systems engineering into representation learning. It maintains two overlapping states:

* **🟢 The Active Buffer:** Uses a fixed projection matrix and an accumulation of historical frequencies. It computes the Epps-Pulley loss and passes clean, stable gradients backward.
* **👻 The Shadow Buffer:** Uses a newly initialized random projection matrix. It accumulates statistics to be used after the swap.

After $K$ steps, the Active Buffer is discarded, the Shadow Buffer is promoted to Active, and a new Shadow Buffer is spun up. **Result: The axes rotate to prevent dimensional collapse, but the network never experiences instability from low batch size.**

## 🛠️ Implementation Preview (WIP)

The full PyTorch training loop and module are currently being implemented. The core mechanism decoupling the gradient graph from the shadow EMA will follow this conceptual structure:

```python
import torch
import torch.nn as nn

class DoubleBufferSIGReg(nn.Module):

    def forward(self, z):
        # 1. ACTIVE BUFFER (Gradient Path)
        z_active = z @ self.active_proj
        # ... Compute loss against Target Gaussian using active buffer  ...
        
        # 2. SHADOW BUFFER (Silent Warm-up)
        with torch.no_grad():
            z_shadow = z @ self.shadow_proj
            # ... Silently aggregate frequencies to mature the shadow buffer ...
            
        # 3. SWAP MECHANISM
        self.current_step += 1
        if self.current_step >= self.swap_steps:
            self._swap_buffers() # Promote shadow to active, spin up new shadow
            
        return loss
```

## 🚀 Roadmap / Next Steps
- [x] Formulate math and architectural geometry.
- [x] Publish initial whitepaper draft.
- [ ] **[In Progress]** Implement PyTorch module with `.detach()` shadow updates.
- [ ] Run empirical baseline on small-scale datasets.
- [ ] Ablation study: Swap frequency ($K$) vs. dimensional collapse.
- [ ] Ablation study: EMA vs. CMA for shadow buffer.

## 📖 Citation
If you find this theoretical framework useful, please consider citing:
```bibtex
@article{db_sigreg_2026,
  title={Double-Buffered Random Projections for Memory-Efficient Distribution Matching in Self-Supervised Learning},
  author={[Your Name]},
  year={2026},
  publisher={GitHub}
}
```
