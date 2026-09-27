# Ternary Relational Neural Networks: Context-Gated Signed Evidence as a Hidden-Unit Primitive

**Author:** Ansh Chhari  
**Repository:** Official PyTorch Implementation  
**Author Website:** [anshchhari.vercel.app](https://anshchhari.vercel.app)  

---

## Abstract

Most conventional neural-network units learn a weight vector and compute a projection of the input. Prototype-based networks instead compare an input representation with learned reference points. This repository presents **Ternary Relational Networks (TRN)** and the **Ternary Relational Unit (TRU)**—an architectural primitive maintaining three learned reference states per hidden unit:
1. **Positive Evidence Prototype ($P$)**
2. **Negative Evidence Prototype ($N$)**
3. **Neutral / Context Prototype ($U$)**

The positive and negative references create signed evidence, while the neutral reference controls how strongly that evidence is expressed via context gating.

---

## Mathematical Formulation

For a layer mapping input $x \in \mathbb{R}^d$ to hidden dimension $h$, each unit maintains reference parameters $P, N, U \in \mathbb{R}^{h \times d}$, parameter scalar logs $\log \tau, \log \gamma$, and bias $b$:

1. **Parameter Softplus Constraints:**
   $$\tau = \text{softplus}(\log \tau) + 10^{-3}, \quad \gamma = \text{softplus}(\log \gamma) + 10^{-3}$$

2. **Mean-Squared Feature Distances:**
   $$d_P = \frac{1}{d} \Vert{}x - P\Vert{}_2^2, \quad d_N = \frac{1}{d} \Vert{}x - N\Vert{}_2^2, \quad d_U = \frac{1}{d} \Vert{}x - U\Vert{}_2^2$$

3. **Prototype Similarities:**
   $$s_P = \exp\left(-\frac{d_P}{\tau}\right), \quad s_N = \exp\left(-\frac{d_N}{\tau}\right), \quad s_U = \exp\left(-\frac{d_U}{\tau}\right)$$

4. **Signed Evidence & Context Gate:**
   $$e = s_P - s_N, \quad g = \sigma\left(\gamma \cdot \left(s_U - 0.5(s_P + s_N)\right)\right)$$

5. **Layer Activation:**
   $$h = \text{LayerNorm}\Big(\tanh(4e) \cdot g + b\Big)$$

---

## Benchmark Results

### 1. Tabular Digits Dataset (5 Random Seeds, 25 Epochs)

Evaluated across 5 random seeds (0, 1, 2, 3, 4) on an 80/20 stratified split with AdamW optimizer (`lr = 0.002`, `weight_decay = 1e-4`):

| Model | Parameters | Clean Accuracy | Noisy Accuracy ($\sigma = 0.20$) | Clean Log-Loss | Noisy Log-Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MLP** | 8,970 | 74.28% ± 3.30% | 68.39% ± 5.35% | 1.6988 ± 0.3595 | 1.7760 ± 0.3155 |
| **DualPrototype** | 17,546 | **92.11% ± 1.34%** | **87.22% ± 0.81%** | **0.3882 ± 0.0344** | **0.5366 ± 0.0307** |
| **TRN (Ours)** | 25,866 | 91.89% ± 1.68% | 86.61% ± 1.54% | 0.3985 ± 0.0484 | 0.5484 ± 0.0434 |

### 2. High-Dimensional MNIST Benchmark (3 Random Seeds, GPU Acceleration)

Evaluated on $28 \times 28$ flattened vision inputs ($d = 784, h = 256$):

| Model | Parameters | Clean Accuracy (Mean) | Noisy Accuracy ($\sigma = 0.20$) |
| :--- | :---: | :---: | :---: |
| **MLP (Matched)** | 181,495 | **97.66%** | **86.92%** |
| **DualPrototype (Matched)** | 188,590 | 94.97% | 66.02% |
| **TRN (Matched)** | 180,750 | 95.43% | 60.47% |

---

## Quick Start & Setup

```bash
git clone [https://github.com/AnshChhari/ternary-relational-networks.git](https://github.com/AnshChhari/ternary-relational-networks.git)
cd ternary-relational-networks
pip install -r requirements.txt
```

### Reproducing Benchmarks

Run the 5-seed Digits benchmark:
```bash
python benchmark_digits.py
```

Run the GPU-accelerated MNIST benchmark:
```bash
python benchmark_mnist.py
```

---

## Citation

```bibtex
@article{chhari2026ternary,
  title={Ternary Relational Neural Networks: Context-Gated Signed Evidence as a Hidden-Unit Primitive},
  author={Chhari, Ansh},
  journal={Exploratory Research Prototype},
  year={2026},
  url={[https://github.com/AnshChhari/ternary-relational-networks](https://github.com/AnshChhari/ternary-relational-networks)}
}
```
