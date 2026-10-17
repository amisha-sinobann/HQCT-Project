# Hybrid Quantum-Classical Transformer (HQCT)

Implementation of the paper:

> **Hybrid Quantum-Classical Attention Mechanisms for Parameter-Efficient Transformers in Natural Language Processing**  
> Amisha Singh, The NorthCap University

This repository provides a complete, end-to-end trainable implementation of:

- **HQCT-K** – Quantum Kernel Attention (primary model in the paper)
- **HQCT-P** – Quantum Projection Attention
- Matched classical Transformer baseline

## Architecture Overview

```
Input embeddings
       │
       ├──────────────────────────────┐
       │                              │
 Classical Value Projection     Quantum Block
       │                              │
       │                    ┌─────────┴─────────┐
       │                    │  Angle Embedding  │
       │                    │  + Variational    │
       │                    │    Ansatz (L=3)   │
       │                    └─────────┬─────────┘
       │                              │
       │              HQCT-K: fidelity kernel κ(q,k)=|⟨ψq|ψk⟩|²
       │              HQCT-P: Pauli-Z expectations as features
       │                              │
       └──────────► Softmax Attention ◄──────────┘
                        │
                   Weighted sum of values
                        │
                   Feed-forward + residual
```

### Quantum Circuit (hardware-efficient ansatz)

- **Encoding**: Angle embedding (`RX` + `RY` per qubit)
- **Ansatz**: `L` layers of `(RX, RY, RZ)` on every qubit + nearest-neighbour `CNOT` chain
- Default configuration (paper): **6 qubits, depth 3**

## Installation

```bash
# Clone / copy the repository
cd hqct

# Create environment (recommended)
python -m venv .venv
source .venv/bin/activate   # or Windows equivalent

# Install dependencies
pip install -r requirements.txt
```

> **Note**: The quantum kernel attention (`hqct-k`) evaluates a circuit for every query-key pair.  
> It is computationally expensive; the paper reports ~3.5 h on an A100 for AG News.  
> For quick experiments use `hqct-p` or the classical baseline, or reduce `max_seq_len` / `n_qubits`.

## Quick Start – Smoke Test

```bash
python scripts/demo.py
```

This runs forward + backward passes on synthetic data for all three attention types and confirms that gradients flow through the quantum parameters.

## Training

### Full-data experiment (matches Table 1)

```bash
# HQCT-K on AG News (main result)
python scripts/train.py \
    --dataset ag_news \
    --attn_type hqct-k \
    --n_qubits 6 \
    --q_layers 3 \
    --epochs 30 \
    --batch_size 32

# Classical baseline
python scripts/train.py --dataset ag_news --attn_type classical

# HQCT-P
python scripts/train.py --dataset ag_news --attn_type hqct-p
```

Supported datasets: `ag_news`, `imdb`, `sst2`, `multinli`.

### Few-shot (Table 2)

```bash
python scripts/train.py \
    --dataset ag_news \
    --attn_type hqct-k \
    --few_shot 16 \
    --epochs 50 \
    --patience 10
```

### Ablations

```bash
# Number of qubits
python scripts/train.py --dataset ag_news --n_qubits 4
python scripts/train.py --dataset ag_news --n_qubits 8

# Circuit depth
python scripts/train.py --dataset ag_news --q_layers 2
python scripts/train.py --dataset ag_news --q_layers 4

# Noise robustness (Fig. 4)
python scripts/train.py --dataset ag_news --noise 0.01
python scripts/train.py --dataset ag_news --noise 0.05
```

## Project Layout

```
hqct/
├── hqct/
│   ├── __init__.py
│   ├── quantum_circuits.py   # Hardware-efficient ansatz, QuantumKernel, QuantumProjection
│   ├── models.py             # HQCT, attention modules, Transformer blocks
│   ├── data.py               # AG News / IMDB / SST-2 / MultiNLI loaders + few-shot
│   └── train.py              # Training loop, early stopping, parameter groups
├── scripts/
│   ├── train.py              # CLI entry point
│   └── demo.py               # Quick smoke test
├── requirements.txt
└── README.md
```

## Key Implementation Details (matching the paper)

| Component              | Implementation                                      |
|------------------------|-----------------------------------------------------|
| Quantum library        | PennyLane 0.34+ (Lightning backend)                 |
| Differentiation        | Parameter-shift rule for quantum, autograd for classical |
| Optimiser              | Adam – lr=1e-3 (classical), 5e-4 (quantum)          |
| Quantum regularisation | L2 on quantum weights only                          |
| Initialisation         | Quantum weights ~ 𝒩(0, 0.01²) (near identity)       |
| Early stopping         | Patience = 5 on validation accuracy                 |
| Sequence lengths       | 128 (AG News, SST-2, MultiNLI), 256 (IMDB)          |

## Expected Results (from the paper)

| Model              | AG News | IMDB  | SST-2 | MultiNLI | Classical Params |
|--------------------|---------|-------|-------|----------|------------------|
| Classical Transformer | 91.2   | 88.4 | 89.1 | 72.3    | 3.85 M          |
| HQCT-K             | **91.8** | **88.9** | 88.7 | **73.1** | **3.31 M**     |
| HQCT-P             | 90.9   | 88.1 | 88.3 | 71.8    | 3.28 M          |

Few-shot (16 examples/class): HQCT-K gains +4–5 absolute points over the classical baseline.

## Notes & Limitations

1. **Computational cost of HQCT-K**  
   The quantum kernel requires a circuit evaluation for every pair of tokens.  
   On a classical simulator this is O(N²) state-vector simulations per attention head.  
   For production use or longer sequences consider:
   - Low-rank / Nyström approximations of the kernel
   - Caching or batched circuit evaluation
   - Running only the quantum attention on a subset of layers

2. **Real quantum hardware**  
   The code is written against PennyLane’s device abstraction.  
   To target real hardware, change the device string in `quantum_circuits.make_device`  
   (e.g. `qml.device("qiskit.ibmq", …)` or a cloud provider plugin) and reduce  
   `n_qubits` / `q_layers` according to the available coherence budget.

3. **Noise model**  
   When `--noise > 0` the code attempts to use Qiskit Aer’s density-matrix simulator  
   with a depolarizing channel. If Qiskit is not installed it falls back to a pure simulator.

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{singh2025hqct,
  title={Hybrid Quantum-Classical Attention Mechanisms for Parameter-Efficient Transformers in Natural Language Processing},
  author={Singh, Amisha},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT
```

## Analysis Dashboard

Generate a self-contained HTML dashboard with paper results, ablation plots, and (optionally) your own training run:

```bash
# Paper figures only
python scripts/analyze.py

# Include a completed training run
python scripts/analyze.py --run_dir runs/ag_news_hqct-k_q6

# Generate + open in browser
python scripts/analyze.py --serve
```

The dashboard includes:
- Main classification table (Table 1)
- Accuracy vs parameter count (Fig. 2)
- Few-shot comparison (Table 2)
- Qubit ablation (Fig. 3)
- Noise robustness curve (Fig. 4)
- Architecture summary cards
- Your experiment metrics & training curves (when `--run_dir` is supplied)
# memory_assist.ai
