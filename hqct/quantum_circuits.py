"""
Quantum circuit definitions for Hybrid Quantum-Classical Transformer (HQCT).

Implements the hardware-efficient ansatz used in both HQCT-P and HQCT-K:
- Angle embedding (Rx / Ry)
- L layers of single-qubit rotations (Rx, Ry, Rz) + nearest-neighbour CNOTs
"""

from typing import Optional, Tuple

import pennylane as qml
from pennylane import numpy as pnp
import torch
import torch.nn as nn


def make_device(n_qubits: int, shots: Optional[int] = None, noise: float = 0.0):
    """
    Create a PennyLane device.
    - noise == 0.0 → pure statevector (Lightning)
    - noise  > 0.0 → density-matrix simulator with depolarizing noise
    """
    if noise > 0.0:
        # Qiskit Aer density-matrix backend with depolarizing noise
        try:
            from qiskit_aer.noise import NoiseModel, depolarizing_error
            noise_model = NoiseModel()
            error = depolarizing_error(noise, 1)
            noise_model.add_all_qubit_quantum_error(error, ["rx", "ry", "rz"])
            error2 = depolarizing_error(noise, 2)
            noise_model.add_all_qubit_quantum_error(error2, ["cx"])
            dev = qml.device(
                "qiskit.aer",
                wires=n_qubits,
                backend="aer_simulator_statevector",
                noise_model=noise_model,
                shots=shots,
            )
        except Exception:
            # Fallback pure simulator if Qiskit not available
            dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
    else:
        # Fast pure-state simulator
        try:
            dev = qml.device("lightning.qubit", wires=n_qubits, shots=shots)
        except Exception:
            dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
    return dev


def hardware_efficient_ansatz(features, weights, n_qubits: int, n_layers: int):
    """
    Hardware-efficient ansatz used throughout the paper.

    Encoding: angle embedding with Rx and Ry on each qubit.
    Ansatz:   L layers of (Rx, Ry, Rz) on every qubit + linear CNOT chain.
    """
    # Angle embedding – features shape: (n_qubits,)
    for i in range(n_qubits):
        qml.RX(features[i], wires=i)
        qml.RY(features[i], wires=i)

    # Variational layers
    for layer in range(n_layers):
        # Single-qubit rotations
        for i in range(n_qubits):
            qml.RX(weights[layer, i, 0], wires=i)
            qml.RY(weights[layer, i, 1], wires=i)
            qml.RZ(weights[layer, i, 2], wires=i)
        # Nearest-neighbour entanglement
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])


class QuantumKernel(nn.Module):
    """
    Quantum kernel κ(q, k) = |⟨ψ(q)|ψ(k)⟩|²

    Used by HQCT-K. The circuit is evaluated for every query-key pair
    (or batched via broadcasting on the classical simulator).
    """

    def __init__(
        self,
        n_qubits: int = 6,
        n_layers: int = 3,
        noise: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.noise = noise

        # Trainable variational parameters: (n_layers, n_qubits, 3)
        # Initialised near identity (narrow Gaussian around 0) as recommended
        init = 0.01 * torch.randn(n_layers, n_qubits, 3)
        self.weights = nn.Parameter(init)

        # Classical linear map from model_dim → n_qubits (done outside)
        # We only keep the quantum part here.

        self._dev = make_device(n_qubits, noise=noise)
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        """Return a QNode that computes fidelity between two feature vectors."""

        @qml.qnode(self._dev, interface="torch", diff_method="parameter-shift")
        def circuit(features_q, features_k, weights):
            # Encode query
            hardware_efficient_ansatz(features_q, weights, self.n_qubits, self.n_layers)
            # Inverse encode key (for fidelity via state overlap)
            # More efficient: prepare |ψ(q)> then apply U(k)† and measure |0…0>
            qml.adjoint(hardware_efficient_ansatz)(
                features_k, weights, self.n_qubits, self.n_layers
            )
            # Probability of all-zero state = |⟨ψ(q)|ψ(k)⟩|²
            return qml.probs(wires=range(self.n_qubits))

        return circuit

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Compute quantum kernel matrix.

        Args:
            q: (batch, n_heads, seq_len, n_qubits) or (seq_len, n_qubits)
            k: same shape as q

        Returns:
            kappa: (batch, n_heads, seq_len, seq_len) kernel matrix
        """
        # Flatten batch & heads for simpler looping (simulator is already slow)
        original_shape = q.shape
        if q.dim() == 4:
            B, H, N, D = q.shape
            q = q.reshape(B * H, N, D)
            k = k.reshape(B * H, N, D)
            flat = True
        else:
            flat = False
            N = q.shape[0]

        # We compute pairwise fidelities. For speed we use a simple double loop
        # on the classical simulator; on real hardware one would use swap-test
        # or parallelised circuits.
        device = q.device
        dtype = q.dtype
        weights = self.weights

        def single_fidelity(fq, fk):
            # fq, fk: (n_qubits,)
            probs = self._circuit(fq, fk, weights)
            return probs[0]  # P(|0…0>)

        # Vectorised over the leading dimension as much as possible
        batch_size = q.shape[0]
        kappa = torch.zeros(batch_size, N, N, device=device, dtype=dtype)

        for b in range(batch_size):
            for i in range(N):
                for j in range(N):
                    kappa[b, i, j] = single_fidelity(q[b, i], k[b, j])

        if flat:
            kappa = kappa.reshape(B, H, N, N)
        return kappa


class QuantumProjection(nn.Module):
    """
    Quantum Projection used by HQCT-P.

    Maps a classical vector → quantum state → vector of Pauli-Z expectations
    (optionally + a few correlators). This replaces the classical W_Q / W_K.
    """

    def __init__(
        self,
        in_dim: int,
        n_qubits: int = 6,
        n_layers: int = 3,
        noise: float = 0.0,
        include_correlators: bool = False,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.include_correlators = include_correlators

        # Classical projection to n_qubits features
        self.pre_map = nn.Linear(in_dim, n_qubits)

        # Variational weights
        init = 0.01 * torch.randn(n_layers, n_qubits, 3)
        self.weights = nn.Parameter(init)

        self._dev = make_device(n_qubits, noise=noise)
        self._circuit = self._build_circuit()

        out_dim = n_qubits
        if include_correlators:
            out_dim += n_qubits - 1  # nearest-neighbour ZZ
        self.out_dim = out_dim

    def _build_circuit(self):
        @qml.qnode(self._dev, interface="torch", diff_method="parameter-shift")
        def circuit(features, weights):
            hardware_efficient_ansatz(features, weights, self.n_qubits, self.n_layers)
            # Pauli-Z expectations
            expvals = [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            if self.include_correlators:
                for i in range(self.n_qubits - 1):
                    expvals.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(i + 1)))
            return expvals

        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., in_dim)
        Returns:
            (..., out_dim) quantum features
        """
        features = self.pre_map(x)  # (..., n_qubits)
        orig_shape = features.shape[:-1]
        features = features.reshape(-1, self.n_qubits)

        outs = []
        for i in range(features.shape[0]):
            o = self._circuit(features[i], self.weights)
            outs.append(torch.stack(o))
        out = torch.stack(outs, dim=0)
        return out.reshape(*orig_shape, self.out_dim)
