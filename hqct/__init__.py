"""HQCT – Hybrid Quantum-Classical Transformer."""

__version__ = "0.1.0"

# Lazy imports so lightweight scripts (e.g. analysis dashboard) don't
# force heavy optional dependencies at import time.
def __getattr__(name):
    if name in ("HQCT", "ClassicalMultiHeadAttention", "QuantumKernelAttention", "QuantumProjectionAttention"):
        from .models import HQCT, ClassicalMultiHeadAttention, QuantumKernelAttention, QuantumProjectionAttention
        return {
            "HQCT": HQCT,
            "ClassicalMultiHeadAttention": ClassicalMultiHeadAttention,
            "QuantumKernelAttention": QuantumKernelAttention,
            "QuantumProjectionAttention": QuantumProjectionAttention,
        }[name]
    if name in ("QuantumKernel", "QuantumProjection", "hardware_efficient_ansatz"):
        from .quantum_circuits import QuantumKernel, QuantumProjection, hardware_efficient_ansatz
        return {
            "QuantumKernel": QuantumKernel,
            "QuantumProjection": QuantumProjection,
            "hardware_efficient_ansatz": hardware_efficient_ansatz,
        }[name]
    if name in ("get_dataloaders", "load_benchmark", "make_few_shot"):
        from .data import get_dataloaders, load_benchmark, make_few_shot
        return {
            "get_dataloaders": get_dataloaders,
            "load_benchmark": load_benchmark,
            "make_few_shot": make_few_shot,
        }[name]
    if name in ("train_model", "evaluate"):
        from .train import train_model, evaluate
        return {"train_model": train_model, "evaluate": evaluate}[name]
    if name == "generate_dashboard":
        from .analysis import generate_dashboard
        return generate_dashboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
