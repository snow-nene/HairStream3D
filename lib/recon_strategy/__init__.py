"""
Hair 3D Reconstruction Strategy Package
========================================
Provides interchangeable reconstruction backends via Strategy Pattern.

Available strategies:
    - NeuralPIFuStrategy: HGPIFuNet_orien based occupancy + orientation fields (original HairStep)
    - LaplacePDEStrategy: Training-free Laplace PDE physical field solver

Usage:
    from lib.recon_strategy import get_strategy

    strategy = get_strategy('neural', opt, cuda)
    strategy.filter(data)
    occ = strategy.query_occ(points, calib)     # [B, 1, N]
    orien = strategy.query_orien(points, calib)  # [B, 3, N]
"""

from .base import BaseReconStrategy
from .neural_pifu import NeuralPIFuStrategy
from .laplace_pde import LaplacePDEStrategy

STRATEGY_REGISTRY = {
    'neural': NeuralPIFuStrategy,
    'laplace': LaplacePDEStrategy,
}


def get_strategy(name: str, opt, cuda) -> BaseReconStrategy:
    """
    Factory function to instantiate a reconstruction strategy by name.

    Args:
        name:  'neural' or 'laplace'
        opt:   BaseOptions namespace
        cuda:  torch.device

    Returns:
        Instantiated BaseReconStrategy subclass
    """
    if name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[name](opt, cuda)


__all__ = [
    'BaseReconStrategy',
    'NeuralPIFuStrategy',
    'LaplacePDEStrategy',
    'get_strategy',
]
