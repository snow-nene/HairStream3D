"""
Abstract base class for 3D hair reconstruction strategies.

All strategies must implement:
    - filter(data): Pre-compute any feature extraction from the 2D hairstep input
    - query_occ(points, calib): Query occupancy field at 3D points -> [B, 1, N] in [0, 1]
    - query_orien(points, calib): Query 3D orientation field at 3D points -> [B, 3, N]

The upstream utilities (gen_mesh_real, hair_synthesis, etc.) are fully strategy-agnostic
as long as the strategy exposes these two query methods.
"""

from abc import ABC, abstractmethod
import torch


class BaseReconStrategy(ABC):
    """
    Abstract strategy for 3D hair reconstruction from HairStep 2D representation.

    Input hairstep tensor:
        4-channel [B, 4, H, W]:
            channels 0-2: strand_map (2D orientation field RGB, normalized to [-1, 1])
            channel  3:   depth_map  (1-channel depth, background padded with -3.0)

    Calib tensor:
        [B, 4, 4] orthographic projection matrix:
            maps 3D world coordinates -> normalized 2D image coordinates

    Occupancy query output:
        [B, 1, N] float tensor in [0, 1]
        0 = outside hair volume, 1 = inside hair volume
        Marching Cubes will extract iso-surface at value 0.5

    Orientation query output:
        [B, 3, N] float tensor, unnormalized 3D direction vector
        Points in the direction a hair strand grows from root to tip
    """

    def __init__(self, opt, cuda: torch.device):
        """
        Args:
            opt:   BaseOptions namespace with all configuration flags
            cuda:  torch.device (e.g. torch.device('cuda:0'))
        """
        self.opt = opt
        self.cuda = cuda

    @abstractmethod
    def filter(self, data: dict) -> None:
        """
        Pre-compute any representation derived from the hairstep 2D input.
        Must be called once per sample before calling query_occ / query_orien.

        Args:
            data: dict with keys:
                'hairstep': [4, H, W] float tensor (single sample, no batch dim)
                'calib':    [4, 4] float tensor
        """
        raise NotImplementedError

    @abstractmethod
    def query_occ(self, points: torch.Tensor, calib: torch.Tensor) -> torch.Tensor:
        """
        Query occupancy (SDF) value at a set of 3D world-space points.

        Args:
            points: [B, 3, N] world-space 3D coordinates
            calib:  [B, 4, 4] camera calibration matrix

        Returns:
            [B, 1, N] occupancy values in [0, 1]
        """
        raise NotImplementedError

    @abstractmethod
    def query_orien(self, points: torch.Tensor, calib: torch.Tensor) -> torch.Tensor:
        """
        Query 3D hair orientation vector at a set of 3D world-space points.

        Args:
            points: [B, 3, N] world-space 3D coordinates
            calib:  [B, 4, 4] camera calibration matrix

        Returns:
            [B, 3, N] 3D orientation (tangent) vectors
        """
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # Adapter methods that match the interface expected by legacy utilities
    # (lib/mesh_util.py gen_mesh_real, lib/hair_util.py export_hair_real)
    # -----------------------------------------------------------------------

    def get_preds(self):
        """
        Compatibility shim for reconstruction() in lib/mesh_util.py.
        Returns the most recently computed query result.
        """
        return [self._last_preds]

    def query(self, points: torch.Tensor, calib: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Generic query dispatcher used inside reconstruction() eval_func closure.
        Subclasses select occ vs orien mode via self._query_mode.

        Returns predictions and stores in self._last_preds.
        """
        if self._query_mode == 'occ':
            self._last_preds = self.query_occ(points, calib)
        else:
            self._last_preds = self.query_orien(points, calib)
        return self._last_preds

    def set_query_mode(self, mode: str):
        """Switch between 'occ' and 'orien' query modes."""
        assert mode in ('occ', 'orien'), f"mode must be 'occ' or 'orien', got {mode}"
        self._query_mode = mode

    _query_mode: str = 'occ'
    _last_preds: torch.Tensor = None
