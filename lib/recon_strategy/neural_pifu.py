"""
Neural PIFu Strategy
=====================
Wraps the original HGPIFuNet_orien network pair (occNet + orienNet) from HairStep.
This is a thin adapter layer that makes the existing neural network backend
conform to the BaseReconStrategy interface.
"""

import torch
from .base import BaseReconStrategy
from lib.model.recon3D import HGPIFuNet_orien


class NeuralPIFuStrategy(BaseReconStrategy):
    """
    Strategy that uses two HGPIFuNet_orien networks:
        - occ_net:   predicts occupancy (gen_orien=False, sigmoid output in [0,1])
        - orien_net: predicts 3D orientation vectors (gen_orien=True, raw R^3 output)

    This is the original HairStep reconstruction method (NeuralHDHair*).
    """

    def __init__(self, opt, cuda: torch.device):
        super().__init__(opt, cuda)
        self.occ_net = self._load_net(gen_orien=False, ckpt=opt.checkpoint_hairstep2occ)
        self.orien_net = self._load_net(gen_orien=True, ckpt=opt.checkpoint_hairstep2orien)

        # cache of the most recently queried network (for get_preds / query shim)
        self._active_net = self.occ_net

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_net(self, gen_orien: bool, ckpt: str) -> HGPIFuNet_orien:
        kind = 'orienNet' if gen_orien else 'occNet'
        net = HGPIFuNet_orien(self.opt, gen_orien=gen_orien).to(device=self.cuda)
        print(f'[NeuralPIFuStrategy] Loading {kind} from: {ckpt}')
        net.load_state_dict(torch.load(ckpt, map_location=self.cuda))
        net.eval()
        return net

    # ------------------------------------------------------------------
    # BaseReconStrategy interface
    # ------------------------------------------------------------------

    def filter(self, data: dict) -> None:
        """
        Extract 2D feature maps from the 4-channel hairstep input for both networks.
        Must be called once per sample before any query.
        """
        image_tensor = data['hairstep'].to(device=self.cuda).unsqueeze(0)  # [1, 4, H, W]
        with torch.no_grad():
            self.occ_net.filter(image_tensor)
            self.orien_net.filter(image_tensor)

    def query_occ(self, points: torch.Tensor, calib: torch.Tensor) -> torch.Tensor:
        """
        Query occupancy network.

        Args:
            points: [B, 3, N]  world-space coordinates
            calib:  [B, 4, 4]  camera calibration matrix

        Returns:
            [B, 1, N] occupancy values in [0, 1]
        """
        self._active_net = self.occ_net
        with torch.no_grad():
            self.occ_net.query(points, calib)
        self._last_preds = self.occ_net.get_preds()
        return self._last_preds[0]

    def query_orien(self, points: torch.Tensor, calib: torch.Tensor) -> torch.Tensor:
        """
        Query orientation network.

        Args:
            points: [B, 3, N]  world-space coordinates
            calib:  [B, 4, 4]  camera calibration matrix

        Returns:
            [B, 3, N] 3D orientation vectors
        """
        self._active_net = self.orien_net
        with torch.no_grad():
            self.orien_net.query(points, calib)
        self._last_preds = self.orien_net.get_preds()
        return self._last_preds[0]

    # ------------------------------------------------------------------
    # Legacy shims for lib/mesh_util.py reconstruction() eval_func
    # ------------------------------------------------------------------

    def query(self, points: torch.Tensor, calib: torch.Tensor, **kwargs) -> torch.Tensor:
        """Dispatch to whichever network is currently active (occ or orien)."""
        if self._query_mode == 'occ':
            return self.query_occ(points, calib)
        else:
            return self.query_orien(points, calib)

    def get_preds(self):
        return [self._last_preds]
