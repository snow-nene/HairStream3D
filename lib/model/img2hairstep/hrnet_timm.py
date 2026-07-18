import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, variant='hrnet_w18', pretrained=False, out_channels=2, decoder_channels=128,
                 multi_scale_supervision=True):
        super(Model, self).__init__()
        self.multi_scale_supervision = multi_scale_supervision
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                'timm is required for HRNet backbone. Install with: pip install timm'
            ) from exc

        self.backbone = timm.create_model(variant, pretrained=pretrained, features_only=True)
        feat_channels = self.backbone.feature_info.channels()

        self.head = nn.Sequential(
            nn.Conv2d(sum(feat_channels), decoder_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )

        # Auxiliary heads for multi-scale supervision on each HRNet branch output
        if self.multi_scale_supervision:
            self.aux_heads = nn.ModuleList()
            for ch in feat_channels:
                self.aux_heads.append(nn.Sequential(
                    nn.Conv2d(ch, ch // 2, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(ch // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch // 2, out_channels, kernel_size=1, stride=1, padding=0),
                ))

    def forward(self, x, return_features=False):
        in_h, in_w = x.shape[2], x.shape[3]
        feats = self.backbone(x)

        target_h = max(f.shape[2] for f in feats)
        target_w = max(f.shape[3] for f in feats)
        upsampled = []
        for feat in feats:
            if feat.shape[2] != target_h or feat.shape[3] != target_w:
                feat = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
            upsampled.append(feat)

        fused = torch.cat(upsampled, dim=1)
        pred = self.head(fused)
        if pred.shape[2] != in_h or pred.shape[3] != in_w:
            pred = F.interpolate(pred, size=(in_h, in_w), mode='bilinear', align_corners=False)

        # Multi-scale auxiliary predictions
        aux_preds = None
        if self.multi_scale_supervision:
            aux_preds = []
            for i, (feat, aux_head) in enumerate(zip(feats, self.aux_heads)):
                aux_pred = aux_head(feat)
                if aux_pred.shape[2] != in_h or aux_pred.shape[3] != in_w:
                    aux_pred = F.interpolate(aux_pred, size=(in_h, in_w), mode='bilinear', align_corners=False)
                aux_preds.append(aux_pred)

        if return_features:
            return pred, aux_preds, feats
        return pred, aux_preds
