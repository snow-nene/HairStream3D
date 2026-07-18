from .UNet import Model as UNetModel
from .hrnet_timm import Model as HRNetModel


def create_img2strand_model(opt):
    backbone = getattr(opt, 'img2strand_backbone', 'unet').lower()
    if backbone == 'unet':
        return UNetModel()

    if backbone == 'hrnet':
        return HRNetModel(
            variant=getattr(opt, 'hrnet_variant', 'hrnet_w18'),
            pretrained=getattr(opt, 'hrnet_pretrained', False),
            out_channels=2,
            decoder_channels=getattr(opt, 'hrnet_decoder_channels', 128),
            multi_scale_supervision=getattr(opt, 'multi_scale_supervision', True),
        )

    raise ValueError('Unsupported img2strand_backbone: {}'.format(backbone))
