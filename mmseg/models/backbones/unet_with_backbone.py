from typing import Optional
from mmcv.runner import BaseModule
from mmseg.models.builder import BACKBONES

from mmseg.models.cls_backbones.resnet import *
from mmseg.models.backbones.unet import *


@BACKBONES.register_module()
class UnetWithResNet(BaseModule):
    def __init__(
        self, 
        model_name, 
        pretrained=True,
        conv_cfg=None,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
    ):
        
        # self.enc_dims = (3, 64, 128, 256, 512)
        super(UnetWithResNet, self).__init__()
        self.encoder: ResNet = eval(model_name)(pretrained)
        self.enc_dims = self.encoder.dims
        self.decoder = nn.ModuleList()
        
        num_stages = len(self.enc_dims)
        for i in range(1, num_stages):
            decoder = UpConvBlock(
                conv_block=BasicConvBlock,
                in_channels=self.enc_dims[i],
                skip_channels=self.enc_dims[i - 1],
                out_channels=self.enc_dims[i - 1],
                num_convs=2,
                stride=1,
                dilation=1,
                with_cp=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                upsample_cfg=upsample_cfg,
                dcn=None,
                plugins=None
            )
            self.decoder.append(decoder)

    def forward(self, x):
        enc_outs = self.encoder(x)
        x = enc_outs[-1]
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)
        return tuple(dec_outs)
        # return tuple(outs)
        
        