from mmseg.models import build_backbone
import torch

cfg = dict(
        type='UnetWithResNet',
        model_name="resnet18",
        # model_name="resnet152",
        # pretrained=True,
        pretrained=False,
        # pretrained=r"D:\pretrained_model\torchvision\resnet18-5c106cde.pth"
    )

backbone = build_backbone(cfg)

if __name__ == "__main__":
    x = torch.rand((1, 3, 224, 224))
    outs = backbone(x)
    for i, out in enumerate(outs):
        print(i, out.shape)