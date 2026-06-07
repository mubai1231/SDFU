import os
tinyimagenet_model_prefix = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../download_ckpts/tinyimagenet_teachers/"
)
from .VPTResNet import (
    ResNet18,
    ResNet50,
    ResNet101,
    VPTResNet18,
    VPTResNet34,
    VPTResNet50,
    VPTResNet101,
    VPTResNet152,
)
from .VPTViT import (
    ViTTiny,
    ViTBase16,
    VPTViTTiny,
    VPTViTBase16,
)

tinyimagenet_model_dict = {
    "VPTResNet18": (
        VPTResNet18,
        tinyimagenet_model_prefix + "VPTResNet18_distill/ckpt_epoch_100.pth"
    ),
    "VPTResNet34": (VPTResNet34, None),
    "VPTResNet50": (
        VPTResNet50,
        tinyimagenet_model_prefix + "VPTResNet50_vanilla/ckpt_epoch_50.pth"
    ),
    "VPTResNet101": (VPTResNet101, None),
    "VPTResNet152": (VPTResNet152, None),
    "ResNet18": (ResNet18, None),
    "ResNet50": (ResNet50, None),
    "ResNet101": (ResNet101, None),
    "ViTTiny": (ViTTiny, None),
    "ViTBase16": (ViTBase16, None),
    "VPTViTTiny": (VPTViTTiny, None),
    "VPTViTBase16": (VPTViTBase16, None),
}