import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from safetensors.torch import load_file as load_safetensors

__all__ = ["ViTTiny", "ViTBase16", "VPTViTTiny", "VPTViTBase16"]

_PTH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pth")


class ViTPromptBlock(nn.Module):

    def __init__(self, dim, bottleneck_dim=None):
        super(ViTPromptBlock, self).__init__()
        if bottleneck_dim is None:
            bottleneck_dim = dim // 4
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, dim)
        self.se = nn.Sequential(
            nn.Linear(dim, dim // 16),
            nn.ReLU(),
            nn.Linear(dim // 16, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        out = self.norm(x)
        out = self.fc1(out)
        out = F.gelu(out)
        out = self.fc2(out)
        gate = self.se(x.mean(dim=1))
        return residual + out * gate.unsqueeze(1)


class ViTWrapper(nn.Module):

    def __init__(self, vit_model, num_classes):
        super(ViTWrapper, self).__init__()
        self.vit = vit_model
        self.vit.head = nn.Linear(vit_model.head.in_features, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.vit.patch_embed(x)
        x = torch.cat((self.vit.cls_token.expand(B, -1, -1), x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)

        for block in self.vit.blocks:
            x = block(x)

        x = self.vit.norm(x)
        pooled = x[:, 0]
        out = self.vit.head(pooled)

        feats = {
            "pooled_feat": pooled,
            "feats": [pooled],
        }
        return out, feats


class VPTViT(nn.Module):

    def __init__(self, vit_model, num_classes, prompt_indices=(3, 6, 9)):
        super(VPTViT, self).__init__()
        self.vit = vit_model
        embed_dim = vit_model.embed_dim

        self.vit.head = nn.Linear(vit_model.head.in_features, num_classes)

        self.prompt_indices = prompt_indices
        self.prompt_blocks = nn.ModuleList([
            ViTPromptBlock(embed_dim) for _ in prompt_indices
        ])

    def forward(self, x):
        B = x.shape[0]
        x = self.vit.patch_embed(x)
        x = torch.cat((self.vit.cls_token.expand(B, -1, -1), x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)

        feats_list = []
        prompt_idx = 0

        for i, block in enumerate(self.vit.blocks):
            x = block(x)
            if i in self.prompt_indices:
                x = self.prompt_blocks[prompt_idx](x)
                prompt_idx += 1
                feats_list.append(x[:, 0])

        x = self.vit.norm(x)
        pooled = x[:, 0]
        out = self.vit.head(pooled)

        feats = {
            "pooled_feat": pooled,
            "feats": feats_list,
        }
        return out, feats


HF_MIRROR = "https://hf-mirror.com"


def _load_pretrained(model, model_name, pretrained_path, use_mirror=True):
    if pretrained_path and os.path.exists(pretrained_path):
        if pretrained_path.endswith(".safetensors"):
            state_dict = load_safetensors(pretrained_path)
        else:
            state_dict = torch.load(pretrained_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")
        return

    if use_mirror:
        os.environ.setdefault("HF_ENDPOINT", HF_MIRROR)

    try:
        pretrained_model = timm.create_model(model_name, pretrained=True)
        model.load_state_dict(pretrained_model.state_dict(), strict=False)
        print(f"Downloaded pretrained weights for {model_name}")
    except Exception as e:
        print(f"Warning: could not download pretrained weights for {model_name}: {e}")
        print(f"Training from scratch.")


def ViTTiny(pretrained_path=None, num_classes=200, **kwargs):
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    if pretrained_path is None:
        pretrained_path = os.path.join(_PTH_DIR, "vit_tiny.safetensors")
    _load_pretrained(model, "vit_tiny_patch16_224", pretrained_path)
    for param in model.parameters():
        param.requires_grad = False
    return ViTWrapper(model, num_classes)


def ViTBase16(pretrained_path=None, num_classes=200, **kwargs):
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    if pretrained_path is None:
        pretrained_path = os.path.join(_PTH_DIR, "vit_base.safetensors")
    _load_pretrained(model, "vit_base_patch16_224", pretrained_path)
    for param in model.parameters():
        param.requires_grad = False
    return ViTWrapper(model, num_classes)


def VPTViTTiny(pretrained_path=None, num_classes=200, **kwargs):
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
    if pretrained_path is None:
        pretrained_path = os.path.join(_PTH_DIR, "vit_tiny.safetensors")
    _load_pretrained(model, "vit_tiny_patch16_224", pretrained_path)
    for param in model.parameters():
        param.requires_grad = False
    return VPTViT(model, num_classes)


def VPTViTBase16(pretrained_path=None, num_classes=200, **kwargs):
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    if pretrained_path is None:
        pretrained_path = os.path.join(_PTH_DIR, "vit_base.safetensors")
    _load_pretrained(model, "vit_base_patch16_224", pretrained_path)
    for param in model.parameters():
        param.requires_grad = False
    return VPTViT(model, num_classes)
