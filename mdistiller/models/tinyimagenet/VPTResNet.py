import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ["ResNet", "VPTResNet18", "VPTResNet34", "VPTResNet50", "VPTResNet101", "VPTResNet152"]

model_urls = {
    "VPTResNet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "VPTResNet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "VPTResNet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "VPTResNet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "VPTResNet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class PromptBlock(nn.Module):
    def __init__(self, in_channels):
        super(PromptBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=5, groups=in_channels, padding=2)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.depthwise_conv(out)
        out = self.conv2(out)
        return x + out * self.se(out)


def add_prompt_blocks(model, num_stages=3):
    for stage in range(num_stages):
        layer = getattr(model, f'layer{stage + 1}')
        last_sub_layer = layer[-1]
        out_channels = last_sub_layer.conv1.out_channels * last_sub_layer.expansion

        prompt_block = PromptBlock(out_channels)

        layer[-1] = nn.Sequential(last_sub_layer, prompt_block)

    return model


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = F.relu(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = F.relu(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            print("ResNet unknown block error !!!")

        return [bn2, bn3, bn4]

    def get_stage_channels(self):
        return [256, 512, 1024, 2048]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        stem = x
        x = self.relu(x)
        x = self.maxpool(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        x = self.avgpool(F.relu(feat4))
        x = x.view(x.size(0), -1)
        avg = x
        out = self.fc(x)

        feats = {}
        feats["pooled_feat"] = avg
        feats["feats"] = [
            F.relu(stem),
            F.relu(feat1),
            F.relu(feat2),
            F.relu(feat3),
            F.relu(feat4),
        ]
        feats["preact_feats"] = [stem, feat1, feat2, feat3, feat4]

        return out, feats


def ResNet18(pretrained=True, num_classes=100, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["VPTResNet18"]))
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def VPTResNet18(pretrained=True, num_classes=100, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["VPTResNet18"]))
    for param in model.parameters():
        param.requires_grad = False
    model = add_prompt_blocks(model, num_stages=3)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def VPTResNet34(pretrained=True, num_classes=100, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["VPTResNet34"]))
    for param in model.parameters():
        param.requires_grad = False
    model = add_prompt_blocks(model, num_stages=3)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def ResNet50(pretrained=True, num_classes=100, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["VPTResNet50"]))
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def VPTResNet50(pretrained=True, num_classes=100, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["VPTResNet50"]))
    for param in model.parameters():
        param.requires_grad = False
    model = add_prompt_blocks(model, num_stages=3)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def ResNet101(pretrained=True, num_classes=100, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["VPTResNet101"]))
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def VPTResNet101(pretrained=True, num_classes=100, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["VPTResNet101"]))
    for param in model.parameters():
        param.requires_grad = False
    model = add_prompt_blocks(model, num_stages=3)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def VPTResNet152(pretrained=True, num_classes=100, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["VPTResNet152"]))
    for param in model.parameters():
        param.requires_grad = False
    model = add_prompt_blocks(model, num_stages=3)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
