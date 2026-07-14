from __future__ import annotations

import torch
import timm
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


def build_convnext_tiny(
    num_classes: int,
    in_channels: int = 5,
    pretrained: bool = False,
) -> nn.Module:
    weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = convnext_tiny(weights=weights)

    first_conv = model.features[0][0]
    if first_conv.in_channels != in_channels:
        replacement = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )
        _initialize_first_conv(replacement, first_conv, in_channels)
        model.features[0][0] = replacement

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


class SmallBeachCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int = 5,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_channels, 32),
            nn.MaxPool2d(2),
            _conv_block(32, 64),
            nn.MaxPool2d(2),
            _conv_block(64, 128),
            nn.MaxPool2d(2),
            _conv_block(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_small_cnn(
    num_classes: int,
    in_channels: int = 5,
    dropout: float = 0.25,
) -> nn.Module:
    return SmallBeachCNN(num_classes=num_classes, in_channels=in_channels, dropout=dropout)


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def build_vit_tiny(
    num_classes: int,
    in_channels: int = 5,
    pretrained: bool = True,
    model_name: str = "vit_tiny_patch16_224",
    dropout: float = 0.0,
) -> nn.Module:
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )
    _adapt_timm_patch_embed(model, in_channels)
    return model


def _adapt_timm_patch_embed(model: nn.Module, in_channels: int) -> None:
    projection = model.patch_embed.proj
    if projection.in_channels == in_channels:
        return

    replacement = nn.Conv2d(
        in_channels=in_channels,
        out_channels=projection.out_channels,
        kernel_size=projection.kernel_size,
        stride=projection.stride,
        padding=projection.padding,
        dilation=projection.dilation,
        groups=projection.groups,
        bias=projection.bias is not None,
    )
    _initialize_first_conv(replacement, projection, in_channels)
    model.patch_embed.proj = replacement


def _initialize_first_conv(replacement: nn.Conv2d, original: nn.Conv2d, in_channels: int) -> None:
    with torch.no_grad():
        if original.weight.shape[1] == 3 and in_channels >= 3:
            replacement.weight[:, :3] = original.weight
            if in_channels > 3:
                extra = original.weight.mean(dim=1, keepdim=True)
                replacement.weight[:, 3:] = extra.repeat(1, in_channels - 3, 1, 1)
        else:
            nn.init.kaiming_normal_(replacement.weight, mode="fan_out", nonlinearity="relu")

        if replacement.bias is not None:
            if original.bias is not None:
                replacement.bias.copy_(original.bias)
            else:
                replacement.bias.zero_()
