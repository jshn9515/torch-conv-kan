import torch
import torch.nn as nn

from kan_convs import KALNConv2DLayer


class SimpleConvKALN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            degree: int = 3,
            groups: int = 1):
        super(SimpleConvKALN, self).__init__()

        self.layers = nn.Sequential(
            KALNConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, degree=degree, groups=1, padding=1, stride=1,
                            dilation=1),
            KALNConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, degree=degree, groups=groups, padding=1,
                            stride=2, dilation=1),
            KALNConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, degree=degree, groups=groups, padding=1,
                            stride=2, dilation=1),
            KALNConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, degree=degree, groups=groups, padding=1,
                            stride=1, dilation=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.output = nn.Linear(layer_sizes[3], num_classes)

        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.output(x)
        return x


class EightSimpleConvKALN(nn.Module):
    def __init__(
            self,
            layer_sizes,
            num_classes: int = 10,
            input_channels: int = 1,
            degree: int = 3,
            groups: int = 1):
        super(EightSimpleConvKALN, self).__init__()

        self.layers = nn.Sequential(
            KALNConv2DLayer(input_channels, layer_sizes[0], kernel_size=3, degree=degree, groups=1, padding=1, stride=1,
                            dilation=1),
            KALNConv2DLayer(layer_sizes[0], layer_sizes[1], kernel_size=3, degree=degree, groups=groups, padding=1,
                            stride=2, dilation=1),
            KALNConv2DLayer(layer_sizes[1], layer_sizes[2], kernel_size=3, degree=degree, groups=groups, padding=1,
                            stride=2, dilation=1),
            KALNConv2DLayer(layer_sizes[2], layer_sizes[3], kernel_size=3, degree=degree, groups=groups, padding=1,
                            stride=1, dilation=1),
            KALNConv2DLayer(layer_sizes[3], layer_sizes[4], kernel_size=3, degree=degree, groups=groups, padding=1,
                            stride=1, dilation=1),
            KALNConv2DLayer(layer_sizes[4], layer_sizes[5], kernel_size=3, degree=degree, groups=groups, padding=1,
                            stride=2, dilation=1),
            KALNConv2DLayer(layer_sizes[5], layer_sizes[6], kernel_size=3, degree=degree, groups=groups, padding=1,
                            stride=1, dilation=1),
            KALNConv2DLayer(layer_sizes[6], layer_sizes[7], kernel_size=3, degree=degree, groups=groups, padding=1,
                            stride=1, dilation=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.output = nn.Linear(layer_sizes[7], num_classes)

        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.output(x)
        return x