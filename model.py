import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvMixer(nn.Module):
    def __init__(self, dim, depth,
        kernel_size = 9, patch_size = 7, num_classes = 1000, activation = nn.GELU):
        super(ConvMixer, self).__init__()
        padding_size = (kernel_size - 1) // 2

        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size = patch_size, stride = patch_size),
            activation(),
            nn.BatchNorm2d(dim)
        )

        self.blocks = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # Depthwise convolution
                    nn.Conv2d(dim, dim, kernel_size, groups = dim, padding = padding_size),
                    activation(),
                    nn.BatchNorm2d(dim)
                )),
                # Pointwise convolution
                nn.Conv2d(dim, dim, kernel_size = 1),
                activation(),
                nn.BatchNorm2d(dim)
            ) for _ in range(depth)]
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, num_classes)
        )


    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


def ConvMixer1536_20(num_classes = 1000):
    return ConvMixer(dim = 1536, depth = 20, kernel_size = 9, patch_size = 7, num_classes = num_classes)

def ConvMixer1536_20_p14_k3(num_classes = 1000):
    return ConvMixer(dim = 1536, depth = 20, kernel_size = 3, patch_size = 14, num_classes = num_classes)

def ConvMixer768_32(num_classes = 1000):
    return ConvMixer(dim = 768, depth = 32, kernel_size = 7, patch_size = 7,  num_classes = num_classes)

def ConvMixer768_32_p14_k3(num_classes = 1000):
    return ConvMixer(dim = 768, depth = 32, kernel_size = 3, patch_size = 14,  num_classes = num_classes)


if __name__ == "__main__":
    net = ConvMixer768_32_p14_k3()
    img = torch.randn(1, 3, 224, 224)

    print(sum(p.numel() for p in net.parameters()))
    assert net(img).shape == (1, 1000)