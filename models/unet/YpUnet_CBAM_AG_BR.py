from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision


class CBAM_Module(nn.Module):
    def __init__(self, channels, reduction,attention_kernel_size=3):
        super(CBAM_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        k=2
        self.conv_after_concat = nn.Conv2d(k, 1,
                                           kernel_size = attention_kernel_size,
                                           stride=1,
                                           padding = attention_kernel_size//2)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        # Spatial attention module
        x = module_input * x
        module_input = x
        b, c, h, w = x.size()
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x


class _BoundaryRefineModule(nn.Module):
    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.BR = _BoundaryRefineModule(F_int)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        x = x * psi
        x = self.BR(x)
        return x


class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, **batchnorm_params):

        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=not (use_batchnorm)),
            nn.ReLU(inplace=True),
        ]

        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_batchnorm=True,
                 attention_kernel_size=3,
                 reduction=16):
        super().__init__()
        self.BR = _BoundaryRefineModule(in_channels)
        self.channel_gate = CBAM_Module(in_channels, reduction, attention_kernel_size)
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        x, skip = x
        # x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.channel_gate(x)
        x = self.block(x)
        return x


class UpSample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return x


class UNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder

        """
    def __init__(self, num_classes=1,  pretrained=False, use_batchnorm=True, freeze_encoder=False):
        """
        :param num_classes:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)

        net = torchvision.models.resnet34(pretrained=pretrained)
        with torch.no_grad():
            pretrained_conv1 = net.conv1.weight.clone()
            # Assign new conv layer with 4 input channels
            net.conv1 = torch.nn.Conv2d(4, 64, 7, 2, 3, bias=False)
            net.conv1.weight[:, :3] = pretrained_conv1
            net.conv1.weight[:, 3] = pretrained_conv1[:, 0]
        self.encoder = net

        decoder_channels = (256, 128, 64, 32, 16)
        encoder_channels = (512, 256, 128, 64, 64)
        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        for layer in self.encoder.parameters():
            layer.requires_grad = not freeze_encoder

        self.relu = nn.ReLU(inplace=True)
        self.up = UpSample()

        self.conv0 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.conv1 = self.encoder.layer1
        self.conv2 = self.encoder.layer2
        self.conv3 = self.encoder.layer3
        self.conv4 = self.encoder.layer4

        self.Att1 = Attention_block(F_g=encoder_channels[0], F_l=encoder_channels[1], F_int=encoder_channels[1])
        self.Att2 = Attention_block(F_g=decoder_channels[0], F_l=encoder_channels[2], F_int=encoder_channels[2])
        self.Att3 = Attention_block(F_g=decoder_channels[1], F_l=encoder_channels[3], F_int=encoder_channels[3])
        self.Att4 = Attention_block(F_g=decoder_channels[2], F_l=encoder_channels[4], F_int=encoder_channels[4])

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm)
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm)
        self.final = nn.Sequential(
            _BoundaryRefineModule(out_channels[4]),
            nn.Conv2d(out_channels[4], num_classes, kernel_size=1)
        )

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        conv0 = self.encoder.conv1(x)
        conv0 = self.encoder.bn1(conv0)
        conv0 = self.encoder.relu(conv0)

        conv1 = self.pool(conv0)
        conv1 = self.conv1(conv1)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        # print(f"conv0:{conv0.size()}")    #[32, 64, 64, 64]
        # print(f"conv1:{conv1.size()}")    #[32, 64, 64, 64]
        # print(f"conv2:{conv2.size()}")    #[32, 128, 32, 32]
        # print(f"conv3:{conv3.size()}")    #[32, 256, 16, 16]
        # print(f"conv4:{conv4.size()}")    #[32, 512, 8, 8]
        conv4 = self.up(conv4)
        conv3 = self.Att1(g=conv4, x=conv3)
        x = self.layer1([conv4, conv3])

        x = self.up(x)
        conv2 = self.Att2(g=x, x=conv2)
        x = self.layer2([x, conv2])

        x = self.up(x)
        conv1 = self.Att3(g=x, x=conv1)
        x = self.layer3([x, conv1])

        x = self.up(x)
        conv0 = self.Att4(g=x, x=conv0)
        x = self.layer4([x, conv0])

        x = self.up(x)
        x = self.layer5([x, None])
        x = self.final(x)

        return x
