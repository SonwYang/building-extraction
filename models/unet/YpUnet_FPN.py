from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


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
    def __init__(self, in_c, mid_c, out_c):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            SEModule(in_c, reduction=16),
            nn.Conv2d(in_c, mid_c, kernel_size=1),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(),
            nn.Conv2d(mid_c, out_c, kernel_size=1),
            nn.ReLU()
        )
        # todo add seblock!

    def forward(self, x):
        return self.conv(x)


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

        self.conv0 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.conv1 = self.encoder.layer1
        self.conv2 = self.encoder.layer2
        self.conv3 = self.encoder.layer3
        self.conv4 = self.encoder.layer4

        self.center = ASPP(encoder_channels[0], encoder_channels[1])

        self.layer1 = DecoderBlock(in_channels[0], out_channels[2], out_channels[2])
        self.layer2 = DecoderBlock(in_channels[1], out_channels[2], out_channels[2])
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2], out_channels[2])
        self.layer4 = DecoderBlock(in_channels[3], out_channels[2], out_channels[2])
        self.layer5 = DecoderBlock(in_channels[3], out_channels[2], out_channels[2])

        # self.dsv1 = self.dsv_layer(out_channels[0], 1)
        # self.dsv2 = self.dsv_layer(out_channels[1], 2)
        # self.dsv3 = self.dsv_layer(out_channels[2], 4)

        self.final = nn.Sequential(
            nn.Conv2d(64 * 4, 32, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[1],
            encoder_channels[1] + decoder_channels[2],
            encoder_channels[2] + decoder_channels[2],
            encoder_channels[3] + decoder_channels[2],
            encoder_channels[3] + decoder_channels[2],
        ]
        return channels

    def dsv_layer(self, in_channels, out_channels):
        layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        return layer

    def forward(self, x):
        conv0 = self.encoder.conv1(x)
        conv0 = self.encoder.bn1(conv0)
        conv0 = self.encoder.relu(conv0)

        conv1 = self.pool(conv0)
        conv1 = self.conv1(conv1)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        center = self.center(conv4)
        # print(f"conv0:{conv0.size()}")    #[32, 64, 64, 64]
        # print(f"conv1:{conv1.size()}")    #[32, 64, 64, 64]
        # print(f"conv2:{conv2.size()}")    #[32, 128, 32, 32]
        # print(f"conv3:{conv3.size()}")    #[32, 256, 16, 16]
        # print(f"conv4:{conv4.size()}")    #[32, 512, 8, 8]
        x1 = self.layer1(center)
        x2 = self.layer2(torch.cat([x1, conv3], 1))
        x3 = self.layer3(torch.cat([x2, conv2], 1))
        x4 = self.layer4(torch.cat([x3, conv1], 1))
        x5 = self.layer5(torch.cat([x4, conv0], 1))

        # x1 = self.dsv1(x1)
        # x2 = self.dsv2(x2)
        # x3 = self.dsv3(x3)

        size = x5.size()[2:]
        out = torch.cat(
            [
                F.upsample_bilinear(x2, size=size),
                F.upsample_bilinear(x3, size=size),
                F.upsample_bilinear(x4, size=size),
                x5
            ],
            1
        )

        out = self.final(out)

        return out

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[1, 2, 3, 4]):
        super(ASPP, self).__init__()
        self.aspp1 = _ASPPModule(inplanes, mid_c, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, mid_c, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, mid_c, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, mid_c, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(mid_c),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(mid_c * 5, mid_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
