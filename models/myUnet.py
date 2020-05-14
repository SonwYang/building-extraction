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


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x


def expansion_layer(in_channels, out_channels, is_upsample):
    if is_upsample:
        e_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode = 'bilinear'))
    else:
        e_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2,
                               padding=1),
            nn.ReLU(inplace=True)
        )
    return e_layer


def contraction_layer(in_channels, out_channels, kernel_size=3):
    c_layer = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return c_layer


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super(AttentionBlock, self).__init__()
        self.theta = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=2, stride=2,
                               bias=False)
        self.phi = nn.Conv2d(in_channels=gating_channels, out_channels=inter_channels, kernel_size=1, stride=1,
                             bias=True)
        self.psi = nn.Conv2d(in_channels=inter_channels, out_channels=1, kernel_size=1, stride=1, bias=True)
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels))

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode='bilinear')
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = F.sigmoid(self.psi(f))

        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode='bilinear')
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y


class UNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder

        """
    def __init__(self, num_classes=1,  pretrained=False, is_upsample=True, freeze_encoder=False):
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

        channels = (512, 256, 128, 64, 32, 1024)

        #attention
        self.attention0 = AttentionBlock(in_channels=channels[3], gating_channels=channels[3], inter_channels=channels[3])
        self.attention1 = AttentionBlock(in_channels=channels[3], gating_channels=channels[2], inter_channels=channels[3])
        self.attention2 = AttentionBlock(in_channels=channels[2], gating_channels=channels[1], inter_channels=channels[2])
        self.attention3 = AttentionBlock(in_channels=channels[1], gating_channels=channels[0], inter_channels=channels[1])
        self.attention4 = AttentionBlock(in_channels=channels[0], gating_channels=channels[5], inter_channels=channels[0])

        self.center = contraction_layer(channels[0], channels[5])
        self.center_up = expansion_layer(channels[5], channels[5], is_upsample=is_upsample)
        self.center_cov = contraction_layer(channels[5], channels[0])

        #decoder
        self.dec4_up = expansion_layer(channels[5], channels[0], is_upsample=is_upsample)
        self.dec4_cov = contraction_layer(channels[0], channels[1])

        self.dec3_up = expansion_layer(channels[0], channels[1], is_upsample=is_upsample)
        self.dec3_cov = contraction_layer(channels[1], channels[2])

        self.dec2_up = expansion_layer(channels[1], channels[2], is_upsample=is_upsample)
        self.dec2_cov = contraction_layer(channels[2], channels[3])

        self.dec1_up = expansion_layer(channels[2], channels[3], is_upsample=is_upsample)
        self.dec1_cov = contraction_layer(channels[3], channels[3])

        self.dec0_up = expansion_layer(channels[2], channels[3], is_upsample=is_upsample)
        self.dec0_cov = contraction_layer(channels[3], channels[4])

        self.final = nn.Sequential(
            nn.Conv2d(channels[4], num_classes, kernel_size=1)
        )

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
        center = self.center(self.pool(conv4))
        center = self.center_up(center)
        conv4 = self.attention4(conv4, center)
        center = self.center_cov(center)

        dec4 = self.dec4_up(torch.cat([conv4, center], dim=1))
        conv3 = self.attention3(conv3, dec4)
        dec4 = self.dec4_cov(dec4)

        dec3 = self.dec3_up(torch.cat([dec4, conv3], dim=1))
        conv2 = self.attention2(conv2, dec3)
        dec3 = self.dec3_cov(dec3)

        dec2 = self.dec2_up(torch.cat([dec3, conv2], dim=1))
        conv1 = self.attention1(conv1, dec2)
        dec2 = self.dec2_cov(dec2)

        dec1 = self.dec1_up(torch.cat([dec2, conv1], dim=1))
        conv0 = self.attention0(conv0, dec1)
        dec1 = self.dec1_cov(dec1)

        dec0 = self.dec0_up(torch.cat([conv0, dec1], dim=1))
        dec0 = self.dec0_cov(dec0)
        out = self.final(dec0)
        return out
