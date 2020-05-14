from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision


class DAF(nn.Module):
    def __init__(self, pretrained=False, use_batchnorm=True, freeze_encoder=False):
        super(DAF, self).__init__()
        net = torchvision.models.resnet34(pretrained=pretrained)
        with torch.no_grad():
            pretrained_conv1 = net.conv1.weight.clone()
            # Assign new conv layer with 4 input channels
            net.conv1 = torch.nn.Conv2d(4, 64, 7, 2, 3, bias=False)
            net.conv1.weight[:, :3] = pretrained_conv1
            net.conv1.weight[:, 3] = pretrained_conv1[:, 0]
        self.encoder = net
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)
        self.conv1 = self.encoder.layer1
        self.conv2 = self.encoder.layer2
        self.conv3 = self.encoder.layer3
        self.conv4 = self.encoder.layer4

        self.down4 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )

        self.fuse1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )

        self.attention4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.Softmax2d()
        )
        self.attention3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.Softmax2d()
        )
        self.attention2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.Softmax2d()
        )
        self.attention1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.Softmax2d()
        )

        self.refine4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.refine3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.refine2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.refine1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )

        self.predict4 = nn.Conv2d(64, 1, kernel_size=1)
        self.predict3 = nn.Conv2d(64, 1, kernel_size=1)
        self.predict2 = nn.Conv2d(64, 1, kernel_size=1)
        self.predict1 = nn.Conv2d(64, 1, kernel_size=1)

        self.predict4_2 = nn.Conv2d(64, 1, kernel_size=1)
        self.predict3_2 = nn.Conv2d(64, 1, kernel_size=1)
        self.predict2_2 = nn.Conv2d(64, 1, kernel_size=1)
        self.predict1_2 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        layer0 = self.encoder.conv1(x)
        layer0 = self.encoder.bn1(layer0)
        layer0 = self.encoder.relu(layer0)

        layer1 = self.pool(layer0)
        layer1 = self.conv1(layer1)
        layer2 = self.conv2(layer1)
        layer3 = self.conv3(layer2)
        layer4 = self.conv4(layer3)

        down4 = F.upsample(self.down4(layer4), size=layer1.size()[2:], mode='bilinear')
        down3 = F.upsample(self.down3(layer3), size=layer1.size()[2:], mode='bilinear')
        down2 = F.upsample(self.down2(layer2), size=layer1.size()[2:], mode='bilinear')
        down1 = self.down1(layer1)

        predict4 = self.predict4(down4)
        predict3 = self.predict3(down3)
        predict2 = self.predict2(down2)
        predict1 = self.predict1(down1)

        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))

        attention4 = self.attention4(torch.cat((down4, fuse1), 1))
        attention3 = self.attention3(torch.cat((down3, fuse1), 1))
        attention2 = self.attention2(torch.cat((down2, fuse1), 1))
        attention1 = self.attention1(torch.cat((down1, fuse1), 1))

        refine4 = self.refine4(torch.cat((down4, attention4 * fuse1), 1))
        refine3 = self.refine3(torch.cat((down3, attention3 * fuse1), 1))
        refine2 = self.refine2(torch.cat((down2, attention2 * fuse1), 1))
        refine1 = self.refine1(torch.cat((down1, attention1 * fuse1), 1))

        predict4_2 = self.predict4_2(refine4)
        predict3_2 = self.predict3_2(refine3)
        predict2_2 = self.predict2_2(refine2)
        predict1_2 = self.predict1_2(refine1)

        predict1 = F.upsample(predict1, size=x.size()[2:], mode='bilinear')
        predict2 = F.upsample(predict2, size=x.size()[2:], mode='bilinear')
        predict3 = F.upsample(predict3, size=x.size()[2:], mode='bilinear')
        predict4 = F.upsample(predict4, size=x.size()[2:], mode='bilinear')

        predict1_2 = F.upsample(predict1_2, size=x.size()[2:], mode='bilinear')
        predict2_2 = F.upsample(predict2_2, size=x.size()[2:], mode='bilinear')
        predict3_2 = F.upsample(predict3_2, size=x.size()[2:], mode='bilinear')
        predict4_2 = F.upsample(predict4_2, size=x.size()[2:], mode='bilinear')

        return predict1, predict2, predict3, predict4, predict1_2, predict2_2, predict3_2, predict4_2
