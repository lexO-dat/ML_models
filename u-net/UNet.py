import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

def checkpoint_sequential(module, x):
    return cp.checkpoint(module, x)

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels,
                          out_channels,
                          kernel_size=3,
                          padding=1,
                          bias=False
                ),
                nn.ReLU(inplace=True)
            )

        def Ct(input_channels, output_channels):
          return nn.ConvTranspose2d(
              input_channels,
              output_channels,
              kernel_size=2,
              stride=2
          )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.conv1 = double_conv(3, 64)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = double_conv(64, 128)
        # self.pool2 = nn.MaxPool2d(2)

        self.conv3 = double_conv(128, 256)
        # self.pool3 = nn.MaxPool2d(2)

        self.conv4 = double_conv(256, 512)
        # self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.conv5 = double_conv(512, 1024)


        # Decoder
        self.up6 = Ct(1024, 512)
        self.conv6 = double_conv(1024, 512)

        self.up7 = Ct(512, 256)
        self.conv7 = double_conv(512, 256)

        self.up8 = Ct(256, 128)
        self.conv8 = double_conv(256, 128)

        self.up9 = Ct(128, 64)
        self.conv9 = double_conv(128, 64)

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def crop_image(self, tensor, target_tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        return tensor[:, :, delta:tensor_size-delta, delta:tensor_size-delta]

    def forward(self, image):
        # Store original size
        # original_size = x.size()[2:]

        # Encoder
        c1 = self.conv1(image)
        # c1 = checkppoint_sequential(self.conv1, p1)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        # c2 = checkpoint_sequential(self.conv2, p1)
        p2 = self.pool(c2)

        c3 = self.conv3(p2) # Corrected from self.conv2 to self.conv3
        # c3 = checkpoint_sequential(self.conv3, p2)
        p3 = self.pool(c3)

        c4 = self.conv4(p3) # Corrected from self.conv2 to self.conv4
        # c4 = checkpoint_sequential(self.conv4, p3)
        p4 = self.pool(c4)

        # Bottleneck
        c5 = self.conv5(p4)
        # c5 = checkpoint_sequential(self.conv5, p4)

        # Decoder
        up_6 = self.up6(c5)
        crop1 = self.crop_image(c4, up_6)
        c6 = self.conv6(torch.cat([up_6, crop1], dim=1))

        up_7 = self.up7(c6)
        crop2 = self.crop_image(c3, up_7)
        c7 = self.conv7(torch.cat([up_7, crop2], dim=1)) # Corrected from up_6, crop1 to up_7, crop2

        up_8 = self.up8(c7)
        crop3 = self.crop_image(c2, up_8)
        c8 = self.conv8(torch.cat([up_8, crop3], dim=1))

        up_9 = self.up9(c8)
        crop4 = self.crop_image(c1, up_9)
        c9 = self.conv9(torch.cat([up_9, crop4], dim=1))

        ''' up_7 = self.up7(c6)
        if up_7.size()[2:] != c3.size()[2:]:
            up_7 = F.interpolate(up_7, size=c3.size()[2:], mode='nearest')
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        if up_8.size()[2:] != c2.size()[2:]:
            up_8 = F.interpolate(up_8, size=c2.size()[2:], mode='nearest')
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        if up_9.size()[2:] != c1.size()[2:]:
            up_9 = F.interpolate(up_9, size=c1.size()[2:], mode='nearest')
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9) '''

        output = self.final_conv(c9)

        return output