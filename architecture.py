import torch.nn as nn


class ColorizerModel(nn.Module):
    """ Colorizer Architecture

        Architecture based on "Colorful Image Colorization" by Zhang, et al.
        https://arxiv.org/pdf/1603.08511.pdf
        https://richzhang.github.io/colorization/
    """

    def __init__(self, input_size=128):
        super(ColorizerModel, self).__init__()

        # Increase channels to 64
        conv1 = [nn.Conv2d(1, 64, kernel_size=3, stride=1,
                 padding=1, bias=True)]
        conv1 += [nn.ReLU(True)]
        # Decrease HxW from 256x256 to 128x128
        conv1 += [nn.Conv2d(64, 64, kernel_size=3, stride=2,
                  padding=1, bias=True)]
        conv1 += [nn.ReLU(True)]
        conv1 += [nn.BatchNorm2d(64)]

        # Increase channels to 128
        conv2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1,
                 padding=1, bias=True)]
        conv2 += [nn.ReLU(True)]
        # Decrease HxW from 128x128 to 64x64
        conv2 += [nn.Conv2d(128, 128, kernel_size=3, stride=2,
                  padding=1, bias=True)]
        conv2 += [nn.ReLU(True)]
        conv2 += [nn.BatchNorm2d(128)]

        # Increase channels to 256
        conv3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1,
                 padding=1, bias=True)]
        conv3 += [nn.ReLU(True)]
        # Decrease HxW from 64x64 to 32x32
        conv3 += [nn.Conv2d(256, 256, kernel_size=3, stride=2,
                  padding=1, bias=True)]
        conv3 += [nn.ReLU(True)]
        conv3 += [nn.BatchNorm2d(256)]

        # Increase channels to 512
        conv4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1,
                 padding=1, bias=True)]
        conv4 += [nn.ReLU(True)]
        conv4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1,
                  padding=1, bias=True)]
        conv4 += [nn.ReLU(True)]
        conv4 += [nn.BatchNorm2d(512)]

        # Dilated Convolution
        conv5 = [nn.Conv2d(512, 512, kernel_size=3, stride=1,
                 padding=2, dilation=2, bias=True)]
        conv5 += [nn.ReLU(True)]
        conv5 += [nn.Conv2d(512, 512, kernel_size=3, stride=1,
                  padding=2, dilation=2, bias=True)]
        conv5 += [nn.ReLU(True)]
        conv5 += [nn.BatchNorm2d(512)]

        # Dilated Convolution
        conv6 = [nn.Conv2d(512, 512, kernel_size=3, stride=1,
                 padding=2, dilation=2, bias=True)]
        conv6 += [nn.ReLU(True)]
        conv6 += [nn.Conv2d(512, 512, kernel_size=3, stride=1,
                  padding=2, dilation=2, bias=True)]
        conv6 += [nn.ReLU(True)]
        conv6 += [nn.BatchNorm2d(512)]

        # Normal Convolution
        conv7 = [nn.Conv2d(512, 512, kernel_size=3, stride=1,
                 padding=1, bias=True)]
        conv7 += [nn.ReLU(True)]
        conv7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1,
                  padding=1, bias=True)]
        conv7 += [nn.ReLU(True)]
        conv7 += [nn.BatchNorm2d(512)]

        # Decrease channels from 512 to 256, increase HxW from 32x32 to 64x64
        conv8 = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2,
                 padding=1, bias=True)]
        conv8 += [nn.ReLU(True)]
        conv8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1,
                  padding=1, bias=True)]
        conv8 += [nn.ReLU(True)]
        conv8 += [nn.BatchNorm2d(256)]

        # Increase channels from 256 to 313
        conv9 = [nn.Conv2d(256, 313, kernel_size=1, stride=1,
                 padding=0, bias=True)]
        # Decrease channels from 313 to 2
        conv9 += [nn.Conv2d(313, 2, kernel_size=1, stride=1,
                  padding=0, dilation=1, bias=False)]

        # Softmax layer
        softmax = [nn.Softmax(dim=1)]

        # Upsample x4 from 64x64 to 256x256
        upsample = [nn.Upsample(scale_factor=4, mode='bilinear')]

        # Create Sequential objects for each convolution layer
        self.conv1 = nn.Sequential(*conv1)
        self.conv2 = nn.Sequential(*conv2)
        self.conv3 = nn.Sequential(*conv3)
        self.conv4 = nn.Sequential(*conv4)
        self.conv5 = nn.Sequential(*conv5)
        self.conv6 = nn.Sequential(*conv6)
        self.conv7 = nn.Sequential(*conv7)
        self.conv8 = nn.Sequential(*conv8)
        self.conv9 = nn.Sequential(*conv9)
        self.softmax = nn.Sequential(*softmax)
        self.upsample = nn.Sequential(*upsample)

    def forward(self, input):

        # Normalize luminance
        output = (input - 50.) / 100.

        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = self.conv7(output)
        output = self.conv8(output)
        output = self.conv9(output)
        output = self.softmax(output)
        output = self.upsample(output)

        # Un-normalize ab
        output = output * 110.

        return output
