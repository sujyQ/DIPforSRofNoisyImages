import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
class Discriminator(nn.Module):
    def __init__(self, in_chn, ndf=64, ngpu=1 ):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        main = nn.Sequential(
            # input is (in_chn) x 64 x 64
            nn.Conv2d(in_chn, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.main = self._init_weights(main)

    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, input):
        return self.main(input)

# sinGAN discriminator
class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, stride):
        super(ConvBlock, self).__init__()
        # self.conv2d = nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)
        # self.bn = nn.BatchNorm2d(out_channel)
        # self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.add_module('conv', nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm', nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',  nn.LeakyReLU(0.2, inplace=True))

class WDiscriminator(nn.Module):
    def __init__(self, nc, nfc, min_nfc, ker_size, num_layer, stride, pad_size):
        super(WDiscriminator, self).__init__()

        self.is_cuda = torch.cuda.is_available()
        N = int(nfc)
        self.head = ConvBlock(nc, N, ker_size, pad_size,1)
        self.body = nn.Sequential()
        for i in range(num_layer-2):
            N = int(nfc/pow(2,(i+1)))
            block = ConvBlock(max(2*N, min_nfc),max(N, min_nfc), ker_size, pad_size,1)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Conv2d(max(N, min_nfc),1,kernel_size= ker_size,stride=1,padding= pad_size)

    def forward(self,x):
        
        # print(x.size())
        x = self.head(x)
        # print(x.size())
        x = self.body(x)
        # print(x.size())
        x = self.tail(x)
        # print(x.size())

        return x


def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

class UNetD(nn.Module):
    def __init__(self, in_chn, wf=32, depth=5, relu_slope=0.2):
        super(UNetD, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        prev_channels = self.get_input_chn(in_chn)
        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.down_path.append(UNetConvBlock(prev_channels, (2**i)*wf, downsample, relu_slope))
            prev_channels = (2**i) * wf

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            prev_channels = (2**i)*wf

        self.last = conv3x3(prev_channels, 1, bias=True)

    def forward(self, x1):
        blocks = []
        for i, down in enumerate(self.down_path):
            if (i+1) < self.depth:
                x1, x1_up = down(x1)
                blocks.append(x1_up)
            else:
                x1 = down(x1)

        for i, up in enumerate(self.up_path):
            x1 = up(x1, blocks[-i-1])

        out = self.last(x1)
        print(out)
        return out

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=True))

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x):
        out = self.block(x)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out