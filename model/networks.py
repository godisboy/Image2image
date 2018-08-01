import torch
import torch.nn as nn
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        input_nc = config['input_nc']
        output_nc = config['output_nc']
        ngf = config['ngf']
        norm = config['norm']
        n_downsample = config['n_downsample']
        n_upsample = config['n_upsample']
        n_resblock = config['n_resblock']
        self.encoder = ResEncoder(input_nc, ngf, norm, n_downsample, n_resblock)
        self.decoder = ResDecoder(output_nc, ngf*4, norm, n_upsample, n_resblock)

    def forward(self, input, common_latent_o):
        encode, common_latent = self.encoder(input)
        decode_input = encode - common_latent + common_latent_o
        return self.decoder(decode_input)


# the Res encoder for disentangling the content
class ResEncoder(nn.Module):
    def __init__(self, input_nc, ngf, norm, n_downsample, n_resblock):
        super(ResEncoder,self).__init__()
        self.model = []
        # 256 x 256
        self.model += [Conv2dBlock(input_nc, ngf, 7, 1, 3, norm=norm)]
        for i in range(n_downsample):
            self.model += [Conv2dBlock(ngf, ngf * 2, 4, 2, 1, norm=norm)]
            ngf *= 2
        # 64 x 64
        for i in range(n_resblock):
            self.model += [ResBlock(ngf, norm=norm)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = ngf
        # common latent layer
        self.common_latent = Avglatentlayer()

    def forward(self, x):
        out = self.model(x)
        return out, self.common_latent(out)


class ResDecoder(nn.Module):
    def __init__(self, output_nc, ngf, norm, n_upsample, n_resblock):
        super(ResDecoder,self).__init__()
        self.model = []
        for i in range(n_resblock):
            self.model += [ResBlock(ngf, norm=norm)]
        # up sampling
        for j in range(n_upsample):
            self.model += [Conv2dBlock(ngf, ngf // 2, 3, 2, conv_padding=1, norm=norm, pad_type='none', transpose=True)]
            ngf //= 2
        self.model += [Conv2dBlock(ngf, output_nc, 7, 1, 3, norm='none', pad_type='reflect', activation='tanh')]
        self.model = nn.Sequential(*self.model)

    def forward(self, input):
        # input = input - common_latent + common_latent_o
        # input_BA = input2 - common_latent_B + common_latent_A
        out = self.model(input)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, conv_padding=0, norm='none', activation='relu', pad_type='zero', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding=conv_padding, output_padding=conv_padding, bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding=conv_padding, bias=self.use_bias)

    def forward(self, x):
        if self.pad is not None:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class Avglatentlayer(nn.Module):
    def __init__(self):
        """
        Input a latent and compute the average representation
        """
        super(Avglatentlayer, self).__init__()
        # self.avglatent = Variable(torch.zeros(1, 64*4, 64, 64).cuda(), volatile=False)
        self.register_buffer('avglatent', torch.zeros(1, 64*4, 64, 64))
        self.threshold = 1000.
        self.count = 1.

    def forward(self, input):
        if self.train():
            if self.count > self.threshold:
                self.count = 1
                self.avglatent = input.data
            else:
                self.count += 1
                self.avglatent = (self.avglatent + input.data) / self.count
        else:
            pass
        return Variable(self.avglatent).cuda()


# Patch discriminator
class PatchD(nn.Module):
    def __init__(self, input_nc, ndf):
        super(PatchD, self).__init__()
        # 256 x 256
        self.layer1 = nn.Sequential(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 128 x 128
        self.layer2 = nn.Sequential(nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1),
                                 nn.BatchNorm2d(ndf*2),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 64 x 64
        self.layer3 = nn.Sequential(nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1),
                                 nn.BatchNorm2d(ndf*4),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 32 x 32
        self.layer4 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
                                    nn.BatchNorm2d(ndf*8),
                                    nn.LeakyReLU(0.2, inplace=True))
        # 31 x 31
        self.layer5 = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
                                    nn.Sigmoid())
        # 30 x 30

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out
