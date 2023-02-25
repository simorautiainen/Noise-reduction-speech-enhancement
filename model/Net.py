import math
import torch
import torch.nn as nn
from model.dual_transf import Dual_Transformer
from torch import cuda, rand

class SPConvTranspose2d(nn.Module):
    # This is sub-pixel conv block
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseBlock(nn.Module):
    # This is the Dilated-Dense Block
    def __init__(self, input_size, depth=5, in_channels=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.LayerNorm(input_size))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class MaskingModule(nn.Module):
    # Masking module must be sepe
    def __init__(self):
        super().__init__()
        self.in_conv = nn.Sequential(
                nn.PReLU(),
                nn.Conv2d(64//2, 64, 1)
        )
        self.left_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.Tanh()
        )
        self.right_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        out = self.in_conv(x)
        out = self.left_block(out) * self.right_block(out)
        out = self.out_conv(out)
        return out
    
class Net(nn.Module):
    def __init__(self, nffts=512):
        super().__init__()
        # self.device = device
        self.in_channels = 1
        self.out_channels = 1
        self.nffts = nffts
        self.nffts_half = math.floor(nffts/2 + .5)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 1)),
            nn.LayerNorm(self.nffts),
            nn.PReLU(64),
            DenseBlock(self.nffts, 4, 64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0,1)),
            nn.LayerNorm(self.nffts_half),
            nn.PReLU(64),
        )


        self.TSTM = Dual_Transformer(64, 64, num_layers=4)
        self.decoder = nn.Sequential(
            DenseBlock(self.nffts_half, 4, 64),
            nn.ConstantPad2d((1, 1, 0, 0), value=0.),
            SPConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(1, 3), r=2),
            nn.LayerNorm(self.nffts),
            nn.PReLU(64),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1))
        )
        # gated output layer
        self.masking_module = MaskingModule()
    def forward(self, x):
        x = x.to(dtype=torch.float)
        x = x.unsqueeze(1)
        x = x.transpose(-1, -2)

        encoder_out = self.encoder(x)  # [b, 64, num_frames, frame_size]

        transformer_out = self.TSTM(encoder_out)  # [b, 64, num_frames, 256]

        masking_module_out = self.masking_module(transformer_out)
        out = encoder_out * masking_module_out
        out = self.decoder(out)

        out = out.transpose(-1, -2)
        return out.squeeze()


def main():
    device = 'cuda' if cuda.is_available() else 'cpu'

    batch_size = 4
    d_time = 62
    d_feature = 512
    x = rand(batch_size,d_time,d_feature)
    x = x.to(device)
    model = Net(512)
    model = model.to(device)
    y_hat = model(x)
    print(y_hat.shape)

if __name__ == "__main__":
    main()