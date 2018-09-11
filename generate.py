from __future__ import print_function
import argparse
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import scipy.misc

# hard-wire the gpu id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('--netG', default='', help='path to trained generator')
parser.add_argument('--nz', type=int, default='100', help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--batch_size', type=int, default='256', help='the batch size used during training')
parser.add_argument('--cuda', default=1, action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus available')

opt = parser.parse_args()
print(opt)

cudnn.benchmark = True

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # sample from Z and feed the first conv
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # output size (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # output size (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # output size (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # output size ngf x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # ouput size 3 x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

# load the pretrained weights
netG = Generator(ngpu).to(device)
netG.load_state_dict(torch.load(opt.netG))
netG.eval()
print(netG)

# generate images from random noise
noise = torch.randn(opt.batch_size, nz, 1, 1, device=device)
generated_samples = netG(noise)

# save one of the generated images as example
scipy.misc.toimage(generated_samples.view(-1, 3, 64, 64)[0].cpu().detach().numpy()).save('./examples/example.png')
