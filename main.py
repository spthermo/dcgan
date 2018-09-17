from __future__ import print_function
import argparse
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils

from logger import Logger

# hard-wire the gpu id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', default='', help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--cuda', default=1, action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus available')
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=50, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.99, help='beta2 for adam. default=0.99')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3

# save feedback
logger = Logger('./logs')

# Computes and stores the average and current value
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.npgu = ngpu
        self.main = nn.Sequential(
            # input size 3 x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # output size ndf x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # output size (ndf * 2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # output size (ndf * 4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # output size (ndf * 8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.npgu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

# weight initialization
def weights_init(w):
    layer = w.__class__.__name__
    if layer.find('Conv') != -1:
        w.weight.data.normal_(0.0, 0.02)
    elif layer.find('BatchNorm') != -1:
        w.weight.data.normal_(1.0, 0.02)
        w.bias.data.fill_(0)

# initiate Generator & Discriminator
netG = Generator(ngpu).to(device)
netG.apply(weights_init)
print(netG)

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
print(netD)

# choose loss function
criterion = nn.BCELoss()

# init z distribution & fake/real labels
z_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizers
optG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
optD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

# data loading process
traindir = os.path.join(opt.dataPath, 'train')

train_loader = torch.utils.data.DataLoader(
        dsets.ImageFolder(traindir, transforms.Compose([
            transforms.CenterCrop(100),
            transforms.Resize(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
        ])),
        batch_size=opt.batchSize, shuffle=True,
        num_workers=opt.workers, pin_memory=True)

errD_all = AverageMeter()
errG_all = AverageMeter()
D_x_all = AverageMeter()
D_G_z1_all = AverageMeter()
D_G_z2_all = AverageMeter()

# start training
for epoch in range(opt.niter):
    t0 = time.time()
    for i, data in enumerate(train_loader,0):

        # train Discriminator with real samples
        optD.zero_grad()
        real_data = data[0].to(device)
        batch_size = real_data.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        out = netD(real_data)
        errD_real = criterion(out, label)
        D_x = out.mean().item()

        # train Discriminator with fake samples
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
	label_f = label.clone()
	label_f.fill_(fake_label)        
	fake_data = netG(noise)
        out = netD(fake_data.detach())
        errD_fake = criterion(out, label)
        D_G_z1 = out.mean().item()
        errD = errD_real + errD_fake
	errD.backward()
        optD.step()

        # update Generator
        optG.zero_grad()
	fake_data = netG(noise)
        out = netD(fake_data)
        errG = criterion(out, label)
        errG.backward()
        D_G_z2 = out.mean().item()
        optG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(train_loader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        errD_all.update(errD.item())
        errG_all.update(errG.item())
        D_x_all.update(D_x)
        D_G_z1_all.update(D_G_z1)
        D_G_z2_all.update(D_G_z2)

    print('Time elapsed Epoch %d: %d seconds'
          % (epoch, time.time() - t0))
    # TensorBoard logging
    # scalar values
    info = {
        'D loss': errD_all.avg,
        'G loss': errG_all.avg,
        'D(x)': D_x_all.avg,
        'D(G(z1))': D_G_z1_all.avg,
        'D(G(z2))': D_G_z2_all.avg
    }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)

    # values and gradients of the parameters (histogram)
    for tag, value in netG.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.cpu().detach().numpy(), epoch)

    # (3) generated image samples
    info = {
        'images': fake_data.view(-1, 3, 64, 64)[:10].cpu().detach().numpy()
    }
    for tag, images in info.items():
        logger.image_summary(tag, images, epoch)

torch.save(netG.state_dict(), './models/netG_e%d.pth' % (epoch+1))
torch.save(netD.state_dict(), './models/netD_e%d.pth' % (epoch+1))
