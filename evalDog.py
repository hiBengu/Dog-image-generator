import numpy
import torch
import torch.nn as nn
import torchvision.utils as vutils
from matplotlib import pyplot as plt
import numpy as np

nz = 100 # size of fenerator input, latent vector
ngf = 64 # size of feature maps in Generator
nc = 3 # number of colour channels
ngpu = 0
showTrainData = True # Show train images

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Generator Class Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

netG = Generator(ngpu) # Defined generator

noise = torch.randn(64, 100, 1, 1, device=device) # Random noise

model = torch.load("models/100e40k5187.pt", map_location=torch.device('cpu')) # Load Model
netG.load_state_dict(model['state_dict'])

out = netG(noise).detach().cpu() # Push noise to network

outGrid = vutils.make_grid(out, padding=2, normalize=True) # Make a grid out of 64 images
npImgGrid = np.asarray(np.transpose(outGrid,(1,2,0)))

plt.figure()
plt.imshow(npImgGrid)
plt.show()
