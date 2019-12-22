from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import cv2

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results

print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

parser = argparse.ArgumentParser(description='Generate fake dog images')
parser.add_argument('--dataroot', default='../dogData',
                    help='Dataset directory')
parser.add_argument('--workers', type=int, default='2',
                    help='number of worker thread for loading the data')
parser.add_argument('--batchSize', type=int, default='128',
                    help='Batch Size, DCGAN paper uses batch size of 128')
parser.add_argument('--imageSize', type=int, default='64',
                    help='Size of the images')
parser.add_argument('--nc', type=int, default='3',
                    help='number of colour channels of images')
parser.add_argument('--nz', type=int, default='100',
                    help='Size of the generator input')
parser.add_argument('--ngf', type=int, default='64',
                    help='size of feature maps in Generator')
parser.add_argument('--ndf', type=int, default='64',
                    help='size of feature maps in Discriminator')
parser.add_argument('--numEpochs', type=int, default='100',
                    help='number of epochs')
parser.add_argument('--lr', type=float, default='0.0002',
                    help='learning rate')
parser.add_argument('--beta1', type=float, default='0.5',
                    help='hyperparameter for adam optimizer')
parser.add_argument('--ngpu', type=int, default='1',
                    help='number of gpus that will be used')
parser.add_argument('--showTrainData', type=int, default='1',
                    help="0 if don't want to see train data")

args = parser.parse_args()

# Create dataset
dataset = dset.ImageFolder(root=args.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(args.imageSize),
                               transforms.CenterCrop(args.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=args.workers)

# Choose device
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

if (args.showTrainData != 0):
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

# Custom Weight Initialization called on netG and netD
def weightsInit(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Class Code
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(args.nz, args.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(args.ngf * 8, args.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(args.ngf * 4, args.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(args.ngf * 2, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(args.ngf, args.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Create the Generator
netG = Generator(args.ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (args.ngpu > 1):
    netG = DataParallel(netG, list(range(ngpu)))

# Apply  the weights_init function to init. weights to mean = 0, stdev = 0.2
netG.apply(weightsInit)

print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = args.ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(args.nc, args.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(args.ndf * 2, args.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(args.ndf * 4, args.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(args.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Create the Discriminator
netD = Discriminator(args.ngpu).to(device)

# Apply  the weights_init function to init. weights to mean = 0, stdev = 0.2
netD.apply(weightsInit)

# Print the Model
print(netD)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use
fixedNoise = torch.randn(64, args.nz, 1, 1, device=device)

# labels
realLabel = 1
fakeLabel = 0

# Setup the Adam optimizer for both networks
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

# Training Loop
img_list = []
G_losses = []
D_losses = []
iters = 0

# For each epoch
for epoch in range(args.numEpochs):
    # For each batch in DataLoader
    for i, data in enumerate(dataloader, 0):
        #####
        # Update Discriminative, maximize log(D(x)) + log(1 - D(G(z)))
        #####
        # Train with all-real batches ###########
        netD.zero_grad()

        # Format batches
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), realLabel, device=device)

        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)

        # Calculate loss on all-real batch
        errD_real = criterion(output, label)

        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch ##############
        # Generate batch of latent vectors
        noise = torch.randn(b_size, args.nz, 1, 1, device=device)

        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fakeLabel)

        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)

        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)

        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake

        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(realLabel)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        ## Show Losses
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, args.numEpochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == args.numEpochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixedNoise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
        iters += 1

## Plot Losses
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("figures/300e40kLoss.png")

npImgList = np.asarray(np.transpose(img_list[0],(1,2,0)))
npImgList = np.expand_dims(npImgList, axis=0)
print(npImgList.shape)

for i in range(1,5):
    first = 1
    print(i, ":")
    img = np.transpose(img_list[i],(1,2,0))
    fig = plt.figure()
    plt.imshow(img)
    if i % 2 == 0:
        plt.savefig("figures/300e40k"+str(i)+"epoch.png")
    img = np.expand_dims(img, axis=0)
    npImgList = np.vstack((npImgList,np.asarray(img)))

np.save("genOut300e40kIm", npImgList)

torch.save(netD, "models/300e40k")
