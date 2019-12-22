import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import cv2
import numpy as np
from dogModel import genNet, disNet, weightsInit
from dogDataLoader import loadData
from dogTrainer import Trainer

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

# Load the Data into dataloader
device, dataloader = loadData(args)

# Create the Generator
netG = genNet(args).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (args.ngpu > 1):
    netG = DataParallel(netG, list(range(ngpu)))

# Apply  the weights_init function to init. weights to mean = 0, stdev = 0.2
netG.apply(weightsInit)

# Create the Discriminator
netD = disNet(args).to(device)

# Apply  the weights_init function to init. weights to mean = 0, stdev = 0.2
netD.apply(weightsInit)

# Create Trainer

Trainer = Trainer(args, dataloader, netG, netD, device)
img_list = []
# For each epoch
for epoch in range(args.numEpochs):
    # For each batch in DataLoader
    img_list = Trainer.train(epoch, img_list)

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
