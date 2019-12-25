import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from dogModel import genNet, disNet, weightsInit
from dogDataLoader import loadData
from dogTrainer import Trainer
from dogFinalize import saveFinish

manualSeed = random.randint(1, 10000)

print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

parser = argparse.ArgumentParser(description='Generate fake dog images')
parser.add_argument('--dataroot', default='../datasets/dogData',
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

# Create Arrays to save in them later
img_list, G_Losses, D_Losses = [], [], []

# For each epoch
for epoch in range(args.numEpochs):
    # For each batch in DataLoader
    img_list, G_Losses, D_Losses = Trainer.train(epoch, img_list, G_Losses, D_Losses)

saveFinish(img_list, G_Losses, D_Losses, netG, manualSeed, netD)
