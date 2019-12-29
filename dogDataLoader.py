import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


def loadData(args):
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

    # Show Training Datas
    if (args.showTrainData != 0):
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()

    return device, dataloader
