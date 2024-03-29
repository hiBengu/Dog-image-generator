import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt


def saveFinish(img_list, G_Losses, D_Losses, netG, manualSeed, netD):
    # save Losses

    torch.save({'state_dict': netG.state_dict()}, "models/10e40kGenCrop"+str(manualSeed)+".pt")
    torch.save({'state_dict': netD.state_dict()}, "models/10e40kDisCrop"+str(manualSeed)+".pt")

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_Losses,label="G")
    plt.plot(D_Losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("figures/10e40kLossCropped"+str(manualSeed)+".png")

    npImgList = np.full((len(img_list),530,530,3),0)
    print(npImgList.shape)
    for i in range(len(img_list)):
        npImgList[i] = np.asarray(np.transpose(img_list[i],(1,2,0)))

    print(npImgList.shape)

    for i in range(npImgList.shape[0]):
        plt.figure()
        plt.imshow(np.transpose(img_list[i],(1,2,0)))
        if (i % 10 == 0 or i == npImgList.shape[0]-1 ):
            plt.savefig("figures/10e40k"+str(manualSeed)+"S"+str(i)+"epoch.png")
        plt.close()

    np.save("genOut10e40kIm"+str(manualSeed)+".npy", npImgList)
