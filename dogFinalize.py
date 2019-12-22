import numpy
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt


def saveFinish(img_list, G_Losses, D_Losses, netG, manualSeed):
    # save Losses
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_Losses,label="G")
    plt.plot(D_Losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("figures/300e40kLoss.png")

    npImgList = np.asarray(np.transpose(img_list,(1,2,0)))
    npImgList = np.expand_dims(npImgList, axis=0)
    print(npImgList.shape)

    for i in range(npImgList.shape[0]):
        img = np.transpose(img_list[i],(1,2,0))
        fig = plt.figure()
        plt.imshow(img)
        if i % 2 == 0:
            plt.savefig("figures/100e40k"+str(manualSeed)+"S"+str(i)+"epoch.png")
        img = np.expand_dims(img, axis=0)
        npImgList = np.vstack((npImgList,np.asarray(img)))

    np.save("genOut100e40kIm"+str(manualSeed)+".npy", npImgList)

    torch.save({'state_dict': netG.state_dict()}, "models/100e40k"+str(manualSeed)+".pt")
