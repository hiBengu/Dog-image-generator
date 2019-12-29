import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils


class Trainer():
    def __init__(self, args, dataloader, genNet, disNet, device):
        self.netG = genNet
        self.netD = disNet
        self.dataloader = dataloader
        self.device = device
        self.numEpochs = args.numEpochs
        self.nz = args.nz
        # Initialize BCELoss function
        self.criterion = nn.BCELoss()

        # Create batch of latent vectors that we will use
        self.fixedNoise = torch.randn(64, self.nz, 1, 1, device=self.device)

        # labels
        self.realLabel = 1
        self.fakeLabel = 0

        # Setup the Adam optimizer for both networks
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

        # Training Loop
        self.iters = 0

    def train(self, epoch, img_list, G_Losses, D_Losses):
        self.G_Losses = G_Losses
        self.D_Losses = D_Losses
        self.img_list = img_list
        for i, data in enumerate(self.dataloader, 0):
            #####
            # Update Discriminative, maximize log(D(x)) + log(1 - D(G(z)))
            #####
            # Train with all-real batches ###########
            self.netD.zero_grad()

            # Format batches
            real_cpu = data[0].to(self.device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), self.realLabel, device=self.device)

            # Forward pass real batch through D
            output = self.netD(real_cpu).view(-1)

            # Calculate loss on all-real batch
            errD_real = self.criterion(output, label)

            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch ##############
            # Generate batch of latent vectors
            noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)

            # Generate fake image batch with G
            fake = self.netG(noise)
            label.fill_(self.fakeLabel)

            # Classify all fake batch with D
            output = self.netD(fake.detach()).view(-1)

            # Calculate D's loss on the all-fake batch
            errD_fake = self.criterion(output, label)

            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake

            # Update D
            self.optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            self.netG.zero_grad()
            label.fill_(self.realLabel)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = self.netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = self.criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            self.optimizerG.step()

            ## Show Losses
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, self.numEpochs, i, len(self.dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            self.G_Losses.append(errG.item())
            self.D_Losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (self.iters % 500 == 0) or ((epoch == self.numEpochs-1) and (i == len(self.dataloader)-1)):
                with torch.no_grad():
                    fake = self.netG(self.fixedNoise).detach().cpu()
                self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            self.iters += 1

        return self.img_list, self.G_Losses, self.D_Losses
