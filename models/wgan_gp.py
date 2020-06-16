import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable
from itertools import chain
import pandas as pd
import time as t
import os
from utils.inception_score import get_inception_score


class Generator(nn.Module):
    def __init__(self, ngpu, nc, nz, ngf):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGAN(nn.Module):
    def __init__(self, args):
        super(DCGAN, self).__init__()

        self.netG = Generator(args.ngpu, args.nc, args.nz, args.ngf).cuda()
        if args.ngpu > 1:
            self.netG = nn.DataParallel(self.netG, list(range(args.ngpu)))
        print(self.netG)

        self.netD = Discriminator(args.ngpu, args.nc, args.ndf).cuda()
        if args.ngpu > 1:
            self.netD = nn.DataParallel(self.netD, list(range(args.ngpu)))
        self.netD.apply(weights_init)
        print(self.netD)

        self.criterion = nn.BCELoss().cuda()
        if args.ngpu > 1:
            self.criterion = nn.DataParallel(self.criterion, list(range(args.ngpu)))

        self.fixed_noise = torch.randn(args.batch_size, args.nz, 1, 1).cuda()

        self.real_label = 1
        self.fake_label = 0

        self.optimizerD = optim.Adam(
            self.netD.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999)
        )
        self.optimizerG = optim.Adam(
            self.netG.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999)
        )

        self.epochs = args.epochs
        self.batch_size = args.batch_size

        self.model_name = args.model
        self.gpu_ids = args.gpuids

        self.latent_size = args.nz
        self.n_channel = args.nc
        self.image_size = args.image_size

    def calculate_gradient_penalty(self, real_image, fake_image):
        alpha = torch.rand(self.batch_size, 1, 1, 1).uniform_(0, 1)
        alpha = alpha.expand(self.batch_size, real_image.size(1), real_images.size(2), real_images.size(3))
        alpha = alpha.cuda()

        interpolated = eta * real_images + ((1 - alpha) * fake_image)
        interpolated = interpolated.cuda()
        interpolated = Variable(interpolated, requires_grad=True)

        prob_interpolated = self.D(interpolated)

        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(
                                  prob_interpolated.size()).cuda(),
                                  create_graph=True, retain_graph=True)[0]
                                  
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lamda

        return gradient_penalty


    def train(self, dataloader):
        self.t_begin = t.time()
        G_losses = []
        D_losses = []
        Wasserstein_D_list = []
        iter_list = []
        iters = 0

        if not os.path.exists("./results/{}/".format(self.model_name)):
            os.makedirs("./results/{}/".format(self.model_name))
        self.file = open(
            "./results/{}/inception_score_graph.txt".format(self.model_name), "w"
        )

        print("Starting Training Loop...")

        for epoch in range(self.epochs):
            for i, data in enumerate(dataloader, 0):

                self.netD.zero_grad()
                real = data[0].cuda()
                batch_size = real.size(0)
                label = torch.full((batch_size,), self.real_label).cuda()
                output = self.netD(real).view(-1)

                d_loss_real = self.criterion(output, label)
                d_loss_real.backward()
                D_x = output.mean().item()

                noise = torch.randn(batch_size, self.latent_size, 1, 1).cuda()

                fake = self.netG(noise)
                label.fill_(self.fake_label)

                output = self.netD(fake.detach()).view(-1)

                d_loss_fake = self.criterion(output, label)
                d_loss_fake.backward()
                D_G_z1 = output.mean().item()


                gradient_penalty = self.calculate_gradient_penalty(real, fake)
                gradient_penalty.backward()

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake


                self.optimizerD.step()

                self.netG.zero_grad()
                label.fill_(self.real_label)

                output = self.netD(fake).view(-1)

                g_loss = self.criterion(output, label)

                g_loss.backward()
                D_G_z2 = output.mean().item()

                self.optimizerG.step()

                if i % 50 == 0:
                    print(
                        "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                        % (
                            epoch,
                            self.epochs,
                            i,
                            len(dataloader),
                            d_loss.item(),
                            g_loss.item(),
                            D_x,
                            D_G_z1,
                            D_G_z2,
                        )
                    )

                if (iters + 1) % 1000 == 0:

                    output = self.calculate_inception_score()
                    self.file.write(output)

                    self.save_training_result_images

                self.save_logs(iters, g_loss, d_lss, Wasserstein_D)

                iters += 1

            self.generate_image()
            self.save_model()
          
        self.file.close()


    def calculate_inception_score(self):
                    
        sample_list = []
        for i in range(10):
            z = self.fixed_noise
            samples = self.netG(z)
            sample_list.append(samples.data.cpu().numpy())

        new_sample_list = list(chain.from_iterable(sample_list))
        print("Calculating Inception Score over 8k generated images")
        inception_score = get_inception_score(
            new_sample_list,
            cuda=True,
            batch_size=32,
            resize=True,
            splits=10,
        )

        time = t.time() - self.t_begin
        print("Inception score: {}".format(inception_score))
        print("Generator iter: {}".format(iters))
        print("Time {}".format(time))

        output = str(iters) + ", " + str(inception_score[0]) + "\n"
                    
        return output

    def save_training_result_images(self):

        z = self.fixed_noise
        samples = self.netG(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()[:64]
        grid = vutils.make_grid(samples)
        if not os.path.exists(
            "./training_result_images/{}/".format(self.model_name)
        ):
            os.makedirs(
                "./training_result_images/{}/".format(self.model_name)
            )
            vutils.save_image(
                grid,
                "./training_result_images/{}/img_generator_iter_{}.png".format(
                    self.model_name, str(iters).zfill(3)
                ),
            )

    def save_logs(self, iters, g_loss, d_lss, Wasserstein_D):
        iter_list.append(iters)
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())
        Wasserstein_D_list.append(Wasserstein_D.item())

        losses = pd.DataFrame(
            {"Steps": iter_list, "G_losses": G_losses, "D_losses": D_losses, "Wasserstein_Distance:" Wasserstein_D_list}
        )

        if not os.path.exists("./results/{}/".format(self.model_name)):
            os.makedirs("./results/{}/".format(self.model_name))
        losses.to_csv("./results/{}/g_and_d_losses.csv".format(self.model_name))


    def generate_image(self):
        if not os.path.exists(
            "./generated_images/{}/real/".format(self.model_name)
        ):
            os.makedirs("./generated_images/{}/real".format(self.model_name))
        if not os.path.exists(
            "./generated_images/{}/fake/".format(self.model_name)
        ):
            os.makedirs("./generated_images/{}/fake".format(self.model_name))

        real = real.mul(0.5).add(0.5)
        fake = fake.mul(0.5).add(0.5)

        for i in range(0, 64):
            vutils.save_image(
                real.data[i, ...].reshape(
                    (1, self.n_channel, self.image_size, self.image_size)
                ),
                os.path.join(
                    "./generated_images/{}/real/generated_%02d.png".format(
                        self.model_name
                    )
                    % i
                ),
            )
            vutils.save_image(
                fake.data[i, ...].reshape(
                    (1, self.n_channel, self.image_size, self.image_size)
                ),
                os.path.join(
                    "./generated_images/{}/fake/generated_%02d.png".format(
                        self.model_name
                    )
                    % i
                ),
            )

    def save_model(self):
        if not os.path.exists("./checkpoint/{}/".format(self.model_name)):
            os.makedirs("./checkpoint/{}/".format(self.model_name))
        torch.save(
            self.netG.state_dict(),
            "checkpoint/{}/netG_epoch_{}.pth".format(self.model_name, self.epochs),
        )
        torch.save(
            self.netD.state_dict(),
            "checkpoint/{}/netD_epoch_{}.pth".format(self.model_name, self.epochs),
        )
