import torch
import torch.nn as nn
from torch.autograd import Variable


def calculate_gradient_penalty(self, netD, real_image, fake_image, lamda, batch_size):
    alpha = torch.rand(batch_size, 1, 1, 1).uniform_(0, 1)
    alpha = alpha.expand(batch_size, real_image.size(1), real_images.size(2), real_images.size(3))
    alpha = alpha.cuda()

    interpolated = eta * real_images + ((1 - alpha) * fake_image)
    interpolated = interpolated.cuda()
    interpolated = Variable(interpolated, requires_grad=True)

    prob_interpolated = netD(interpolated)

    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(
                                  prob_interpolated.size()).cuda(),
                                  create_graph=True, retain_graph=True)[0]
                                  
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamda

    return gradient_penalty
