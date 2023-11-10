import torch

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        # log-normal common noise-schedule variance of pixels for every image
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # loss function weight
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        # Images augmentations - geometrical transformations
        # The augmenation label gives infomation about the transformations and is feeded to the nn as a conditional variable
        # When none this indicates that no transformations have been applied to the images and therefore allow for non-leaky
        #augmentations during inference.
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        # Sampling the gaussian with independent components and common variance 
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, None, class_labels=labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss
