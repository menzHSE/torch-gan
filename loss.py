# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

import torch
import torch.nn as nn

# We define two different loss functions, one for the real images
# and one for the fake images.

# We use the binary cross entropy loss (BCELoss) for both since
# the discriminator is a binary classifier and has a sigmoid
# activation function in the last layer.


# smoothing class=1 to [0.8, 1.1]
def smooth_real_labels(y, dev):
    return y - 0.2 + (torch.rand(y.shape, device=dev) * 0.3)


# smoothing class=0 to [0.0, 0.2]
def smooth_fake_labels(y, dev):
    return y + (torch.rand(y.shape, device=dev) * 0.2)


def bce_loss_real(predictions, smooth=False, device=torch.device("cpu")):
    criterion = nn.BCELoss()
    # we use the label 1 for real images

    # Add label smoothing in real loss to
    # prevent discriminator becoming too strong too quickly

    if smooth:
        real_labels = smooth_real_labels(torch.ones_like(predictions), device)
    else:
        real_labels = torch.ones_like(predictions)

    return criterion(predictions, real_labels)


def bce_loss_fake(predictions, smooth=False, device=torch.device("cpu")):
    criterion = nn.BCELoss()
    # we use the label 0 for fake images

    # Add label smoothing in fake loss to
    # prevent discriminator becoming too strong too quickly
    if smooth:
        fake_labels = smooth_fake_labels(torch.zeros_like(predictions), device)
    else:
        fake_labels = torch.zeros_like(predictions)

    return criterion(predictions, fake_labels)
