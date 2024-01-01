# Markus Enzweiler - markus.enzweiler@hs-esslingen.de
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class BaseNet(nn.Module):
    """Base class for neural network models."""

    def __init__(self, device=torch.device("cpu")):
        super(BaseNet, self).__init__()
        self.device = device

    def save(self, fname):
        # Extract the directory path from the file name
        dir_path = os.path.dirname(fname)

        # Ensure the directory exists
        utils.ensure_folder_exists(dir_path)

        # Save the model
        torch.save(self.state_dict(), fname)

    def load(self, fname, device):
        self.load_state_dict(torch.load(fname, map_location=self.device))
        self.eval()

class Generator(BaseNet):
    """
    A convolutional generator based on the DCGAN architecture.

    This generator employs a fully convolutional architecture, adhering to the guidelines
    set forth in the DCGAN paper ("Unsupervised Representation Learning with Deep Convolutional
    Generative Adversarial Networks", https://arxiv.org/pdf/1511.06434.pdf). The design is
    specifically tailored for generative adversarial networks, with a focus on producing
    high-quality synthetic images. The architecture leverages transposed convolutional layers
    to upscale latent space representations into detailed and coherent images.

    Guidelines from the DCGAN paper:    
        * Replace any pooling layers with strided convolutions (discriminator) 
          and fractional-strided convolutions (generator).
        * Use batchnorm in both the generator and the discriminator.
        * Remove fully connected hidden layers for deeper architectures.
        * Use ReLU activation in generator for all layers except for the output, which uses Tanh.
        * Use LeakyReLU activation in the discriminator for all layers.

    """

    def __init__(self, num_latent_dims, num_img_channels, max_num_filters=64, device=torch.device("cpu")):
        super(Generator, self).__init__(device)
            
        self.num_latent_dims = num_latent_dims
        self.num_img_channels = num_img_channels
        self.max_num_filters = max_num_filters

        # we assume B x #img_channels x 28 x 28 input 
        # Todo: add input shape attribute to the model to make it more flexible

        # C x H x W
        img_input_shape = (num_img_channels, 28, 28)

        # half the number of filters in each layer
        self.num_filters_1 = self.max_num_filters
        self.num_filters_2 = self.num_filters_1 // 2
        self.num_filters_3 = self.num_filters_2 // 2

        # we push the latent vector through a linear layer to project it
        # to a higher dimensional space
        self.lat_proj = nn.Linear(self.num_latent_dims, self.num_filters_1)
         
        # we assume input shape of (num_filters_1, 1, 1)

        # Output num_filters_1 x 4 x 4
        self.conv1 = nn.ConvTranspose2d(self.num_filters_1,   self.num_filters_1,    kernel_size=4, 
            stride=1, padding=0, bias=False)
        # Output num_filters_2 x 7 x 7   
        self.conv2 = nn.ConvTranspose2d(self.num_filters_1,   self.num_filters_2,    kernel_size=3, 
            stride=2, padding=1, bias=False)
        # Output num_filters_3 x 14 x 14   
        self.conv3 = nn.ConvTranspose2d(self.num_filters_2,   self.num_filters_3,    kernel_size=4, 
            stride=2, padding=1, bias=False)
        # Output num_filters_4 x 28 x 28
        self.conv4 = nn.ConvTranspose2d(self.num_filters_3,   self.num_img_channels, kernel_size=4, 
            stride=2, padding=1, bias=False)
       
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(self.num_filters_1)
        self.bn2 = nn.BatchNorm2d(self.num_filters_2)
        self.bn3 = nn.BatchNorm2d(self.num_filters_3)      
    
    def forward(self, x):

        # Assuming x has a shape of [batch_size, num_latent_dims, 1, 1]
        # Flatten x to the shape [batch_size, num_latent_dims]
        x = x.view(x.size(0), -1)
        # Pass it through the linear layer
        x = self.lat_proj(x) 
        # reshape the latent vector to a 4D tensor of shape (batch_size, num_filters_1, 1, 1)
  
        x = x.view(len(x), self.num_filters_1, 1, 1)    
        
        # push it through the layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))) 
             
        # We use a tanh activation in the last layer as suggested in the DCGAN paper. 
        # This ensures that the output is in the range [-1, 1] and thus matches the
        # range of the real images (see dataset.py).

        x = F.tanh(         self.conv4(x)) 
        return x
    
class Discriminator(BaseNet):
    """
    A convolutional discriminator based on the DCGAN architecture.

    This discriminator is fully convolutional and follows the design principles outlined in
    the DCGAN paper ("Unsupervised Representation Learning with Deep Convolutional Generative
    Adversarial Networks", https://arxiv.org/pdf/1511.06434.pdf). The architecture is
    optimized for generative adversarial networks and is suitable for discriminating
    between real and generated images.


    Guidelines from the DCGAN paper:    
        * Replace any pooling layers with strided convolutions (discriminator) 
          and fractional-strided convolutions (generator).
        * Use batchnorm in both the generator and the discriminator.
        * Remove fully connected hidden layers for deeper architectures.
        * Use ReLU activation in generator for all layers except for the output, which uses Tanh.
        * Use LeakyReLU activation in the discriminator for all layers.
    """

    def __init__(self, num_img_channels, max_num_filters=64, device=torch.device("cpu")):
        super(Discriminator, self).__init__(device)
              
        self.num_img_channels = num_img_channels  
        self.max_num_filters = max_num_filters   

        # we assume B x #img_channels x 28 x 28 input 
        # Todo: add input shape attribute to the model to make it more flexible

        # C x H x W
        img_input_shape = (num_img_channels, 28, 28)

        # double the number of filters in each layer
        num_filters_3 = max_num_filters 
        num_filters_2 = num_filters_3   // 2
        num_filters_1 = num_filters_2   // 2

        # we assume input shape of (num_img_channels, 28, 28)

        # Output num_filters_1 x 16 x 16
        self.conv1 = nn.Conv2d(num_img_channels, num_filters_1, kernel_size=4, stride=2, padding=1, bias=False)
        # Output num_filters_2 x 8 x 8
        self.conv2 = nn.Conv2d(num_filters_1,    num_filters_2, kernel_size=4, stride=2, padding=1, bias=False)
        # Output num_filters_3 x 4 x 4 
        self.conv3 = nn.Conv2d(num_filters_2,    num_filters_3, kernel_size=3, stride=2, padding=1, bias=False)
     
        # In the final layer we want to have a single output value, i.e. (batch_size, 1, 1, 1).
        # For that, we use 1 output channel, a kernel size of 4 and stride of 1 with no padding.
        # Output 1 x 1 x 1
        self.conv4 = nn.Conv2d(num_filters_3,   1, kernel_size=4, stride=1, padding=0, bias=False)                            

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(num_filters_1)
        self.bn2 = nn.BatchNorm2d(num_filters_2)
        self.bn3 = nn.BatchNorm2d(num_filters_3)
       
    def forward(self, x):        
        # slope of the leaky ReLU activation function is set to 0.2)
        # as suggested in the DCGA)N paper
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)      

        # From https://arxiv.org/pdf/1511.06434.pdf:
        # "For the discriminator, the last convolution layer
        # "is flattened and then fed into a single sigmoid output"

        # Sigmoid activation as suggested in the DCGAN paper 
        x = F.sigmoid(self.conv4(x))

        # Flatten the output of the last convolutional layer
        # (all dimensions except batch dimension)
        x = torch.flatten(x, start_dim=1)
        
        return x
    
