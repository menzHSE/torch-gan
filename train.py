# Markus Enzweiler - markus.enzweiler@hs-esslingen.de

# This is an implementation of a deep convolutional GAN in PyTorch. 
# The architecture is based on the DCGAN paper ("Unsupervised Representation 
# Learning with Deep Convolutional Generative Adversarial Networks", 
# https://arxiv.org/pdf/1511.06434.pdf). 

import argparse
import time
import numpy as np
import time

import torch
import torch.optim as optim
import torchvision
from torchinfo import summary

import dataset
import model
import device
import utils
import loss

# Print model summaries
def print_model_summaries(G, D, num_latent_dims, num_img_channels, img_size):
    # add a single batch dimension to the input shape
    G_input_shape = (1,) + (num_latent_dims, 1, 1)
    D_input_shape = (1,) + (num_img_channels, img_size[0], img_size[1])
    
    # print summary and correctly flush the stream
    model_stats_G = summary(G, input_size=G_input_shape, row_settings=["var_names"])
    print("", flush=True)
    time.sleep(1)

    model_stats_D = summary(D, input_size=D_input_shape, row_settings=["var_names"])
    print("", flush=True)
    time.sleep(1)

# Save models
def save_model(model, fname, epoch):
    fname_full = f"{fname}_{epoch:03d}.pth"
    utils.ensure_folder_exists(fname_full)
    model.save(fname_full)

# Save a grid of images
def save_image_grid(images, fname, epoch):
    fname_full = f"{fname}_{epoch:03d}.jpg"

    # normalize from [-1, 1] to [0, 1]
    #images = (images + 1.0) / 2.0

    # use torchvision.utils.save_image to save the grid
    utils.ensure_folder_exists(fname)
    torchvision.utils.save_image(images, fname_full, nrow=8, padding=2, normalize=False)
   
# Weights initialization as suggested in the DCGAN paper:
# Zero-centered Normal distribution with standard deviation 0.02
def init_weights(model):
    mu = 0.0
    sigma = 0.02
    if isinstance(model, torch.nn.Conv2d) or isinstance(model, torch.nn.ConvTranspose2d):
        torch.nn.init.normal_  (model.weight, mu, sigma)
    if isinstance(model, torch.nn.BatchNorm2d):
        torch.nn.init.normal_  (model.weight, mu, sigma)
        torch.nn.init.constant_(model.bias, 0)

# check gradients
def check_grads(model, model_name):
    grads = []
    for p in model.parameters():
        if not p.grad is None:
            grads.append(float(p.grad.mean()))

    grads = np.array(grads)
    if grads.any() and grads.mean() > 100:
        print(f"WARNING! gradients mean is over 100 ({model_name})")
    if grads.any() and grads.max() > 100:
        print(f"WARNING! gradients max is over 100 ({model_name})")

# Sample from the generator
def sample_from_generator(G, num_images, dev):
    # Sample "num_images" random vectors of dimension G.num_latent_dims from the latent space
    latent_vectors = utils.sample_latent_vectors(num_images, G.num_latent_dims, dev)
    # and forward through the generator
    fake_img = G(latent_vectors)
    return fake_img


# Train the discriminator for one step
def train_step_D(G, D, optimizer_D, images):
   
    batch_size = images.shape[0]

    # Check gradients
    check_grads(D, "Discriminator")

    # Reset gradients in the optimizer
    optimizer_D.zero_grad()

    ### Train on real images ###

    # Get outputs of the discriminator for the real images
    real_img_output = D(images)
    #print(f"real image output : {real_img_output}")

    # Compute the loss on the real images
    loss_real = loss.bce_loss_real(real_img_output, smooth=True, device=dev)

    ### Train on fake images ###

    # Generate and train on fake data   
    # Let the generator create "batch_size" fake images from random latent vectors
    fake_img = sample_from_generator(G, batch_size, dev)

    # Get outputs of the discriminator for the fake images
    fake_img_output = D(fake_img)
    #print(f"fake image output : {fake_img_output}")

    # Compute the loss on the fake images
    loss_fake = loss.bce_loss_fake(fake_img_output, smooth=True, device=dev)

    ### Optimize ###

    # Average the two losses 
    loss_avg = (loss_fake + loss_real) / 2

    # Gradients and optimizer step
    loss_avg.backward()
    optimizer_D.step()

    return loss_avg

# Train the generator for one step
def train_step_G(G, D, optimizer_G, images):
   
    batch_size = images.shape[0]

    # Check gradients
    check_grads(G, "Generator")

    # Reset gradients in the optimizer  
    optimizer_G.zero_grad()

    # We want to generate fake images and 
    # train the generator with the output
    # of the discriminator

    # Generate fake images

    # Let the generator create "batch_size" fake images from random latent vectors
    fake_img = sample_from_generator(G, batch_size, dev)

    # Get outputs of the discriminator for the fake images
    fake_img_output = D(fake_img)

    # We want to make these images "real"
    loss_g = loss.bce_loss_real(fake_img_output, smooth=True, device=dev)

    # Gradients and optimizer step
    loss_g.backward()
    optimizer_G.step()
    return loss_g

# Train one epoch
def train_epoch(dev, epoch, G, D, train_loader, optimizer_G, optimizer_D):
        
     # We keep track of the losses
    running_loss_G = 0.0
    running_loss_D = 0.0
    throughput_list = []

    # For each batch, we alternate between training the discriminator and the generator.

    for batch_count, (images, _) in enumerate(train_loader):

        batch_size = images.shape[0]

        # We measure the throughput            
        throughput_tic = time.perf_counter()

        # Transfer the data to the device
        images = images.to(dev)
                    
        # Train discriminator
        loss_D = train_step_D(G, D, optimizer_D, images)
        # Train generator
        loss_G = train_step_G(G, D, optimizer_G, images)
        
        # Keep track of stats
        throughput_toc = time.perf_counter()
        throughput = batch_size / (throughput_toc - throughput_tic)
        throughput_list.append(throughput)
        running_loss_D += loss_D
        running_loss_G += loss_G

        if batch_count > 0 and (batch_count % 10 == 0):
            print(f"Epoch {epoch:4d}: "
                  f"Loss_D {(running_loss_D / batch_count):6.5f} | "
                  f"Loss_G {(running_loss_G / batch_count):6.5f} | "
                  f"Batch {batch_count:5d} | "
                  f"Throughput {throughput:10.2f} images/second", end="\r")         
                        
    return running_loss_G, running_loss_D, throughput_list, batch_count

# Train the Generator and the Discriminator
def train(dev, batch_size, num_epochs, learning_rate, dataset_name, num_latent_dims, max_num_filters):

    # Image size 
    img_size = (28, 28)

    # Get the data
    train_loader, _, _, num_img_channels = dataset.get_loaders(dataset_name, img_size, batch_size)

    # Instantiate the Generator and Discriminator
    G = model.Generator(num_latent_dims, num_img_channels, max_num_filters, device=dev)
    D = model.Discriminator(num_img_channels, max_num_filters, device=dev)

    # Print model summaries
    print_model_summaries(G, D, num_latent_dims, num_img_channels, img_size)

    # Optimizers (we use the same learning rate for both networks)
    betas=[0.5, 0.999] # as suggested in the DCGAN paper

    # We use the AdamW optimizer with the same learning rate for both networks
    # and the betas suggested in the DCGAN paper
    optimizer_G = optim.AdamW(G.parameters(), lr=learning_rate, betas=betas)
    optimizer_D = optim.AdamW(D.parameters(), lr=learning_rate, betas=betas)
    
    # Weights initialization as suggested in the DCGAN paper
    #G = G.apply(init_weights)
    #D = D.apply(init_weights)

    # We transfer the models to the device and set them to training mode
    G.to(dev)
    D.to(dev)
    G.train()
    D.train()

    # We keep a fixed set of latent vectors to generate images from
    # during training. This allows us to see how the generator improves
    # over time.
    num_images = 64
    fixed_latent_vectors = utils.sample_latent_vectors(num_images, G.num_latent_dims, dev)
    # save some generated images in the untrained state    
    fname_img = f"gen_images/{dataset_name}/G_filters_{G.max_num_filters:04d}_dims_{G.num_latent_dims:04d}"

    # Let the generator create "batch_size" fake images from our fixed latent vectors
    fake_img = G(fixed_latent_vectors)
    save_image_grid(fake_img, fname_img, -1)

    #### TRAINING LOOP ####

    # We train the GAN for num_epochs epochs
    for epoch in range(num_epochs):

        # train one epoch
        tic = time.perf_counter()
        running_loss_G, running_loss_D, throughput_list, batch_count = train_epoch(
            dev, epoch, G, D, train_loader, optimizer_G, optimizer_D)
        toc = time.perf_counter()
       
        # print epoch stats   
        samples_per_sec = torch.mean(torch.tensor(throughput_list))              
        print(
            f"Epoch {epoch:4d}: "
            f"Loss_D {(running_loss_D.item() / batch_count):6.5f} | "
            f"Loss_G {(running_loss_G.item() / batch_count):6.5f} | "
            f"Throughput {samples_per_sec.item():10.2f} images/second | ",
            f"Time {toc - tic:8.3f} (s)"
        )    

        # save the models
        fname_G = f"models/{dataset_name}/G_filters_{G.max_num_filters:04d}_dims_{G.num_latent_dims:04d}"
        fname_D = f"models/{dataset_name}/D_filters_{D.max_num_filters:04d}"
        save_model(G, fname_G, epoch)
        save_model(D, fname_D, epoch)

        # Save progress images
        num_images = 64
        fname_img = f"gen_images/{dataset_name}/G_filters_{G.max_num_filters:04d}_dims_{G.num_latent_dims:04d}"
        
        # Generate images from the fixed latent vectors
        fake_img = G(fixed_latent_vectors)        
        save_image_grid(fake_img, fname_img, epoch)
              

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a VAE with PyTorch.")

    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU (cuda/mps) acceleration")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batchsize", type=int, default=128, help="Batch size for training")
    parser.add_argument("--max_filters", type=int, default=128, help="Maximum number of filters in the convolutional layers")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--dataset", type=str, choices=['mnist', 'cifar-10', 'cifar-100', 'celeb-a'], default='mnist', 
                        help="Select the dataset to use (mnist, cifar-10, cifar-100, celeb-a)")
    parser.add_argument("--latent_dims", type=int, default=100, help="Number of latent dimensions (positive integer)")


     
    args = parser.parse_args()
  
    # Autoselect the device to use
    # We transfer our model and data later to this device. If this is a GPU
    # PyTorch will take care of everything automatically.
    dev = torch.device('cpu')
    if not args.cpu:
        dev = device.autoselectDevice(verbose=1)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Options: ")
    print(f"  Device: {'GPU' if not args.cpu else 'CPU'}")
    print(f"  Seed: {args.seed}")
    print(f"  Batch size: {args.batchsize}")
    print(f"  Max number of filters: {args.max_filters}")
    print(f"  Number of epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Number of latent dimensions: {args.latent_dims}")

    train(dev, args.batchsize, args.epochs, args.lr, args.dataset, args.latent_dims, args.max_filters)
