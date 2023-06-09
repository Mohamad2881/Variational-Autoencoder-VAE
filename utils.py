import torch
from torch import nn

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

def vae_loss_fn(imgs, z_mean, z_log_var, decoded, weight=1):
    
    # Sum across latent dimension
    KL_div_loss  = -0.5 * torch.sum((z_log_var + 1 - z_mean**2 - torch.exp(z_log_var)), dim=1)

    # Take average across batch_size dimension
    KL_div_loss = KL_div_loss.mean()
    
    # print(KL_div_loss)

    # Reconstruction loss
    loss_fn = nn.functional.mse_loss
    # print()
    # calculate mse pixel wise and sum over all pixels and take mean across batch size
    recon_loss = loss_fn(imgs, decoded, reduction='none')
    recon_loss = recon_loss.view(imgs.size(0), -1).sum(1).mean()
    
    # print(recon_loss)

    loss = weight*recon_loss + KL_div_loss

    return loss


def generate_images_from_decoder(model, latent_dim, num_images, device):
    model = model.to(device)
    model.eval()

    with torch.inference_mode():
        random_z = torch.randn(num_images, latent_dim).to(device)
        new_imgs = model.decoder(random_z)
    
    return new_imgs

def generate_images_from_vae(model, images, device):
    model = model.to(device)
    images = images.to(device)
    model.eval()

    with torch.inference_mode():
        _, _, _, new_imgs = model(images)
    
    return new_imgs


def encode_imaages(model, train_dataloader, device):
    model = model.to(device)
    model.eval()
    all_embeddings = torch.Tensor().to(device)

    for (images, _) in tqdm(train_dataloader):
        images = images.to(device)
        
        with torch.inference_mode():
            # Encoder
            encoded = model.encoder(images) 
            # Get mu and sigma from encoder
            z_mean, z_log_var = model.z_mean(encoded), model.z_log_var(encoded)  
            # sample a latent vector z
            z = model.reparameterize(z_mean, z_log_var)

        all_embeddings = torch.cat((all_embeddings, z))    
    
    return all_embeddings


def plot_imgs(imgs, nrows=None, ncols=5, title=None):
    """ 
    Args:
        imgs: tensor of shape: (batch_size, C, H, W)
    """
    
    # send imgs to cpu if needed
    if imgs.is_cuda:
        imgs = imgs.cpu()

    # Make channels last
    imgs = imgs.permute(0, 2, 3, 1)

    if nrows is None:
        nrows = int(np.ceil(len(imgs) // ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4, nrows*4))
    fig.suptitle(title, fontsize=30)
    for (img, ax) in zip(imgs, axs.flatten()):
        ax.imshow(img, cmap='gray')


def plot_latent_dims(embeddings, nrows=None, ncols=10, title=None):
    # send imgs to cpu if needed
    if embeddings.is_cuda:
        embeddings = embeddings.cpu()
    
    if nrows is None:
        nrows = int(np.ceil(embeddings.shape[1] // ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4, nrows*4))
    fig.suptitle(title, fontsize=30)
    
    for (i, ax) in enumerate(axs.flatten()):
        ax.hist(embeddings[:, i])
