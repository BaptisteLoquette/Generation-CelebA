import torch
from torch import nn
import matplotlib.pyplot as plt
from VAE_CelebA import AutoEncoder
import pandas as pd
import numpy as np
import os
from PIL import Image
import torchvision.transforms.functional as TF

device  =   "cuda:0" if torch.cuda.is_available() else "cpu"
model   =   AutoEncoder()
model.load_state_dict(torch.load("models/VAEKinCelebAv0.1.pt"))
model.to(device)

df  =   pd.read_csv("list_attr_celeba.csv")

def get_imgs_latent(dset_path = "../Datasets/CelebA/img_align_celeba/", save=False, path="images_latent.npy"):
    images_latent   =   {}
    model.eval()
    for image_name in df["image_id"].tolist():
        path2image  =   os.path.join(dset_path, image_name)
        img         =   TF.to_tensor(Image.open(path2image))

        mu, logvar  =   model.encoder(img.reshape(1, -1))       #   Encoding the image
        latent      =   model.z(mu, logvar)                     #   Reparametrize the encoding to get the latent vector corresponding to the image

        images_latent[image_name]   =   latent.cpu().detach()
    
    if save:
        np.save(path, images_latent)
    
    return images_latent

def get_attr_latent(images_latent=None, images_latent_path="images_latent.npy", save=False, save_path="attr_latent.npy"):

    if images_latent:
        latent_attr =   {}  #   Initialize the dict

        for attr in list(df.columns)[1:]:   #   Iterating over the attributes
            latent  =   0                   #   Initializing the attribute latent vector representation
            images_ids  =   df[df[attr] == 1]["image_id"].tolist()  #   Retrieving all the images id corresponding with the attribute

            for i, id in enumerate(images_ids):
                latent   +=  images_latent[id]      # Iteratively adding the latent vectors...

            latent_attr[attr]   =   latent/(i+1)    # ...Then dividing to get the Empirical Mean of the latent representation of the attr
    
    elif images_latent_path:
        images_latent   =   np.load(images_latent_path, allow_pickle=True).item()
        latent_attr =   {}  #   Initialize the dict

        for attr in list(df.columns)[1:]:   #   Iterating over the attributes
            latent  =   0                   #   Initializing the attribute latent vector representation
            images_ids  =   df[df[attr] == 1]["image_id"].tolist()  #   Retrieving all the images id corresponding with the attribute

            for i, id in enumerate(images_ids):
                latent   +=  images_latent[id]      # Iteratively adding the latent vectors...

            latent_attr[attr]   =   latent/(i+1)    # ...Then dividing to get the Empirical Mean of the latent representation of the attr
    
    if save:
        np.save(save_path, latent_attr)
    return latent_attr