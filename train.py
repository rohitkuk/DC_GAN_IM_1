# Imports 
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from model import Discrimiator, Generator, initialize_wieghts
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torchvision.utils import make_grid
import os
import shutil
from IPython import get_ipython
import wandb

wandb.init(project="gans", entity="rohitkuk")

shutil.rmtree("logs") if os.path.isdir("logs") else ""

# Hyper Paramerts
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS   = 5
NOISE_DIM    = 100
IMG_DIM      = 64
lr           = 2e-4
BATCH_SIZE   = 128
MAPS_GEN     = 64
MAPS_DISC    = 64
IMG_CHANNELS = 3
FIXED_NOISE  = torch.randn(64, NOISE_DIM, 1, 1).to(DEVICE)


# Transforms
Trasforms = transforms.Compose([
    transforms.Resize(IMG_DIM),
    transforms.CenterCrop(IMG_DIM),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5))
    ])


# Data Loaders
train_dataset = datasets.ImageFolder(root = 'img_align_celeba', transform=Trasforms)
train_loader   = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last=True)


# Model Initializes
generator = Generator(noise_channels=NOISE_DIM, img_channels=IMG_CHANNELS, maps=MAPS_GEN).to(DEVICE)
discremenator = Discrimiator(num_channels=IMG_CHANNELS, maps=MAPS_DISC).to(DEVICE)


# weights Initialize
initialize_wieghts(generator)
initialize_wieghts(discremenator)

# discremenator.apply(initialize_wieghts)
# generator.apply(initialize_wieghts)


# Loss and Optimizers
gen_optim = optim.Adam(params = generator.parameters(), lr=lr, betas=(0.5, 0.999))
disc_optim = optim.Adam(params = discremenator.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()


# Tensorboard Implementation
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")

wandb.watch(generator)
wandb.watch(discremenator)


# Code for COLLAB TENSORBOARD VIEW
try:
    get_ipython().magic("%load_ext tensorboard")
    get_ipython().magic("%tensorboard --logdir logs")
except:
    pass

# training
discremenator.train()
generator.train()
step = 0

for epoch in range(1, NUM_EPOCHS+1):
    tqdm_iter = tqdm(enumerate(train_loader), total = len(train_loader), leave = False)

    for batch_idx, (data, _) in tqdm_iter:
        data = data.to(DEVICE)
        batch_size = data.shape[0]
        
        # ====================== Training the Discremnator===============
        latent_noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(DEVICE)
        fake_img = generator(latent_noise)
        
        disc_fake = discremenator(fake_img.detach()).reshape(-1)
        disc_real = discremenator(data).reshape(-1)

        disc_fake_loss = criterion(disc_fake, torch.zeros_like(disc_fake))
        disc_real_loss = criterion(disc_real, torch.ones_like(disc_real))
        disc_loss = (disc_fake_loss+disc_real_loss)/2

        discremenator.zero_grad()
        disc_loss.backward()
        disc_optim.step()

        # ====================== Training the Generator===============
        # gen_img  = generator(latent_noise)
        output = discremenator(fake_img).reshape(-1)
        gen_loss = criterion(output, torch.ones_like(output))
        generator.zero_grad()
        gen_loss.backward()
        gen_optim.step()
        
        # Logger
        tqdm_iter.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        tqdm_iter.set_postfix(disc_loss = "{0:.4f}".format(disc_loss.item()), gen_loss = "{0:.4f}".format(gen_loss.item()))

        # for Tensorboard
        if batch_idx % 50 == 0:
            GAN_gen = generator(FIXED_NOISE)
            
            img_grid_real = make_grid(data[:32], normalize=True)
            img_grid_fake = make_grid(GAN_gen[:32], normalize=True)

            writer_real.add_image("Real", img_grid_real, global_step=step)
            writer_fake.add_image("Fake", img_grid_fake, global_step=step)
            wandb.log({"Discremenator Loss": disc_loss.item(), "Generator Loss": gen_loss.item()})
            wandb.log({"img": [wandb.Image(img_grid_fake, caption=step)]})
            step +=1 
            torch.save(generator.state_dict(), os.path.join(wandb.run.dir, 'dc_gan_model_gen.pt'))
            torch.save(discremenator.state_dict(), os.path.join(wandb.run.dir, 'dc_gan_model_disc.pt'))

