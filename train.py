
# Imports 
import torch
import torch.nn as nn
from torch.optim import optimizer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from model import Discrimiator, Generator, initialize_wieghts
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torchvision.utils import make_grid



# Hyper Paramerts
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS   = 1
NOISE_DIM    = 100
IMG_DIM      = 64
lr           = 4e-4
BATCH_SIZE   = 128
MAPS_GEN     = 64
MAPS_DISC    = 64
IMG_CHANNELS = 1
FIXED_NOISE  = torch.randn(32, NOISE_DIM, 1, 1).to(DEVICE)


# Transforms
Trasforms = transforms.Compose([
    transforms.Resize(IMG_DIM),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.5 for i in range(IMG_CHANNELS)],
        std  = [0.5 for i in range(IMG_CHANNELS)]
    )
])


# Data Loaders
train_dataset = datasets.MNIST(root = 'dataset', download=True, transform=Trasforms)
train_loader   = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, drop_last=True)


# Model Initializes
generator = Generator(noise_channels=NOISE_DIM, img_channels=IMG_CHANNELS, maps=MAPS_GEN).to(DEVICE)
discremenator = Discrimiator(num_channels=IMG_CHANNELS, maps=MAPS_DISC).to(DEVICE)


# weights Initialize
initialize_wieghts(generator)
initialize_wieghts(discremenator)


# Loss and Optimizers
gen_optim = optim.Adam(params = generator.parameters(), lr=lr, betas=(0.5, 0.999))
disc_optim = optim.Adam(params = discremenator.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()


# Tensorboard Implementation
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")



# training
discremenator.train()
generator.train()
step = 1

for epoch in range(1, NUM_EPOCHS+1):
    tqdm_iter = tqdm(enumerate(train_loader), total = len(train_loader), leave = False)

    for batch_idx, (data, _) in tqdm_iter:
        data = data.to(DEVICE)
        batch_size = data.shape[0]
        
        # ====================== Training the Discremnator===============
        latent_noise = torch.randn(batch_size, NOISE_DIM, 1, 1).to(DEVICE)
        fake_img = generator(latent_noise)
        
        disc_fake = discremenator(fake_img)
        disc_real = discremenator(data)

        disc_fake_loss = criterion(disc_fake, torch.zeros_like(disc_fake))
        disc_real_loss = criterion(disc_real, torch.zeros_like(disc_real))
        disc_loss = (disc_fake_loss+disc_real_loss)/2

        discremenator.zero_grad()
        disc_loss.backward(retain_graph = True)
        disc_optim.step()

        # ====================== Training the Generator===============
        gen_img = generator(latent_noise)
        gen_loss = criterion(gen_img, torch.ones_like(gen_img))
        generator.zero_grad()
        gen_loss.backward()
        gen_optim.step()
        
        # Logger
        tqdm_iter.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        tqdm_iter.set_postfix(disc_loss = "{0:.0}".format(disc_loss.item()), gen_loss = "{0:.0%}".format(gen_loss.item()))

        # for Tensorboard
        if batch_idx % 400 == 0:
            GAN_gen = generator(FIXED_NOISE)
            
            img_grid_real = make_grid(data[:32], normalize=True)
            img_grid_fake = make_grid(GAN_gen[:32], normalize=True)

            writer_real.add_image("Real", img_grid_real, global_step=1)
            writer_fake.add_image("Fake", img_grid_fake, global_step=1)
            step +=1 






        



