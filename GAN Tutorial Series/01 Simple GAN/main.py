import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
from time import sleep

# Things to try:
# 1. What happens with a larger network?
# 2. Better Normalization with BatchNorm
# 3. Different learning rate (is there a better one?)
# 4. Change architecture to a CNN => Tutorial #2

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128), nn.LeakyReLU(0.1), nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),  # 28x28x1 => 784
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


def img(img, e, i):
    img = img.cpu().numpy()
    plt.figure(figsize=(25, 25))
    plt.imshow(np.transpose(img, (1, 2, 0)), interpolation="nearest", cmap="gray")
    plt.savefig(f"image/e{e}-i{i}.png")
    plt.show()


# Hyperparams etc.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64  # Try 128, 256, etc
image_dim = 28 * 28 * 1  # H * W * C
batch_size = 32
num_epochs = 50

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
dataset = datasets.MNIST(root="./../_datasets/dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
step = 0

# Train Loop
with trange(num_epochs, unit="e") as tepoch:
    for epoch in tepoch:
        for batch_idx, (real, _) in enumerate(loader):
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]

            # Train Discriminator
            # Maximise log(D(real)) + log(1 - D(G(z))
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            # Train Generator
            # Minimize min log(1 - D(G(z))) => max log(D(G(z)))
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()
            tepoch.set_postfix(lossD=lossD.item(), lossG=lossG.item())

            if batch_idx % 100 == 0:
                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                    data = real.reshape(-1, 1, 28, 28)
                    img_grid_fake = torchvision.utils.make_grid(
                        fake, normalize=True, padding=5
                    )
                    img(img_grid_fake, epoch + 1, batch_idx)
