import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import glob
import random
import pickle
import numpy as np
import time
from skimage.transform import resize
from torch.utils.data import SubsetRandomSampler
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
# Load data
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use VAE for autoencoder
class ProjectedLidarDataset(Dataset):
    def __init__(self, input_lidar, downsample_factor):
        data = torch.from_numpy(input_lidar).float()
        self.num_samples = torch.tensor(data.shape[0], dtype=torch.int64)
        self.height = torch.tensor(data.shape[1], dtype=torch.int64)
        self.width = torch.tensor(data.shape[2], dtype=torch.int64)
        self.shape = torch.tensor(data.shape, dtype=torch.int64)
        self.downsample_factor = downsample_factor

        # normalize the data
        self.data_mu = torch.mean(data)
        self.data_std = torch.std(data)
        self.data = (data - self.data_mu)/(self.data_std + 1e-6)
        self.data = (self.data + 3)/6
        self.data = torch.clamp(self.data, 0, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index]

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=1024):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
        )

        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        y = self.encoder(x)
        mu = self.fc_mu(y)
        log_var = self.fc_var(y)
        return mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def load_lidar_data(num_images=1000, input_h=1280, input_w=1920, downsample=4) -> ProjectedLidarDataset:
    data_path = '/home/meowater/Documents/ssd_drive/lidar_projected/'
    data_list = glob.glob(os.path.join(data_path, '*/*.pkl'), recursive=True)

    selected_list = random.sample(data_list, num_images)

    start_time = time.time()

    reduced_height = input_h // downsample
    reduced_width = input_w // downsample

    output_data = np.zeros((num_images, reduced_height, reduced_width))

    for ind, fn in enumerate(selected_list):
        with open(fn, 'rb') as handle:
            projected_lidar = pickle.load(handle)

        output_data[ind] = resize(projected_lidar["lidar_projection"],
                                  output_shape=(reduced_height, reduced_width),
                                  anti_aliasing=True)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print("Elapsed time to load the lidar data: ", elapsed_time)
    output_data = ProjectedLidarDataset(output_data, downsample)
    return output_data


# data loader ofr the data
def create_dataloader(input_lidar: ProjectedLidarDataset, batch_size=8, num_workers=4, train_ratio=0.9):
    data_indices = torch.arange(input_lidar.num_samples)

    num_train_samples = torch.round(train_ratio * input_lidar.num_samples).int()

    train_indices = data_indices[:num_train_samples]
    val_indices = data_indices[num_train_samples:]


    # shuffle the indices and split the data
    train_subset = SubsetRandomSampler(train_indices)
    val_subset = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(input_lidar, batch_size=batch_size, num_workers=num_workers, sampler=train_subset)
    val_loader = DataLoader(input_lidar, batch_size=batch_size, num_workers=num_workers, sampler=val_subset)

    return train_loader, val_loader


def loss_function(recon_x, x, mu, log_var, beta=0.1):

    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + beta * KLD



def vae_train(training_data, input_dim, epochs=40, lr=0.00001,
              latent_dim=512, hidden_dim=1024,
              early_stopping_trials = 20, loss_beta=0.1):
    best_loss = float('inf')

    epochs_without_improvement = 0
    model = VAE(input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0

        model.train()

        for curr_batch in tqdm(training_data, desc=f"Epoch {epoch+1}/{epochs}"):
            x = curr_batch.view(-1, input_dim).to(device)

            # Forward
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(x)

            total_loss = loss_function(recon_batch, x, mu, log_var, beta=loss_beta)

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / training_data.__len__()
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_trials:
                print("Early stopping as the loss stopped improving. ")

                return model, losses
        if (epoch + 1) % 5 == 0:
            print(f"For epoch {epoch + 1}/{epochs}, avg_loss = {avg_loss}.")

    return model, losses

def visualize_lidar_data(train_model, testing_data, downsample, recon_height, recon_width, num_images2show=6):

    train_model.eval()
    with torch.no_grad():

        x = torch.FloatTensor(testing_data[0:num_images2show]).to(device)
        x_flatten = x.view(num_images2show, -1)

        recon, _, _ = train_model(x_flatten)
        recon = recon.view(num_images2show, recon_height, recon_width)

        fig, axes = plt.subplots(2, num_images2show, figsize=(3*num_images2show, 6))

        for curr_img in range(num_images2show):
            axes[0, curr_img].imshow(x[curr_img].cpu(), cmap='viridis')
            axes[0, curr_img].axis('off')

            if curr_img == 3:
                axes[0, curr_img].set_title("Original")

            axes[1, curr_img].imshow(recon[curr_img].cpu(), cmap='viridis')
            axes[1, curr_img].axis('off')
            if curr_img == 3:
                axes[0, curr_img].set_title("Reconstructed")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    lidar_data = load_lidar_data(num_images=10)
    num_samples, height, width = lidar_data.shape
    input_size_flatten = height * width

    load_model = '/home/meowater/Documents/ssd_drive/VAE_model/lidar_data_vae_model.pkl'

    with open(load_model, 'rb') as f:
        trained_model = pickle.load(f)

    # train_data_loader, _ = create_dataloader(input_lidar=lidar_data, batch_size=8)
    # trained_model, losses = vae_train(train_data_loader, input_dim=input_size_flatten, latent_dim=256, hidden_dim=512)
    test_data = load_lidar_data(num_images=6)
    visualize_lidar_data(trained_model, test_data, downsample=4, recon_height=height, recon_width=width, num_images2show=6)



    # save_path = '/home/meowater/Documents/ssd_drive/VAE_model/'
    # os.makedirs(save_path, exist_ok=True)
    #
    # output_fn = os.path.join(save_path, 'lidar_data_vae_model2.pkl')
    # with open(output_fn, 'wb') as handle:
    #     pickle.dump(trained_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

