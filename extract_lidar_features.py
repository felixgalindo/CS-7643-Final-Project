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

# Class to include the lidar data
# Preprocess the data
class ProjectedLidarDataset(Dataset):
    def __init__(self, input_lidar, downsample_factor, transform_method, threshold):
        data = torch.from_numpy(input_lidar).float()

        self.num_samples = torch.tensor(data.shape[0], dtype=torch.int64)
        self.height = torch.tensor(data.shape[1], dtype=torch.int64)
        self.width = torch.tensor(data.shape[2], dtype=torch.int64)
        self.shape = torch.tensor(data.shape, dtype=torch.int64)
        self.downsample_factor = downsample_factor

        non_zero_mask = data != 0
        large_value_mask = data < threshold
        self.non_zero_mask = torch.logical_and(non_zero_mask, large_value_mask)
        self.zeros_mask = ~self.non_zero_mask

        if transform_method == "normalization_without_mask":
            self.data_mu = torch.mean(data)
            self.data_std = torch.std(data)
            self.data = (data - self.data_mu) / (self.data_std + 1e-8)
            self.data = (self.data + 3) / 6
            self.data = torch.clamp(self.data, 0, 1)
            self.data[self.zeros_mask] = 0
        elif transform_method == "normalization_with_mask":

            self.data_mu = torch.mean(data[self.non_zero_mask])
            self.data_std = torch.std(data[self.non_zero_mask])
            self.data = (data - self.data_mu) / (self.data_std + 1e-8)
            self.data = (self.data + 3) / 6
            self.data = torch.clamp(self.data, 0, 1)
            self.data[self.zeros_mask] = 0
        elif transform_method == "log_normalization_with_mask":
            data = torch.log(data + 1e-6)
            self.data_mu = torch.mean(data[self.non_zero_mask])
            self.data_std = torch.std(data[self.non_zero_mask])
            self.data = (data - self.data_mu) / (self.data_std + 1e-8)
            self.data = (self.data + 3) / 6
            self.data = torch.clamp(self.data, 0, 1)
            self.data[self.zeros_mask] = 0


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index]

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=(1024, 512)):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.BatchNorm1d(hidden_dim[0]),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.BatchNorm1d(hidden_dim[1]),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim[1], hidden_dim[1]),
            nn.BatchNorm1d(hidden_dim[1]),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
        )

        # Latent space
        self.fc_mu = nn.Linear(hidden_dim[1], latent_dim)
        self.fc_var = nn.Linear(hidden_dim[1], latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim[1]),
            nn.BatchNorm1d(hidden_dim[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[1]),
            nn.BatchNorm1d(hidden_dim[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[0]),
            nn.BatchNorm1d(hidden_dim[0]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim[0], input_dim),
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

class CAE(nn.Module):
    def __init__(self, input_channel_num = 1):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(

            nn.Conv2d(input_channel_num, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.25),
            nn.LeakyReLU(),

            nn.Conv2d(32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.LeakyReLU(),

            nn.Conv2d(64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.25),
            nn.LeakyReLU(),

            nn.Conv2d(128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout(0.25),
            nn.LeakyReLU(),

            nn.Conv2d(256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.25),
            nn.LeakyReLU(),

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),


            nn.ConvTranspose2d(32, out_channels=input_channel_num, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),

        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

def load_lidar_data(num_images=1000, input_h=1280, input_w=1920, downsample=4, threshold=10,
                    transform_method="normalization_without_mask") -> ProjectedLidarDataset:
    data_path = '/home/meowater/Documents/ssd_drive/lidar_projected/'
    data_list = glob.glob(os.path.join(data_path, '*/*.pkl'), recursive=True)
    print(len(data_list))
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
    output_data = ProjectedLidarDataset(output_data, downsample, threshold=threshold, transform_method=transform_method)
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

    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + beta * kld



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

def cae_train(training_data, epochs=40, lr=0.00001):

    model = CAE().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    losses = []
    best_loss = float('inf')
    best_state = None
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch in tqdm(training_data, desc=f"Epoch {epoch+1}/{epochs}"):

            batch = batch.to(device)
            batch = batch.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / training_data.__len__()
        losses.append(avg_loss)

        schedular.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict()

        if (epoch + 1) % 5 == 0:
            print(f"For epoch {epoch + 1}/{epochs}, avg_loss = {avg_loss}.")

    model.load_state_dict(best_state)
    return model, losses

def visualize_vae_result(train_model, testing_data, downsample, recon_height, recon_width, num_images2show=6):

    train_model.eval()
    with torch.no_grad():

        x = torch.FloatTensor(testing_data[0:num_images2show]).to(device)
        x_flatten = x.view(num_images2show, -1)

        recon, _, _ = train_model(x_flatten)
        recon = recon.view(num_images2show, recon_height, recon_width)

        fig, axes = plt.subplots(2, num_images2show, figsize=(3*num_images2show, 6))

        for curr_img in range(num_images2show):
            axes[0, curr_img].imshow(x[curr_img].cpu(), cmap='viridis', vmin=0, vmax=1)
            axes[0, curr_img].axis('off')

            if curr_img == 3:
                axes[0, curr_img].set_title("Original")

            axes[1, curr_img].imshow(recon[curr_img].cpu(), cmap='viridis', vmin=0, vmax=1)
            axes[1, curr_img].axis('off')
            if curr_img == 3:
                axes[0, curr_img].set_title("Reconstructed")

        plt.tight_layout()
        plt.show()

def visualize_cae_result(cae_model, testing_data, downsample, recon_height, recon_width, num_images2show=6):

    cae_model.eval()
    with torch.no_grad():

        x = torch.FloatTensor(testing_data[0:num_images2show]).to(device)
        reshaped_x = x.unsqueeze(1)
        recon= cae_model(reshaped_x)
        print(f"Reconstructed shape is {recon.shape}")
        recon = recon.squeeze(1)
        recon = recon.view(num_images2show, recon_height, recon_width)
        print(f"Reconstructed shape after squeeze is {recon.shape}")
        fig, axes = plt.subplots(2, num_images2show, figsize=(3*num_images2show, 6))

        for curr_img in range(num_images2show):
            axes[0, curr_img].imshow(x[curr_img].cpu(), cmap='viridis', vmin=0, vmax=1)
            axes[0, curr_img].axis('off')

            if curr_img == 3:
                axes[0, curr_img].set_title("Original")

            axes[1, curr_img].imshow(recon[curr_img].cpu(), cmap='viridis', vmin=0, vmax=1)
            axes[1, curr_img].axis('off')
            if curr_img == 3:
                axes[0, curr_img].set_title("Reconstructed")

        plt.tight_layout()
        plt.show()

def visualize_lidar_data_no_model(input_lidar, num_images2show=6, vmin=0, vmax=1):
    x = torch.FloatTensor(input_lidar[0:num_images2show]).to(device)
    fig, axes = plt.subplots(1, num_images2show, figsize=(3 * num_images2show, 3))

    for curr_img in range(num_images2show):
        axes[curr_img].imshow(x[curr_img].cpu(), cmap='viridis', vmin=vmin, vmax=vmax)
        axes[curr_img].axis('off')

        if curr_img == 3:
            axes[curr_img].set_title("Original")
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

