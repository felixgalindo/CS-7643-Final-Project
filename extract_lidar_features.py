import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Load data
device = "cuda" if torch.cuda.is_available() else "cpu"


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
            nn.ReLU(),

            nn.Conv2d(32, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),
            nn.ReLU(),

            nn.Conv2d(64, out_channels=128, kernel_size=2, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout(0.25),
            nn.ReLU(),

            nn.Conv2d(128, out_channels=256, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(256),
            nn.Dropout(0.25),
            nn.ReLU(),

            nn.Conv2d(256, out_channels=512, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.Dropout(0.25),
            nn.ReLU(),

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, out_channels=256, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, out_channels=128, kernel_size=3, stride=3, padding=0, output_padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, out_channels=64, kernel_size=2, stride=2, padding=2, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, out_channels=32, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, out_channels=input_channel_num, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)



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
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0

        model.train()

        for curr_batch in training_data:
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

def cae_train(training_data: DataLoader, epochs=40, lr=0.00001):

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



if __name__ == "__main__":
    downsample_factor = 4
    threshold = 10
    transform_method = "normalization_without_mask"

    # lidar_data = load_lidar_data(num_images=10, downsample=downsample_factor)
    #
    # # convert lidar data to dataset
    # lidar_data = ProjectedLidarDataset(lidar_data,
    #                                    downsample=downsample_factor,
    #                                    threshold=threshold,
    #                                    transform_method=transform_method)
    # num_samples, height, width = lidar_data.shape
    # input_size_flatten = height * width
    #
    #
    # train_data_loader, _ = create_dataloader(input_lidar=lidar_data, batch_size=8)
    # # trained_model, losses = vae_train(train_data_loader, input_dim=input_size_flatten, latent_dim=256, hidden_dim=512)
    #
    #
    #
    # output_data = ProjectedLidarDataset(output_data, downsample, threshold=threshold, transform_method=transform_method)
    #
    # visualize_lidar_data(trained_model, test_data, downsample=4, recon_height=height, recon_width=width, num_images2show=6)
    #


    # save_path = '/home/meowater/Documents/ssd_drive/VAE_model/'
    # os.makedirs(save_path, exist_ok=True)
    #
    # output_fn = os.path.join(save_path, 'lidar_data_vae_model2.pkl')
    # with open(output_fn, 'wb') as handle:
    #     pickle.dump(trained_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

