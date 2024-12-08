import torch
import matplotlib.pyplot as plt

def visualize_vae_result(train_model, testing_data, recon_height, recon_width, num_images2show=6, device='cuda'):

    train_model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(testing_data[0:num_images2show]).to(device)
        x_flatten = x.view(num_images2show, -1)

        recon, _, _ = train_model(x_flatten)
        recon = recon.view(num_images2show, recon_height, recon_width)

        plot_original_reconstruction_lidar(x, recon, num_images2show)

def visualize_cae_result(cae_model, testing_data, recon_height, recon_width, num_images2show=6, device='cuda'):

    cae_model.eval()
    with torch.no_grad():

        x = torch.FloatTensor(testing_data[0:num_images2show]).to(device)
        reshaped_x = x.unsqueeze(1)
        recon= cae_model(reshaped_x)
        print(f"Reconstructed shape is {recon.shape}")
        recon = recon.squeeze(1)
        recon = recon.view(num_images2show, recon_height, recon_width)
        print(f"Reconstructed shape after squeeze is {recon.shape}")

        plot_original_reconstruction_lidar(x, recon, num_images2show)

def visualize_lidar_data_no_model(input_lidar, num_images2show=6, colormap='viridis', vmin=0, vmax=1, device='cuda'):
    x = torch.FloatTensor(input_lidar[0:num_images2show]).to(device)
    fig, axes = plt.subplots(1, num_images2show, figsize=(3 * num_images2show, 3))

    for curr_img in range(num_images2show):
        axes[curr_img].imshow(x[curr_img].cpu(), cmap=colormap, vmin=vmin, vmax=vmax)
        axes[curr_img].axis('off')

        if curr_img == 3:
            axes[curr_img].set_title("Original")
    plt.tight_layout()
    plt.show()


def plot_original_reconstruction_lidar(x_og, x_recon, num_col, colormap='nipy_spectral', cmap_min=0, cmap_max=1):
    fig, axes = plt.subplots(2, num_col, figsize=(3 * num_col, 6))

    for curr_img in range(num_col):
        axes[0, curr_img].imshow(x_og[curr_img].cpu(), cmap=colormap, vmin=cmap_min, vmax=cmap_max)
        axes[0, curr_img].axis('off')

        if curr_img == 3:
            axes[0, curr_img].set_title("Original")

        axes[1, curr_img].imshow(x_recon[curr_img].cpu(), cmap=colormap, vmin=cmap_min, vmax=cmap_max)
        axes[1, curr_img].axis('off')
        if curr_img == 3:
            axes[1, curr_img].set_title("Reconstructed")

    plt.tight_layout()
    plt.show()
