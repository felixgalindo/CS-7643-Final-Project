import torch
import matplotlib.pyplot as plt
import os
from PIL import Image
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

def visualize_cae_result_with_image(cae_model, input_data, input_fn, recon_height, recon_width, device='cuda'):

    cae_model.eval()
    with torch.no_grad():

        x = torch.FloatTensor(input_data).to(device)
        reshaped_x = x.unsqueeze(1)
        recon= cae_model(reshaped_x)
        print(f"Reconstructed shape is {recon.shape}")
        recon = recon.squeeze(1)
        recon = recon.view(1, recon_height, recon_width)
        print(f"Reconstructed shape after squeeze is {recon.shape}")

        base_name = os.path.basename(input_fn)
        context_name = base_name.split('_camera_image')[0]

        base_name = base_name.replace(context_name + '_', '')
        base_name = base_name.replace('.pkl', '.jpg')
        base_name = base_name.replace('_camera_1_', '_camera-1_')

        img_dir = str(os.path.dirname(input_fn).replace('lidar_projected', 'compressed_camera_images2'))
        img_path = os.path.join(img_dir, base_name)
        print(img_path)
        plot_original_reconstruction_lidar_with_img(x, recon, img_path)

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


        axes[0, curr_img].set_title("Original")

        axes[1, curr_img].imshow(x_recon[curr_img].cpu(), cmap=colormap, vmin=cmap_min, vmax=cmap_max)
        axes[1, curr_img].axis('off')
        axes[1, curr_img].set_title("Reconstructed")

    plt.tight_layout()
    plt.show()


def plot_original_reconstruction_lidar_with_img(x_og, x_recon, img_path, colormap='nipy_spectral', cmap_min=0, cmap_max=1):
    fig, axes = plt.subplots(1, 3, figsize=(6, 3))

    img = Image.open(img_path)
    axes[0].imshow(img.resize((320, 320)))
    axes[0].axis('off')
    axes[0].set_title("Camera image")


    axes[1].imshow(x_og.squeeze().cpu(), cmap=colormap, vmin=cmap_min, vmax=cmap_max)
    axes[1].axis('off')
    axes[1].set_title("Original\nLiDAR")

    axes[2].imshow(x_recon.squeeze().cpu(), cmap=colormap, vmin=cmap_min, vmax=cmap_max)
    axes[2].axis('off')
    axes[2].set_title("Reconstructed\nLiDAR")

    plt.tight_layout()
    plt.show()