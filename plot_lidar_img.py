from extract_lidar_features import *
from visualization_lidar import *
from load_lidar_data import *

downsample_factor = 4
threshold = 10
transform_method = "normalization_without_mask"

# Load data to numpy array
lidar_data, selected_list = load_lidar_data(num_images=20, input_h=1280, input_w=1920, downsample=downsample_factor)
# Convert the data to the Dataset class
lidar_data = ProjectedLidarDataset(lidar_data, downsample=downsample_factor, transform_method=transform_method, threshold=threshold)


# Calculate the input size
num_samples, height, width = lidar_data.shape

# Create dataloader
train_data_loader = create_dataloader(input_lidar=lidar_data, batch_size=4)


model_path = '/home/meowater/Documents/ssd_drive/CAE_models/lidar_data_cae_model.pkl'

with open(model_path, 'rb') as f:
    model, _ = pickle.load(f)

for ind in range(20):
    input_data = lidar_data[ind].unsqueeze(0)
    visualize_cae_result_with_image(model, input_data, selected_list[ind], recon_height=height, recon_width=width)


