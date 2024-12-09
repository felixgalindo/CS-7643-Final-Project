from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import glob
import os
import random
import time
import pickle
from skimage.transform import resize
from torch.utils.data import SubsetRandomSampler

# Class to include the lidar data
# Preprocess the data
class ProjectedLidarDataset(Dataset):
    def __init__(self, input_lidar, downsample, transform_method, threshold):
        data = torch.from_numpy(input_lidar).float()

        self.num_samples = torch.tensor(data.shape[0], dtype=torch.int64)
        self.height = torch.tensor(data.shape[1], dtype=torch.int64)
        self.width = torch.tensor(data.shape[2], dtype=torch.int64)
        self.shape = torch.tensor(data.shape, dtype=torch.int64)
        self.downsample_factor = downsample

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
        else:
            self.data = data


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index]


def load_lidar_data(num_images=1000,
                    input_h=1280, input_w=1920, downsample=4) -> np.array:

    data_path = '/home/meowater/Documents/ssd_drive/lidar_projected/'
    data_list = glob.glob(os.path.join(data_path, '*/*.pkl'), recursive=True)
    selected_list = random.sample(data_list, num_images)

    start_time = time.time()

    reduced_height = round(input_h // downsample)
    # reduced_width = round(input_w // downsample)

    output_data = np.zeros((num_images, reduced_height, reduced_height))

    for ind, fn in enumerate(selected_list):
        with open(fn, 'rb') as handle:
            projected_lidar = pickle.load(handle)

        output_data[ind] = resize(projected_lidar["lidar_projection"],
                                  output_shape=(reduced_height, reduced_height),
                                  anti_aliasing=True)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print("Elapsed time to load the lidar data: ", elapsed_time)
    return output_data



# data loader ofr the data
def create_dataloader(input_lidar: ProjectedLidarDataset, batch_size: int =8, num_workers: int =4) \
        -> torch.utils.data.DataLoader:
    # divide up the data into train and validation
    data_indices = torch.arange(input_lidar.num_samples)

    # shuffle the indices and split the data
    shuffled_indices = SubsetRandomSampler(data_indices)
    train_loader = DataLoader(input_lidar, batch_size=batch_size, num_workers=num_workers, sampler=shuffled_indices)

    return train_loader