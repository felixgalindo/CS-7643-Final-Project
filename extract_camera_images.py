import glob
import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2
import re
from io import BytesIO
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
from tqdm import tqdm

# Set up raw data file and save path
dataset_dir = '/home/meowater/Documents/ssd_drive/training/'
save_path = '/home/meowater/Documents/ssd_drive/compressed_camera_images_test/'


def read(tag: str, context_name) -> dd.DataFrame:
    """Creates a Dask DataFrame for the component specified by its tag."""
    paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/*{context_name}.parquet')
    return dd.read_parquet(paths)

def extract_camera_images(fullpath):
    file_path, base_name = os.path.split(fullpath)
    context_name = base_name.replace('.parquet', '')

    # read the data
    cam_image_df = read('camera_image', context_name)

    # create output path
    output_path = os.path.join(save_path, context_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterate through the rows to extract image data
    counter = 0
    for i, (_, r) in enumerate(cam_image_df.iterrows()):
        # Create component dataclasses for the raw data
        cam_image = v2.CameraImageComponent.from_dict(r)
        timestamp = cam_image.key.frame_timestamp_micros

        # only save the image from front camera
        if cam_image.key.camera_name == 1:
            output_fn = os.path.join(output_path, f'camera_image_camera-{cam_image.key.camera_name}_frame-{counter}_timestamp-{timestamp}.jpg')
            counter += 1
            img = Image.open(BytesIO(cam_image.image))
            img.save(output_fn, 'JPEG', quality=50, optimize=True)

# Find the list of file to process through
file_list = glob.glob(dataset_dir + 'camera_image/*parquet', recursive=True)



for fn in file_list:
    extract_camera_images(fn)

