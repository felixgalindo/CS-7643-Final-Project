import glob
import tensorflow as tf
import dask.dataframe as dd
from PIL.ImageDraw import ImageDraw
from waymo_open_dataset import v2
import re
import cv2
from io import BytesIO
import os
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import PIL.Image as Image
from PIL import ImageDraw
import pickle
from tqdm import tqdm


dataset_dir = '/home/meowater/Documents/ssd_drive/training/'
save_path = '/home/meowater/Documents/ssd_drive/cam_box_per_image_test/'
os.makedirs(save_path, exist_ok=True)


file_list = glob.glob(dataset_dir + '/camera_box/*parquet', recursive=True)

def read(tag: str, context_name) -> dd.DataFrame:
  """Creates a Dask DataFrame for the component specified by its tag."""
  paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/*{context_name}.parquet')
  return dd.read_parquet(paths)

def extract_box_per_image(input_fn):
    file_path, base_name = os.path.split(input_fn)
    context_name = base_name.replace('.parquet', '')

    cam_image_df = read('camera_image', context_name)
    cam_box_df = read('camera_box', context_name)

    df = v2.merge(cam_box_df, cam_image_df, left_group=True)

    output_path = os.path.join(save_path, context_name)
    os.makedirs(output_path, exist_ok=True)

    for i, (_, r) in enumerate(df.iterrows()):
        # Create component dataclasses for the raw data
        cam_box = v2.CameraBoxComponent.from_dict(r)
        # Only output box data for camera 1
        if cam_box.key.camera_name == 1:
            cam_image = v2.CameraImageComponent.from_dict(r)
            box_coordinates = []

            output_fn = f"{context_name}_camera_image_camera_{cam_image.key.camera_name}_frame-{frame_counter}_timestamp-{cam_image.key.frame_timestamp_micros}"
            for (x, y, w, h) in zip(cam_box.box.center.x, cam_box.box.center.y, cam_box.box.size.x, cam_box.box.size.y):
                box_coordinates.append((x, y, w, h))

            output = {
                "context_name": context_name,
                "camera_name": cam_image.key.camera_name,
                "timestamp": cam_image.key.frame_timestamp_micros,
                "feature_tensor_fn": output_fn + ".pt",
                "box_type": cam_box.type,
                "box_loc": box_coordinates,
            }

            pkl_fn = os.path.join(output_path, output_fn + ".pkl")

            with open(pkl_fn, 'wb') as f:
                pickle.dump(output, f)


for fn in tqdm(file_list):
    frame_counter = extract_box_per_image(fn)
