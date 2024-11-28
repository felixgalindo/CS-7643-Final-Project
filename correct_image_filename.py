import glob
import pickle
import os
from tqdm import tqdm

# Change the path to where the camera images are
image_path = '/home/meowater/Documents/ssd_drive/compressed_camera_images/'
img_list = glob.glob(image_path + '*/*.jpg', recursive=True)

# Change the path to where the name correction pkl files are
correction_path = '/home/meowater/Documents/ssd_drive/name_correction/'
correction_list = glob.glob(os.path.join(correction_path, '*correction.pkl'), recursive=True)

# load the pickle file that contains the correct name
for correction_fn in tqdm(correction_list):
    _, base_name = os.path.split(correction_fn)

    context_name = base_name.split('_correction')[0]
    with open(correction_fn, 'rb') as handle:
      timestamp_list = pickle.load(handle)

    for timestamps in timestamp_list:
        search_str = os.path.join(image_path, context_name, 'camera*timestamp-' + timestamps["pose_timestamp"] + '*.jpg')
        matching_fn = glob.glob(search_str)

        # only change the name if there is a matching file
        if len(matching_fn) == 1:
            matching_fn = str(matching_fn[0])
            new_fn = matching_fn.replace(timestamps["pose_timestamp"], timestamps["frame_timestamp"])

            os.rename(matching_fn, new_fn)