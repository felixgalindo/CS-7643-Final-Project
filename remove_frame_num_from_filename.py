import glob
import pickle
import os
from tqdm import tqdm

# Change the path to where the files are
# search_dir = '/home/meowater/Documents/ssd_drive/compressed_camera_images2/'
# file_list = glob.glob(search_dir + '*/*.jpg', recursive=True)
search_dir = '/home/meowater/Documents/ssd_drive/cam_box_per_image/'
file_list = glob.glob(search_dir + '*/*.pkl', recursive=True)


# load the pickle file that contains the correct name
for fn in tqdm(file_list):
    _, base_name = os.path.split(fn)
    name_parts = base_name.split('_timestamp')[0]
    remove_pattern = name_parts.split('_')[-1]
    frame_str = remove_pattern + '_'

    new_fn = fn.replace(frame_str, '')
    os.rename(fn, new_fn)
