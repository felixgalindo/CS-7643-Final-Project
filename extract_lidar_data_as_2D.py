import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2
import glob
import os
import pickle
from waymo_open_dataset.utils.range_image_utils import build_camera_depth_image
from waymo_open_dataset.v2.perception.utils import lidar_utils
from tqdm import tqdm

dataset_dir = '/home/meowater/Documents/ssd_drive/training/'
search_path = '/home/meowater/Documents/ssd_drive/training/lidar/'
file_list = glob.glob(search_path + '*.parquet')
save_path = '/home/meowater/Documents/ssd_drive/lidar_projected/'

def read(tag: str, context_name) -> dd.DataFrame:
  """Creates a Dask DataFrame for the component specified by its tag."""
  paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/*{context_name}.parquet')
  return dd.read_parquet(paths)

def get_extrinsic(calibration):
  """Projects from vehicle coordinate system to image with global shutter.

  Arguments:
    calibration: Camera calibration details (including intrinsics/extrinsics).
  Returns:
    Array of shape [N, 3], with the latter dimension composed of (u, v, ok).
  """

  # Populate camera image metadata. Velocity and latency stats are filled with
  # zeroes.
  extrinsic = tf.reshape(
      tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32),
      [1, 4, 4])

  # Perform projection and return projected image coordinates (u, v, ok).
  return extrinsic
save_path
def get_data_frames(context_name):
    cam_img_df = read('camera_image', context_name)
    lidar_df = read('lidar', context_name)
    lidar_calib_df = read('lidar_calibration', context_name)
    camera_calib_df = read('camera_calibration', context_name)
    projections_df = read('lidar_camera_projection', context_name)

    return {
        "cam_img_df": cam_img_df,
        "lidar_df": lidar_df,
        "lidar_calib_df": lidar_calib_df,
        "camera_calib_df": camera_calib_df,
        "projections_df": projections_df,
    }

def extract_data(cam_img_df, lidar_df, projections_df, camera_calib_df, lidar_calib_df):
    # only use the first items
    _, r_cam_calib = next(iter(camera_calib_df.iterrows()))
    cam_config = v2.CameraCalibrationComponent.from_dict(r_cam_calib)

    _, r_lidar_calib = next(iter(lidar_calib_df.iterrows()))
    lidar_config = v2.LiDARCalibrationComponent.from_dict(r_lidar_calib)

    camera_extrinsic = get_extrinsic(cam_config)
    camera_image_size = [cam_config.height, cam_config.width]

    basic_info_list = []
    lidar_projection_list = None
    for i, ((_, r_cam), (_, r_lidar), (_, r_project)) in enumerate(zip(cam_img_df.iterrows(),
                                                       lidar_df.iterrows(),
                                                       projections_df.iterrows())):

        cam_img = v2.CameraImageComponent.from_dict(r_cam)

        if cam_img.key.camera_name == 1:
            # get basic info
            basic_info = {
                "context_name": context_name,
                "camera_name": cam_img.key.camera_name,
                "timestamp": str(cam_img.key.frame_timestamp_micros),
            }
            basic_info_list.append(basic_info)

            lidar = v2.LiDARComponent.from_dict(r_lidar)
            projections = v2.LiDARCameraProjectionComponent.from_dict(r_project)
            camera_image_size = [cam_config.height, cam_config.width]
            lidar_coords = lidar_utils.convert_range_image_to_cartesian(lidar.range_image_return1, lidar_config)


            project_array = projections.range_image_return1.tensor

            lidar_coords = tf.expand_dims(lidar_coords, axis=0)
            project_array = tf.expand_dims(project_array, axis=0)

            lidar2d = build_camera_depth_image(lidar_coords, camera_extrinsic,
                                    camera_projection=project_array,
                                    camera_image_size=camera_image_size,
                                    camera_name=1)
            if lidar_projection_list is None:
                lidar_projection_list = lidar2d
            else:
                lidar_projection_list = tf.concat([lidar_projection_list, lidar2d], axis=0)

    return lidar_projection_list, basic_info_list
for fn in tqdm(file_list[14:]):
    file_path, base_name = os.path.split(fn)
    context_name = base_name.replace('.parquet', '')
    dataframes = get_data_frames(context_name)
    lidar_projected, basic_infos = extract_data(dataframes["cam_img_df"], dataframes["lidar_df"],
                           dataframes["projections_df"], dataframes["camera_calib_df"],
                           dataframes["lidar_calib_df"])

    lidar_projected = lidar_projected.numpy()

    sub_folder = os.path.join(save_path, context_name)
    if not os.path.exists(sub_folder):
        os.mkdir(sub_folder)

    for ind, basic_info in enumerate(basic_infos):
        output_basename = context_name + '_camera_image_camera_1_timestamp-' + basic_info["timestamp"] + '.pkl'
        output_fn = os.path.join(sub_folder, output_basename)

        basic_info["lidar_projection"] = lidar_projected[ind, :, :].squeeze()

        with open(output_fn, 'wb') as f:
            pickle.dump(basic_info, f)


