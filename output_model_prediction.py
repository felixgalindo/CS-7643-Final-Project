import optuna
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from mm_fusion_detector import MMFusionDetector, MMFusionDetectorDataset, custom_collate
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch.nn.functional as F
from torchvision.ops import box_iou
from torchvision.ops import generalized_box_iou_loss
import matplotlib.pyplot as plt
import cv2
import re
from trainer import convert_to_corners
import pickle

def parse_model_params_from_filename(filename):
    """
    Extract model parameters from the filename.
    """
    pattern = r"dim(\d+)_heads(\d+)_layers(\d+)_epochs(\d+)_lr([0-9.e+-]+)_wd([0-9.e+-]+)_alpha([0-9.e+-]+)_beta([0-9.e+-]+)_delta([0-9.e+-]+)_boxacc([0-9.e+-]+)\.pth"
    match = re.search(pattern, filename)
    if not match:
        raise ValueError("Filename does not match expected pattern.")

    params = {
        "model_dim": int(match.group(1)),
        "num_heads": int(match.group(2)),
        "num_layers": int(match.group(3)),
        "epochs": int(match.group(4)),
        "lr": float(match.group(5)),
        "weight_decay": float(match.group(6)),
        "alpha": float(match.group(7)),
        "beta": float(match.group(8)),
        "delta": float(match.group(9)),
        "box_accuracy": float(match.group(10)),
    }
    return params


def load_model_from_file(model_path, device):
    """
    Load the model from a .pth file, parsing parameters from the filename.
    """
    filename = os.path.basename(model_path)
    params = parse_model_params_from_filename(filename)
    model = MMFusionDetector(
        model_dim=params["model_dim"],
        num_heads=params["num_heads"],
        num_layers=params["num_layers"],
        alpha=params["alpha"],
        beta=params["beta"],
        delta=params["delta"],
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, params

def generate_predictions(
    model, dataloader, output_dir="visualizations", num_images=5, image_size=(1920, 1280)

):
    """
    Visualize predictions by overlaying predicted and ground truth boxes on images.
    Includes IoU and L1 values for matched predictions.
    Processes only images with 1-3 ground truth vehicle boxes.

    Args:
        model: Trained model.
        dataloader: DataLoader providing input data with image paths.
        output_dir: Directory to save visualizations.
        num_images: Number of images to visualize.
        image_size: Tuple containing the (width, height) of the images for scaling predictions.
    """
    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        processed_images = 0
        for i, (features, ground_truth) in enumerate(tqdm(dataloader)):
            if processed_images >= num_images:
                break

            # Filter ground truth for vehicle classes (class 1)
            vehicle_boxes = []
            vehicle_classes = []
            for cls, box in zip(ground_truth["classes"], ground_truth["boxes"]):
                if cls == 1:  # Only vehicles
                    vehicle_classes.append(cls)
                    vehicle_boxes.append(box)

            # Skip if no vehicle boxes found
            if not vehicle_boxes:
                continue

            # Convert lists to tensors
            ground_truth["classes"] = torch.tensor(vehicle_classes)
            ground_truth["boxes"] = torch.tensor(vehicle_boxes)

            processed_images += 1

            try:
                sub_dir = os.path.join(output_dir, ground_truth["context_name"])
                os.makedirs(sub_dir, exist_ok=True)

                name_prefix = ground_truth["feature_tensor_fn"]
                pattern = name_prefix.split("_timestamp")[0]
                pattern = pattern.split("_camera_1")[-1]

                # Load the image
                prediction_fn = os.path.join(
                    sub_dir,
                    ground_truth["feature_tensor_fn"]
                        .replace(pattern, "")
                        .replace(".pt", "_image_only_model_preds.pkl")
                )

                # Predictions
                predicted_classes, predicted_boxes = model(features)

                # Scale predictions to image size
                predicted_boxes = predicted_boxes * torch.tensor(
                    [image_size[0], image_size[1], image_size[0], image_size[1]]
                )

                # Convert predicted and ground truth boxes to corner format
                predicted_boxes_corners = convert_to_corners(predicted_boxes[0])
                gt_boxes_corners = convert_to_corners(ground_truth["boxes"])

                # IoU Matching
                iou_matrix = box_iou(predicted_boxes_corners, gt_boxes_corners)

                matched_indices = linear_sum_assignment(iou_matrix.cpu().numpy(), maximize=True)
                matched_pred_indices, matched_gt_indices = matched_indices

                # Save visualization
                predictions = {
                    "predicted_boxes": predicted_boxes,
                    "predicted_boxes_corners": predicted_boxes_corners,
                    "matched_pred_indices": matched_pred_indices,
                    "matched_gt_indices": matched_gt_indices,
                    "gt_boxes_corners": gt_boxes_corners,
                }

                with open(prediction_fn, "wb") as f:
                    pickle.dump(predictions, f)

            except Exception as e:
                print(f"Error processing image {processed_images}: {e}")



if __name__ == "__main__":

    data_type = "Image_Lidar"
    # Define constants
    if data_type == "Image":
        MODEL_PATH = "/home/meowater/Documents/ssd_drive/models/Image/best_model_trial_0_dim256_heads8_layers4_epochs10_lr9.3e-05_wd1.6e-05_alpha2.4e+00_beta3.8e+01_delta4.4e+01_boxacc0.2802.pth"
        input_dim = 3840
    elif data_type == "Lidar":
        MODEL_PATH = "/home/meowater/Documents/ssd_drive/models/Lidar/"
        input_dim = 512
    elif data_type == "Image_Lidar":
        MODEL_PATH = "/home/meowater/Documents/ssd_drive/models/Image_Lidar/best_model_trial_2_dim128_heads8_layers4_epochs10_lr6.8e-05_wd1.1e-05_alpha4.4e+01_beta3.3e+01_delta1.5e+00_boxacc0.3290.pth"
        input_dim = 4352

    DATA_DIR = "/home/meowater/Documents/ssd_drive/cam_box_per_image"
    PT_DIR = "/home/meowater/Documents/ssd_drive/image_features_more_layers"
    LIDAR_DIR = "/home/meowater/Documents/ssd_drive/lidar_projected_cae_resized"
    OUTPUT_DIR = "/home/meowater/Documents/ssd_drive/prediction_" + data_type
    IMG_DIR = "/home/meowater/Documents/ssd_drive/compressed_camera_images2"
    # Parse model parameters from the file name
    params = parse_model_params_from_filename(MODEL_PATH)

    # Initialize the model with parsed parameters
    model = MMFusionDetector(
        input_dim=input_dim,
        model_dim=params["model_dim"],
        num_heads=params["num_heads"],
        num_layers=params["num_layers"],
        alpha=params["alpha"],
        beta=params["beta"],
        delta=params["delta"],
    )

    # Load model checkpoint
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint)
    model.eval()

    # Prepare the dataset and dataloader

    dataset = MMFusionDetectorDataset(DATA_DIR, PT_DIR, LIDAR_DIR, data_type=data_type)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    print("DataLoader initialized.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    generate_predictions(model, dataloader, output_dir=OUTPUT_DIR, num_images=len(dataset.valid_samples))


