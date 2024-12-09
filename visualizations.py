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

def visualize_predictions(
    model, dataloader, img_dir, output_dir="visualizations", num_images=5, image_size=(1920, 1280)
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
        for i, (features, ground_truth) in enumerate(dataloader):
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

            # Skip if the number of GT boxes is not in the range [1, 3]
            num_gt_boxes = len(vehicle_boxes)
            if num_gt_boxes < 1 or num_gt_boxes > 3:
                continue

            processed_images += 1
            print(f"\nProcessing Image {processed_images}/{num_images} (Image Index: {i})")

            try:

                img_name_prefix = ground_truth["feature_tensor_fn"].replace(ground_truth["context_name"] + '_', '')
                pattern = img_name_prefix.split("_timestamp")[0]
                pattern = pattern.split("_camera_1")[-1]

                # Load the image
                image_path = os.path.join(
                    img_dir,
                    ground_truth["context_name"],
                    ground_truth["feature_tensor_fn"]
                        .replace(ground_truth["context_name"] + "_", "")
                        .replace(pattern, "")
                        .replace("_camera_1_", "_camera-1_")
                        .replace(".pt", ".jpg")
                )

                img = cv2.imread(image_path)
                if img is None:
                    print(f"Warning: Could not read image at {image_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Predictions
                predicted_classes, predicted_boxes = model(features)

                # Scale predictions to image size
                predicted_boxes = predicted_boxes * torch.tensor(
                    [image_size[0], image_size[1], image_size[0], image_size[1]]
                )

                # Convert predicted and ground truth boxes to corner format
                predicted_boxes_corners = convert_to_corners(predicted_boxes[0])
                gt_boxes_corners = convert_to_corners(ground_truth["boxes"])

                # Debugging: Print box shapes and ranges
                print("Predicted Boxes (Corners):", predicted_boxes_corners)
                print("Ground Truth Boxes (Corners):", gt_boxes_corners)

                # IoU Matching
                iou_matrix = box_iou(predicted_boxes_corners, gt_boxes_corners)
                print("IoU Matrix:", iou_matrix)

                matched_indices = linear_sum_assignment(iou_matrix.cpu().numpy(), maximize=True)
                matched_pred_indices, matched_gt_indices = matched_indices

                print("matched_pred_indices:", matched_pred_indices)
                print("matched_gt_indices", matched_gt_indices)

                # Draw ground truth boxes
                for gt_box in gt_boxes_corners:
                    x1, y1, x2, y2 = gt_box.cpu().numpy()
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
                    cv2.putText(
                        img,
                        "GT",
                        (int(x1), max(int(y1) - 20, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 0, 0),
                        2,
                    )

                # Draw matched predictions
                for pred_idx, gt_idx in zip(matched_pred_indices, matched_gt_indices):
                    # Retrieve IoU
                    iou = iou_matrix[pred_idx, gt_idx].item()

                    # Retrieve boxes for visualization
                    pred_box = predicted_boxes_corners[pred_idx].cpu().numpy()
                    gt_box = gt_boxes_corners[gt_idx].cpu().numpy()

                    # Calculate L1 distance
                    l1_value = np.abs(pred_box - gt_box).sum()

                    # Log IoU and L1 for debugging
                    print(f"Matched Pair: Pred Box {pred_idx}, GT Box {gt_idx}, IoU: {iou:.4f}, L1: {l1_value:.2f}")

                    # Draw predicted box
                    x1, y1, x2, y2 = pred_box
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    cv2.putText(
                        img,
                        f"IoU: {iou:.4f}, L1: {l1_value:.2f}",
                        (int(x1), max(int(y1) - 20, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )

                # Save visualization
                output_path = os.path.join(output_dir, f"image_{processed_images}.jpg")
                plt.imshow(img)
                plt.axis('off')
                plt.savefig(output_path, bbox_inches='tight')
                plt.close()
                print(f"Visualization saved to {output_path}")

            except Exception as e:
                print(f"Error processing image {processed_images}: {e}")



if __name__ == "__main__":

    # Define constants
    MODEL_PATH = "./data/models/best_model_trial_4_dim512_heads8_layers4_epochs1_lr5.5e-05_wd3.1e-05_alpha2.2e+01_beta4.2e+01_delta3.1e+01_boxacc0.1294.pth"
    DATA_DIR = "./dataset/cam_box_per_image"
    PT_DIR = "./data/image_features_more_layers"
    LIDAR_DIR = "./dataset/lidar_projected_cae_resized"
    OUTPUT_DIR = "./data/visualizations/"
    NUM_IMAGES = 500

    # Parse model parameters from the file name
    params = parse_model_params_from_filename(MODEL_PATH)

    # Initialize the model with parsed parameters
    model = MMFusionDetector(
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

    dataset = MMFusionDetectorDataset(DATA_DIR, PT_DIR, LIDAR_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    print("DataLoader initialized.")

    # Run visualization
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    visualize_predictions(model, dataloader, img_dir=IMG_DIR, output_dir=OUTPUT_DIR, num_images=5)
    print("Visualization completed.")

