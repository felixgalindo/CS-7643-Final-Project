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


def visualize_predictions(model, dataloader, output_dir="visualizations", num_images=5, image_size=(1920, 1280)):
    """
    Visualize predictions by overlaying ground truth and matched predicted boxes on images.
    Includes IoU, L1 values, and class annotations

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

            # Process only images with 1 to 3 GT boxes
            num_gt_boxes = len(vehicle_boxes)
            if num_gt_boxes < 1 or num_gt_boxes > 3:
                continue

            processed_images += 1

            print(f"\nProcessing Image {processed_images}/{num_images} (Image Index: {i})")
            print("Ground Truth Keys:", ground_truth.keys())
            print("Number of GT Boxes:", num_gt_boxes)
            print("Filtered GT Boxes:", ground_truth["boxes"])
            print("Filtered GT Classes:", ground_truth["classes"])

            try:
                # Load the image
                image_path = (
                    "./dataset/compressed_camera_images2/"
                    + ground_truth["context_name"]
                    + "/"
                    + ground_truth["feature_tensor_fn"]
                        .replace(ground_truth["context_name"] + "_", "")
                        .replace("camera_image_camera_", "camera_image_camera-")
                        .replace(".pt", ".jpg")
                )
                print("Image Path:", image_path)

                img = cv2.imread(image_path)
                if img is None:
                    print(f"Warning: Could not read image at {image_path}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Draw Ground Truth Boxes
                for box, cls in zip(ground_truth["boxes"].cpu().numpy(), ground_truth["classes"].cpu().numpy()):
                    x1, y1, x2, y2 = box
                    print(f"GT Box: ({x1}, {y1}, {x2}, {y2}), Class: {cls}")
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(
                        img,
                        f"GT: C{cls}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )

                # Predictions
                print("Running Model Prediction...")
                predicted_classes, predicted_boxes = model(features)
                print("Raw Predicted Classes:", predicted_classes)
                print("Raw Predicted Boxes:", predicted_boxes)

                # Scale predictions to image size
                predicted_boxes = predicted_boxes * torch.tensor(
                    [image_size[0], image_size[1], image_size[0], image_size[1]]
                )

                print("Scaled Predicted Boxes:", predicted_boxes)

                predicted_boxes_corners = convert_to_corners(predicted_boxes[0])
                print("Predicted Boxes Shape (corners):", predicted_boxes_corners.shape)

                # Convert GT to corners
                gt_boxes_corners = convert_to_corners(ground_truth["boxes"])
                print("GT Boxes Shape (corners):", gt_boxes_corners.shape)

                # IoU Matching
                print("Performing IoU Matching...")
                iou_matrix = box_iou(predicted_boxes_corners, gt_boxes_corners)
                print("IoU Matrix:", iou_matrix)

                matched_indices = linear_sum_assignment(-iou_matrix.cpu().numpy(), maximize=True)
                matched_pred_indices, matched_gt_indices = matched_indices

                print("Matched Indices (Predicted -> GT):", matched_indices)

                # Filter matched predictions and GT
                matched_pred_boxes = predicted_boxes_corners[matched_pred_indices]
                matched_gt_boxes = gt_boxes_corners[matched_gt_indices]
                matched_pred_classes = predicted_classes[0][matched_pred_indices].argmax(dim=1)
                matched_gt_classes = ground_truth["classes"][matched_gt_indices]

                # Draw Predicted Boxes with IoU ≥ 0.5
                for box, cls, gt_box, iou in zip(
                        matched_pred_boxes.cpu().numpy(),
                        matched_pred_classes.cpu().numpy(),
                        matched_gt_boxes.cpu().numpy(),
                        iou_matrix[matched_pred_indices, matched_gt_indices].cpu().numpy(),
                ):
                    if iou >= 0.1:  # Only plot matches with IoU ≥ 0.5
                        x1, y1, x2, y2 = box
                        l1_loss = np.abs(np.array([x1, y1, x2, y2]) - gt_box).sum()

                        print(f"Predicted Box: ({x1}, {y1}, {x2}, {y2}), Class: {cls}, IoU: {iou:.2f}, L1: {l1_loss:.2f}")
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(
                            img,
                            f"Pred: C{cls}, IoU: {iou:.2f}, L1: {l1_loss:.2f}",
                            (int(x1), int(y1) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 0, 0),
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
    MODEL_PATH = "./data/models/best_model_trial_0_dim256_heads8_layers4_epochs10_lr9.3e-05_wd1.6e-05_alpha2.4e+00_beta3.8e+01_delta4.4e+01_boxacc0.2802.pth"
    DATA_DIR = "./dataset/cam_box_per_image"
    PT_DIR = "./data/image_features_more_layers"
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

    dataset = MMFusionDetectorDataset(DATA_DIR, PT_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    print("DataLoader initialized.")

    # Run visualization
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    visualize_predictions(model, dataloader, output_dir=OUTPUT_DIR, num_images=NUM_IMAGES)
    print("Visualization completed.")

