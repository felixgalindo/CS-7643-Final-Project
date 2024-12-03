import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from pedestrian_detector_dataset import PedestrianDetectorDataset, custom_collate
from scipy.optimize import linear_sum_assignment
import numpy as np


class MMFusionPedestrianDetector(nn.Module):
    def __init__(self, model_dim=256, num_classes=3, num_heads=8, num_layers=6):
        super(MMFusionPedestrianDetector, self).__init__()

        # Create embedding projectors
        self.cnn_projector = nn.Linear(2048, model_dim)

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(model_dim, num_heads),
            num_layers
        )

        # Predictors
        self.box_predictor = nn.Linear(model_dim, 4)  # x, y, width, height
        self.class_predictor = nn.Linear(model_dim, num_classes)  # Pedestrian, vehicle, padding

        # Positional Encoder
        self.positional_encoder = nn.Parameter(torch.zeros(1, 500, model_dim))

    def forward(self, camera_features):
        """
        Forward propagation through the model.
        """
        # Flatten spatial dimensions
        batch_size, channels, height, width = camera_features.size()
        camera_features = camera_features.view(batch_size, channels, -1).permute(0, 2, 1)

        # Create embeddings
        embeddings = self.cnn_projector(camera_features)

        # Add positional encodings
        positional_encodings = self.positional_encoder[:, :embeddings.size(1), :]
        transformer_input = embeddings + positional_encodings

        # Transformer output
        transformer_out = self.transformer(transformer_input)
        classes = self.class_predictor(transformer_out)
        boxes = self.box_predictor(transformer_out)

        return classes, boxes


def compute_iou(box1, box2):
    """
    Compute IoU between two bounding boxes.
    """
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]

    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]

    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection_area = inter_width * inter_height

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0  # Avoid division by zero
    return intersection_area / union_area


def compute_iou_matrix(pred_boxes, gt_boxes):
    """
    Compute IoU matrix between predicted and ground truth boxes.
    """
    num_pred = pred_boxes.shape[0]
    num_gt = gt_boxes.shape[0]
    iou_matrix = np.zeros((num_pred, num_gt), dtype=np.float32)

    for i in range(num_pred):
        for j in range(num_gt):
            iou_matrix[i, j] = compute_iou(pred_boxes[i], gt_boxes[j])

    return iou_matrix


def compute_losses(predicted_classes, predicted_boxes, ground_truth_classes, ground_truth_boxes):
    """
    Compute classification, bounding box regression, and cardinality losses.

    Args:
        predicted_classes: Predicted class logits for the batch.
        predicted_boxes: Predicted bounding boxes for the batch.
        ground_truth_classes: Ground truth class labels for the batch.
        ground_truth_boxes: Ground truth bounding boxes for the batch.

    Returns:
        Tuple of (class_loss, box_loss, cardinality_loss) for the batch.
    """
    device = predicted_classes.device  # Ensure computations happen on the correct device

    #Init losses
    class_loss = 0.0
    box_loss = 0.0
    cardinality_loss = 0.0
    batch_size = ground_truth_classes.size(0)

    for b in range(batch_size):
        gt_classes = ground_truth_classes[b, :].detach().cpu().numpy()
        gt_boxes = ground_truth_boxes[b, :].detach().cpu().numpy()
        pred_boxes = predicted_boxes[b, :].detach().cpu().numpy()
        pred_classes = predicted_classes[b, :]

        # Mask valid ground truth objects
        valid_gt_mask = gt_classes > 0 #0 is padding
        gt_classes = gt_classes[valid_gt_mask]
        gt_boxes = gt_boxes[valid_gt_mask]

        if len(gt_classes) == 0:
            continue  # Skip if no valid ground truth

        # Compute IoU matrix which tells you overlap between predicted and ground truth boxes
        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

        # Match predictions to ground truth
        #Since, the order of the predictions may differ from ground truth, this will run Hungarian algorithm to find the best index matches 
        row_indices, col_indices = linear_sum_assignment(-iou_matrix)

        matched_pred_classes = pred_classes[row_indices]
        matched_gt_classes = torch.tensor(gt_classes[col_indices], dtype=torch.long).to(device)
        matched_pred_boxes = predicted_boxes[b, row_indices]
        matched_gt_boxes = torch.tensor(gt_boxes[col_indices], dtype=torch.float).to(device)

        # Compute classification and box losses
        class_loss += nn.CrossEntropyLoss(ignore_index=0)(matched_pred_classes, matched_gt_classes).mean()
        box_loss += nn.SmoothL1Loss()(matched_pred_boxes, matched_gt_boxes).mean()

        # Compute cardinality loss
        num_gt_objects = len(gt_classes)
        num_pred_objects = len(row_indices)  # Number of matched predictions
        cardinality_penalty = abs(num_pred_objects - num_gt_objects)
        cardinality_loss += torch.tensor(cardinality_penalty, dtype=torch.float, device=device)

    # Normalize losses by batch size
    class_loss /= batch_size
    box_loss /= batch_size
    cardinality_loss /= batch_size

    return class_loss, box_loss, cardinality_loss


def evaluate_model(val_loader, model):
    """
    Evaluate the model on the validation set.

    Returns:
        Tuple of (avg_class_loss, avg_box_loss, avg_cardinality_loss).
    """
    total_class_loss = 0.0
    total_box_loss = 0.0
    total_cardinality_loss = 0.0

    with torch.no_grad():
        for batch_features, batch_ground_truth in val_loader:
            predicted_classes, predicted_boxes = model(batch_features)
            class_loss, box_loss, cardinality_loss = compute_losses(
                predicted_classes, predicted_boxes,
                batch_ground_truth["classes"], batch_ground_truth["boxes"]
            )
            total_class_loss += class_loss.item()
            total_box_loss += box_loss.item()
            total_cardinality_loss += cardinality_loss.item()

    avg_class_loss = total_class_loss / len(val_loader)
    avg_box_loss = total_box_loss / len(val_loader)
    avg_cardinality_loss = total_cardinality_loss / len(val_loader)

    return avg_class_loss, avg_box_loss, avg_cardinality_loss



def train_model(train_loader, val_loader, model_dim=256, num_epochs=20, learning_rate=1e-4):
    """
    Train the MMFusionPedestrianDetector model.

    Args:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        model_dim (int): Model dimension.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for optimizer.

    Returns:
        nn.Module: Trained model.
    """
    model = MMFusionPedestrianDetector(model_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        #Init losses
        epoch_class_loss = 0.0
        epoch_box_loss = 0.0
        epoch_cardinality_loss = 0.0

        model.train()
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for batch_features, batch_ground_truth in train_loader:
                optimizer.zero_grad()

                # Forward pass
                predicted_classes, predicted_boxes = model(batch_features)

                # Compute losses
                class_loss, box_loss, cardinality_loss = compute_losses(
                    predicted_classes,
                    predicted_boxes,
                    batch_ground_truth["classes"],
                    batch_ground_truth["boxes"]
                )
                # Iterate through each batch element
                for b in range(batch_ground_truth["classes"].size(0)):  # Iterate over batch size
                    # Extract ground truth and predictions for the current element
                    gt_classes = batch_ground_truth["classes"][b]
                    pred_classes = torch.argmax(predicted_classes[b], dim=-1)  # Get predicted class labels

                    # Check if there's any non-pedestrian class (2) in ground truth or predictions
                    if 2 in gt_classes or 2 in pred_classes:
                        print(f"Batch index {b}:")
                        print("Ground truth classes:", gt_classes.tolist())
                        print("Predicted classes:", pred_classes.tolist())


                # Apply log transformation to reduce scale
                box_loss = torch.log(1 + box_loss)

                # Backprop and optimization
                total_loss = class_loss + box_loss + cardinality_loss
                total_loss.backward()
                optimizer.step()

                # Add up the losses
                epoch_class_loss += class_loss.item()
                epoch_box_loss += box_loss.item()
                epoch_cardinality_loss += cardinality_loss.item()

                # Update progress bar
                pbar.set_postfix({
                    "Class Loss": f"{class_loss.item():.4f}",
                    "Box Loss": f"{box_loss.item():.4f}",
                    "Cardinality Loss": f"{cardinality_loss.item():.4f}"
                })
                pbar.update(1)

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Class Loss: {epoch_class_loss:.4f}, "
              f"Box Loss: {epoch_box_loss:.4f}, "
              f"Cardinality Loss: {epoch_cardinality_loss:.4f}")

        # Run validation for the model
        model.eval()
        val_class_loss, val_box_loss, val_cardinality_loss = evaluate_model(val_loader, model)
        print(f"Validation Class Loss: {val_class_loss:.4f}, "
              f"Box Loss: {val_box_loss:.4f}, "
              f"Cardinality Loss: {val_cardinality_loss:.4f}")

    return model




if __name__ == "__main__":
    # Dataset directories
    pt_dir = os.path.expanduser("./data/image_features")
    pkl_dir = os.path.expanduser("./dataset/cam_box_per_image")

    # Initialize dataset
    dataset = PedestrianDetectorDataset(pkl_dir, pt_dir)

    # Split the datasets for training,validation,testing
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=custom_collate)

    # Train the model
    print("Starting training...")
    trained_model = train_model(train_loader, val_loader)

    # Test the model
    print("Evaluating on test set...")
    test_class_loss, test_box_loss, test_cardinality_loss = evaluate_model(test_loader, trained_model)
    print(f"Test - Class Loss: {test_class_loss:.4f}, Box Loss: {test_box_loss:.4f}, Cardinality Loss: {test_cardinality_loss:.4f}")
