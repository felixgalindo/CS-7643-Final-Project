import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from pedestrian_detector_dataset import PedestrianDetectorDataset, custom_collate
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch.nn.functional as F
from torchvision.ops import box_iou

def sinusoidal_positional_encoding(seq_len, model_dim):
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
    pe = torch.zeros(seq_len, model_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # Add batch dimension

# In the model initialization
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
        #self.box_predictor = nn.Linear(model_dim, 4)  # x, y, width, height
        self.box_predictor = nn.Sequential(
            nn.Linear(model_dim, 4),  # x, y, width, height
            nn.Sigmoid()  # Constrain outputs to [0, 1]
        )
        self.class_predictor = nn.Linear(model_dim, num_classes)  # Pedestrian, vehicle, padding

        # Positional Encoder
        seq_len = 500  # Adjust based on expected input size
        self.register_buffer("positional_encoder", sinusoidal_positional_encoding(seq_len, model_dim))


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
        positional_encodings = self.positional_encoder[:, :embeddings.size(1), :].to(embeddings.device)
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

def calculate_loss(predicted_classes, predicted_boxes, gt_classes, gt_boxes, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
    """
    Calculate classification, box regression, cardinality losses, and unmatched penalties.

    Args:
        predicted_classes: Tensor of predicted class logits.
        predicted_boxes: Tensor of predicted bounding boxes.
        gt_classes: Ground truth class labels.
        gt_boxes: Ground truth bounding boxes.

    Returns:
        total_loss, class_loss, box_loss, cardinality_loss, unmatched_penalty
    """
    device = predicted_classes.device
    batch_size = predicted_classes.size(0)
    image_width, image_height = 224, 224  # Example dimensions
    total_class_loss, total_box_loss, total_cardinality_loss, total_unmatched_penalty = 0.0, 0.0, 0.0, 0.0

    for b in range(batch_size):
        #print(f"Processing batch {b+1}/{batch_size}...")

        # Ground truth for the batch
        gt_classes_b = gt_classes[b][gt_classes[b] > 0]  # Exclude padding
        gt_boxes_b = gt_boxes[b][gt_classes[b] > 0]

        if gt_classes_b.numel() == 0:
            total_unmatched_penalty += delta * predicted_classes.size(1)  # Apply unmatched penalty
            #print(f"Batch {b+1}: No ground truth, applying unmatched penalty.")
            continue

        # Diagnostics for ground truth boxes
        gt_box_min, gt_box_max = gt_boxes_b.min(), gt_boxes_b.max()
        #print(f"Batch {b+1}: Ground Truth Boxes Range: min={gt_box_min:.4f}, max={gt_box_max:.4f}")

        # Check normalization for ground truth
        if not (torch.max(gt_boxes_b) <= 1.0 and torch.min(gt_boxes_b) >= 0.0):
            #print(f"Batch {b+1}: Ground truth boxes are not normalized, normalizing now.")
            gt_boxes_b = gt_boxes_b / torch.tensor([image_width, image_height, image_width, image_height], device=device)

        # Diagnostics for predicted boxes
        # pred_box_min, pred_box_max = predicted_boxes[b].min(), predicted_boxes[b].max()
        # print(f"Batch {b+1}: Predicted Boxes Range: min={pred_box_min:.4f}, max={pred_box_max:.4f} before norm")
        # if not (torch.max(predicted_boxes[b]) <= 1.0 and torch.min(predicted_boxes[b]) >= 0.0):
        #     print(f"Batch {b+1}: Predictde boxes are not normalized, normalizing now.")
        #     predicted_boxes[b]  = predicted_boxes[b] / torch.tensor([image_width, image_height, image_width, image_height], device=device)
        #     pred_box_min, pred_box_max = predicted_boxes[b].min(), predicted_boxes[b].max()
        #     print(f"Batch {b+1}: Predicted Boxes Range: min={pred_box_min:.4f}, max={pred_box_max:.4f} after norm")

        # Ensure ground truth boxes are clamped to valid range
        gt_boxes_b = torch.clamp(gt_boxes_b, min=0.0, max=1.0)

        # Compute pairwise box distances
        l1_distances = torch.cdist(predicted_boxes[b], gt_boxes_b, p=1)  # L1 distances

        # Use Hungarian matching to find the best assignment
        cost_matrix = l1_distances.detach().cpu().numpy()
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched_pred_classes = predicted_classes[b, row_indices]
        matched_gt_classes = torch.tensor(gt_classes_b[col_indices], dtype=torch.long, device=device)
        matched_pred_boxes = predicted_boxes[b][row_indices]
        matched_gt_boxes = gt_boxes_b[col_indices]

        # Loss calculations
        class_loss = nn.CrossEntropyLoss()(matched_pred_classes, matched_gt_classes).mean()
        box_loss = nn.SmoothL1Loss()(matched_pred_boxes, matched_gt_boxes).mean()
        cardinality_loss = abs(len(row_indices) - len(gt_classes_b))

        # Debugging unmatched predictions
        unmatched_preds = len(predicted_boxes[b]) - len(row_indices)
        unmatched_gts = len(gt_classes_b) - len(col_indices)
        if unmatched_preds > 0:
            unmatched_pred_boxes = predicted_boxes[b][len(row_indices):]
            distances_to_nearest_gt = torch.min(torch.cdist(unmatched_pred_boxes, gt_boxes_b, p=1), dim=1).values
            #print(f"Batch {b+1}: Min distances for unmatched predictions: {distances_to_nearest_gt}")

        # Update totals
        total_class_loss += class_loss
        total_box_loss += box_loss
        total_cardinality_loss += cardinality_loss
        total_unmatched_penalty += delta * (unmatched_preds + unmatched_gts)/49

        #print(f"Batch {b+1}: Class Loss = {class_loss:.4f}, Box Loss = {box_loss:.4f}, Cardinality Loss = {cardinality_loss:.4f}")
        #print(f"Batch {b+1}: Unmatched Predictions = {unmatched_preds}, Unmatched Ground Truths = {unmatched_gts}")

    # Normalize by batch size
    total_class_loss /= batch_size
    total_box_loss /= batch_size
    total_cardinality_loss /= batch_size
    total_unmatched_penalty /= batch_size

    total_loss = (
        alpha * total_class_loss
        + beta * total_box_loss
        + gamma * total_cardinality_loss
        + delta * total_unmatched_penalty
    )

    #print(f"Total Loss: {total_loss:.4f}, Average Class Loss: {total_class_loss:.4f}, Average Box Loss: {total_box_loss:.4f}, Average Cardinality Loss: {total_cardinality_loss:.4f}, Average Unmatched Penalty: {total_unmatched_penalty:.4f}")

    return total_loss, total_class_loss, total_box_loss, total_cardinality_loss, total_unmatched_penalty


def evaluate_model(model, data_loader, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
    model.eval()
    total_class_loss, total_box_loss, total_cardinality_loss, total_unmatched_loss = 0.0, 0.0, 0.0, 0.0
    num_batches = 0

    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Evaluating") as pbar:
            for batch_features, batch_ground_truth in data_loader:
                predicted_classes, predicted_boxes = model(batch_features)
                valid_mask = batch_ground_truth["classes"] > 0  # Exclude padded entries

                _, class_loss, box_loss, cardinality_loss, unmatched_loss = calculate_loss(
                    predicted_classes=predicted_classes,
                    predicted_boxes=predicted_boxes,
                    gt_classes=batch_ground_truth["classes"],
                    gt_boxes=batch_ground_truth["boxes"],
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    delta=delta
                )

                # Accumulate losses
                total_class_loss += class_loss
                total_box_loss += box_loss
                total_cardinality_loss += cardinality_loss
                total_unmatched_loss += unmatched_loss
                num_batches += 1
                pbar.update(1)

    # Average losses
    avg_class_loss = total_class_loss / num_batches
    avg_box_loss = total_box_loss / num_batches
    avg_cardinality_loss = total_cardinality_loss / num_batches
    avg_unmatched_loss = total_unmatched_loss / num_batches

    print(f"Validation Results: Class Loss = {avg_class_loss:.4f}, "
          f"Box Loss = {avg_box_loss:.4f}, "
          f"Cardinality Loss = {avg_cardinality_loss:.4f}, "
          f"Unmatched Loss = {avg_unmatched_loss:.4f}")


def train_model(model, optimizer, train_loader, val_loader, num_epochs, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
    model.train()
    for epoch in range(num_epochs):
        epoch_class_loss, epoch_box_loss, epoch_cardinality_loss, epoch_unmatched_loss = 0.0, 0.0, 0.0, 0.0
        num_batches = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for batch_features, batch_ground_truth in train_loader:
                optimizer.zero_grad()

                # Forward pass
                predicted_classes, predicted_boxes = model(batch_features)

                # Calculate losses
                total_loss, class_loss, box_loss, cardinality_loss, unmatched_loss = calculate_loss(
                    predicted_classes=predicted_classes,
                    predicted_boxes=predicted_boxes,
                    gt_classes=batch_ground_truth["classes"],
                    gt_boxes=batch_ground_truth["boxes"],
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    delta=delta
                )

                # Backward pass
                total_loss.backward()
                optimizer.step()

                # Accumulate losses
                epoch_class_loss += class_loss.item()
                epoch_box_loss += box_loss.item()
                epoch_cardinality_loss += cardinality_loss
                epoch_unmatched_loss += unmatched_loss
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    "Class Loss": f"{epoch_class_loss / num_batches:.4f}",
                    "Box Loss": f"{epoch_box_loss / num_batches:.4f}",
                    "Cardinality Loss": f"{epoch_cardinality_loss / num_batches:.4f}",
                    "Unmatched Loss": f"{epoch_unmatched_loss / num_batches:.4f}"
                })
                pbar.update(1)

        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"Class Loss = {epoch_class_loss / num_batches:.4f}, "
              f"Box Loss = {epoch_box_loss / num_batches:.4f}, "
              f"Cardinality Loss = {epoch_cardinality_loss / num_batches:.4f}, "
              f"Unmatched Loss = {epoch_unmatched_loss / num_batches:.4f}")

        # Evaluate on validation data
        print("Validating on validation data...")
        evaluate_model(model, val_loader, alpha, beta, gamma, delta)

    return model


if __name__ == "__main__":
    # Dataset directories
    pt_dir = os.path.expanduser("./data/image_features")
    pkl_dir = os.path.expanduser("./dataset/cam_box_per_image")

    # Initialize dataset
    dataset = PedestrianDetectorDataset(pkl_dir, pt_dir)

    # Split the datasets for training, validation, and testing
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=custom_collate)

    # Initialize model and optimizer
    model_dim = 256
    model = MMFusionPedestrianDetector(model_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    print("Starting training...")
    trained_model = train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20
    )

    # Evaluate the model on the test set
    print("Evaluating on test set...")
    evaluate_model(trained_model, test_loader, alpha=1.0, beta=0.02, gamma=1.0, delta=1.0)
