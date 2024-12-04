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
from torchvision.ops import generalized_box_iou_loss

def sinusoidal_positional_encoding(seq_len, model_dim):
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
    pe = torch.zeros(seq_len, model_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # Add batch dimension

# In the model initialization
class MMFusionPedestrianDetector(nn.Module):
    def __init__(self, model_dim=256, num_classes=3, num_heads=8, num_layers=6, num_queries=20):
        super(MMFusionPedestrianDetector, self).__init__()

        # Encoder
        self.cnn_projector = nn.Linear(2048, model_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(model_dim, num_heads), num_layers
        )

        # Decoder
        self.num_queries = num_queries
        self.object_queries = nn.Parameter(torch.randn(num_queries, model_dim))  # Learnable queries
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(model_dim, num_heads), num_layers
        )

        # Prediction heads
        self.box_predictor = nn.Sequential(nn.Linear(model_dim, 4), nn.Sigmoid())
        self.class_predictor = nn.Linear(model_dim, num_classes)

        # Positional Encoder
        seq_len = 500
        self.register_buffer("positional_encoder", sinusoidal_positional_encoding(seq_len, model_dim))

    def forward(self, camera_features):
        # Flatten spatial dimensions
        batch_size, channels, height, width = camera_features.size()
        camera_features = camera_features.view(batch_size, channels, -1).permute(0, 2, 1)

        # Encoder
        embeddings = self.cnn_projector(camera_features)
        positional_encodings = self.positional_encoder[:, :embeddings.size(1), :].to(embeddings.device)
        encoder_out = self.encoder(embeddings + positional_encodings)  # [batch_size, seq_len, model_dim]

        # Adjust encoder output shape for the decoder
        encoder_out = encoder_out.permute(1, 0, 2)  # [seq_len, batch_size, model_dim]

        # Decoder
        object_queries = self.object_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_queries, model_dim]
        object_queries = object_queries.permute(1, 0, 2)  # [num_queries, batch_size, model_dim]
        decoder_out = self.decoder(object_queries, encoder_out)  # [num_queries, batch_size, model_dim]
        decoder_out = decoder_out.permute(1, 0, 2)  # [batch_size, num_queries, model_dim]

        # Predictions
        classes = self.class_predictor(decoder_out)  # [batch_size, num_queries, num_classes]
        boxes = self.box_predictor(decoder_out)  # [batch_size, num_queries, 4]

        return classes, boxes



# def compute_iou(box1, box2):
#     """
#     Compute IoU between two bounding boxes.
#     """
#     box1_x1, box1_y1 = box1[0], box1[1]
#     box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]

#     box2_x1, box2_y1 = box2[0], box2[1]
#     box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]

#     inter_x1 = max(box1_x1, box2_x1)
#     inter_y1 = max(box1_y1, box2_y1)
#     inter_x2 = min(box1_x2, box2_x2)
#     inter_y2 = min(box1_y2, box2_y2)

#     inter_width = max(0, inter_x2 - inter_x1)
#     inter_height = max(0, inter_y2 - inter_y1)
#     intersection_area = inter_width * inter_height

#     box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
#     box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
#     union_area = box1_area + box2_area - intersection_area

#     if union_area == 0:
#         return 0.0  # Avoid division by zero
#     return intersection_area / union_area

def convert_to_corners(boxes):
    """
    Converts bounding boxes from [x, y, w, h] to [x1, y1, x2, y2].
    
    Args:
        boxes: Tensor of shape (N, 4) where each box is [x, y, w, h].
    
    Returns:
        Tensor of shape (N, 4) where each box is [x1, y1, x2, y2].
    """
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def calculate_loss(
    predicted_classes,
    predicted_boxes,
    gt_classes,
    gt_boxes,
    alpha=1.0,
    beta=1.0,
    delta=1.0,
    iou_threshold=0.5,
):
    """
    Calculate losses and metrics

    Args:
        predicted_classes: Tensor of predicted class logits.
        predicted_boxes: Tensor of predicted bounding boxes.
        gt_classes: Ground truth class labels.
        gt_boxes: Ground truth bounding boxes.

    Returns:
        total_loss: Weighted sum of all losses.
        weighted_class_loss, weighted_giou_loss, weighted_unmatched_penalty: Individual weighted losses.
        total_class_accuracy, total_box_accuracy: Metrics for classification and box regression.
        f1_metrics: Tuple containing (F1 score, precision, recall).
    """
    device = predicted_classes.device
    batch_size = predicted_classes.size(0)
    num_queries = predicted_classes.size(1)

    # Normalize ground truth boxes 
    image_width, image_height = 1920, 1280
    gt_boxes = gt_boxes / torch.tensor([image_width, image_height, image_width, image_height], device=device)
    gt_boxes = torch.clamp(gt_boxes, min=0.0, max=1.0)

    # Initialize metrics
    total_class_loss, total_giou_loss, total_unmatched_penalty = 0.0, 0.0, 0.0
    total_class_accuracy, total_box_accuracy = 0.0, 0.0
    true_positives, false_positives, false_negatives = 0, 0, 0

    # Compute IoU matrices 
    iou_matrices = [
        box_iou(convert_to_corners(predicted_boxes[b]), convert_to_corners(gt_boxes[b][gt_classes[b] > 0]))
        for b in range(batch_size)
    ]

    # Perform Hungarian matching 
    matched_indices = [
        linear_sum_assignment(
            iou_matrix[:, : (gt_classes[b] > 0).sum().item()].detach().cpu().numpy(), 
            maximize=True
        )
        for b, iou_matrix in enumerate(iou_matrices)
    ]

    row_indices = [indices[0] for indices in matched_indices]
    col_indices = [indices[1] for indices in matched_indices]

    # Parallelize loss and metric calculations
    for b in range(batch_size):
        gt_classes_b = gt_classes[b][gt_classes[b] > 0]
        if len(gt_classes_b) == 0:
            false_positives += num_queries
            continue

        # Identify unmatched predictions and ground truths
        unmatched_pred_indices = set(range(num_queries)) - set(row_indices[b])
        unmatched_gt_indices = set(range(len(gt_classes_b))) - set(col_indices[b])

        # Compute penalties for unmatched predictions and ground truths
        unmatched_pred_penalty = len(unmatched_pred_indices) * delta
        unmatched_gt_penalty = len(unmatched_gt_indices) * delta

        # Matched predictions and ground truths
        matched_pred_classes = predicted_classes[b, row_indices[b]]
        matched_gt_classes = gt_classes_b[col_indices[b]]
        matched_pred_boxes = predicted_boxes[b][row_indices[b]]
        matched_gt_boxes = gt_boxes[b][gt_classes[b] > 0][col_indices[b]]

        # Loss calculations
        class_loss = nn.CrossEntropyLoss()(matched_pred_classes, matched_gt_classes).mean()
        giou_loss = generalized_box_iou_loss(
            convert_to_corners(matched_pred_boxes), convert_to_corners(matched_gt_boxes)
        ).mean()

        # Update totals
        total_class_loss += class_loss
        total_giou_loss += giou_loss
        total_unmatched_penalty += unmatched_pred_penalty + unmatched_gt_penalty

        # Metrics calculations
        predicted_labels = matched_pred_classes.argmax(dim=1)
        correct_predictions = (predicted_labels == matched_gt_classes).sum().item()
        total_class_accuracy += correct_predictions / len(matched_gt_classes)

        correct_boxes = (iou_matrices[b][row_indices[b], col_indices[b]] >= iou_threshold).sum().item()
        total_box_accuracy += correct_boxes / len(matched_gt_boxes)

        # Update F1 score metrics
        true_positives += correct_boxes
        false_positives += len(row_indices[b]) - correct_boxes
        false_negatives += len(gt_classes_b) - correct_boxes

    # Normalize losses and metrics by batch size
    total_class_loss /= batch_size
    total_giou_loss /= batch_size
    total_unmatched_penalty /= batch_size
    total_class_accuracy /= batch_size
    total_box_accuracy /= batch_size

    # Weighted losses
    weighted_class_loss = alpha * total_class_loss
    weighted_giou_loss = beta * total_giou_loss
    weighted_unmatched_penalty = delta * total_unmatched_penalty

    # Combine losses with weights
    total_loss = weighted_class_loss + weighted_giou_loss + weighted_unmatched_penalty

    # Compute F1 score, precision, recall
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return (
        total_loss,
        weighted_class_loss,
        weighted_giou_loss,
        weighted_unmatched_penalty,
        total_class_accuracy,
        total_box_accuracy,
        (f1_score, precision, recall),
    )



def evaluate_model(model, data_loader,  alpha=10, beta=10, delta=.1, iou_threshold=0.5):
    model.eval()
    total_class_loss, total_box_loss, total_cardinality_loss, total_unmatched_loss = 0.0, 0.0, 0.0, 0.0
    total_class_accuracy, total_box_accuracy = 0.0, 0.0
    total_f1_score, total_precision, total_recall = 0.0, 0.0, 0.0
    num_batches = 0

    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Evaluating") as pbar:
            for batch_features, batch_ground_truth in data_loader:
                # Forward pass and calculate losses and metrics
                predicted_classes, predicted_boxes = model(batch_features)
                (
                    total_loss,
                    class_loss,
                    box_loss,
                    #cardinality_loss,
                    unmatched_loss,
                    class_accuracy,
                    box_accuracy,
                    (f1_score, precision, recall),
                ) = calculate_loss(
                    predicted_classes=predicted_classes,
                    predicted_boxes=predicted_boxes,
                    gt_classes=batch_ground_truth["classes"],
                    gt_boxes=batch_ground_truth["boxes"],
                    alpha=alpha,
                    beta=beta,
                    #gamma=gamma,
                    delta=delta,
                    iou_threshold=iou_threshold,
                )

                # Accumulate metrics
                total_class_loss += class_loss
                total_box_loss += box_loss
                #total_cardinality_loss += cardinality_loss
                total_unmatched_loss += unmatched_loss
                total_class_accuracy += class_accuracy
                total_box_accuracy += box_accuracy
                total_f1_score += f1_score
                total_precision += precision
                total_recall += recall
                num_batches += 1

                # Update progress bar with averaged metrics
                pbar.set_postfix({
                    "Avg Class Loss": f"{total_class_loss / num_batches:.4f}",
                    "Avg Box Loss": f"{total_box_loss / num_batches:.4f}",
                    #"Avg Cardinality Loss": f"{total_cardinality_loss / num_batches:.4f}",
                    "Avg Unmatched Loss": f"{total_unmatched_loss / num_batches:.4f}",
                    "Avg Class Accuracy": f"{total_class_accuracy / num_batches:.4f}",
                    "Avg Box Accuracy": f"{total_box_accuracy / num_batches:.4f}",
                    "Avg F1 Score": f"{total_f1_score / num_batches:.4f}",
                })
                pbar.update(1)

    # Final averaged metrics
    avg_class_loss = total_class_loss / num_batches
    avg_box_loss = total_box_loss / num_batches
    #avg_cardinality_loss = total_cardinality_loss / num_batches
    avg_unmatched_loss = total_unmatched_loss / num_batches
    avg_class_accuracy = total_class_accuracy / num_batches
    avg_box_accuracy = total_box_accuracy / num_batches
    avg_f1_score = total_f1_score / num_batches
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches

    print(f"Validation Results: "
          f"Class Loss = {avg_class_loss:.4f}, "
          f"Box Loss = {avg_box_loss:.4f}, "
          #f"Cardinality Loss = {avg_cardinality_loss:.4f}, "
          f"Unmatched Loss = {avg_unmatched_loss:.4f}, "
          f"Class Accuracy = {avg_class_accuracy:.4f}, "
          f"Box Accuracy = {avg_box_accuracy:.4f}, "
          f"F1 Score = {avg_f1_score:.4f}, "
          f"Precision = {avg_precision:.4f}, "
          f"Recall = {avg_recall:.4f}")


def train_model(model, optimizer, train_loader, val_loader, num_epochs, alpha=10, beta=10, delta=.1, iou_threshold=0.5):
    model.train()
    for epoch in range(num_epochs):
        epoch_class_loss, epoch_box_loss, epoch_cardinality_loss, epoch_unmatched_loss = 0.0, 0.0, 0.0, 0.0
        epoch_class_accuracy, epoch_box_accuracy = 0.0, 0.0
        epoch_f1_score, epoch_precision, epoch_recall = 0.0, 0.0, 0.0
        num_batches = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for batch_features, batch_ground_truth in train_loader:
                optimizer.zero_grad()

                # Forward pass
                predicted_classes, predicted_boxes = model(batch_features)

                # Calculate losses and metrics
                (
                    total_loss,
                    class_loss,
                    box_loss,
                    #cardinality_loss,
                    unmatched_loss,
                    class_accuracy,
                    box_accuracy,
                    (f1_score, precision, recall),
                ) = calculate_loss(
                    predicted_classes=predicted_classes,
                    predicted_boxes=predicted_boxes,
                    gt_classes=batch_ground_truth["classes"],
                    gt_boxes=batch_ground_truth["boxes"],
                    alpha=alpha,
                    beta=beta,
                    #gamma=gamma,
                    delta=delta,
                    iou_threshold=iou_threshold,
                )

                # Backward pass
                total_loss.backward()
                optimizer.step()

                # Accumulate metrics
                epoch_class_loss += class_loss
                epoch_box_loss += box_loss
                #epoch_cardinality_loss += cardinality_loss
                epoch_unmatched_loss += unmatched_loss
                epoch_class_accuracy += class_accuracy
                epoch_box_accuracy += box_accuracy
                epoch_f1_score += f1_score
                epoch_precision += precision
                epoch_recall += recall
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    "Class Loss": f"{epoch_class_loss / num_batches:.4f}",
                    "Box Loss": f"{epoch_box_loss / num_batches:.4f}",
                    #"Cardinality Loss": f"{epoch_cardinality_loss / num_batches:.4f}",
                    "Unmatched Loss": f"{epoch_unmatched_loss / num_batches:.4f}",
                    "Class Accuracy": f"{epoch_class_accuracy / num_batches:.4f}",
                    "Box Accuracy": f"{epoch_box_accuracy / num_batches:.4f}",
                    "F1 Score": f"{epoch_f1_score / num_batches:.4f}",
                })
                pbar.update(1)

        # Normalize metrics by the number of batches
        avg_class_loss = epoch_class_loss / num_batches
        avg_box_loss = epoch_box_loss / num_batches
        #avg_cardinality_loss = epoch_cardinality_loss / num_batches
        avg_unmatched_loss = epoch_unmatched_loss / num_batches
        avg_class_accuracy = epoch_class_accuracy / num_batches
        avg_box_accuracy = epoch_box_accuracy / num_batches
        avg_f1_score = epoch_f1_score / num_batches
        avg_precision = epoch_precision / num_batches
        avg_recall = epoch_recall / num_batches

        # Print epoch results
        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"Class Loss = {avg_class_loss:.4f}, "
              f"Box Loss = {avg_box_loss:.4f}, "
              #f"Cardinality Loss = {avg_cardinality_loss:.4f}, "
              f"Unmatched Loss = {avg_unmatched_loss:.4f}, "
              f"Class Accuracy = {avg_class_accuracy:.4f}, "
              f"Box Accuracy = {avg_box_accuracy:.4f}, "
              f"F1 Score = {avg_f1_score:.4f}, "
              f"Precision = {avg_precision:.4f}, "
              f"Recall = {avg_recall:.4f}")

        # Evaluate on validation data
        evaluate_model(model, val_loader, alpha, beta, delta, iou_threshold)

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

    # # Initialize model and optimizer
    # model_dim = 256
    # model = MMFusionPedestrianDetector(model_dim)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # # Initialize model and optimizer
    # model_dim = 128  
    # num_layers = 4  
    # num_heads = 4  
    # model = MMFusionPedestrianDetector(model_dim, num_heads=num_heads, num_layers=num_layers)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Higher initial learning rate

    # Initialize model and optimizer
    model_dim = 128  
    num_layers = 4   
    num_heads = 4    
    model = MMFusionPedestrianDetector(model_dim, num_heads=num_heads, num_layers=num_layers)

    # Optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

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
    evaluate_model(trained_model, test_loader, alpha=1.0, beta=0.02, delta=1.0)
