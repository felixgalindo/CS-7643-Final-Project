import optuna
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pedestrian_detector import MMFusionDetector, PedestrianDetectorDataset, custom_collate
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

def calculate_loss_metrics(
    predicted_classes,
    predicted_boxes,
    gt_classes,
    gt_boxes,
    alpha=1.0,
    beta=5.0,
    delta=2.0,
    iou_threshold=0.5,
):
    """
    Calculate losses and metrics with enhanced debugging.
    """
    device = predicted_classes.device
    batch_size = predicted_classes.size(0)
    num_queries = predicted_classes.size(1)

    # Normalize ground truth boxes
    image_width, image_height = 1920, 1280
    gt_boxes = gt_boxes / torch.tensor([image_width, image_height, image_width, image_height], device=device)
    gt_boxes = torch.clamp(gt_boxes, min=0.0, max=1.0)

    # print(f"Predicted classes shape: {predicted_classes.shape}")
    # print(f"Predicted boxes shape: {predicted_boxes.shape}")
    # print(f"Ground truth classes shape: {gt_classes.shape}")
    # print(f"Ground truth boxes shape: {gt_boxes.shape}")
    # print(f"GT classes before filtering: {gt_classes}")

    # Filter out padding (-1) values from ground truth
    valid_gt_mask = gt_classes >= 0
    filtered_gt_classes = [gt_classes[b][valid_gt_mask[b]] for b in range(batch_size)]
    filtered_gt_boxes = [gt_boxes[b][valid_gt_mask[b]] for b in range(batch_size)]

    # print(f"Filtered GT classes: {[c.tolist() for c in filtered_gt_classes]}")
    # print(f"Filtered GT boxes: {[len(b) for b in filtered_gt_boxes]}")

    # Validate class range
    for b, c in enumerate(filtered_gt_classes):
        if not torch.all((c >= 0) & (c < predicted_classes.size(-1))):
            raise ValueError(
                f"Batch {b}: GT classes contain values out of range! Classes: {c.tolist()}."
            )

    #print("Predicted classes: ", predicted_classes)
    #print("GT classes: ", filtered_gt_classes)
    # Compute IoU matrices
    iou_matrices = [
        box_iou(
            convert_to_corners(predicted_boxes[b]),
            convert_to_corners(filtered_gt_boxes[b]) if len(filtered_gt_boxes[b]) > 0 else torch.zeros((0, 4), device=device),
        )
        for b in range(batch_size)
    ]
    #print("iou_matrices: ", iou_matrices)
    matched_indices = [
        linear_sum_assignment(
            iou_matrix[:, : len(filtered_gt_classes[b])].detach().cpu().numpy(),
            maximize=True
        ) if len(filtered_gt_classes[b]) > 0 else ([], [])
        for b, iou_matrix in enumerate(iou_matrices)
    ]

    # print("matched_indices: ", matched_indices)

    total_class_loss, total_box_loss, total_giou_loss = 0.0, 0.0, 0.0
    total_class_accuracy, total_box_accuracy = 0.0, 0.0
    true_positives, false_positives, false_negatives = 0, 0, 0

    for b, (row_indices, col_indices) in enumerate(matched_indices):
        if len(filtered_gt_classes[b]) == 0:
            print(f"Batch {b}: No valid ground truth classes found!")
            false_positives += num_queries
            continue

        # unmatched_pred_indices = set(range(num_queries)) - set(row_indices)
        # unmatched_gt_indices = set(range(len(filtered_gt_classes[b]))) - set(col_indices)

        # unmatched_pred_penalty = len(unmatched_pred_indices) * delta
        # unmatched_gt_penalty = len(unmatched_gt_indices) * delta

        matched_pred_classes = predicted_classes[b, row_indices]
        matched_gt_classes = torch.tensor(filtered_gt_classes[b])[col_indices].to(device)

        #print(f"Batch {b}: Matched predicted classes shape: {matched_pred_classes.shape}")
        # print(f"Batch {b}: Matched ground truth classes shape: {matched_gt_classes.shape}")
        # print(f"Batch {b}: Matched ground truth classes values: {matched_gt_classes.tolist()}")

        # Classification loss
        class_loss = nn.CrossEntropyLoss()(matched_pred_classes, matched_gt_classes).mean()

        # GIoU loss
        giou_loss = generalized_box_iou_loss(
            convert_to_corners(predicted_boxes[b, row_indices]),
            convert_to_corners(filtered_gt_boxes[b][col_indices]),
        ).mean()

        # print("Predicted boxes: ", predicted_boxes[b, row_indices])
        # print("GT boxes: ", filtered_gt_boxes[b][col_indices])
        # L1 loss
        l1_loss = F.l1_loss(predicted_boxes[b, row_indices], filtered_gt_boxes[b][col_indices], reduction="mean")

        # Combine GIoU loss and L1 loss as box loss
        #box_loss = giou_loss + l1_loss

        # Update totals
        total_class_loss += class_loss
        total_box_loss += l1_loss
        total_giou_loss += giou_loss

        # Metrics calculations
        predicted_labels = matched_pred_classes.argmax(dim=1)
        correct_predictions = (predicted_labels == matched_gt_classes).sum().item()
        total_class_accuracy += correct_predictions / len(matched_gt_classes)

        correct_boxes = (iou_matrices[b][row_indices, col_indices] >= iou_threshold).sum().item()
        total_box_accuracy += correct_boxes / len(filtered_gt_boxes[b])

        # Update F1 score metrics
        true_positives += correct_boxes
        false_positives += len(row_indices) - correct_boxes
        false_negatives += len(filtered_gt_classes[b]) - correct_boxes

    total_class_loss /= batch_size
    total_box_loss /= batch_size
    total_giou_loss /= batch_size
    #total_unmatched_penalty /= batch_size
    total_class_accuracy /= batch_size
    total_box_accuracy /= batch_size

    # Weighted losses
    weighted_class_loss = alpha * total_class_loss
    weighted_box_loss = beta * total_box_loss
    weighted_giou_loss = delta * total_giou_loss

    # Combine losses
    total_loss = weighted_class_loss + weighted_box_loss + weighted_giou_loss

    # F1 metrics
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    # print(f"Loss Summary: Class Loss={total_class_loss}, Box Loss={total_box_loss}, Total Loss={total_loss}")
    # print(f"F1 Score Metrics: Precision={precision}, Recall={recall}, F1 Score={f1_score}")

    return (
        total_loss,
        weighted_class_loss,
        weighted_box_loss,
        weighted_giou_loss,
        total_class_accuracy,
        total_box_accuracy,
        (f1_score, precision, recall),
    )

def evaluate_model(model, data_loader, alpha=1.0, beta=5.0, delta=2.0, iou_threshold=0.5, trial=1):
    """
    Evaluate the model on the validation dataset.

    :param model: The model to evaluate
    :param data_loader: The data loader for the validation set
    :param alpha: Weight for the class loss
    :param beta: Weight for the box loss
    :param delta: Weight for the unmatched loss
    :param iou_threshold: IoU threshold for bounding box matching
    :return: Scalar validation loss value
    """
    model.eval()
    
    # Initialize variables to store the accumulated metrics
    total_class_loss, total_box_loss, total_unmatched_loss = 0.0, 0.0, 0.0
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
                    unmatched_loss,
                    class_accuracy,
                    box_accuracy,
                    (f1_score, precision, recall),
                ) = calculate_loss_metrics(
                    predicted_classes=predicted_classes,
                    predicted_boxes=predicted_boxes,
                    gt_classes=batch_ground_truth["classes"],
                    gt_boxes=batch_ground_truth["boxes"],
                    alpha=alpha,
                    beta=beta,
                    delta=delta,
                    iou_threshold=iou_threshold
                )

                # Accumulate metrics
                total_class_loss += class_loss.item()
                total_box_loss += box_loss.item()
                total_unmatched_loss += unmatched_loss.item()
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
                    "Avg GIoU Loss": f"{total_unmatched_loss / num_batches:.4f}",
                    "Avg Class Accuracy": f"{total_class_accuracy / num_batches:.4f}",
                    "Avg Box Accuracy": f"{total_box_accuracy / num_batches:.4f}",
                    "Avg F1 Score": f"{total_f1_score / num_batches:.4f}",
                })
                pbar.update(1)

    # Calculate average metrics
    avg_class_loss = total_class_loss / num_batches
    avg_box_loss = total_box_loss / num_batches
    avg_unmatched_loss = total_unmatched_loss / num_batches
    avg_class_accuracy = total_class_accuracy / num_batches
    avg_box_accuracy = total_box_accuracy / num_batches
    avg_f1_score = total_f1_score / num_batches
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches

    # Print the final averaged metrics
    print(f"Trial {trial}: ,"
          f"Validation Results: ,"
          f"Class Loss = {avg_class_loss:.4f}, "
          f"Box Loss = {avg_box_loss:.4f}, "
          f"GIoU Loss = {avg_unmatched_loss:.4f}, "
          f"Class Accuracy = {avg_class_accuracy:.4f}, "
          f"Box Accuracy = {avg_box_accuracy:.4f}, "
          f"F1 Score = {avg_f1_score:.4f}, "
          f"Precision = {avg_precision:.4f}, "
          f"Recall = {avg_recall:.4f}")

    # Return the total loss as the scalar value for optimization
    total_loss = avg_class_loss + avg_box_loss + avg_unmatched_loss  
    return total_loss



def train_model(model, optimizer, scheduler, train_loader, val_loader, num_epochs, alpha=1.0, beta=5, delta=2.0, iou_threshold=0.5, trial=1):
    model.train()
    for epoch in range(num_epochs):
        epoch_class_loss, epoch_box_loss, epoch_unmatched_loss = 0.0, 0.0, 0.0, 0.0
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
                    unmatched_loss,
                    class_accuracy,
                    box_accuracy,
                    (f1_score, precision, recall),
                ) = calculate_loss_metrics(
                    predicted_classes=predicted_classes,
                    predicted_boxes=predicted_boxes,
                    gt_classes=batch_ground_truth["classes"],
                    gt_boxes=batch_ground_truth["boxes"],
                    alpha=alpha,
                    beta=beta,
                    delta=delta,
                    iou_threshold=iou_threshold,
                )

                # Backward pass
                total_loss.backward()
                optimizer.step()

                # Accumulate metrics
                epoch_class_loss += class_loss
                epoch_box_loss += box_loss
                epoch_unmatched_loss += unmatched_loss
                epoch_class_accuracy += class_accuracy
                epoch_box_accuracy += box_accuracy
                epoch_f1_score += f1_score
                epoch_precision += precision
                epoch_recall += recall
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    "Trial": f"{trial}",
                    "Class Loss": f"{epoch_class_loss / num_batches:.4f}",
                    "Box Loss": f"{epoch_box_loss / num_batches:.4f}",
                    "GIoU Loss": f"{epoch_unmatched_loss / num_batches:.4f}",
                    "Class Accuracy": f"{epoch_class_accuracy / num_batches:.4f}",
                    "Box Accuracy": f"{epoch_box_accuracy / num_batches:.4f}",
                    "F1 Score": f"{epoch_f1_score / num_batches:.4f}",
                })
                pbar.update(1)

            scheduler.step()  # Update learning rate

        # Normalize metrics by the number of batches
        avg_class_loss = epoch_class_loss / num_batches
        avg_box_loss = epoch_box_loss / num_batches
        avg_unmatched_loss = epoch_unmatched_loss / num_batches
        avg_class_accuracy = epoch_class_accuracy / num_batches
        avg_box_accuracy = epoch_box_accuracy / num_batches
        avg_f1_score = epoch_f1_score / num_batches
        avg_precision = epoch_precision / num_batches
        avg_recall = epoch_recall / num_batches

        # Print epoch results
        print(f"Trial {trial}: "
              f"Epoch {epoch + 1}/{num_epochs}: "
              f"Class Loss = {avg_class_loss:.4f}, "
              f"Box Loss = {avg_box_loss:.4f}, "
              f"GIoU Loss = {avg_unmatched_loss:.4f}, "
              f"Class Accuracy = {avg_class_accuracy:.4f}, "
              f"Box Accuracy = {avg_box_accuracy:.4f}, "
              f"F1 Score = {avg_f1_score:.4f}, "
              f"Precision = {avg_precision:.4f}, "
              f"Recall = {avg_recall:.4f}")

        # Evaluate on validation data
        evaluate_model(model, val_loader, alpha, beta, delta, iou_threshold)

    return model

def objective(trial):
    """
    Define the hyperparameter search space and the training loop for Optuna optimization.
    """
    # Sample hyperparameters
    model_dim = trial.suggest_categorical('model_dim', [128, 256, 512])  
    num_heads = trial.suggest_int('num_heads', 2, 4)  
    num_layers = trial.suggest_int('num_layers', 4, 8) 
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3) 
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)  
    alpha = trial.suggest_loguniform('alpha', 0.1, 100)  
    beta = trial.suggest_loguniform('beta', 0.1, 100)  
    delta = trial.suggest_loguniform('delta', 0.1, 100)

    # Print out the trial number and the hyperparameters being tested
    print(f"Running trial {trial.number} with hyperparameters:")
    print(f"  model_dim = {model_dim}, num_heads = {num_heads}, num_layers = {num_layers}")
    print(f"  lr = {lr}, weight_decay = {weight_decay}")
    print(f"  alpha = {alpha}, beta = {beta}, delta = {delta}")

    # Initialize model
    model = MMFusionDetector(
        model_dim=model_dim, 
        num_heads=num_heads, 
        num_layers=num_layers, 
        alpha=alpha, 
        beta=beta, 
        delta=delta
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    # Dataset directories
    pt_dir = "./data/image_features_more_layers"
    pkl_dir = "./dataset/cam_box_per_image"

    # Initialize dataset
    dataset = PedestrianDetectorDataset(pkl_dir, pt_dir)

    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16, collate_fn=custom_collate)

    # Train the model using the train_model function
    trained_model = train_model(model, optimizer, scheduler, train_loader, val_loader, num_epochs=10, trial=trial.number)

    # Validation loss is returned by the evaluate_model function
    validation_loss = evaluate_model(trained_model, val_loader, trial=trial.number)
    
    # Return validation loss 
    return validation_loss  


def hypertune():
    print("Hypertuning started")

    # Create an Optuna study
    study = optuna.create_study(direction='minimize')  
    study.optimize(objective, n_trials=10, n_jobs=5)  

    # Print the best hyperparameters
    print(f"Best hyperparameters: {study.best_params}")


if __name__ == "__main__":
    hypertune()
    # # Dataset directories
    # pt_dir = os.path.expanduser("./data/image_features_more_layers")
    # pkl_dir = os.path.expanduser("./dataset/cam_box_per_image")

    # # Initialize dataset
    # dataset = PedestrianDetectorDataset(pkl_dir, pt_dir)

    # # Split the datasets for training, validation, and testing
    # train_size = int(0.7 * len(dataset))
    # val_size = int(0.2 * len(dataset))
    # test_size = len(dataset) - train_size - val_size
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # # Create DataLoaders
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16, collate_fn=custom_collate)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16, collate_fn=custom_collate)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=16, collate_fn=custom_collate)

    # # # Initialize model and optimizer
    # # model_dim = 256
    # # model = MMFusionDetector(model_dim)
    # # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # # # Initialize model and optimizer
    # # model_dim = 128  
    # # num_layers = 4  
    # # num_heads = 4  
    # # model = MMFusionPedestrianDetector(model_dim, num_heads=num_heads, num_layers=num_layers)
    # # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  

    # # Initialize model and optimizer
    # model_dim = 256  
    # num_layers = 6
    # num_heads = 8    
    # model = MMFusionPedestrianDetector(model_dim, num_heads=num_heads, num_layers=num_layers)

    # # Optimizer 
    # optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)

    # # Learning rate scheduler
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # # Train the model
    # print("Starting training...")
    # trained_model = train_model(
    #     model=model,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_epochs=20
    # )

    # # Evaluate the model on the test set
    # print("Evaluating on test set...")
    # evaluate_model(trained_model, test_loader)