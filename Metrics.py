# Metrics
import numpy as np
import matplotlib.pyplot as plt
import os
from tabulate import tabulate

# Function to calculate IoU between two boxes
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    intersection_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0.0
    return intersection_area / union_area

# Function to evaluate bounding box accuracy using IoU
def evaluate_boxes(predicted_boxes, ground_truth_boxes, iou_threshold=0.5):
    assert len(predicted_boxes) == len(ground_truth_boxes)

    iou_scores = []
    true_positives = 0

    for pred_box, gt_box in zip(predicted_boxes, ground_truth_boxes):
        iou = calculate_iou(pred_box, gt_box)
        iou_scores.append(iou)

        if iou >= iou_threshold:
            true_positives += 1

    mean_iou = np.mean(iou_scores) if iou_scores else 0.0
    precision = true_positives / len(predicted_boxes) if predicted_boxes else 0.0
    recall = true_positives / len(ground_truth_boxes) if ground_truth_boxes else 0.0

    return {
        "mean_iou": mean_iou,
        "precision": precision,
        "recall": recall
    }

# Function to evaluate metrics for all images
def evaluate_group_of_images(predictions, ground_truths, iou_threshold=0.5):
    """
    Evaluate metrics for all images.

    """
    all_metrics = {"mean_iou": [], "precision": [], "recall": []}

    for image_name in predictions.keys():
        predicted_boxes = predictions[image_name]
        ground_truth_boxes = ground_truths.get(image_name, [])
        
        metrics = evaluate_boxes(predicted_boxes, ground_truth_boxes, iou_threshold)
        for key in metrics:
            all_metrics[key].append(metrics[key])

    # Calculate mean metrics across all images
    mean_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    return mean_metrics

def plot_comparison(metrics_model1, metrics_model2, model_names):
    """
    Plot comparison between two models.
    
    """
    labels = list(metrics_model1.keys())
    values_model1 = list(metrics_model1.values())
    values_model2 = list(metrics_model2.values())

    x = np.arange(len(labels))  # Label locations
    width = 0.3  # Bar width

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, values_model1, width, label=model_names[0], color='skyblue')
    bars2 = ax.bar(x + width/2, values_model2, width, label=model_names[1], color='orange')

    ax.set_ylabel('Score')
    ax.set_title('Comparison of Metrics Between Models')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Adding value annotations
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset text by 3 points
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.ylim(0, 1)  # Limit y-axis to [0, 1] for normalized metrics
    plt.show()

def display_metrics_table(metrics_model1, metrics_model2, model_names):
    """
    Display a comparison table for metrics.

    Parameters:
        metrics_model1: dict, metrics for model 1
        metrics_model2: dict, metrics for model 2
        model_names: list of strings, names of the two models
    """
    headers = ["Metric", model_names[0], model_names[1]]
    data = [
        [key, f"{metrics_model1[key]:.2f}", f"{metrics_model2[key]:.2f}"]
        for key in metrics_model1.keys()
    ]
    print(tabulate(data, headers=headers, tablefmt="fancy_grid"))

# Example ground truth and predictions for two models
ground_truths = {
    "image1.jpg": [[50, 50, 100, 100], [200, 200, 250, 250]],
    "image2.jpg": [[60, 60, 90, 90], [185, 185, 245, 245]],
}

predictions_fusion = {
    "image1.jpg": [[50, 50, 100, 100], [200, 200, 250, 250]],
    "image2.jpg": [[55, 55, 95, 95], [190, 190, 240, 240]],
}

predictions_mrcnn = {
    "image1.jpg": [[48, 48, 102, 102], [198, 198, 252, 252]],
    "image2.jpg": [[57, 57, 93, 93], [188, 188, 242, 242]],
}

# Evaluate metrics for both models
metrics_model1 = evaluate_group_of_images(predictions_fusion, ground_truths, iou_threshold=0.5)
metrics_model2 = evaluate_group_of_images(predictions_mrcnn, ground_truths, iou_threshold=0.5)

# Display metrics table
display_metrics_table(metrics_model1, metrics_model2, model_names=["Fusion", "Musked_r_cnn"])
# Plot comparison
plot_comparison(metrics_model1, metrics_model2, model_names=["Fusion", "Musked_r_cnn"])