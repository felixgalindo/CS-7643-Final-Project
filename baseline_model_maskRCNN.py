#Musk R CNN
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
# Load Pretrained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()  

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
def load_image(image_path):
    # Read image with OpenCV
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    # Convert to tensor and normalize
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)  # Add batch dimension
    return img, img_tensor
def get_predictions(img_tensor, threshold=0.7):
    # Get model predictions
    outputs = model(img_tensor)[0]

    # Filter predictions by score threshold
    pred_boxes = outputs['boxes'].detach().cpu().numpy()
    pred_scores = outputs['scores'].detach().cpu().numpy()
    pred_classes = outputs['labels'].detach().cpu().numpy()
    pred_masks = outputs['masks'].detach().cpu().numpy()

    # Keep only predictions above the threshold and 3 as cars
    high_conf_indices = (pred_scores > threshold) & (pred_classes == 3)
    pred_boxes = pred_boxes[high_conf_indices]
    pred_classes = pred_classes[high_conf_indices]
    pred_masks = pred_masks[high_conf_indices]
    return pred_boxes, pred_classes, pred_masks
def visualize_predictions(img, boxes, masks, labels, class_names):
    # Apply masks and draw bounding boxes on the image
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)

        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Label the bounding box
        label = class_names[labels[i]]
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Apply mask
        mask = masks[i, 0] > 0.5
        img[mask] = [0, 255, 0]  # Green mask

    plt.imshow(img)
    plt.axis("off")
    plt.show()
# Define class names for COCO dataset
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", 
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Function to process a folder of images and return bounding boxes
def process_image_folder(folder_path):
    all_boxes = {}  # Dictionary to store bounding boxes for each image
    
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        
        if image_path.endswith(".jpg") or image_path.endswith(".png"):
            print(f"Processing {filename}...")
            
            original_img, img_tensor = load_image(image_path)
            # Get bounding boxes for the current image
            boxes, labels, masks = get_predictions(img_tensor, threshold=0.7)
            all_boxes[filename] = boxes
            
            # Optionally visualize the bounding boxes
            # visualize_predictions(original_img, boxes, masks, labels, COCO_CLASSES)
    
    return all_boxes

# path to your folder
folder_path = r"C:\Users\pwang\OneDrive\Documents\GitHub\CS-7643-Final-Project\sample_images"
process_image_folder(folder_path)
# # Get predictions
# boxes, labels, masks = get_predictions(img_tensor, threshold=0.7)
# print(boxes)
