import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pedestrian_detector_dataset import PedestrianDetectorDataset, custom_collate


class MMFusionPedestrianDetector(nn.Module):
    def __init__(self, model_dim=256, num_classes=2, num_heads=8, num_layers=6):
        super(MMFusionPedestrianDetector, self).__init__()

        # Create embedding projectors
        self.cnn_projector = nn.Linear(2048, model_dim)

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(model_dim, num_heads), num_layers
        )

        # Create predictors
        self.box_predictor = nn.Linear(model_dim, 4)  # x, y, width, height
        self.class_predictor = nn.Linear(model_dim, num_classes)  # Pedestrian, non-pedestrian

        # Positional Encoder
        self.positional_encoder = nn.Parameter(torch.zeros(1, 500, model_dim))

    def forward(self, camera_features):
        # Create embeddings
        camera_embeddings = self.cnn_projector(camera_features)
        fused_embeddings = camera_embeddings

        # Add positional encodings
        fused_embeddings += self.positional_encoder[:, :fused_embeddings.size(1), :]

        # Get transformer output and make predictions
        transformer_output = self.transformer(fused_embeddings)
        classes = self.class_predictor(transformer_output)
        boxes = self.box_predictor(transformer_output)

        return classes, boxes


def train_model(dataloader, model_dim=256, num_classes=2, num_epochs=20, learning_rate=1e-4):
    # Initialize the model, loss functions, and optimizer
    model = MMFusionPedestrianDetector(model_dim, num_classes)
    class_loss_function = nn.CrossEntropyLoss()
    box_loss_function = nn.SmoothL1Loss()  # for bounding box regression loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    model.train()
    for epoch in range(num_epochs):
        epoch_class_loss = 0.0
        epoch_box_loss = 0.0

        for batch_features, batch_ground_truth in dataloader:
            optimizer.zero_grad()

            # Forward pass
            predicted_classes, predicted_boxes = model(batch_features)
            ground_truth_classes = batch_ground_truth["classes"]
            ground_truth_boxes = batch_ground_truth["boxes"]

            # Compute loss
            class_loss = class_loss_function(
                predicted_classes.view(-1, num_classes), ground_truth_classes.view(-1)
            )
            box_loss = box_loss_function(predicted_boxes, ground_truth_boxes)
            total_loss = class_loss + box_loss

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            epoch_class_loss += class_loss.item()
            epoch_box_loss += box_loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs} - Class Loss: {epoch_class_loss:.4f}, Box Loss: {epoch_box_loss:.4f}")

    return model


if __name__ == "__main__":
    # Define dataset directories
    pt_dir = os.path.expanduser("./data/image_features")
    pkl_dir = os.path.expanduser("./dataset/cam_box_per_image")

    # Initialize dataset and dataloader
    dataset = PedestrianDetectorDataset(pkl_dir, pt_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate,
    )

    # Train the model
    trained_model = train_model(dataloader, model_dim=256, num_classes=2, num_epochs=20, learning_rate=1e-4)
