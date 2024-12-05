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

class MMFusionPedestrianDetector(nn.Module):
    def __init__(self, model_dim=256, num_heads=8, num_layers=6, num_queries=100, num_classes=2, alpha=10, beta=10, delta=.1):
        super(MMFusionPedestrianDetector, self).__init__()
        self.model_dim = model_dim
        self.num_queries = num_queries
        self.alpha=alpha
        self.delta=delta
        self.beta1=beta

        # Project features to model_dim
        self.featureProjector = nn.Linear(3840, model_dim)

        # Positional encoding only for encoder input (image features)
        self.encoderPositionalEncoder = sinusoidal_positional_encoding(49, model_dim)

        # Transformer (Encoder and Decoder)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(model_dim, num_heads), num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(model_dim, num_heads), num_layers
        )

        # Learned object queries for the decoder (removed positional encoding on queries)
        self.objectQueries = nn.Parameter(torch.randn(num_queries, model_dim))

        # Prediction heads
        self.boxPredictor = nn.Sequential(nn.Linear(model_dim, 4), nn.Sigmoid())
        self.classPredictor = nn.Linear(model_dim, num_classes)

    def forward(self, batchFeatures):
        """
        Forward pass using pre-extracted features.

        :param batchFeatures: Tensor of shape [batch_size, 1, 3840, 7, 7].
        :return: Predicted classes and boxes.
        """
        batchFeatures = batchFeatures.squeeze(1)  # Shape: [batch_size, 1, 3840, 7, 7]
        batchSize, _, height, width = batchFeatures.size()
        seq_len = height * width  # Should be 49 for 7x7

        # Reshape batch features for transformer
        batchFeatures = batchFeatures.view(batchSize, seq_len, -1)  # [batch_size, 49, 3840]

        # Project features 
        projectedFeatures = self.featureProjector(batchFeatures)  # [batch_size, seq_len, model_dim]

        # Encoder
        encoderPositionalEncodings = self.encoderPositionalEncoder.to(projectedFeatures.device)
        encoderOut = self.encoder(projectedFeatures + encoderPositionalEncodings)  # [batch_size, seq_len, model_dim]
        encoderOut = encoderOut.permute(1, 0, 2)  # [seq_len, batch_size, model_dim]

        # Decoder
        objectQueries = self.objectQueries.unsqueeze(0).expand(batchSize, -1, -1)  # [batch_size, num_queries, model_dim]
        decoderOut = self.decoder(objectQueries.permute(1, 0, 2), encoderOut)  # [num_queries, batch_size, model_dim]
        decoderOut = decoderOut.permute(1, 0, 2)  # [batch_size, num_queries, model_dim]

        # Predictions
        predictedClasses = self.classPredictor(decoderOut)  # [batch_size, num_queries, num_classes]
        predictedBoxes = self.boxPredictor(decoderOut)  # [batch_size, num_queries, 4]

        return predictedClasses, predictedBoxes


