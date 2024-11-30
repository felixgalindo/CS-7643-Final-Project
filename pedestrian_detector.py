import os
import io
import torch
import torch.nn as nn
import pickle


class MMFusionPedestrianDetector(nn.Module):
    def __init__(self, modelDim=256, nClasses=2, nHeads=8, nLayers=6):
        super(MMFusionPedestrianDetector, self).__init__()

        # Create embedding projectors
        self.cnnProjector = nn.Linear(2048, modelDim)
        # self.lidarProjector = nn.Linear(2048, modelDim)

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                modelDim, nHeads), nLayers)

        # Create predictors
        self.boxPredictor = nn.Linear(modelDim, 4)  # x,y,width,height
        self.classPredictor = nn.Linear(
            modelDim, 2)  # Pedistrian, non-pedstrian

        # Positional Encoder
        self.positionalEncoder = nn.Parameter(torch.zeros(500), modelDim)

    def forward(self, cameraFeatures):  # , lidarFeatures):

        # Create embeddings
        cameraEmbeddings = self.cnnProjector(cameraFeatures)
        # lidarEmbeddings = self.cnnProjector(lidarFeatures)
        fusedEmbeddings = cameraEmbeddings
        # fusedEmbeddings = torch.cat([cameraEmbeddings,lidarEmbeddings], dim=1)

        # Add the positional encodings
        fusedEmbeddings = self.positionalEncoder[:fusedEmbeddings.size(1)]

        # Get transformer output and make predictions
        transformerOut = self.transformer(fusedEmbeddings)
        classes = self.classPredictor(transformerOut)
        boxes = self.boxPredictor(transformerOut)

        return classes, boxes


def TrainModel(cameraFeatures, lidarFeatures=None, modelDim=256,
               nClasses=2, batchSize=10, nEpochs=20, lr=1e-4):

    # Intialize
    model = MMFusionPedestrianDetector(modelDim)
    classLossFunction = nn.CrossEntropyLoss()
    boxLossFunction = nn.SmoothL1Loss()  # for bounding box regression loss
    optimizer = optim.Adam(model.paramaters(), lr)

    # Train the model
    model.train()
    for e in range(nEpochs):
        classLoss = 0.0
        boxLoss = 0.0

        # Fwd Pass and compute loss
        predictedClasses, predictedBoxes = model(
            cameraFeatures)  # ,lidarFeatures)
        # classLoss = classLossFunction()
        classLoss = classificationLossFunction(
            predictedClasses.view(-1, nClasses), groundTruthClasses.view(-1)
        )
        boxLoss = boxLossFunction(predictedBoxes, groundTruthBoxes)
        totalLoss = classLoss + boxLoss

        # Now run backward pass
        totalLoss.backward()
        optimizer.step()
    return model


if __name__ == "__main__":
    ptDir = os.path.expanduser(
        os.path.dirname(
            os.path.abspath(__file__)) +
        "/data/image_features")
    pklDir = os.path.expanduser(
        os.path.dirname(
            os.path.abspath(__file__)) +
        "/dataset/cam_box_per_image")
