import os
import io
import torch
import torch.nn as nn
import pickle

class MMFusionPedestrianDetector(nn.Module):
    def __init__(self, modelDim=256, nClasses=2, nHeads=8,nLayers=6 ):
        super(PedestrianDetector, self).__init__()

        #Create embedding projectors
        self.cnnProjector = nn.Linear(2048, modelDim)
        #self.lidarProjector = nn.Linear(2048, modelDim)

        #Transformer
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(modelDim,nHeads),nLayers)

        #Create predictors
        self.boxPredictor = nn.Linear(modelDim,4) # x,y,width,height
        self.classPredictor = nn.Linear(modelDim,2) # Pedistrian, non-pedstrian

        #Positional Encoder
        self.positionalEncoder = nn.Parameter(torch.zeros(500), modelDim)

    def forward(self, cameraFeatures):#, lidarFeatures):

        #Create embeddings
        cameraEmbeddings = self.cnnProjector(cameraFeatures)
        #lidarEmbeddings = self.cnnProjector(lidarFeatures)
        fusedEmbeddings = cameraEmbeddings
        #fusedEmbeddings = torch.cat([cameraEmbeddings,lidarEmbeddings], dim=1)

        #Add the positional encodings
        fusedEmbeddings = self.positionalEncoder[:fusedEmbeddings.size(1)]

        #Get transformer output and make predictions
        transformerOut = self.transformer(fusedEmbeddings)
        classes = self.classPredictor(transformerOut)
        boxes = self.boxPredictor(transformerOut)

        return classes,boxes

def TrainModel(cameraFeatures, lidarFeatures=None, modelDim=256, nClasses=2, batchSize=10, nEpochs=20,lr=1e-4):

    #Intialize 
    model = MMFusionPedestrianDetector(modelDim)
    classLossFunction = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.paramaters(), lr)

    #Train the model
    model.train()
    for e in range(nEpochs):
        classLoss= 0.0
        boxLoss = 0.0

        #Fwd Pass and compute loss
        predictedClasses, predictedBoxes = model(cameraFeatures)#,lidarFeatures)
        #classLoss = classLossFunction()
        boxLoss = boxLossFunction(predictedBoxes)
        totalLoss = classLoss + boxLoss

        #Now run backward pass
        totalLoss.backward()
        optimizer.step()
    return model

def AlignFeaturesAndGroundTruth(pklDir, ptDir):
    """
    Align .pkl ground truth data with .pt image features
    Args:
        pklDir: Directory containing .pkl files with ground truth data.
        ptDir: Directory containing .pt files with image features.
    Returns:
        features: List of image feature tensors.
        gndTruth: List of dictionaries with ground truth classes and boxes.
    """
    features = []
    gndTruth = []
    missing = 0
    total = 0
    # Traverse all .pkl files in the directory
    for root, _, files in os.walk(pklDir):
        for pklFile in files:
            if pklFile.endswith('.pkl'):
                # Load ground truth from the .pkl file
                pklPath = os.path.join(root, pklFile)
                with open(pklPath, 'rb') as f:
                    groundTruth = pickle.load(f)

                # Extract relative folder name and base file name
                relativeFolder = os.path.relpath(root, pklDir)
                pklFilename = os.path.splitext(pklFile)[0]  # Remove .pkl extension

                containingFolder = os.path.basename(root)
                #print(f"File: {pklFile} | Containing Folder: {containingFolder}")

                # Construct .pt file path
                ptFolder = os.path.join(ptDir, relativeFolder)
                ptFilePath = os.path.join(ptFolder, f"{pklFilename}.pt")
                ptFilePath = ptFilePath.replace(f"{containingFolder}_", "")
                ptFilePath = ptFilePath.replace("camera_image_camera_", "camera_image_camera-")
                #print(ptFilePath)
                #print(containingFolder)

                # Verify the .pt file exists
                if os.path.exists(ptFilePath):
                    # Load the .pt tensor
                    imgFeatureTensor = torch.load(ptFilePath)

                    # Append to the lists
                    features.append(imgFeatureTensor)
                    gndTruth.append(groundTruth)
                else:
                    print(f"Warning: Missing .pt file for {pklPath}")
                    missing += 1

                total += 1
    print("missing: ", missing)
    print("total: ", total)

    return features, gndTruth

if __name__ == "__main__":
    ptDir = os.path.expanduser(os.path.dirname(os.path.abspath(__file__)) +"/data/image_features")
    pklDir = os.path.expanduser(os.path.dirname(os.path.abspath(__file__)) + "/dataset/cam_box_per_image")

    print(pklDir)
    print(ptDir)

    # Align features and ground truth
    features, gndTruth = AlignFeaturesAndGroundTruth(pklDir, ptDir)

    # Example: Inspect the first entries
    if features and gndTruth:
        print("Example Image Features:")
        print(features[0])
        print("Example Ground Truth:")
        print(gndTruth[0])


