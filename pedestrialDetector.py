import os
import io
import torch
import torch.nn as nn
import pickle

class PedestrianDetector(nn.Module):
    def __init__(self, modelDim=256, nClasses=2, nHeads=8,nLayers=6 ):
        super(PedestrianDetector, self).__init__()

        #Create embedding projectors
        self.cnnProjector = nn.Linear(2048, modelDim)
        #self.lidarProjector = nn.Linear(2048, modelDim)

        #Transformer
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(modelDim,nHeads),nLayers)

        #Create predictors
        self.boxPredictor = nn.Linear(modelDim,6) # x,y,x,len,width,height
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

if __name__ == "__main__":
    #inDir = os.path.expanduser(os.path.dirname(os.path.abspath(__file__)) +"/dataset/compressed_camera_images")
    #outDir = os.path.expanduser(os.path.dirname(os.path.abspath(__file__)) + "/data/compressed_camera_images")
    pickleFile = os.path.expanduser(os.path.dirname(os.path.abspath(__file__)) +"/dataset/camera_box_list_20241127.pkl")

    #os.makedirs(outDir, exist_ok=True)
    # Open the .pkl file in read-binary mode
    print(pickleFile)
    with open(pickleFile, 'rb') as file:
        data = pickle.load(file)

    for d in data:
        print(d)

    # groundTruth = []
    # # Initialize the Pedestrian Detector
    # featureExtractor = PedestrianDetector(groundTruth)


