import os
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms
import torch.nn.functional as F


class ImgFeatureExtractor:
    def __init__(self, inDir, outDir, model):
        self.inDir = inDir
        self.outDir = outDir

        self.imgTransformer = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to 224x224
            transforms.ToTensor(),  # Convert PIL image into PyTorch tensor
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]  # Normalize using ImageNet stats
            ),
        ])

        # Extract multiple layers
        self.model = model
        self.feature_layers = {
            "layer1": self.model.layer1,
            "layer2": self.model.layer2,
            "layer3": self.model.layer3,
            "layer4": self.model.layer4,
        }

        self.model.eval()

    def extractFeatures(self, imagePath):
        """
        Extract features from multiple layers of the model for one image.

        :param imagePath: Path to the image file.
        :return: Combined feature tensor.
        """
        image = Image.open(imagePath).convert("RGB")  # Open image from file path
        imageTensor = self.imgTransformer(image).unsqueeze(0)  # Transform and add batch dimension

        features = []
        with torch.no_grad():
            # Forward pass through ResNet initial layers
            x = self.model.conv1(imageTensor)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            # Extract features from all layers
            for layer_name, layer_module in self.feature_layers.items():
                x = layer_module(x)
                # Upsample to (7, 7) resolution if needed
                if x.size(-1) != 7 or x.size(-2) != 7:
                    x = F.interpolate(x, size=(7, 7), mode='bilinear', align_corners=False)
                features.append(x)

        # Concatenate features along the channel dimension
        combined_features = torch.cat(features, dim=1)  # Shape: [batch_size, 3840, 7, 7]
        #print(f"Features extracted and combined: {combined_features.shape}")
        return combined_features


    def parseJPGImages(self):
        """
        Extract features from all `.jpg` images in the input directory and subdirectories
        and save them as PyTorch Tensors.
        """
        for root, dirs, files in os.walk(self.inDir):
            image_files = [f for f in files if f.endswith('.jpg')]

            for imageFile in tqdm(image_files, desc=f"Processing Images in {root}"):
                imagePath = os.path.join(root, imageFile)
                features = self.extractFeatures(imagePath)

                relativePath = os.path.relpath(root, self.inDir)
                outputDir = os.path.join(self.outDir, relativePath)
                os.makedirs(outputDir, exist_ok=True)

                outputFile = os.path.join(outputDir, imageFile.replace('.jpg', '.pt'))
                torch.save(features, outputFile)
                #print(f"Saved feature shape: {features.shape}")

                #print(f"Features saved to {outputFile}")


if __name__ == "__main__":
    inDir = os.path.expanduser(os.path.dirname(os.path.abspath(__file__)) + "/dataset/compressed_camera_images2")
    outDir = os.path.expanduser(os.path.dirname(os.path.abspath(__file__)) + "/data/image_features_more_layers")

    os.makedirs(outDir, exist_ok=True)

    # Initialize the feature extractor with ResNet-50
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    featureExtractor = ImgFeatureExtractor(inDir, outDir, resnet50)

    # Process all images
    featureExtractor.parseJPGImages()
