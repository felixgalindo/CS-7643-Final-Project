import os
import io
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms


class ImgFeatureExtractor:
    def __init__(self, inDir, outDir, model):
        self.inDir = inDir
        self.outDir = outDir

        self.imgTransformer = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image 224x224
            transforms.ToTensor(),  # Converts PIL image into PyTorch tensor
            transforms.Normalize(
                mean=[
                    0.485, 0.456, 0.406], std=[
                    0.229, 0.224, 0.225]),  # Normalize image
        ])

        # Load the pre-trained model and remove the last two layers (AvgPool
        # and Fully Connected)
        self.model = model
        self.model = torch.nn.Sequential(*list(model.children())[:-2])
        self.model.eval()

    def extractFeatures(self, imagePath):
        """
        Extracts features from an image file.

        :param imagePath: Path to the image file.
        :return: Numpy array containing features.
        """
        image = Image.open(imagePath).convert(
            "RGB")  # Open image from file path
        imageTensor = self.imgTransformer(image).unsqueeze(
            0)  # Transform and add batch dimension

        with torch.no_grad():
            features = self.model(imageTensor)
            features = features.squeeze().numpy()  # Remove batch dimension
        print("Features extracted!")
        return features

    def parseJPGImages(self):
        """
        Extract features from all `.jpg` images in the input directory and subdirectories
        and save them as PyTorch Tensors
        """
        for root, dirs, files in os.walk(self.inDir):
            image_files = [f for f in files if f.endswith('.jpg')]

            for imageFile in tqdm(
                    image_files, desc=f"Processing Images in {root}"):
                imagePath = os.path.join(root, imageFile)
                features = self.extractFeatures(imagePath)

                relativePath = os.path.relpath(root, self.inDir)
                outputDir = os.path.join(self.outDir, relativePath)
                os.makedirs(outputDir, exist_ok=True)

                outputFile = os.path.join(
                    outputDir, imageFile.replace(
                        '.jpg', '.pt'))
                torch.save(features, outputFile)

                print(f"Features saved to {outputFile}")


if __name__ == "__main__":
    inDir = os.path.expanduser(
        os.path.dirname(
            os.path.abspath(__file__)) +
        "/dataset/compressed_camera_images2")
    outDir = os.path.expanduser(
        os.path.dirname(
            os.path.abspath(__file__)) +
        "/data/image_features")

    os.makedirs(outDir, exist_ok=True)

    # Initialize the feature extractor
    featureExtractor = ImgFeatureExtractor(
        inDir, outDir, models.resnet50(pretrained=True))

    # Process all images
    featureExtractor.parseJPGImages()
