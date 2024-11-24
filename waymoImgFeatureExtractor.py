import os
import io
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms
#from waymo_open_dataset.protos import dataset_pb2  # For Frame protobuf
from waymo_open_dataset import dataset_pb2 as open_dataset
import tensorflow as tf


class WaymoImgFeatureExtractor:
    def __init__(self, inDir, outDir, modelName="resnet50"):
        self.inDir = inDir
        self.outDir = outDir
        self.modelName = modelName

        self.imgTransformer = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to 224x224
            transforms.ToTensor(),  # Converts PIL image into PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize image
        ])

        # Load the pre-trained model and remove the last two layers (AvgPool and Fully Connected)
        model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(model.children())[:-2])
        self.model.eval()

    def extractFeatures(self, image):
        """
        Extracts features from an image.

        :param image: Raw image in bytes.
        :return: Numpy array of features.
        """
        image = Image.open(io.BytesIO(image)).convert("RGB")
        imageTensor = self.imgTransformer(image).unsqueeze(0)

        with torch.no_grad():
            features = self.model(imageTensor)
            features = features.squeeze().numpy()  # Remove batch dimension
        print("Features extracted!")
        return features

    def parseTFRecord(self, fileName):
        """
        Extracts features from a Waymo Perception Dataset TFRecord file and saves them as PyTorch Tensors.
        :param fileName: TFRecord file name.
        """
        filePath = os.path.join(self.inDir, fileName)
        dataset = tf.data.TFRecordDataset(filePath)

        # List to store extracted features
        featuresList = []

        # Process each record in the TFRecord
        for raw_record in tqdm(dataset, desc=f"Processing {fileName}"):
            frame = open_dataset.Frame()
            frame.ParseFromString(raw_record.numpy())

            # Extract images from the frame
            for image_data in frame.images:
                camera_name = open_dataset.CameraName.Name.Name(image_data.name)
                print(f"Processing image from camera: {camera_name}")

                # Extract image features
                image_bytes = image_data.image  # Raw image bytes
                features = self.extractFeatures(image_bytes)
                featuresList.append((camera_name, features))

        # Save the features as a tensor in a .pt file
        outputFile = os.path.join(self.outDir, fileName.replace('.tfrecord', '.pt'))
        torch.save(featuresList, outputFile)

        print(f"Features extracted and saved to {outputFile}")

    def parseAllTFRecords(self):
        """
        Process all TFRecord files in the input directory and extract features from images
        """
        tfrecord_files = [f for f in os.listdir(self.inDir) if f.endswith('.tfrecord')]

        for fileName in tfrecord_files:
            print(f"Processing file: {fileName}")
            self.parseTFRecord(fileName)


if __name__ == "__main__":
    inDir = os.path.expanduser("~/CS-7643-Final-Project/waymo_open_dataset_v_1_4_3/individual_files/validation")
    outDir = os.path.expanduser("~/CS-7643-Final-Project/waymo_open_dataset_v_1_4_3/individual_files/validation")

    # Ensure output directory exists
    os.makedirs(outDir, exist_ok=True)

    # Initialize the feature extractor
    featureExtractor = WaymoImgFeatureExtractor(inDir, outDir)

    # Process all TFRecords in the directory
    featureExtractor.parseAllTFRecords()
