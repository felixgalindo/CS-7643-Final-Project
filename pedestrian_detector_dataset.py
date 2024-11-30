import os
import pickle
import torch
from torch.utils.data import DataLoader, Dataset


class PedestrianDetectorDataset(Dataset):
    def __init__(self, features, ground_truth):
        """
        Args:
            features: List of feature tensors (.pt files).
            ground_truth: List of dictionaries with classes and boxes (.pkl files).
        """
        self.features = features
        self.ground_truth = ground_truth

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.
        """
        feature_tensor = self.features[idx]
        ground_truth = self.ground_truth[idx]

        # Extract ground truth components
        boxes = torch.tensor(ground_truth["boxes"], dtype=torch.float32)
        classes = torch.tensor(ground_truth["classes"], dtype=torch.long)

        return feature_tensor, boxes, classes

    def __len__(self):
        return len(self.features)


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
                pklFilename = os.path.splitext(
                    pklFile)[0]  # Remove .pkl extension

                containingFolder = os.path.basename(root)
                # print(f"File: {pklFile} Containing Folder: {containingFolder}")

                # Construct .pt file path
                ptFolder = os.path.join(ptDir, relativeFolder)
                ptFilePath = os.path.join(ptFolder, f"{pklFilename}.pt")
                ptFilePath = ptFilePath.replace(f"{containingFolder}_", "")
                ptFilePath = ptFilePath.replace(
                    "camera_image_camera_", "camera_image_camera-")
                # print(ptFilePath)
                # print(containingFolder)

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


# Main script
if __name__ == "__main__":
    ptDir = os.path.expanduser(
        os.path.dirname(
            os.path.abspath(__file__)) +
        "/data/image_features")
    pklDir = os.path.expanduser(
        os.path.dirname(
            os.path.abspath(__file__)) +
        "/dataset/cam_box_per_image")

    print("pklDir:", pklDir)
    print("ptDir:", ptDir)

    # Align features and ground truth
    features, ground_truth = AlignFeaturesAndGroundTruth(pklDir, ptDir)

    # Create dataset and dataloader
    dataset = PedestrianDetectorDataset(features, ground_truth)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4)

    # Iterate through batches
    for batch_features, batch_ground_truth in dataloader:
        print("Batch features shape:", batch_features.shape)
        print("Batch ground truth:", batch_ground_truth)
