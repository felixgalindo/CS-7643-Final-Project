import os
import torch
from torch.utils.data import Dataset, DataLoader
import pickle


class PedestrianDetectorDataset(Dataset):
    def __init__(self, pkl_dir, pt_dir):
        """
        Initialize the dataset.
        Args:
            pkl_dir: Directory containing .pkl files with ground truth data.
            pt_dir: Directory containing .pt files with image features.
        """
        self.pkl_dir = pkl_dir
        self.pt_dir = pt_dir
        self.valid_samples = self._build_valid_samples()

        if not self.valid_samples:
            raise ValueError("No valid samples found. Please check your data directories.")

        print(f"Dataset initialized with {len(self.valid_samples)} valid samples.")
        print(f"First valid sample: {self.valid_samples[0]}")
        print(f"Type of first sample: {type(self.valid_samples[0])}")

    def _build_valid_samples(self):
        """
        Build a list of valid samples with matched .pkl and .pt files.
        Returns:
            valid_samples: List of tuples (pkl_path, pt_path).
        """
        valid_samples = []

        for root, _, files in os.walk(self.pkl_dir):
            for file in files:
                if file.endswith(".pkl"):
                    pkl_path = os.path.join(root, file)
                    pt_path = self._construct_pt_path(pkl_path)

                    if os.path.exists(pt_path):
                        valid_samples.append((pkl_path, pt_path))

        return valid_samples

    def _construct_pt_path(self, pkl_path):
        """
        Construct the corresponding .pt file path for a given .pkl file path.
        Args:
            pkl_path: Path to the .pkl file.
        Returns:
            Corresponding .pt file path.
        """
        relative_folder = os.path.relpath(os.path.dirname(pkl_path), self.pkl_dir)
        pkl_filename = os.path.splitext(os.path.basename(pkl_path))[0]

        containing_folder = os.path.basename(os.path.dirname(pkl_path))
        pt_folder = os.path.join(self.pt_dir, relative_folder)
        pt_file_path = os.path.join(pt_folder, f"{pkl_filename}.pt")
        pt_file_path = pt_file_path.replace(f"{containing_folder}_", "")
        pt_file_path = pt_file_path.replace("camera_image_camera_", "camera_image_camera-")

        return pt_file_path

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        pkl_path, pt_path = self.valid_samples[idx]

        # Load the pickle file
        with open(pkl_path, "rb") as file:
            ground_truth = pickle.load(file)

        if "box_type" in ground_truth and "box_loc" in ground_truth:
            ground_truth["classes"] = ground_truth["box_type"]
            ground_truth["boxes"] = ground_truth["box_loc"]
        else:
            print(f"Missing 'box_type' or 'box_loc' in ground_truth for {pkl_path}")
            print(f"Ground truth content: {ground_truth}")
            return None

        # Normalize classes
        ground_truth["classes"] = [
            1 if cls == 1 else (0 if cls == 0 else 2) for cls in ground_truth["classes"]
        ]

        # Validate and clean up box structures
        for i, box in enumerate(ground_truth["boxes"]):
            if len(box) != 4:
                print(f"Unexpected box length at index {i}: {box} (length: {len(box)}) in {pkl_path}")
            if not isinstance(box, (tuple, list)):
                print(f"Unexpected box type at index {i}: {type(box)} in {pkl_path}")

        # Load the .pt file
        image_features = torch.load(pt_path, weights_only=False)

        return image_features, ground_truth #classes [batch size, 7x7=49, num classes=3] boxes [batch size, 7x7=49, num classes=4(x,y,w,h)] 
    
def custom_collate(batch):
    batch = [item for item in batch if item is not None]  # Filter out None samples

    image_features = [item[0] for item in batch]
    ground_truth = [item[1] for item in batch]

    # Ensure all image features are tensors
    image_features = [torch.tensor(feat) if not isinstance(feat, torch.Tensor) else feat for feat in image_features]

    # Stack image features into a single tensor
    image_features = torch.stack(image_features)

    max_objects = max(len(gt["classes"]) for gt in ground_truth)
    padded_classes = []
    padded_boxes = []

    #Add padding to make all tensors in the batch same size
    for gt in ground_truth:
        num_objects = len(gt["classes"])

        padded_classes.append(torch.tensor(gt["classes"] + [0] * (max_objects - num_objects)))
        padded_boxes.append(torch.tensor(gt["boxes"] + [[0, 0, 0, 0]] * (max_objects - num_objects)))

    padded_classes = torch.stack(padded_classes)
    padded_boxes = torch.stack(padded_boxes)

    return image_features, {"classes": padded_classes, "boxes": padded_boxes} 

if __name__ == "__main__":
    pt_dir = os.path.expanduser("./data/image_features")
    pkl_dir = os.path.expanduser("./dataset/cam_box_per_image")

    print("pkl_dir:", pkl_dir)
    print("pt_dir:", pt_dir)

    # Initialize dataset and dataloader
    dataset = PedestrianDetectorDataset(pkl_dir, pt_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate,
    )

    # Test print
    for batch_features, batch_ground_truth in dataloader:
        print("Batch features shape:", batch_features.shape)
        print("Batch classes shape:", batch_ground_truth["classes"].shape)
        print("Batch boxes shape:", batch_ground_truth["boxes"].shape)
