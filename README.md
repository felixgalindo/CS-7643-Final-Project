
# Multimodal Pedestrian Detection with Waymo Dataset

This project focuses on **late fusion multimodal pedestrian detection** using **Waymo’s Open Dataset**, leveraging **transformers** to combine features from lidar and camera data.


## **Setup Instructions**

### **Prerequisites**
Ensure you have the following installed:
- **Conda**: [Installation Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

## **Step 1: Clone the Repository**

```bash
git clone https://github.com/125918700/CS-7643-Final-Project.git
cd ./CS-7643-Final-Project
```

## **Step 2: Create and Activate Conda Environment**

```bash
conda create -n mm-fusion-ped-det python=3.10
conda activate mm-fusion-ped-det
```

## **Step 3: Install Dependencies**

### **3.1 Install TensorFlow**

To ensure compatibility with Waymo tools, install TensorFlow **2.11.0**:

```bash
pip install tensorflow==2.11.0
```

### **3.2 Install Core Libraries**

```bash
pip install torch torchvision torchaudio transformers numpy
```

### **3.3 Install Waymo Open Dataset Tools**

For TensorFlow **2.11+**, install Waymo Open Dataset Tools using pip:

```bash
pip install waymo-open-dataset-tf-2-11-0
```

## **Step 4: Verify Installation**

Run the following script to verify your environment setup:

```bash
python -c "
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
import torch

print('TensorFlow Version:', tf.__version__)
print('PyTorch Version:', torch.__version__)
print('Waymo Open Dataset Installed Successfully!')
"
```

## **Usage**

Once the environment is set up, you can run the project scripts for pedestrian detection:

```bash
python main.py
```

## **Contributing**

Please fork the repository and submit a pull request for any improvements.

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


