import json
import os
from PIL import Image

class ArtifactDataset:
    def __init__(self, data_path: str):
        """
        Initializes the dataset loader.
        Assumes data_path points to a JSON file containing evaluation metadata:
        [
            {
                "image_path": "data/images/student1.jpg",
                "student_id": "123",
                "artifact_type": "Origami",
                "rubric": "1: No effort, 5: Perfect folds and presentation",
                "ground_truth_score": 4
            }, ...
        ]
        """
        self.data_path = data_path
        self.data = []
        
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        else:
            print(f"Warning: Dataset file {data_path} not found. Returning empty dataset.")
            print("Please create this file or generate a sample dataset.")
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item.get("image_path")
        
        try:
            # Handle absolute or relative paths gracefully based on the json directory
            base_dir = os.path.dirname(self.data_path)
            full_image_path = os.path.join(base_dir, image_path) if not os.path.isabs(image_path) else image_path
            image = Image.open(full_image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = None
            
        return {
            "image": image,
            "metadata": item
        }
