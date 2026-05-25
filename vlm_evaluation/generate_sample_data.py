import json
import os
from PIL import Image, ImageDraw

SAMPLE_DATA = [
    {
        "image_path": "sample_origami.jpg",
        "student_id": "S001",
        "artifact_type": "Origami",
        "rubric": {
            "skill": "creativity",
            "dimension": "originality",
            "max": 5,
            "criteria": "1: No recognizable shape, 5: Perfect folds with clean edges and symmetry"
        },
        "ground_truth_score": 4,
    },
    {
        "image_path": "sample_drawing.jpg",
        "student_id": "S002",
        "artifact_type": "Drawing",
        "rubric": {
            "skill": "creativity",
            "dimension": "composition",
            "max": 5,
            "criteria": "1: No effort, 5: Detailed and creative composition"
        },
        "ground_truth_score": 3,
    },
    {
        "image_path": "sample_model.jpg",
        "student_id": "S003",
        "artifact_type": "Clay Model",
        "rubric": {
            "skill": "problem_solving",
            "dimension": "execution",
            "max": 5,
            "criteria": "1: Unrecognizable, 5: Realistic and well-finished model"
        },
        "ground_truth_score": 5,
    },
]


def create_dummy_image(path, size=(224, 224), color=(200, 100, 50)):
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 174, 174], outline=(255, 255, 255), width=3)
    draw.ellipse([80, 80, 144, 144], fill=(100, 200, 100))
    img.save(path)
    print(f"Created {path}")


def main():
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(output_dir, "sample_data")
    os.makedirs(data_dir, exist_ok=True)

    for item in SAMPLE_DATA:
        image_path = os.path.join(data_dir, item["image_path"])
        create_dummy_image(image_path)
        item["image_path"] = os.path.join("sample_data", item["image_path"])

    json_path = os.path.join(output_dir, "sample_dataset.json")
    with open(json_path, "w") as f:
        json.dump(SAMPLE_DATA, f, indent=2)

    print(f"Sample dataset saved to {json_path}")


if __name__ == "__main__":
    main()
