import torch
import cv2
import numpy as np
import json
import codecs
import sys
from model import TrafficSignCNN
import torchvision.transforms as transforms
from PIL import Image

def predict(image_path):
    # 1. Load label names
    try:
        label_json = codecs.open("DataProfiling/label_names.json", 'r', encoding='utf-8').read()
        label_names = json.loads(label_json)
    except FileNotFoundError:
        print("Error: label_names.json not found in DataProfiling directory.")
        return

    # 2. Load model
    device = torch.device('cpu')
    model = TrafficSignCNN(43)
    try:
        model.load_state_dict(torch.load('serialized_data/model.pt', map_location=device))
        model.to(device)
        model.eval()
    except FileNotFoundError:
        print("Error: model.pt not found in serialized_data directory.")
        return

    # 3. Load and Pre-process Image
    try:
        # Using the same logic as DatasetLoader.py
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return
        
        image = cv2.resize(image, (32, 32))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension

        # 4. Predict
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
            confidence = torch.nn.functional.softmax(output, dim=1)[0][prediction].item()

        # 5. Result
        class_name = label_names.get(str(prediction), "Unknown")
        print("-" * 30)
        print(f"Prediction for: {image_path}")
        print(f"Class ID: {prediction}")
        print(f"Label: {class_name}")
        print(f"Confidence: {confidence:.2%}")
        print("-" * 30)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        # Default to a test image if no argument provided
        test_img = "data/Test/00000.png"
        import os
        if os.path.exists(test_img):
            print(f"Testing with default image: {test_img}")
            predict(test_img)
        else:
            print("Please provide an image path.")
    else:
        predict(sys.argv[1])
