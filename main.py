import cv2
import os
from character_recognition import load_custom_model, plate_preparing

try:
    # Load image
    image_path = "./samples/plaque_20250228_110441_514426.jpg"
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found at {image_path}")
        image_path = input("Please enter the correct path to the license plate image: ")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
        
    # Load model with custom loader
    model_path = "data/lp-character/char-rec-model.h5"
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        model_path = input("Please enter the correct path to the model file: ")
        
    print(f"Loading model from {model_path}...")
    model = load_custom_model(model_path)
    
    # Process image
    plate = plate_preparing(image, "_the_0_license_plate_detected", model)
    print("==>> " + plate)
    
except Exception as e:
    print(f"Error: {str(e)}")