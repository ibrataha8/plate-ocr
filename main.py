import cv2
import os
from character_recognition import load_custom_model, plate_preparing
arabic_chars = "أبجدهوزحطيكلمنصعفضقرسشتثخذضظغ" #all arabic letters.
def format_plat(input_string, arabic_chars):
    """Splits a string based on the presence of Arabic characters and processes them correctly."""
    print(input_string)
    split_result = []
    current_part = ""

    for char in input_string:
        if char in arabic_chars:
            if current_part:
                split_result.append(current_part)
            split_result.append(char)
            current_part = ""
        else:
            current_part += char

    if current_part:
        split_result.append(current_part)

    # Vérifier le nombre de caractères arabes
    arabic_count = sum(1 for char in split_result if char in arabic_chars)

    # Si deux caractères arabes existent et que le premier est "أ", on remplace "أ" par "1" au lieu de le supprimer
    if arabic_count == 2:
        for i in range(len(split_result)):
            if split_result[i] in arabic_chars:
                if split_result[i] == "أ":
                    split_result[i] = "1"  # Remplace "أ" par "1"
                break

    # Fusion correcte des nombres sans ajouter d'espaces inutiles
    new_split_result = []
    temp_part = ""

    for part in split_result:
        if part in arabic_chars:
            if temp_part:
                new_split_result.append(temp_part)
                temp_part = ""
            new_split_result.append(part)
        else:
            temp_part += part  # Conserver les nombres ensemble sans espace

    if temp_part:
        new_split_result.append(temp_part)

    # Traitement final
    if len(new_split_result) > 0 and new_split_result[0].endswith("1"):
        new_split_result[0] = new_split_result[0][:-1]  # Supprime le dernier "1" de new_split_result[0]

    if len(new_split_result) > 2  and new_split_result[2].startswith("1"):
        new_split_result[2] = new_split_result[2][1:]  # Supprime le premier "1" de new_split_result[2]

    return " | ".join(new_split_result)
try:
    # Load image
    image_path = "./samples/six.jpg"
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found at {image_path}")
        image_path = input("Please enter the correct path to the license plate image: ")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Create output directory for debug images
    output_dir = "tmp/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original image for reference
    cv2.imwrite(f"{output_dir}/original_input.png", image)
        
    # Load model with custom loader
    model_path = "data/lp-character/char-rec-model.h5"
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        model_path = input("Please enter the correct path to the model file: ")
        
    print(f"Loading model from {model_path}...")
    model = load_custom_model(model_path)
    
    # Process image
    basename = os.path.splitext(os.path.basename(image_path))[0]
    plate = plate_preparing(image, basename, model)
    print("==>> " +format_plat(plate, arabic_chars))
    # print("==>> " +(plate)):
    
except Exception as e:
    print(f"Error: {str(e)}")