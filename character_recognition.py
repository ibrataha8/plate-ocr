import cv2
import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# Custom objects for model loading (unchanged)
class CustomDepthwiseConv2D(keras.layers.DepthwiseConv2D):
    def __init__(self, kernel_size, strides=(1, 1), padding='valid', 
                 depth_multiplier=1, data_format=None, dilation_rate=(1, 1), 
                 activation=None, use_bias=True, depthwise_initializer='glorot_uniform', 
                 bias_initializer='zeros', depthwise_regularizer=None, 
                 bias_regularizer=None, activity_regularizer=None, 
                 depthwise_constraint=None, bias_constraint=None, groups=1, **kwargs):
        # Ignore 'groups' parameter
        super().__init__(
            kernel_size=kernel_size, 
            strides=strides, 
            padding=padding,
            depth_multiplier=depth_multiplier, 
            data_format=data_format, 
            dilation_rate=dilation_rate, 
            activation=activation, 
            use_bias=use_bias, 
            depthwise_initializer=depthwise_initializer, 
            bias_initializer=bias_initializer, 
            depthwise_regularizer=depthwise_regularizer, 
            bias_regularizer=bias_regularizer, 
            activity_regularizer=activity_regularizer, 
            depthwise_constraint=depthwise_constraint, 
            bias_constraint=bias_constraint, 
            **kwargs
        )
        
    def get_config(self):
        config = super().get_config()
        # Adding 'groups' to maintain compatibility
        config.update({'groups': 1})
        return config

def load_custom_model(model_path):
    """Custom model loader with special handling for compatibility issues"""
    try:
        # Direct custom objects approach
        custom_objects = {
            'DepthwiseConv2D': CustomDepthwiseConv2D,
        }
        return keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    except Exception as e:
        print(f"Loading with custom objects failed: {str(e)}")
        
        # Try with TF custom objects registry
        try:
            tf.keras.utils.get_custom_objects().update({'DepthwiseConv2D': CustomDepthwiseConv2D})
            return keras.models.load_model(model_path, compile=False)
        except Exception as e2:
            print(f"Loading with TF custom objects failed: {str(e2)}")
            
            # Last resort: Load with separate model loading logic
            try:
                print("Attempting alternative model loading approach...")
                # Use lower level functions if needed
                # This approach might vary depending on your TF version
                return tf.saved_model.load(model_path)
            except Exception as e3:
                print(f"All loading attempts failed: {str(e3)}")
                raise ValueError("Unable to load the model. You may need to retrain it with your current TensorFlow version.")

def plate_preparing(license_plate_image, basename, model):
    output_dir = "tmp/output"
    os.makedirs(output_dir, exist_ok=True)
    print(f'Processing image {output_dir}/{basename}.png')
    
    # Save the original image for debugging
    cv2.imwrite(f"{output_dir}/{basename}_original.png", license_plate_image)
    
    # Resize image - adjustment for better stability
    # Maintain height/width ratio
    h, w, _ = license_plate_image.shape
    new_width = 600
    new_height = int(h * (new_width / w))
    resized_image = cv2.resize(license_plate_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Save the resized image for debugging
    cv2.imwrite(f"{output_dir}/{basename}_resized.png", resized_image)
    
    # Use the full image
    image = resized_image
    
    # Convert to grayscale and process
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # IMPROVED: Apply contrast enhancement before blurring
    # This helps with digit '1' detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # IMPROVED: Try different thresholding methods
    # 1. Adaptive thresholding with adjusted parameters (more sensitive)
    thresh1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 8)
    
    # 2. Otsu's thresholding for comparison
    _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. IMPROVED: Add a simple threshold to catch thin features like digit '1' and Arabic 'alif'
    _, thresh3 = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Combine the three methods for better results
    thresh = cv2.bitwise_or(cv2.bitwise_or(thresh1, thresh2), thresh3)
    
    # Save thresholding for debugging
    cv2.imwrite(f"{output_dir}/{basename}_thresh.png", thresh)
    
    # IMPROVED: Apply morphological operations to improve character integrity
    # This helps preserve thin parts and close small gaps
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(f"{output_dir}/{basename}_morph.png", thresh)
    
    # Find connected components
    _, labels = cv2.connectedComponents(thresh)
    mask = np.zeros(thresh.shape, dtype="uint8")
    
    # IMPROVED: Adjust filtering thresholds to be more permissive, especially for thin characters
    total_pixels = image.shape[0] * image.shape[1]
    lower = total_pixels // 300  # Smaller value to include smaller characters like '1' and 'alif'
    upper = total_pixels // 8    # Larger value to include big regions
    
    # Loop through unique components
    for (_, label) in enumerate(np.unique(labels)):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
   
        if numPixels > lower and numPixels < upper:
            mask = cv2.add(mask, labelMask)
    
    # Save the mask for debugging
    cv2.imwrite(f"{output_dir}/{basename}_mask.png", mask)
    
    # Find contours
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # IMPROVED: Better bounding box filtering
    min_valid_width = 5  # Minimum width to be considered valid (for digit '1' or Arabic 'alif')
    boundingBoxes = []
    
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # Filter out very small boxes that might be noise
        if w >= min_valid_width and h > 20:
            boundingBoxes.append((x, y, w, h))
    
    # Sort bounding boxes
    def compare(rect1, rect2):
        # IMPROVED: More flexible line detection based on character height
        avg_height = sum([r[3] for r in boundingBoxes]) / len(boundingBoxes) if boundingBoxes else 30
        threshold = avg_height / 3  # Adaptive threshold based on average character height
        
        if abs(rect1[1] - rect2[1]) > threshold:
            return rect1[1] - rect2[1]
        else:
            return rect1[0] - rect2[0]
    
    boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare))
    
    # Character recognition parameters
    TARGET_WIDTH = 96
    TARGET_HEIGHT = 96
    chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'alif', 'ba', 'dal', 'ha', 'jim', 'mim', 'qaf', 'shin', 'waw']
    
    # Arabic character mapping to Unicode
    arabic_map = {
        'alif': 'أ',
        'ba': 'ب',
        'dal': 'د',
        'ha': 'ه',
        'jim': 'ج',
        'mim': 'م',
        'qaf': 'ق',
        'shin': 'ش',
        'waw': 'و'
    }
    
    # IMPROVED: Lower confidence threshold for better recall
    lp_threshold = 0.3
    
    print("Recognizing characters...")
    
    # Create a debug image to visualize boxes
    debug_img = resized_image.copy()
    
    # List to store characters and their positions for later processing
    char_boxes = []
    
    # IMPROVED: Pre-analyze the set of bounding boxes for better separator detection
    avg_width = sum([r[2] for r in boundingBoxes]) / len(boundingBoxes) if boundingBoxes else 20
    std_width = np.std([r[2] for r in boundingBoxes]) if len(boundingBoxes) > 1 else 10
    width_threshold = min(avg_width * 0.4, std_width)  # Adaptive threshold for separator detection
    
    # IMPROVED: Precompute expected positioning for Algerian plates
    # Find vertical bars (separators)
    potential_separators = []
    for i, rect in enumerate(boundingBoxes):
        x, y, w, h = rect
        aspect_ratio = w / h
        if aspect_ratio < 0.25 and w < width_threshold:
            potential_separators.append(i)
    
    # Process each character
    for i, rect in enumerate(boundingBoxes):
        x, y, w, h = rect
        
        # IMPROVED: Better separator detection logic
        aspect_ratio = w / h
        
        # A separator typically has a very small width-to-height ratio
        # But be careful not to misidentify digit '1' or Arabic 'alif' as a separator
        is_separator = False
        is_thin_char = False  # Could be '1' or 'alif'
        is_in_arabic_position = False  # Used to help determine if this is in the position where Arabic letter should be
        
        # Check if this is a thin vertical character (potential '1' or 'alif')
        if aspect_ratio < 0.25:  # Thin character
            # Extract the region and analyze it
            roi = mask[y:y+h, x:x+w]
            pixel_density = cv2.countNonZero(roi) / (w * h)
            
            # True separators typically have very uniform high density
            if pixel_density > 0.8 and w < width_threshold:
                is_separator = True
            else:
                # Could be a digit '1' or Arabic 'alif'
                is_thin_char = True
        
        # IMPROVED: Check if this might be in the Arabic letter position (usually between two separators)
        if len(potential_separators) >= 2:
            # If there are at least 2 potential separators, check if this is between them
            if i > potential_separators[0] and i < potential_separators[-1]:
                # This character is between the first and last separator - likely an Arabic letter
                is_in_arabic_position = True
        
        # Draw rectangle and number for debugging (different colors)
        color = (0, 0, 255) if is_separator else (0, 255, 0)  # Red for separators, green for characters
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(debug_img, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Store character/separator information
        char_box = {
            'rect': rect,
            'is_separator': is_separator,
            'is_thin_char': is_thin_char,
            'is_in_arabic_position': is_in_arabic_position,
            'index': i,
            'x': x  # For later sorting
        }
        
        # If it's a definite separator, continue to the next character
        if is_separator:
            print(f"Detected separator at position {i} (x={x})")
            char_boxes.append(char_box)
            continue
            
        # Process the character for recognition
        crop = mask[y:y+h, x:x+w]
        crop = cv2.bitwise_not(crop)
        rows = crop.shape[0]
        columns = crop.shape[1]
        
        # IMPROVED: Better padding to center characters
        paddingY = (TARGET_HEIGHT - rows) // 2 if rows < TARGET_HEIGHT else int(0.15 * rows)
        paddingX = (TARGET_WIDTH - columns) // 2 if columns < TARGET_WIDTH else int(0.15 * columns)
       
        # Ensure padding is at least 1 pixel
        paddingY = max(1, paddingY)
        paddingX = max(1, paddingX)
        
        crop = cv2.copyMakeBorder(crop, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)
        
        # Save each character for debugging
        cv2.imwrite(f"{output_dir}/{basename}_char_{i}.png", crop)
        
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)    
        crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))
        
        # Prepare data for prediction
        crop = crop.astype("float") / 255.0
        crop = img_to_array(crop)
        crop = np.expand_dims(crop, axis=0)
        
        # Make prediction with error handling
        try:
            predictions = model.predict(crop, verbose=0)
            classe = np.argmax(predictions, axis=1)
            confidence = predictions[0][int(classe[0])]
            
            # Get the predicted character
            predicted_char = chars[int(classe[0])]
            
            # Get second best prediction for ambiguous cases
            top_indices = np.argsort(predictions[0])[::-1]  # Sort in descending order
            second_best_idx = top_indices[1] if len(top_indices) > 1 else classe[0]
            second_best_char = chars[int(second_best_idx)]
            second_best_confidence = predictions[0][int(second_best_idx)]
            
            print(f"Character {i}: class={predicted_char}, confidence={confidence:.4f} | 2nd best: {second_best_char}, confidence={second_best_confidence:.4f}")
            
            # IMPROVED: Enhanced distinction between '1' and 'alif'
            # Use position, context, and competing predictions to better disambiguate
            
            # Is this a case where we need to decide between '1' and 'alif'?
            is_ambiguous_thin_char = is_thin_char and (
                (predicted_char == '1' and second_best_char == 'alif') or 
                (predicted_char == 'alif' and second_best_char == '1')
            )
            
            if is_ambiguous_thin_char:
                print(f"Ambiguous thin character at position {i}")
                
                # Use position as a strong hint - in Algerian plates:
                # - If it's in the Arabic letter zone, it's very likely 'alif'
                # - If it's outside, it's likely '1'
                
                if is_in_arabic_position:
                    # Force it to be 'alif' if in the middle section
                    char_box['is_arabic'] = True
                    char_box['raw_character'] = 'alif'
                    char_box['character'] = arabic_map['alif']
                    char_box['confidence'] = max(confidence, 0.65)  # Boost confidence a bit due to contextual evidence
                    print(f"Disambiguated to 'alif' based on position in the plate")
                else:
                    # Keep it as model predicted but with position contextual awareness
                    if predicted_char == 'alif' and len(potential_separators) >= 2:
                        # If we're confident in our separator detection, and this is outside the middle section
                        # but still predicted as 'alif', double-check
                        if confidence < 0.65:  # Model is not super confident
                            # Force it to be '1' as it's out of position for an Arabic letter
                            char_box['is_arabic'] = False
                            char_box['raw_character'] = '1'
                            char_box['character'] = '1'
                            char_box['confidence'] = 0.6  # Assign reasonable confidence
                            print(f"Disambiguated to '1' based on position in the plate")
                        else:
                            # Model is very confident it's 'alif' despite position, keep it
                            char_box['is_arabic'] = True
                            char_box['raw_character'] = 'alif'
                            char_box['character'] = arabic_map['alif']
                            char_box['confidence'] = confidence
                    else:
                        # Use the model's prediction
                        if predicted_char == 'alif':
                            char_box['is_arabic'] = True
                            char_box['raw_character'] = 'alif'
                            char_box['character'] = arabic_map['alif']
                        else:
                            char_box['is_arabic'] = False
                            char_box['raw_character'] = predicted_char
                            char_box['character'] = predicted_char
                        char_box['confidence'] = confidence
            
            elif confidence >= lp_threshold:
                # Process normal confident predictions
                if predicted_char in arabic_map:
                    char_box['is_arabic'] = True
                    char_box['raw_character'] = predicted_char
                    char_box['character'] = arabic_map[predicted_char]
                else:
                    char_box['is_arabic'] = False
                    char_box['raw_character'] = predicted_char
                    char_box['character'] = predicted_char
                    
                char_box['confidence'] = confidence
                print(f"Character detected: {predicted_char} (confidence: {confidence:.4f})")
            else:
                # Handle low confidence predictions
                # If it looks like a thin character, make an educated guess
                if is_thin_char:
                    if is_in_arabic_position:
                        char_box['is_arabic'] = True
                        char_box['raw_character'] = 'alif'
                        char_box['character'] = arabic_map['alif']
                        char_box['confidence'] = 0.5  # Assign reasonable confidence
                    else:
                        char_box['is_arabic'] = False
                        char_box['raw_character'] = '1'
                        char_box['character'] = '1'
                        char_box['confidence'] = 0.5  # Assign reasonable confidence
                else:
                    char_box['is_arabic'] = False
                    char_box['raw_character'] = '?'
                    char_box['character'] = '?'
                    char_box['confidence'] = confidence
                    
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            # Special handling for potential thin characters
            if is_thin_char:
                if is_in_arabic_position:
                    char_box['is_arabic'] = True
                    char_box['raw_character'] = 'alif'
                    char_box['character'] = arabic_map['alif']
                    char_box['confidence'] = 0.5
                else:
                    char_box['is_arabic'] = False
                    char_box['raw_character'] = '1'
                    char_box['character'] = '1'
                    char_box['confidence'] = 0.5
            else:
                char_box['is_arabic'] = False
                char_box['raw_character'] = '?'
                char_box['character'] = '?'
                char_box['confidence'] = 0
            
        char_boxes.append(char_box)
    
    # Save debug image with boxes
    cv2.imwrite(f"{output_dir}/{basename}_contours.png", debug_img)
    
    # Sort character boxes by x position
    char_boxes.sort(key=lambda box: box['x'])
    
    # IMPROVED: Better Algerian license plate format identification
    # Standard Algerian plate format: NNNNN|A|NN or NNNN|A|NN
    # Where:
    # - N is a digit (0-9)
    # - A is an Arabic letter
    # - | is a separator
    
    # First, identify all separators and Arabic letters
    separator_indices = [i for i, box in enumerate(char_boxes) if box.get('is_separator', False)]
    arabic_indices = [i for i, box in enumerate(char_boxes) if box.get('is_arabic', False) and 'character' in box]
    
    # Create a structure to analyze the plate
    plate_chars = []
    
    # IMPROVED: Better plate structure recognition with special handling for Arabic 'alif'
    if len(char_boxes) >= 5:  # Minimum characters for a valid plate
        # IMPROVED: Use a structured approach to interpret Algerian license plates format
        if len(separator_indices) >= 2:
            # Standard case with at least two separators detected
            first_sep = separator_indices[0]
            second_sep = separator_indices[1]
            
            # Group 1: Digits before first separator (usually 4-5 digits)
            first_group = []
            for i in range(first_sep):
                if 'character' in char_boxes[i] and not char_boxes[i].get('is_separator', False):
                    first_group.append(char_boxes[i]['character'])
            
            # Group 2: Arabic letter between separators
            middle_group = []
            for i in range(first_sep + 1, second_sep):
                if 'character' in char_boxes[i] and not char_boxes[i].get('is_separator', False):
                    # Force this to be an Arabic character if not already
                    if not char_boxes[i].get('is_arabic', False) and char_boxes[i]['character'] == '1':
                        # This is a case where '1' should probably be 'alif'
                        middle_group.append(arabic_map['alif'])
                    else:
                        middle_group.append(char_boxes[i]['character'])
            
            # Group 3: Final digits after second separator
            last_group = []
            for i in range(second_sep + 1, len(char_boxes)):
                if 'character' in char_boxes[i] and not char_boxes[i].get('is_separator', False):
                    last_group.append(char_boxes[i]['character'])
            
            # Validation: Check and correct typical patterns
            
            # If first group is missing a digit (Algerian plates usually have 4-5 digits at start)
            if len(first_group) < 4:
                # Add leading '1' if it looks like it's needed
                if first_group and first_group[0] in "56789":
                    first_group.insert(0, '1')
            
            # If middle group is empty (no Arabic letter detected)
            if not middle_group and len(first_group) >= 4:
                # Insert 'alif' if it's the most probable missing letter
                middle_group.append(arabic_map['alif'])
            
            # Combine all groups
            plate_chars = first_group + middle_group + last_group
            
        elif len(arabic_indices) > 0:
            # We have at least one Arabic letter - use it as reference to structure the plate
            arabic_idx = arabic_indices[0]
            
            # Digits before the Arabic letter
            for i in range(arabic_idx):
                if not char_boxes[i].get('is_separator', False) and 'character' in char_boxes[i]:
                    plate_chars.append(char_boxes[i]['character'])
            
            # The Arabic letter itself
            plate_chars.append(char_boxes[arabic_idx]['character'])
            
            # Digits after the Arabic letter
            for i in range(arabic_idx + 1, len(char_boxes)):
                if not char_boxes[i].get('is_separator', False) and 'character' in char_boxes[i]:
                    plate_chars.append(char_boxes[i]['character'])
        else:
            # No clear structure - add all characters in order
            for box in char_boxes:
                if not box.get('is_separator', False) and 'character' in box:
                    plate_chars.append(box['character'])
    else:
        # Too few characters - just use what we have
        for box in char_boxes:
            if not box.get('is_separator', False) and 'character' in box:
                plate_chars.append(box['character'])
    
    # FINAL VALIDATION: Additional rules for Algerian plates
    
    # If we have an Arabic letter in the plate, make sure it's appropriately placed
    arabic_char_positions = [i for i, c in enumerate(plate_chars) if c in arabic_map.values()]
    
    if arabic_char_positions:
        arabic_pos = arabic_char_positions[0]
        
        # Typical Algerian format should have the Arabic letter after digit position 4 or 5
        if arabic_pos < 4 and len(plate_chars) > 5:
            # Move the Arabic letter to the correct position
            arabic_char = plate_chars.pop(arabic_pos)
            
            # Determine if it should be after position 4 or 5
            if len(plate_chars) >= 6:  # Likely needs 5 digits before
                insert_pos = 5
            else:
                insert_pos = 4
                
            # Safety check for insertion position
            insert_pos = min(insert_pos, len(plate_chars))
            plate_chars.insert(insert_pos, arabic_char)
    
    # Combine the final plate characters
    vehicle_plate = ''.join(plate_chars)
    
    # Special case handling for common patterns:
    
    # If after all our processing, we still have a digit '1' in the Arabic position:
    if len(vehicle_plate) >= 5:
        # Find potential Arabic position (after 4-5 digits)
        for i in range(4, min(6, len(vehicle_plate))):
            # If this position has '1' and is followed by more digits, it's likely 'alif'
            if i < len(vehicle_plate) and vehicle_plate[i] == '1' and i+1 < len(vehicle_plate) and vehicle_plate[i+1].isdigit():
                # Replace the '1' with 'alif'
                vehicle_plate = vehicle_plate[:i] + arabic_map['alif'] + vehicle_plate[i+1:]
                break
    
    return vehicle_plate


# Add img_to_array function from Keras for compatibility
def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array."""
    if data_format is None:
        data_format = 'channels_last'
        
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
        
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x