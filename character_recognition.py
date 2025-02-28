import cv2
import functools
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# Custom objects for model loading
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
    
    # Resize the image
    resized_image = cv2.resize(license_plate_image, (600, 100), interpolation=cv2.INTER_AREA)
    h, w, _ = resized_image.shape
    
    # Crop the image
    top = 10
    right = 50
    bottom = 20
    left = 80
    image = resized_image[top:h-bottom, left:w-right]
    # image = resized_image
    
    # Convert to grayscale and process
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15)
    
    # Find connected components
    _, labels = cv2.connectedComponents(thresh)
    mask = np.zeros(thresh.shape, dtype="uint8")
    total_pixels = image.shape[0] * image.shape[1]
    lower = total_pixels // 80
    upper = total_pixels // 20
    
    # Loop over the unique components
    for (_, label) in enumerate(np.unique(labels)):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
   
        if numPixels > lower and numPixels < upper:
            mask = cv2.add(mask, labelMask)
    
    # Find contours
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    
    # Sort bounding boxes
    def compare(rect1, rect2):
        if abs(rect1[1] - rect2[1]) > 20:
            return rect1[1] - rect2[1]
        else:
            return rect1[0] - rect2[0]
    
    boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare))
    
    # Character recognition parameters
    TARGET_WIDTH = 96
    TARGET_HEIGHT = 96
    chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'alif', 'ba', 'dal', 'ha', 'jim', 'mim', 'qaf', 'shin', 'waw']
    lp_threshold = 0.6
    
    print("Recognizing characters...")
    vehicle_plate = ""
    
    # Process each character
    for rect in boundingBoxes:
        x, y, w, h = rect
        crop = mask[y:y+h, x:x+w]
        crop = cv2.bitwise_not(crop)
        rows = crop.shape[0]
        columns = crop.shape[1]
        
        paddingY = (TARGET_HEIGHT - rows) // 2 if rows < TARGET_HEIGHT else int(0.17 * rows)
        paddingX = (TARGET_WIDTH - columns) // 2 if columns < TARGET_WIDTH else int(0.45 * columns)
       
        crop = cv2.copyMakeBorder(crop, paddingY, paddingY, paddingX, paddingX, cv2.BORDER_CONSTANT, None, 255)
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
            print(classe)
            if predictions[0][int(classe[0])] >= lp_threshold:
                character = chars[int(classe[0])]
                vehicle_plate += character
                print(f"Detected character: {character} (confidence: {predictions[0][int(classe[0])]})")
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            continue
   
    return vehicle_plate

# Add img_to_array function from Keras for compatibility
def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
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

if __name__ == "__main__":
    try:
        # Load image
        image_path = "samples/_the_0_license_plate_detected.png"
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