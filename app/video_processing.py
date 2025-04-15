import logging

import string 
import random 
import cv2
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

def get_random_letter(hand_landmarks):
    return random.choice(string.ascii_uppercase)

def crop_hand_region(image, hand_landmarks, padding=20):
    """
    Crops the region around the hand using the landmarks.
    
    Args:
        image (np.ndarray): Original image (HWC).
        hand_landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList)
        padding (int): Extra pixels around the hand to include.
        
    Returns:
        Cropped image (np.ndarray)
    """
    image_height, image_width, _ = image.shape
    
    # Extract landmark positions in pixels
    x_coords = [int(landmark.x * image_width) for landmark in hand_landmarks.landmark]
    y_coords = [int(landmark.y * image_height) for landmark in hand_landmarks.landmark]

    # Bounding box
    x_min = max(min(x_coords) - padding, 0)
    x_max = min(max(x_coords) + padding, image_width)
    y_min = max(min(y_coords) - padding, 0)
    y_max = min(max(y_coords) + padding, image_height)

    # Crop and return
    return image[y_min:y_max, x_min:x_max]

def preprocess(image_array, hand_landmark, size):
    image_array = crop_hand_region(image_array, hand_landmark)
    width, height = size
    resized_image = cv2.resize(image_array, (width, height))
    return np.expand_dims(resized_image, axis=0)

def post_process_result(result):
    alphabet = list(string.ascii_uppercase)
    predicted_index = np.argmax(result)
    return alphabet[predicted_index]

def get_letter(model, hand_landmarks, image_array):
    input_image = preprocess(image_array, hand_landmarks, (244, 244))
    prediction = model.predict(input_image, verbose=0)
    predicted_letter = post_process_result(prediction)
    return predicted_letter

