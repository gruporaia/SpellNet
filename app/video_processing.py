import logging

import string 
import random 
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

def get_random_letter(hand_landmarks):
    return random.choice(string.ascii_uppercase)

def post_process_result(result, language):
    if language == 'libras':
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']
    else:
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    logger.warning(result)
    predicted_index = np.argmax(result)
    return alphabet[predicted_index]

def get_letter(model, image_array, language):
    prediction = model.predict(image_array, verbose=0)
    predicted_letter = post_process_result(prediction, language)
    return predicted_letter

