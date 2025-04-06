import logging
import string 
import random 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

# TODO: replace thios with the letter detection model
def get_letter(hand_landmarks):
    return random.choice(string.ascii_letters)