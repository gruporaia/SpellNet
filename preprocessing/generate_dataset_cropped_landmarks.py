import cv2
import mediapipe as mp
import os

max_num_hands: int = 1
min_detection_confidence: float = 0.5
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )

def _draw_hand_landmarks(image_array, hand_landmarks):
        """
        Draw hand landmarks on a copy of the image.
        """
        image_copy = image_array.copy()
        mp_drawing.draw_landmarks(
            image_copy,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )
        return image_copy

def _get_hand_landmarks(image_array):
    """
    Detect hand landmarks in an image.

    Returns:
        NormalizedLandmarkList or None if no hand is detected.
    """
    
    if image_array is None:
        print("Erro: imagem 'hands.jpg' não encontrada.")
        return None

    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
    
    return None


def _get_hand_crop_box(image_array, hand_landmarks):
    """
    Calculate a padded bounding box around hand landmarks.
    """
    h, w, _ = image_array.shape
    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
    x_min = max(min(x_coords) - 100, 0)
    x_max = min(max(x_coords) + 100, w)
    y_min = max(min(y_coords) - 100, 0)
    y_max = min(max(y_coords) + 100, h)
    return (x_min, y_min, x_max, y_max)

def _crop_hand_region_with_box(image_array, crop_box):
        """
        Crop the hand region from the image based on bounding box.
        """
        x_min, y_min, x_max, y_max = crop_box
        cropped = image_array[y_min:y_max, x_min:x_max]
        #if self.show_img:
            #self._show_image(cropped, title='cropped')
        return cropped

def _resize(image_array):
        """
        Resize an image to the configured width and height.
        """
        size = (224, 224)
        resized = cv2.resize(image_array, size)

        return resized

def preprocess_image(image_array):
    landmarks = _get_hand_landmarks(image_array)
    if landmarks is None:
         return None
    image_landmarks = _draw_hand_landmarks(image_array, landmarks)
    crop_box = _get_hand_crop_box(image_landmarks, landmarks)
    cropped_image = _crop_hand_region_with_box(image_landmarks, crop_box)
    resize = _resize(cropped_image)
    
    return resize

roots = ['asl1', 'asl2', 'asl3', 'asl4']
output_root = 'asl_landmarks'
for input_root in roots:  # roots é uma lista de diretórios
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(root, file)
                
                rel_path = os.path.relpath(input_path, input_root)
                output_path = os.path.join(output_root, input_root, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                image = cv2.imread(input_path)
                if image is not None:
                    try:
                        resized_image = preprocess_image(image)
                        if resized_image is not None: 
                            cv2.imwrite(output_path, resized_image)
                    except Exception as e:
                            print(f"Erro ao processar imagem {input_path}: {e}")

                else:
                    print(f"Erro ao ler imagem: {input_path}")