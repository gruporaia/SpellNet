"""
SignLink Hand Landmark Preprocessing Module

This module provides tools to preprocess hand images for machine learning models,
detecting hand landmarks with MediaPipe, cropping the hand region, resizing,
and optionally applying an input preprocessing function (e.g., MobileNet preprocessing).

Classes:
    - SignLinkPreprocessingResponse: Dataclass to store preprocessing outputs.
    - SignLinkPreprocessing: Main class for full hand image preprocessing.

Functions:
    - show_image: Utility function to display an image with OpenCV.
    - cv2_cam_loop: Webcam capture loop to apply preprocessing live.
    - simple_image_test: Test function to apply preprocessing on a static image.
"""
import os 
import cv2
import traceback

import numpy as np
import mediapipe as mp

from dataclasses import dataclass

from mediapipe.framework.formats import landmark_pb2
from tensorflow.keras.applications.mobilenet import preprocess_input


@dataclass
class SignLinkPreprocessingResponse:
    """
    Response structure for the preprocessing pipeline.
    
    Attributes:
        original_image (np.array): Original input image.
        model_input_image (np.array): Final preprocessed image (ready for ML model).
        image_with_hand_landmarks (np.array): Image with hand landmarks drawn.
        final_image_has_hand_landmark (bool): Whether a hand landmark was detected.
    """
    original_image: np.array
    model_input_image: np.array
    image_with_hand_landmarks: np.array
    final_image_has_hand_landmark: bool


class SignLinkPreprocessing:
    """
    Full pipeline for preprocessing hand images for ML models.
    """
    def __init__(
            self,
            final_preprocessing_fn=None,
            max_num_hands: int = 1,
            min_detection_confidence: float = 0.5,
            padding: int = 100,
            width: int = 200,
            height: int = 200,
            show_img_through_process: bool = False
        ):
        """
        Initialize preprocessing class with configuration options.

        Args:
            final_preprocessing_fn (callable, optional): Final image preprocessing function. Defaults to identity.
            max_num_hands (int): Max number of hands to detect. Defaults to 1.
            min_detection_confidence (float): Minimum detection confidence. Defaults to 0.5.
            padding (int): Pixels to pad around detected hand. Defaults to 100.
            width (int): Output width after resizing. Defaults to 200.
            height (int): Output height after resizing. Defaults to 200.
            show_img_through_process (bool): If True, shows intermediate images.
        """
        self.final_preprocessing_fn = final_preprocessing_fn if final_preprocessing_fn is not None else lambda x: x

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )

        self.padding = padding
        self.width = width
        self.height = height
        self.show_img = show_img_through_process

    def _show_image(self, image, title='image'):
        """
        Display an image using OpenCV (for debugging).
        """
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _resize(self, image_array):
        """
        Resize an image to the configured width and height.
        """
        size = (self.width, self.height)
        resized = cv2.resize(image_array, size)
        if self.show_img:
            self._show_image(resized, title='resized')
        return resized

    def _get_hand_landmarks(self, image_array):
        """
        Detect hand landmarks in an image.

        Returns:
            NormalizedLandmarkList or None if no hand is detected.
        """
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
        return None

    def _draw_hand_landmarks(self, image_array, hand_landmarks):
        """
        Draw hand landmarks on a copy of the image.
        """
        image_copy = image_array.copy()
        self.mp_drawing.draw_landmarks(
            image_copy,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )
        if self.show_img:
            self._show_image(image_copy, title='with landmarks')
        return image_copy

    def _get_hand_crop_box(self, image_array, hand_landmarks):
        """
        Calculate a padded bounding box around hand landmarks.
        """
        h, w, _ = image_array.shape
        x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
        y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
        x_min = max(min(x_coords) - self.padding, 0)
        x_max = min(max(x_coords) + self.padding, w)
        y_min = max(min(y_coords) - self.padding, 0)
        y_max = min(max(y_coords) + self.padding, h)
        return (x_min, y_min, x_max, y_max)

    def _crop_hand_region_with_box(self, image_array, crop_box):
        """
        Crop the hand region from the image based on bounding box.
        """
        x_min, y_min, x_max, y_max = crop_box
        cropped = image_array[y_min:y_max, x_min:x_max]
        if self.show_img:
            self._show_image(cropped, title='cropped')
        return cropped

    def _transform_landmarks_to_cropped_image(self, hand_landmarks, crop_box, original_size, target_size):
        """
        Adjust hand landmarks to the new cropped and resized image coordinates.
        """
        h, w = original_size
        x_min, y_min, x_max, y_max = crop_box
        crop_w = x_max - x_min
        crop_h = y_max - y_min
        target_w, target_h = target_size

        transformed_landmarks = []
        for lm in hand_landmarks.landmark:
            x_px = lm.x * w
            y_px = lm.y * h

            x_crop = x_px - x_min
            y_crop = y_px - y_min

            x_norm = x_crop / crop_w
            y_norm = y_crop / crop_h

            new_lm = landmark_pb2.NormalizedLandmark()
            new_lm.x = x_norm
            new_lm.y = y_norm
            new_lm.z = lm.z  # Z coordinate is kept unchanged

            transformed_landmarks.append(new_lm)

        return landmark_pb2.NormalizedLandmarkList(landmark=transformed_landmarks)

    def _apply_final_model_preprocessing(self, image_array):
        final_image = self.final_preprocessing_fn(image_array)

        if self.show_img:
            self._show_image(final_image, title='final image')

        return final_image 

    def model_input_image_full_preprocessing(self, image_array):
        """
        Full pipeline to preprocess an input image for a model.

        Steps:
            - Detect hand landmarks
            - Draw landmarks
            - Crop hand region
            - Resize image
            - Transform landmarks to new image size
            - Apply final preprocessing function

        Returns:
            SignLinkPreprocessingResponse: Full information about the preprocessing.
        """
        try:
            hand_landmarks = self._get_hand_landmarks(image_array)
            if hand_landmarks is None:
                return SignLinkPreprocessingResponse(
                    original_image=image_array,
                    image_with_hand_landmarks=image_array,
                    model_input_image=None,
                    final_image_has_hand_landmark=False
                )

            image_with_landmarks = self._draw_hand_landmarks(image_array, hand_landmarks)

            crop_box = self._get_hand_crop_box(image_array, hand_landmarks)
            cropped_hand = self._crop_hand_region_with_box(image_array, crop_box)
            resized_hand = self._resize(cropped_hand)

            transformed_landmarks = self._transform_landmarks_to_cropped_image(
                hand_landmarks,
                crop_box,
                (image_array.shape[0], image_array.shape[1]),
                (self.width, self.height)
            )

            resized_hand_with_landmarks = self._draw_hand_landmarks(resized_hand, transformed_landmarks)

            model_input_image = self._apply_final_model_preprocessing(resized_hand_with_landmarks)

            return SignLinkPreprocessingResponse(
                original_image=image_array,
                image_with_hand_landmarks=image_with_landmarks,
                model_input_image=model_input_image,
                final_image_has_hand_landmark=True
            )

        except Exception as e:
            print(f'Exception: {e}, Traceback: {traceback.format_exc()}')

            return SignLinkPreprocessingResponse(
                original_image=image_array,
                image_with_hand_landmarks=None,
                model_input_image=None,
                final_image_has_hand_landmark=False
            )


def show_image(image, title: str = 'image'):
    """
    Display a single image in a new OpenCV window.
    """
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cv2_cam_loop():
    """
    Live camera loop to detect hands and save the output video.
    Press 'q' to quit the camera loop.
    """
    cam = cv2.VideoCapture(0)

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

    preprocessing = SignLinkPreprocessing(preprocess_input, show_img_through_process=False)

    while True:
        ret, frame = cam.read()

        response = preprocessing.model_input_image_full_preprocessing(frame)

        if response.final_image_has_hand_landmark:
            out.write(response.image_with_hand_landmarks)
            cv2.imshow('Camera', response.image_with_hand_landmarks)
        else:
            out.write(frame)
            cv2.imshow('Camera', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    out.release()
    cv2.destroyAllWindows()


def simple_image_test():
    """
    Simple static image test to preprocess one image from file.
    """
    image_name = 'image2.png'
    image_root_path = r'C:\Users\Bernardo\Documents\signlink\SignLink-dev\app\test_img'
    image = cv2.imread(os.path.join(image_root_path, image_name))

    preprocessing = SignLinkPreprocessing(preprocess_input, show_img_through_process=True)
    image_processed = preprocessing.model_input_image_full_preprocessing(image)


if __name__ == '__main__':
    simple_image_test()
