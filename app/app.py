import av
import os
import cv2
import time 
import queue 
import logging
import base64

import numpy as np
import mediapipe as mp

import streamlit as st

from tensorflow import keras 

from streamlit_webrtc import webrtc_streamer, WebRtcMode

from video_processing import get_letter, get_random_letter
from preprocessing.signlink_preprocessing import SignLinkPreprocessing, SignLinkPreprocessingResponse
from tensorflow.keras.applications.mobilenet import preprocess_input

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

callback_results = queue.Queue()
green = "#33FF70"
red = "#FF5733"

st.set_page_config(page_title="SignLink", layout="centered")

if 'language' not in st.session_state:
    st.session_state['language'] = 'asl'

# Instruction sidebar
st.sidebar.title("Instruções")
st.sidebar.write(
    "1. Digite a palavra que gostaria de soletrar utilizando a linguagem de sinais.\n"
    "2. Selecione uma opção no menu suspenso: ASL (American Sign Language) ou LIBRAS (Linguagem Brasileira de Sinais).\n"
    "3. Utilize a webcam para transmitir vídeo em tempo real.\n"
    "4. Iremos capturar sua mão e reconhecer as letras."
)


st.title("🖐️ SignLink")
st.write("#### Aprenda linguagem de sinais de maneira interativa!")

# Input of the word to be spelled
palavra = st.text_input("Digite a palavra:", key="input_text")            

# Option combo-box
option = st.selectbox(
    "Escolha uma opção:", 
    ["ASL (American Sign Language)", "LIBRAS (Linguagem Brasileira de Sinais)"]
)


if option == 'ASL (American Sign Language)':
    st.session_state['language'] = 'asl'
elif option == 'LIBRAS (Linguagem Brasileira de Sinais)':
    st.session_state['language'] = 'libras'
else:
    st.session_state['language'] = 'asl'


if st.session_state['language'] == 'asl':
    model_path = './model/mobilenet_cecilia_aug_heavy.keras'
elif st.session_state['language'] == 'libras':
    model_path = './model/mobilenet_dani_libras_aug_heavy.keras'

# Caching model
cache_key = 'signlink_model'
if cache_key in st.session_state:
    model = st.session_state[cache_key]
else:
    model = keras.models.load_model(model_path)
    model.predict(np.zeros((1, 224, 224, 3))) # Avoid delay during first real frame inference
    st.session_state[cache_key] = model

# Initilizing hand detection mediapipe modules
preprocesing = SignLinkPreprocessing(
    final_preprocessing_fn=preprocess_input,
    max_num_hands=2,
    min_detection_confidence=0.5
)

last_infer_time = 0
inference_interval = 1 # seconds

def make_video_frame_callback(model, language):
    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        global last_infer_time
        try:
            img = frame.to_ndarray(format="bgr24")
            final_displayed_image = img.copy()

            preprocessing_response: SignLinkPreprocessingResponse = preprocesing.model_input_image_full_preprocessing(img)
            if preprocessing_response.final_image_has_hand_landmark:
                final_displayed_image = preprocessing_response.image_with_hand_landmarks
                
                # Skipping inference every 2 out of 3 frames
                current_time = time.time()
                if current_time - last_infer_time > inference_interval:
                    letter = get_letter(model, np.expand_dims(preprocessing_response.model_input_image, axis=0), language)
                    logger.warning("Current language", language)
                    logger.warning("Number of classes:", model.output_shape[-1])
                    last_infer_time = current_time
                    logger.warning(f"Letter: {letter}")
                    # Updating the queue, putting the letter that was discovered by the detection model 
                    if not callback_results.empty():
                        callback_results.get()
                    callback_results.put(letter)
                else:
                    logger.info("Skipping frame")
                    
            return av.VideoFrame.from_ndarray(final_displayed_image, format="bgr24")

        except Exception as e:
            logger.warning(f"Error processing image: {e}")
            return frame
    return video_frame_callback

def define_baseline(palavra):
    # Set initial cache configuration
    st.session_state["word"] = palavra.upper()
    st.session_state["current_letter_index"] = 0 
    st.session_state["victory_mapping"] = {i: False for i in range(len(palavra))}
    st.session_state["colors"] = {(i, letter): red  for i, letter in enumerate(palavra)}


def get_current_index():
    # Get current letter: idicates in which state of the word the user is
    return st.session_state["current_letter_index"]

def get_current_letter():
    current_index = get_current_index()

    # In case user already finished the game
    if current_index >= len(st.session_state["word"]):
        return 
    
    return st.session_state["word"][current_index]

def verify(result):
    # Verify if the letter is valid
    current_index = get_current_index()
    letter = get_current_letter()
    
    if letter is None:
        return 
    
    # Setting the session cached data
    # If the the returned letter from the detection model is the same as the letter to be spelled
    # and if the current letter is not yet successfully spelled
    # We set the color of it to green (success), mark it as succes in the mapping and 
    # increment to go to the following letter of the word
    if (
        result is not None 
        and result.lower() == st.session_state["word"][current_index].lower()
        and not st.session_state["victory_mapping"][current_index]
    ):
        st.session_state["colors"][current_index, letter] = green 
        st.session_state["victory_mapping"][current_index] = True
        st.session_state["current_letter_index"] += 1


def get_sample_image():
    # Get current letter: idicates in which state of the word the user is
    letter = get_current_letter()
    if letter is not None:
        letter = letter.lower()

        language = st.session_state['language']
        logger.warning(f'Language option: {language}')

        path = {file.split('.')[0]: file for file in os.listdir(os.path.join('samples', language))}
        if letter in path:
            logger.warning(f"Path: {os.path.join('samples', language, path[letter])}")
            with open(os.path.join('samples', language, path[letter]), 'rb') as f:
                b64_data = base64.b64encode(f.read()).decode()

            return b64_data

def put_word(word_area, cur_letter, sample_image_hover_html):
    # Adding letter by letter, each one with the mapped color
    letters_html = " ".join([
        f"<span style='font-size:24px; font-weight:bold; color:{st.session_state['colors'].get((i, letra), '#FF5733')}'>{letra}</span>"
        for i, letra in enumerate(st.session_state["word"])
    ])

    word_area.markdown(
        f"### Palavra soletrada: {letters_html}",
        unsafe_allow_html=True
    )
    image_base64 = get_sample_image()
    sample_area.markdown(sample_image_hover_html.format(image_base64=image_base64), unsafe_allow_html=True)

    letter_area.markdown(f'##### Letra soletrada: {cur_letter}')

sample_image_hover_html = """
<style>
.tooltip {{
  cursor: pointer;
  font-weight: bold;
  color: #ffb645;
}}

.tooltip .tooltip-image {{
  visibility: hidden;
  width: 200px;
  position: absolute;
  z-index: 1;
  top: 100%;
  left: 50%;
  margin-left: -100px;
  border: 1px solid #ccc;
  background-color: white;
  padding: 5px;
  box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
}}

.tooltip:hover .tooltip-image {{
  visibility: visible;
}}
</style>

<div class="tooltip">
  Precisa de uma dica para fazer a letra? Passe mouse aqui!
  <div class="tooltip-image">
    <img src="data:image/png;base64,{image_base64}" width="200">
  </div>
</div>
"""

if palavra:
    # Only if word is new we want to get the inital cache data
    if (
        "word" not in st.session_state
        or "word" in st.session_state and st.session_state["word"] != palavra
    ):
        define_baseline(palavra) 
    
    # Where input word will be displayed
    word_area = st.empty()
    letter_area = st.empty()
    sample_area = st.empty()

    put_word(word_area, "", sample_image_hover_html)

    ctx = webrtc_streamer(
        key="webcam",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=make_video_frame_callback(model, st.session_state['language']),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

    # Where Win text will be displayed
    win_area = st.empty()

    if ctx.state.playing:
        while True:
            # Getting the letter from detection model
            gest = callback_results.get()
            
            # Verify if this letter satisfies the current application state 
            verify(gest)
            
            # Display the word again, changing letter to green if valid
            put_word(word_area, gest, sample_image_hover_html)
            
            # After all letters of the word are covered, we display victory message
            if all(st.session_state["victory_mapping"].values()):
                 win_area.markdown("### ✨ Parabens, você soletrou corretamente!!! ✨")