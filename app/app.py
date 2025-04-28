import av
import cv2
import time 
import queue 
import logging

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
model_path = './model/mobilenet_aug.keras'

st.set_page_config(page_title="SingLink", layout="centered")

# Caching model
cache_key = 'signlink_model'
if cache_key in st.session_state:
    model = st.session_state[cache_key]
else:
    model = keras.models.load_model(model_path)
    model.predict(np.zeros((1, 200, 200, 3))) # Avoid delay during first real frame inference
    st.session_state[cache_key] = model 

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

# Initilizing hand detection mediapipe modules
preprocesing = SignLinkPreprocessing(
    final_preprocessing_fn=preprocess_input,
    max_num_hands=2,
    min_detection_confidence=0.5
)

last_infer_time = 0
inference_interval = 1 # seconds

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
                letter = get_letter(model, preprocessing_response.model_input_image)
                
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

def define_baseline(palavra):
    # Set initial cache configuration
    st.session_state["word"] = palavra.upper()
    st.session_state["current_letter_index"] = 0 
    st.session_state["victory_mapping"] = {i: False for i in range(len(palavra))}
    st.session_state["colors"] = {(i, letter): red  for i, letter in enumerate(palavra)}


def verify(result):
    # Verify if the letter is valid

    # Get current letter: idicates in which state of the word the user is
    current_index = st.session_state["current_letter_index"]

    # In case user already finished the game
    if current_index >= len(st.session_state["word"]):
        return 
    
    letter = st.session_state["word"][current_index]

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

def put_word(word_area, cur_letter):
    # Adding letter by letter, each one with the mapped color
    letters_html = " ".join([
        f"<span style='font-size:24px; font-weight:bold; color:{st.session_state['colors'].get((i, letra), '#FF5733')}'>{letra}</span>"
        for i, letra in enumerate(st.session_state["word"])
    ])

    word_area.markdown(
        f"### Palavra soletradada: {letters_html}",
        unsafe_allow_html=True
    )

    letter_area.markdown(f'##### Letra soletrada: {cur_letter}')

   
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

    put_word(word_area, "")

    ctx = webrtc_streamer(
        key="webcam",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
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
            put_word(word_area, gest)
            
            # After all letters of the word are covered, we display victory message
            if all(st.session_state["victory_mapping"].values()):
                 win_area.markdown("### ✨ Parabens, você soletrou corretamente!!! ✨")