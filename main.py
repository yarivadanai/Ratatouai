# Standard library imports
from collections import deque
import io
import base64
import threading
from typing import List, Tuple, Dict, Any

# Related third-party imports
from PIL import Image
import streamlit as st  
from streamlit_js_eval import streamlit_js_eval

# Local application/library specific imports
import Utils.model_utils as mu
from Utils.streamlit_utils import AVATAR, LOGO_PATH, handle_image, StreamHandler, resize_image
from recipe_types import dietary_preferences, intolerances, cuisines

IMAGE_HEIGHT_RATIO = 0.75

ROLE_USER = "user"
TYPE_IMAGE = "image"
TYPE_TEXT = "text"


st.set_page_config(
    page_title="RatatouAI",
    page_icon="🐀",
    layout="wide"
)

def get_window_dimensions() -> Tuple[int, int]:
    """
    Get the dimensions of the browser window.

    Returns:
        Tuple[int, int]: The height and width of the window.
    """
    page_height = streamlit_js_eval(js_expressions='window.outerHeight', key='HEIGHT', want_output=True) or 800
    page_width = streamlit_js_eval(js_expressions='window.innerWidth', key='WIDTH', want_output=True) or 1000
    return int(page_height), int(page_width)


def submit_models():
    if st.session_state.models_changed:
        if st.session_state.pswrd == st.secrets["pswrd"]["key"]:
            for model_type in mu.SUPPORTED_MODELS.keys():
                st.session_state[f"{model_type}_model"] = st.secrets[model_type]["model"]
                st.session_state[f"{model_type}_key"] = st.secrets[model_type]["key"]
            
            st.session_state.use_recipes_db = False
            #st.session_state.db_key = st.secrets["X-RapidAPI-Key"]

        models_obj = mu.Models((st.session_state['vision_model'], st.session_state['vision_key']),
                            (st.session_state['reasoning_model'], st.session_state['reasoning_key']),
                            (st.session_state['formatting_model'], st.session_state['formatting_key']))
        
        with st.spinner("Verifying API keys..."):
            if models_obj.verify_api_keys():
                st.session_state.models = models_obj
                st.success("API keys verified successfully.")
            else:
                st.error("Invalid API keys. Please try again.")

        st.session_state.models_changed = False

def model_changed():
    st.session_state.models_changed = True

def setup_models_and_db() -> bool:
    with st.expander("**Select the models and enter API keys**"):
        for model_type in mu.SUPPORTED_MODELS.keys():
            st.selectbox(f"Select {model_type} Model", mu.SUPPORTED_MODELS[model_type], key=f"{model_type}_model", on_change=model_changed)
            st.text_input(f"Insert API Key for {model_type} model", key=f"{model_type}_key", value="", type="password", on_change=model_changed)
        
        st.divider()
        st.session_state.use_recipes_db = False
                    
        #st.checkbox("Use Recipes DB", key="use_recipes_db", value=True)
        #st.text_input("Insert API Key for Recipes DB", key="db_key", type="password", disabled=not st.session_state.use_recipes_db)
        #st.divider()""" #TODO
        
        st.text_input("**Or enter Password to use our own models and keys**", key="pswrd", type="password", on_change=model_changed)
        st.button("Submit", on_click=submit_models)



def add_uploaded_images(container: st.container, images: List[str], uploaded_files: List[any], page_width: int, page_height: int) -> None:
    """
    Adds uploaded images to the Streamlit container and the images list.

    Args:
        container (st.container): The Streamlit container to display the images.
        images (List[str]): The list to store the image strings.
        uploaded_files (List[any]): The list of uploaded image files.
        page_width (int): The width of the page.
        page_height (int): The height of the page.
    """
    for img in uploaded_files:
        add_uploaded_image(container, images, img, page_width, page_height)

def add_uploaded_image(container: st.container, images: List[str], img, page_width: int, page_height: int) -> None:
    """
    Adds uploaded image to the Streamlit container and the images list.

    Args:
        container (st.container): The Streamlit container to display the images.
        img (List[str]): The list to store the image strings.
        img: The uploaded image file.
        page_width (int): The width of the page.
        page_height (int): The height of the page.
    """
    img_str = handle_image(img)
    images.append(img_str)
    st.session_state.messages.append({"type": "image", "role": "user", "content": img_str})
    with container.chat_message("user"):
        image = Image.open(img)
        image = resize_image(image, page_width, page_height)
        st.image(image)

def run():
    """
    Main function to run the application.
    """
    hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """

    if 'messages' not in st.session_state:
        st.session_state.messages = deque(maxlen=50)
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = 0
    if 'camera_input_key' not in st.session_state:
        st.session_state.camera_input_key = 0
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'models_changed' not in st.session_state:
        st.session_state.models_changed = True
    if  'models' not in st.session_state:
        st.session_state.models = None    
    if 'excludeCuisine' not in st.session_state:
        st.session_state.excludeCuisine = []
    if 'dietary_preferences' not in st.session_state:
        st.session_state.dietary_preferences = []
    if 'intolerances' not in st.session_state:
        st.session_state.intolerances = []


    page_height, page_width = get_window_dimensions()



    container_height = int(page_height * IMAGE_HEIGHT_RATIO)
    container = st.container(border=True, height=container_height)

    with container:
        st.markdown("<p style='text-align: center;'><img src='data:image/png;base64,{}'>\
                    </p>".format(base64.b64encode(open(LOGO_PATH, 'rb').read()).decode()), unsafe_allow_html=True)

        st.markdown("<h1 style='text-align: center;'>Welcome to RatatouAI!</h1>", unsafe_allow_html=True)

    with st.sidebar:
        #if 'models' not in st.session_state and not setup_models():
        setup_models_and_db()

        with st.expander("**Share your dietary prefrences**"):
            st.multiselect("What are your dietary_preferences", dietary_preferences, key="dietary_preferences")
            st.multiselect("Do you have any intolerances?",intolerances, key="intolerances")
            st.multiselect("Exclude these cousines:",cuisines, key="exclude_cousines")


    if  st.session_state.models is None:
        st.info("Please select the models and enter your API keys in the side bar fields.")
        return   

    for message in st.session_state.messages:
        with container.chat_message(message["role"], avatar=AVATAR if message["role"] == "assistant" else None):
            if message["type"] == "image":
                st.image(Image.open(io.BytesIO(base64.b64decode(message["content"]))))
            else:
                st.markdown(message["content"])

    images = []

    with st.expander("Upload images of the food ingredients you have"):
        uploaded_files = st.file_uploader("", type=["jpg", "jpeg", "png"],
                                          accept_multiple_files=True,
                                          key=f'file_uploader_key_{st.session_state["file_uploader_key"]}',)
                                          #disabled=st.session_state['running']) TODO

        if len(uploaded_files) > 0:
            add_uploaded_images(container, images, uploaded_files, page_width, page_height)
            st.session_state['file_uploader_key'] += 1

    with st.expander("Take a picture of the food ingredients you have"):
        img_file_buffer = st.camera_input("", key=f'camera_input_key_{st.session_state["camera_input_key"]}',)
                                          #disabled=st.session_state['running']) TODO

        if img_file_buffer is not None:
            add_uploaded_image(container, images, img_file_buffer, page_width, page_height)
            st.session_state['camera_input_key'] += 1

    stream_handler = StreamHandler(container, st.session_state)
    rat_brain = mu.RatBrain(stream_handler, st.session_state.models)

    semaphore=threading.Semaphore(1)
    if len(images) > 0:

        semaphore.acquire()
        rat_brain.run_full_chain(images)
        semaphore.release()

    input_container = st.container()

    user_prompt = input_container.chat_input("Say something") #, disabled=st.session_state['running']) TODO: Fix this

    if user_prompt:
        st.session_state.messages.append({"type": "text", "role": "user", "content": user_prompt})
        with container.chat_message("user"):
            st.write(user_prompt)
        with st.spinner("Thinking..."):
            rat_brain.reply(user_prompt)


if __name__ == "__main__":
    run()
