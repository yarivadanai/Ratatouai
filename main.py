# Standard library imports
from collections import deque
import io
import base64
from typing import List, Tuple, Dict, Any

# Related third-party imports
from PIL import Image
import streamlit as st  
from streamlit_js_eval import streamlit_js_eval

# Local application/library specific imports
import Utils.model_utils as mu
from Utils.streamlit_utils import AVATAR, LOGO_PATH, handle_image, StreamHandler, resize_image

IMAGE_HEIGHT_RATIO = 0.75

ROLE_USER = "user"
TYPE_IMAGE = "image"
TYPE_TEXT = "text"

MODELS = ["gpt-4o", "claude-3-opus-20240229"]


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


def get_model_and_key(name: str) -> Tuple[str, str]:
    """
    Get the selected model and API key for a given name.

    Args:
        name (str): The name of the model.

    Returns:
        Tuple[str, str]: The selected model and API key.
    """
    model = st.selectbox(f"Select {name} Model", MODELS)
    key = st.text_input(f"Insert API Key for {name} model", key=f"{name}_key", type="password")
    return model, key


def setup_models() -> bool:
    """
    Set up the models and API keys.

    Returns:
        bool: True if the API keys are verified successfully, False otherwise.
    """
    with st.sidebar:
        st.markdown("**Select the models and enter your API keys**")
        models: Dict[str, Dict[str, Any]] = {}
        for name in ['vision', 'reasoning', 'formatting']:
            models[name] = {}
            models[name]['model'], models[name]['key'] = get_model_and_key(name)
        st.markdown("**Or enter Password to use our own models and key**")
        pswrd = st.text_input("Password", key="pswrd", type="password")
        if st.button("Submit"):
            if pswrd == st.secrets["pswrd"]["key"]:
                for name in ['vision', 'reasoning', 'formatting']:
                    models[name]['model'] = st.secrets[name]["model"]
                    models[name]['key'] = st.secrets[name]["key"]
            models_obj = mu.Models((models['vision']['model'], models['vision']['key']),
                                   (models['reasoning']['model'], models['reasoning']['key']),
                                   (models['formatting']['model'], models['formatting']['key']))
            with st.spinner("Verifying API keys..."):
                if models_obj.verify_api_keys():
                    st.session_state.models = models_obj
                    st.success("API keys verified successfully.")
                    return True
                else:
                    st.error("Invalid API keys. Please try again.")
                    return False


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

    page_height, page_width = get_window_dimensions()
    if 'models' not in st.session_state:
        st.info("Please select the models and enter your API keys in the side bar fields.")


    container_height = int(page_height * IMAGE_HEIGHT_RATIO)
    container = st.container(border=True, height=container_height)

    with container:
        st.markdown("<p style='text-align: center;'><img src='data:image/png;base64,{}'>\
                    </p>".format(base64.b64encode(open(LOGO_PATH, 'rb').read()).decode()), unsafe_allow_html=True)

        st.markdown("<h1 style='text-align: center;'>Welcome to RatatouAI!</h1>", unsafe_allow_html=True)

    if 'models' not in st.session_state and not setup_models():
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
                                          key=f'file_uploader_key_{st.session_state["file_uploader_key"]}',
                                          disabled=st.session_state['running'])

        if len(uploaded_files) > 0:
            add_uploaded_images(container, images, uploaded_files, page_width, page_height)
            st.session_state['file_uploader_key'] += 1

    with st.expander("Take a picture of the food ingredients you have"):
        img_file_buffer = st.camera_input("", key=f'camera_input_key_{st.session_state["camera_input_key"]}',
                                          disabled=st.session_state['running'])

        if img_file_buffer is not None:
            add_uploaded_image(container, images, img_file_buffer, page_width, page_height)
            st.session_state['camera_input_key'] += 1

    stream_handler = StreamHandler(container, st.session_state)
    rat_brain = mu.RatBrain(stream_handler, st.session_state.models)

    if len(images) > 0:
        st.session_state['running'] = True
        rat_brain.run_full_chain(images)

    input_container = st.container()

    user_prompt = input_container.chat_input("Say something", disabled=st.session_state['running'])

    if user_prompt:
        st.session_state.messages.append({"type": "text", "role": "user", "content": user_prompt})
        with container.chat_message("user"):
            st.write(user_prompt)
        with st.spinner("Thinking..."):
            rat_brain.reply(user_prompt)


if __name__ == "__main__":
    run()
