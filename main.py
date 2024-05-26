# Standard library imports
from collections import deque
import io
import base64
from typing import List

# Related third-party imports
from PIL import Image
import streamlit as st  

# Local application/library specific imports
import Utils.model_utils as mu
from Utils.streamlit_utils import handle_image, StreamHandler, resize_image
from langchain_core.prompts import ChatPromptTemplate
from streamlit_js_eval import streamlit_js_eval

IS_LOCAL = False
IMAGE_HEIGHT_RATIO = 0.75
if IS_LOCAL:
    LOGO_PATH = "/Users/yarivadan/projects/VSProjects/Ratatouai/static/logo.png"
else:
    LOGO_PATH = "./static/logo.png"
ROLE_USER = "user"
TYPE_IMAGE = "image"
TYPE_TEXT = "text"


st.set_page_config(
    page_title="RatatouAI",
    page_icon="🐀",
    layout="wide"
)


def run(): 
    # Hide the top right menu
    hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
    
    if 'secret_keys' not in st.session_state:
        # Check if API keys were given via secrets
        #try:
            #if "openai" in st.secrets and "key" in st.secrets["openai"] and "anthropic" in st.secrets and "key" in st.secrets["anthropic"]:
                #openai_api_key = st.secrets["openai"]["key"]
                #anthropic_api_key = st.secrets["anthropic"]["key"]
        #except:
        # Get API keys from the side bar fields
        with st.sidebar:
            openai_api_key = st.text_input("OpenAI API Key", key="openai_api_key", type="password")
            "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
            anthropic_api_key = st.text_input("Anthropic API Key", key="anthropic_api_key", type="password")
            "[Get an Anthropic API key](https://www.anthropic.com/api)" 
            st.info("Please enter your API keys in the side bar fields.")
            pswrd=st.text_input("Enter Password to use our own keys", key="pswrd", type="password")
           
            if pswrd==st.secrets["pswrd"]["key"]:
                openai_api_key = st.secrets["openai"]["key"]
                anthropic_api_key = st.secrets["anthropic"]["key"]
            if st.button("Verify Keys"):
                if mu.verify_api_keys(openai_api_key, anthropic_api_key):
                    st.session_state.secret_keys={}
                    st.session_state.secret_keys['openai_api_key'] = openai_api_key
                    st.session_state.secret_keys['anthropic_api_key'] = anthropic_api_key
                    st.success("API keys verified successfully.")
                else:
                    st.error("Invalid API keys. Please try again.")
                    return

       
    if 'messages' not in st.session_state:
        st.session_state.messages=deque(maxlen=50)
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key=0
    if 'camera_input_key' not in st.session_state:
        st.session_state.camera_input_key=0
    if 'running' not in st.session_state:
        st.session_state.running=False

    
    page_height = int(streamlit_js_eval(js_expressions='window.outerHeight', key='HEIGHT', want_output=True))
    page_width = int(streamlit_js_eval(js_expressions='window.innerWidth', key='WIDTH', want_output=True))

    container_height = int(page_height * IMAGE_HEIGHT_RATIO)
    container = st.container(border=True, height=container_height)

    with container:
        # Display the image
        st.markdown("<p style='text-align: center;'><img src='data:image/png;base64,{}'>\
                    </p>".format(base64.b64encode(open(LOGO_PATH, 'rb').read()).decode()), unsafe_allow_html=True)
        
        st.markdown("<h1 style='text-align: center;'>Welcome to RatatouAI!</h1>", unsafe_allow_html=True)

    for message in st.session_state.messages:
        with container.chat_message(message["role"]):
            if message["type"] == "image":
                st.image(Image.open(io.BytesIO(base64.b64decode(message["content"]))))
            else:
                st.markdown(message["content"])
    
    images=[]

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
            add_uploaded_images(container, images, img_file_buffer, page_width, page_height)
            st.session_state['camera_input_key'] += 1

    stream_handler = StreamHandler(container, st.session_state)

    if len(images) > 0:

        #with container.chat_message("assistant"):
        st.session_state['running']=True
        mu.run_full_chain(images, stream_handler) 

    input_container = st.container()

    user_prompt = input_container.chat_input("Say something", disabled=st.session_state['running'])
    
    if user_prompt:
        st.session_state.messages.append( {"type": "text", "role": ROLE_USER, "content": user_prompt})
        with container.chat_message(ROLE_USER):
            st.write(user_prompt)
        with st.spinner("Thinking..."):
            mu.reply(user_prompt, stream_handler)

def add_uploaded_images(container: st.container, images: List[str], uploaded_files: List[any], page_width: int, page_height: int) -> None:
    """Adds uploaded images to the Streamlit container and the images list."""
    for img in uploaded_files:
        img_str = handle_image(img)
        images.append(img_str)
        st.session_state.messages.append({"type": TYPE_IMAGE, "role": ROLE_USER, "content": img_str})

        with container.chat_message(ROLE_USER):
            image = Image.open(img)
            # Resize the image
            image = resize_image(image, page_width, page_height)
            st.image(image)


# Run the application
if __name__ == "__main__":
    run()