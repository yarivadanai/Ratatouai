# Standard library imports
from collections import deque
import io
import base64
import threading
from typing import List, Tuple, Dict, Any
import json

# Related third-party imports
from PIL import Image
import streamlit as st  
from streamlit_js_eval import streamlit_js_eval
from streamlit_tags import st_tags

# Local application/library specific imports
import Utils.model_utils as mu
from Utils.streamlit_utils import AVATAR, LOGO_PATH, handle_image, StreamHandler, resize_image
from recipe_types import dietary_preferences, intolerances, cuisines

# Constants
IMAGE_HEIGHT_RATIO = 0.75
ROLE_USER = "user"
TYPE_IMAGE = "image"
TYPE_TEXT = "text"

# Streamlit page configuration
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
    try:
        page_height = streamlit_js_eval(js_expressions='window.outerHeight', key='HEIGHT', want_output=True) or 800
        page_width = streamlit_js_eval(js_expressions='window.innerWidth', key='WIDTH', want_output=True) or 1000
        return int(page_height), int(page_width)
    except Exception as e:
        st.error(f"Error getting window dimensions: {e}")
        return 800, 1000

def submit_models():
    """
    Submit and verify the selected models and API keys.
    """
    if st.session_state.models_changed:
        if st.session_state.pswrd == st.secrets["pswrd"]["key"]:
            for model_type in mu.SUPPORTED_MODELS.keys():
                st.session_state[f"{model_type}_model"] = st.secrets[model_type]["model"]
                st.session_state[f"{model_type}_key"] = st.secrets[model_type]["key"]
            st.session_state["tavili"] = st.secrets["tavili"]["key"]
            st.session_state.use_recipes_db = False

        models_obj = mu.Models(
            (st.session_state['vision_model'], st.session_state['vision_key']),
            (st.session_state['reasoning_model'], st.session_state['reasoning_key']),
            (st.session_state['formatting_model'], st.session_state['formatting_key'])
        )
        
        with st.spinner("Verifying API keys..."):
            if models_obj.verify_api_keys():
                st.session_state.models = models_obj
                st.success("API keys verified successfully.")
            else:
                st.error("Invalid API keys. Please try again.")

        st.session_state.models_changed = False

def model_changed():
    """
    Mark the models as changed.
    """
    st.session_state.models_changed = True

def setup_models_and_db() -> bool:
    """
    Setup the models and database configuration.
    """
    with st.expander("**Select the models and enter API keys**"):
        for model_type in mu.SUPPORTED_MODELS.keys():
            st.selectbox(f"Select {model_type} Model", mu.SUPPORTED_MODELS[model_type], key=f"{model_type}_model", on_change=model_changed)
            st.text_input(f"Insert API Key for {model_type} model", key=f"{model_type}_key", value="", type="password", on_change=model_changed)
        
        st.divider()
        st.session_state.use_recipes_db = False
        st.text_input("**Or enter Password to use our own models and keys**", key="pswrd", type="password", on_change=model_changed)
        st.button("Submit", on_click=submit_models)

def add_uploaded_images(container: st.container, images: List[str], uploaded_files: List[Any], page_width: int, page_height: int) -> None:
    """
    Adds uploaded images to the Streamlit container and the images list.

    Args:
        container (st.container): The Streamlit container to display the images.
        images (List[str]): The list to store the image strings.
        uploaded_files (List[Any]): The list of uploaded image files.
        page_width (int): The width of the page.
        page_height (int): The height of the page.
    """
    for img in uploaded_files:
        add_uploaded_image(container, images, img, page_width, page_height)

def add_uploaded_image(container: st.container, images: List[str], img: Any, page_width: int, page_height: int) -> None:
    """
    Adds an uploaded image to the Streamlit container and the images list.

    Args:
        container (st.container): The Streamlit container to display the images.
        images (List[str]): The list to store the image strings.
        img (Any): The uploaded image file.
        page_width (int): The width of the page.
        page_height (int): The height of the page.
    """
    try:
        img_str = handle_image(img)
        images.append(img_str)
        st.session_state.messages.append({"type": "image", "role": "user", "content": img_str})
        with container.chat_message("user"):
            image = Image.open(img)
            image = resize_image(image, page_width, page_height)
            st.image(image)
    except Exception as e:
        st.error(f"Error processing uploaded image: {e}")
    
    
def initialize_session_state():
    """Initializes the session state variables."""
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
    if 'models' not in st.session_state:
        st.session_state.models = None    
    if 'exclude_cuisines' not in st.session_state:
        st.session_state.exclude_cuisines = []
    if 'dietary_preferences' not in st.session_state:
        st.session_state.dietary_preferences = []
    if 'intolerances' not in st.session_state:
        st.session_state.intolerances = []
    if 'ingredients' not in st.session_state:
        st.session_state.ingredients = []
    if 'tags_key' not in st.session_state:
        st.session_state.tags_key = 0
    if 'recipes_candidate_list' not in st.session_state:
        st.session_state.recipes_candidate_list = {}
    if 'recipes_requested_recipes' not in st.session_state: 
        st.session_state.requested_recipes = {}
    if 'recipes_recipes_data' not in st.session_state: 
        st.session_state.recipes_recipes_data = None   
    if 'expander_open' not in st.session_state:
        st.session_state.expander_open = True

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
    st.markdown(hide_default_format, unsafe_allow_html=True)

    initialize_session_state()

    page_height, page_width = get_window_dimensions()
    container_height = int(page_height * IMAGE_HEIGHT_RATIO)
    container = st.container(border=True, height=container_height)

    with container:
        st.markdown("<p style='text-align: center;'><img src='data:image/png;base64,{}'>\
                    </p>".format(base64.b64encode(open(LOGO_PATH, 'rb').read()).decode()), unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>Welcome to RatatouAI!</h1>", unsafe_allow_html=True) 
    with st.sidebar:
        setup_models_and_db()

        with st.expander("**Share your dietary preferences**"):
            st.multiselect("What are your dietary preferences", dietary_preferences, key="dietary_preferences")
            st.multiselect("Do you have any intolerances?", intolerances, key="intolerances")
            st.multiselect("Exclude these cuisines:", cuisines, key="exclude_cuisines")

    if st.session_state.models is None:
        st.info("Please select the models and enter your API keys in the side bar fields.")
        return  

    display_messages(container)

    stream_handler = StreamHandler(container, st.session_state)
    rat_brain = mu.RatBrain(stream_handler, st.session_state.models)
    
    input_container = st.container()
    user_prompt = input_container.chat_input("Say something")

    if user_prompt:
        handle_user_prompt(container, user_prompt, rat_brain)
    with st.container(border=True):
        st.session_state.ingredients = st_tags(
            label='**Please add your food items. You can add items directly, or use the image and camera uploaders below.**',
            text='Type and press enter to add more',
            value=st.session_state.ingredients,
            key=f"tags_{st.session_state.tags_key}",
            maxtags=-1
        )
            
        images = []

        with st.expander("Upload images of the food ingredients you have", icon=":material/add_photo_alternate:"):
            uploaded_files = st.file_uploader(
                "", type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key=f'file_uploader_key_{st.session_state["file_uploader_key"]}'
            )

            if len(uploaded_files) > 0:
                add_uploaded_images(container, images, uploaded_files, page_width, page_height)
                st.session_state['file_uploader_key'] += 1

        with st.expander("Take a picture of the food ingredients you have", icon=":material/add_a_photo:"):
            img_file_buffer = st.camera_input("", key=f'camera_input_key_{st.session_state["camera_input_key"]}')

            if img_file_buffer is not None:
                add_uploaded_image(container, images, img_file_buffer, page_width, page_height)
                st.session_state['camera_input_key'] += 1



    process_images(images, container, rat_brain)  
    feeling_lucky = input_container.button("Feeling Lucky - Suggest me 10 recipes!", disabled=len(st.session_state.ingredients)== 0 , icon="😋", type="primary") 

    if feeling_lucky:
        fetch_recipes(rat_brain)

    review_recipes(rat_brain)

    if st.session_state.recipes_recipes_data is not None:
        with st.spinner("Fetching the recipes information..."):
            st.json(st.session_state.recipes_recipes_data)

def display_messages(container):
    """Displays the messages in the session state."""
    for message in st.session_state.messages:
        with container.chat_message(message["role"], avatar=AVATAR if message["role"] == "assistant" else None):
            if message["type"] == "image":
                st.image(Image.open(io.BytesIO(base64.b64decode(message["content"]))))
            else:
                st.markdown(message["content"])

def process_images(images, container, rat_brain):
    """Processes the uploaded images and extracts ingredients."""
    semaphore = threading.Semaphore(1)
    if len(images) > 0:
        semaphore.acquire()
        try:
            ingredients = rat_brain.get_ingredients_from_images(images)
            st.session_state.ingredients.extend(ingredients)
            st.session_state.tags_key += 1
            st.rerun()
        except Exception as e:
            st.error(f"Error processing images: {e}")
        finally:
            semaphore.release()

def handle_user_prompt(container, user_prompt, rat_brain):
    """Handles the user prompt and generates a response."""
    st.session_state.messages.append({"type": "text", "role": "user", "content": user_prompt})
    with container.chat_message("user"):
        st.write(user_prompt)
    with st.spinner("Thinking..."):
        try:
            rat_brain.reply(user_prompt)
        except Exception as e:
            st.error(f"Error processing user prompt: {e}")

def fetch_recipes(rat_brain):
    """Fetches recipes based on the ingredients."""
    with st.spinner("Getting the list of recipes for you..."):
        try:
            recipes = rat_brain.get_recipes_from_ingredients(st.session_state.ingredients)

            for recipe in recipes:
                if recipe['title'] not in st.session_state.recipes_candidate_list:
                    st.session_state.recipes_candidate_list[recipe['title']] = {
                        'description': recipe['description'],
                        'ingredients': recipe['ingredients'],
                        'checked': False
                    }
        except Exception as e:
            st.error(f"Error fetching recipes: {e}")

def review_recipes(rat_brain):
    """Displays the recipes for review and selection."""
    with st.expander("Review the recipes and select the ones you want to make", expanded=st.session_state.expander_open, icon=":material/checklist:") as expander:

        col1, col2, col3 = st.columns([1, 1, 8])

        with col1:
            if st.button("📜", help="Submit selected recipes"):
                st.session_state.requested_recipes = [
                    (title, recipe['description'], recipe['ingredients'])
                    for title, recipe in st.session_state.recipes_candidate_list.items()
                    if recipe['checked']
                ]
                if len(st.session_state.requested_recipes) > 0:
                    with st.spinner():
                        try:
                            st.session_state.recipes_recipes_data = rat_brain.get_recipes_from_titles(st.session_state.requested_recipes)
                        except Exception as e:
                            st.error(f"Error fetching detailed recipes: {e}")

        with col2:
            if st.button("🗑️", help="Delete selected recipes"):
                selected_recipes = [
                    title for title, recipe in st.session_state.recipes_candidate_list.items()
                    if recipe['checked']
                ]
                for recipe in selected_recipes:
                    del st.session_state.recipes_candidate_list[recipe]

        for title, recipe in st.session_state.recipes_candidate_list.items():
            checked = st.checkbox(
                f"{title} - {recipe['description']} - {recipe['ingredients']})",
                value=recipe['checked'],
                key=f"recipe_{title}"
            )
            st.session_state.recipes_candidate_list[title]['checked'] = checked


if __name__ == "__main__":
    run()