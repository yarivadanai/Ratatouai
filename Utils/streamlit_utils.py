from PIL import Image
import io
import base64
import streamlit as st
from typing import Callable
import jsonpatch

# Constants
IS_LOCAL = False
MAX_IMAGE_SIZE = 1080
WIDTH_RATIO = 0.6
if IS_LOCAL:
    LOGO_PATH = "/Users/yarivadan/projects/VSProjects/Ratatouai/static/logo.png"
else:
    LOGO_PATH = "./static/logo.png"
AVATAR=Image.open(LOGO_PATH)

class StreamHandler:
    """Handles streaming of messages in a Streamlit app."""

    def __init__(self, container, session_state):
        """Initializes the StreamHandler with a container and a session state."""
        self.container = container
        self.messages = session_state.messages
        self.session_state = session_state

    def stream(self, streamer: Callable, persist: bool = False, isjason=False) -> str:
        """
        Streams messages from a streamer. If persist is True, the messages are persisted.
        For jsonoutputparser objects - use a diff method and apply jsonpatch to connect the diffs
        """
        response = ""
        if persist:
            response = self.container.chat_message("assistant", avatar=AVATAR).write_stream(streamer)
            self.messages.append({"type": "text", "role": "assistant", "content": response})
        elif isjason is True:
            initial_json = {}
            draft = self.container.empty()
            for diff in streamer:
                if isinstance(diff, list):
                    initial_json = jsonpatch.apply_patch(initial_json, diff)
                    draft.write(initial_json)
                    response = initial_json
        else:
            #draft = self.container.empty()
            for message in streamer:
                response += message
                #draft.write({response})
            
        return response

    def write(self, output: str, persist: bool = False) -> None:
        """Writes a message. If persist is True, the message is persisted."""
        if persist:
            self.container.chat_message("assistant", avatar=AVATAR).write(output)
            self.messages.append({"type": "text", "role": "assistant", "content": output})
        else:
            draft = self.container.empty()
            draft.write(output)

    def close(self):
        """Closes the stream."""
        self.session_state['running'] = False


def handle_image(img: Image) -> str:
    """
    Opens the image, resizes it, and converts it to base64.
    :param img: The image to handle.
    :return: The base64 encoded image.
    """
    # Open the image and resize it
    image = Image.open(img)
    ratio = MAX_IMAGE_SIZE / max(image.width, image.height) 
    image = image.resize((int(image.width * ratio), int(image.height * ratio)))

    # Convert the image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_str


def resize_image(img: Image, width: int, height: int) -> Image:
    """
    Resizes an image while maintaining its aspect ratio.
    :param img: The image to resize.
    :param width: The desired width of the image.
    :param height: The desired height of the image.
    :return: The resized image.
    """
    # Calculate the aspect ratio of the image
    aspect_ratio = img.width / img.height

    # If the image is wider than it is tall, resize the image to have a width equal to the size of the grid cell
    if aspect_ratio > 1:
        new_width = int(width * WIDTH_RATIO)
        new_height = int((new_width / img.width) * img.height)
    # If the image is taller than it is wide, resize the image to have a height equal to the size of the grid cell
    else:
        new_height = int(height * WIDTH_RATIO)
        new_width = int((new_height / img.height) * img.width)

    # Resize the image, maintaining the aspect ratio
    new_img = img.resize((new_width, new_height))

    return new_img


