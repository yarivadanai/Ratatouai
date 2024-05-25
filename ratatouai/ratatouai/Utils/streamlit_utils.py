from PIL import Image
import io
import base64
import streamlit as st
from typing import Callable

# Constants
MAX_IMAGE_SIZE = 1080
WIDTH_RATIO = 0.6

class StreamHandler:
    """Handles streaming of messages in a Streamlit app."""

    def __init__(self, container, session_state):
        """Initializes the StreamHandler with a container and a session state."""
        self.container = container
        self.draft = container.empty()
        self.messages = session_state.messages
        self.session_state = session_state

    def stream(self, streamer: Callable, persist: bool = False) -> str:
        """Streams messages from a streamer. If persist is True, the messages are persisted."""
        if persist:
            response = self.container.chat_message("assistant").write_stream(streamer)
            self.messages.append({"type": "text", "role": "assistant", "content": response})
        else:
            response = self.draft.write_stream(streamer)
        return response

    def write(self, output: str, persist: bool = False) -> None:
        """Writes a message. If persist is True, the message is persisted."""
        if persist:
            self.container.chat_message("assistant").write(output)
            self.messages.append({"type": "text", "role": "assistant", "content": output})
        else:
            self.draft.write(output)

    def close(self):
        """Closes the stream."""
        self.session_state['running'] = False
    
    def get_openai_key(self):
        return self.session_state.secret_keys['openai_api_key']

    def get_anthropic_key(self):
        return self.session_state.secret_keys['anthropic_api_key']
    

def handle_image(img: Image) -> str:
    """Opens the image, resizes it, and converts it to base64."""
    # Open the image and resize it
    image = Image.open(img)
    ratio = max(image.width, image.height) // MAX_IMAGE_SIZE
    image = image.resize((image.width // ratio, image.height // ratio))

    # Convert the image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_str

def resize_image(img: Image, width: int, height: int) -> Image:
    """Resizes an image while maintaining its aspect ratio."""
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

