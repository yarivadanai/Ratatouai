# Standard library imports
import json
from typing import Generator, List, Callable

# Third-party library imports
from anthropic import APIStatusError
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langsmith import traceable

# Local application/library specific imports
import Utils.prompts as prompts 
from Utils.streamlit_utils import StreamHandler

# Constants
LLAMA3_MODEL = "llama3"
CLAUDE3_OPUS_MODEL = "claude-3-opus-20240229"

class ImageAnswer(BaseModel):
    """Model for image analysis results."""
    thinking: str = Field(description="The step by step thinking and analysis of the image.")
    answer: List[str] = Field(description="The list of food items in the image. Each food item is a separate entry in the list.")

class Recipe(BaseModel):
    """Model for recipe details."""
    title: str = Field(description="The name of the recipe.")
    description: str = Field(description="A short description of the recipe.")
    ingredients: List[str] = Field(description="The ingredients for the recipe.")
    instructions: List[str] = Field(description="The cooking instructions for the recipe.")

class RecipesLists(BaseModel):
    """Model for a list of recipes."""
    recipes_lists: List[Recipe] = Field(description="List of 3 Recipe items.")

class RecipeReflection(BaseModel):
    """Model for recipe reflection details."""
    nutrition: str = Field(description="Detailed nutrition and calories data for the recipe.")
    missing: str = Field(description="The list of missing ingredients required by the recipe but not found in the provided list of ingredients.")
    suggestions: str = Field(description="Suggestions on how this recipe can be made healthier.")


def get_local_model():
    """Returns an instance of the Ollama model."""
    return Ollama(model=LLAMA3_MODEL)

def get_claude3_opus(anthropic_key):
    """Returns an instance of the ChatAnthropic model with a specific model name."""
    return ChatAnthropic(temperature=0, model_name=CLAUDE3_OPUS_MODEL, api_key=anthropic_key)

def get_chat_gpt(openai_api_key, model_name="gpt-4o"):
    """Returns an instance of the ChatOpenAI model with a specified model name."""
    return ChatOpenAI(temperature=0, model_name=model_name, api_key=openai_api_key)


@traceable
def run_full_chain(images: List[str], stream_handler: StreamHandler):
    """Runs the full chain of image analysis and chat models."""
    vision = get_chat_gpt(stream_handler.get_openai_key())
    claude3_chat = get_claude3_opus(stream_handler.get_anthropic_key())
    food_items = set()

    stream_handler.write("Looking for food ingredients...", True)

    for img in images:
        answer = ImageAnswer.parse_raw(analyzeImage(img, vision, stream_handler))
        for item in answer.answer : 
            food_items.add(item)
    
    if len(food_items) == 0:
        stream_handler.write("Sorry - couldn't find any food ingredients in the images :(", True)
        return

    stream_handler.write("Found the following food items:", True)
    ingredients = ", ".join(food_items)
    format_ingredients(ingredients, claude3_chat, stream_handler)

    stream_handler.write("RatatouAIng 3 recipes with these ingredients...", True) 

    recipes = getRecipes(ingredients, claude3_chat, stream_handler)

    for recipe in recipes:
        recipe = recipe.json()

        missing_ingredients_prompt = get_prompt_template(prompts.system_ingredients_gaps_prompt_template, 
                                                        prompts.human_ingredients_gaps_prompt_templatee)
        reflection_prompt = get_prompt_template(prompts.system_recipe_reflection_prompt_templatee, 
                                                prompts.human_recipe_reflection_prompt_template)
        formatted_recipe_prompt = get_prompt_template(prompts.system_content_formatting_prompt_template, 
                                                      prompts.human_final_formatting_prompt_template)

        missing_ingredients = ({"recipe": RunnablePassthrough(),"ingredients_list": RunnablePassthrough() } | 
                               missing_ingredients_prompt | claude3_chat | StrOutputParser())
        reflection = ({"recipe": RunnablePassthrough()} | reflection_prompt | claude3_chat | StrOutputParser())
        chain = ({"missing_ingredients": missing_ingredients, "reflection" : reflection, "recipe": RunnablePassthrough()} | 
                 formatted_recipe_prompt | claude3_chat | StrOutputParser())

        stream_handler.stream(stream_chain(chain, {"recipe": recipe, "ingredients_list": ingredients}), True) 
    stream_handler.close()


@traceable
def analyzeImage(img: str, chat: BaseLLM, stream_handler: StreamHandler) -> ImageAnswer:
    """Analyzes an image and returns an ImageAnswer object."""
    system_message = SystemMessagePromptTemplate.from_template(prompts.system_img2ingredients_prompt_template)

    human_message = HumanMessage(
        content=[
            {"type": "text", "text": "Here is an image:\n"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}", },},
            {"type": "text", "text": "\n" + prompts.human_img2ingredients_prompt_template}, 
            ])

    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    output_parser = JsonOutputParser(pydantic_object=ImageAnswer)
    
    chain = chat_prompt | chat | output_parser
    return handle_json(stream_handler.stream(stream_chain(chain, {})))

@traceable
def format_ingredients(ingredients: str, chat: BaseLLM, stream_handler: StreamHandler):
    """Formats the ingredients and returns the result."""
    result = execute_chain(prompts.system_content_formatting_prompt_template, 
                    prompts.human_ingredients_formatting_prompt_template, chat, 
                    StrOutputParser(), {"ingredients": ingredients}, stream_handler, True)
    return result

@traceable
def getRecipes(ingredients: str, chat: BaseLLM, stream_handler: StreamHandler) -> RecipesLists:
    """Gets recipes based on the ingredients and returns a RecipesLists object."""
    recipes = execute_chain(prompts.system_ingredients2recipe_prompt_template, 
                    prompts.human_ingredients2recipe_prompt_template, chat, 
                    JsonOutputParser(pydantic_object=RecipesLists), 
                    {"ingredients_list": ingredients, "recipe_answer_outputformat" : prompts.recipeslist_answer_outputformat}, 
                    stream_handler)

    return RecipesLists.parse_raw(json.dumps(recipes[-1])).recipes_lists

@traceable 
def get_prompt_template(system_template, human_template):
    """Gets a prompt template based on the system and human templates."""
    system_message = SystemMessagePromptTemplate.from_template(system_template)
    human_message = HumanMessagePromptTemplate.from_template(human_template)

    return ChatPromptTemplate.from_messages([system_message, human_message])

@traceable 
def execute_chain(system_template: str, human_template: str, chat: BaseLLM, output_parser: Callable, 
                  params_dict: dict, stream_handler: StreamHandler, persist: bool = False) -> str:
    """Executes a chain of operations and returns the result."""
    chat_prompt = get_prompt_template(system_template, human_template)
    chain = chat_prompt | chat | output_parser
    return stream_handler.stream(stream_chain(chain, params_dict), persist)

def reply(user_prompt: str, stream_handler: StreamHandler) -> str:
    """Replies to a user prompt and returns the result."""
    chat = get_claude3_opus(stream_handler.get_anthropic_key())
    messages = []
    last_10_messages = list(stream_handler.messages)#[-10:]
        
    for m in last_10_messages:
        if m["type"] == "text":
            messages.append((m["role"], m["content"]))

    promptTmplt = ChatPromptTemplate.from_messages(messages)  
    chain = promptTmplt | chat | StrOutputParser()
    return stream_handler.stream(stream_chain(chain, {}), True)

def stream_chain(chain: Callable, params_dict: dict) -> Generator:
    """Streams a chain of operations and yields the result."""
    for chunk in chain.stream(params_dict):
        yield chunk

def handle_json(output: List) -> str:
    """Handles a JSON output and returns the result."""
    return json.dumps(output[-1])
