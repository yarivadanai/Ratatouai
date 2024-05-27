import json
import logging
from typing import Generator, List, Callable, Tuple
from anthropic import APIStatusError
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel as PydanticBaseModel, Field as PydanticField
from langchain_openai import ChatOpenAI
from langsmith import traceable
from Utils.streamlit_utils import StreamHandler

# Standard library imports
import Utils.prompts as prompts

# Constants
LLAMA3_MODEL = "llama3"
CLAUDE3_OPUS_MODEL = "claude-3-opus-20240229"
GPT4O_MODEL = "gpt-4o"


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


class Models:
    """Class for the RatatouAI brain."""

    def __init__(self, vision_model: Tuple[str, str], reasoning_model: Tuple[str, str], formatting_model: Tuple[str, str]):
        models = {
            GPT4O_MODEL: ChatOpenAI,
            CLAUDE3_OPUS_MODEL: ChatAnthropic,
            LLAMA3_MODEL: Ollama
        }
        self.vision = models[vision_model[0]](temperature=0, model_name=vision_model[0], api_key=vision_model[1])
        self.reasoning = models[reasoning_model[0]](temperature=0, model_name=reasoning_model[0], api_key=reasoning_model[1])
        self.formatting = models[formatting_model[0]](temperature=0, model_name=formatting_model[0], api_key=formatting_model[1])

    def verify_api_keys(self) -> bool:
        """Verifies the API keys and returns True if they are valid."""
        try:
            self.vision.invoke("Hello RatatouAI!")
            self.reasoning.invoke("Hello RatatouAI!")
            self.formatting.invoke("Hello RatatouAI!")
            return True
        except APIStatusError as e:
            logging.error("An error occurred:", exc_info=True)
            return False

    def get_vision_model(self) -> BaseLLM:
        """Returns the vision model."""
        return self.vision

    def get_reasoning_model(self) -> BaseLLM:
        """Returns the reasoning model."""
        return self.reasoning

    def get_formatting_model(self) -> BaseLLM:
        """Returns the formatting model."""
        return self.formatting


class RatBrain:
    """Class for the RatatouAI brain."""

    def __init__(self, stream_handler: StreamHandler, model: Models):
        self.vision_model = model.get_vision_model()
        self.reasoning_model = model.get_reasoning_model()
        self.formatting_model = model.get_formatting_model()
        self.stream_handler = stream_handler

    @traceable
    def run_full_chain(self, images: List[str]) -> None:
        """Runs the full chain of image analysis and chat models."""
        food_items = set()

        self.stream_handler.write("Looking for food ingredients...", True)

        for img in images:
            answer = ImageAnswer.parse_raw(self._analyze_image(img))
            for item in answer.answer:
                food_items.add(item)

        if len(food_items) == 0:
            self.stream_handler.write("Sorry - couldn't find any food ingredients in the images :(", True)
            return

        self.stream_handler.write("Found the following food items:", True)
        ingredients = ", ".join(food_items)
        self._format_ingredients(ingredients)

        self.stream_handler.write("RatatouAIng 3 recipes with these ingredients...", True)

        recipes = self._get_recipes(ingredients)

        for recipe in recipes:
            recipe = recipe.json()
            """
            missing_ingredients_prompt = self._get_prompt_template(
                prompts.system_ingredients_gaps_prompt_template,
                prompts.human_ingredients_gaps_prompt_template
            )
            reflection_prompt = self._get_prompt_template(
                prompts.system_recipe_reflection_prompt_template,
                prompts.human_recipe_reflection_prompt_template
            )
            formatted_recipe_prompt = self._get_prompt_template(
                prompts.system_content_formatting_prompt_template,
                prompts.human_final_formatting_prompt_template
            )

            missing_ingredients = (
                {"recipe": RunnablePassthrough(), "ingredients_list": RunnablePassthrough()} |
                missing_ingredients_prompt | self.reasoning_model | StrOutputParser()
            )
            reflection = ({"recipe": RunnablePassthrough()} | reflection_prompt | self.reasoning_model | StrOutputParser())
            chain = (
                {"missing_ingredients": missing_ingredients, "reflection": reflection, "recipe": RunnablePassthrough()} |
                formatted_recipe_prompt | self.formatting_model | StrOutputParser()
            )"""

            missing_ingredients=self._execute_chain(prompts.system_ingredients_gaps_prompt_template,
                prompts.human_ingredients_gaps_prompt_template, self.reasoning_model, StrOutputParser(),
                {"recipe": recipe, "ingredients_list": ingredients})
            
            reflection=self._execute_chain(prompts.system_recipe_reflection_prompt_template,
                prompts.human_recipe_reflection_prompt_template, self.reasoning_model, StrOutputParser(),
                {"recipe": recipe})
            
            self._execute_chain(prompts.system_content_formatting_prompt_template,
                prompts.human_final_formatting_prompt_template, self.formatting_model, StrOutputParser(),
                {"missing_ingredients": missing_ingredients, "reflection": reflection, "recipe": recipe}, True)


            #self.stream_handler.stream(self._stream_chain(chain, {"recipe": recipe, "ingredients_list": ingredients}), True)

        self.stream_handler.close()

    @traceable
    def _analyze_image(self, img: str) -> ImageAnswer:
        """Analyzes an image and returns an ImageAnswer object."""
        system_message = SystemMessagePromptTemplate.from_template(prompts.system_img2ingredients_prompt_template)

        human_message = HumanMessage(
            content=[
                {"type": "text", "text": "Here is an image:\n"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}", }},
                {"type": "text", "text": "\n" + prompts.human_img2ingredients_prompt_template},
            ])

        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        output_parser = JsonOutputParser(pydantic_object=ImageAnswer, diff=True)

        chain = chat_prompt | self.vision_model | output_parser
        return self._handle_json(self.stream_handler.stream(self._stream_chain(chain, {})))

    @traceable
    def _format_ingredients(self, ingredients: str) -> str:
        """Formats the ingredients and returns the result."""
        result = self._execute_chain(
            prompts.system_content_formatting_prompt_template,
            prompts.human_ingredients_formatting_prompt_template,
            self.formatting_model,
            StrOutputParser(),
            {"ingredients": ingredients},
            True
        )
        return result

    @traceable
    def _get_recipes(self, ingredients: str) -> List[Recipe]:
        """Gets recipes based on the ingredients and returns a list of Recipe objects."""
        recipes = self._execute_chain(
            prompts.system_ingredients2recipe_prompt_template,
            prompts.human_ingredients2recipe_prompt_template,
            self.reasoning_model,
            JsonOutputParser(pydantic_object=RecipesLists, diff=True),
            {"ingredients_list": ingredients, "recipe_answer_outputformat": prompts.recipeslist_answer_outputformat}
        )

        return RecipesLists.parse_raw(json.dumps(recipes)).recipes_lists

    @traceable
    def _get_prompt_template(self, system_template: str, human_template: str) -> ChatPromptTemplate:
        """Gets a prompt template based on the system and human templates."""
        system_message = SystemMessagePromptTemplate.from_template(system_template)
        human_message = HumanMessagePromptTemplate.from_template(human_template)

        return ChatPromptTemplate.from_messages([system_message, human_message])

    @traceable
    def _execute_chain(self, system_template: str, human_template: str, chat: BaseLLM, output_parser: Callable,
                       params_dict: dict, persist: bool = False) -> str:
        """Executes a chain of operations and returns the result."""
        chat_prompt = self._get_prompt_template(system_template, human_template)
        chain = chat_prompt | chat | output_parser

        return self.stream_handler.stream(self._stream_chain(chain, params_dict), persist)

    @traceable
    def reply(self, user_prompt: str) -> str:
        """Replies to a user prompt and returns the result. Includes the last 10~ messages in the conversation."""
        chat = self.reasoning_model
        messages = [(m["role"], m["content"]) for m in self.stream_handler.messages if m["type"] == "text"][:-1]
        messages.insert(0,("system", prompts.system_chitchat_prompt_template))
        messages.append(("user", user_prompt))

        prompt_template = ChatPromptTemplate.from_messages(messages)
        chain = prompt_template | chat | StrOutputParser()
        return self.stream_handler.stream(self._stream_chain(chain, {}), True)

    @traceable
    def _stream_chain(self, chain: Callable, params_dict: dict) -> Generator:
        """Streams a chain of operations and yields the result."""
        try:
            for chunk in chain.stream(params_dict):
                yield chunk
        except APIStatusError as ase:
            logging.error("An error occurred while calling the model:", exc_info=True)
        except Exception as e:
            logging.error("An error occurred while calling the model:", exc_info=True)
            return

    def _handle_json(self, output: List) -> str:
        """Handles a JSON output and returns the result."""
        return json.dumps(output)
