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
import concurrent.futures

# Standard library imports
import Utils.prompts as prompts

DEBUG_USE_MOCK_INGREDIENS=False
USE_RECIPES_DB = False  


# Constants
LLAMA3_MODEL = "llama3.1"
CLAUDE_MODEL = "claude-3.5-sonnet"
GPT4O_MODEL = "gpt-4o"
GEMMA_MODEL = "gemma2"

SUPPORTED_MODELS = {"vision" : [GPT4O_MODEL, CLAUDE_MODEL],
           "reasoning" : [LLAMA3_MODEL, CLAUDE_MODEL, GPT4O_MODEL, GEMMA_MODEL],
           "formatting" : [LLAMA3_MODEL, CLAUDE_MODEL, GPT4O_MODEL, GEMMA_MODEL]}

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
            CLAUDE_MODEL: ChatAnthropic,
            LLAMA3_MODEL: Ollama,
            GEMMA_MODEL: Ollama,
        }
        self.vision = models[vision_model[0]](
            temperature=0.2,
            model=vision_model[0],
            api_key=vision_model[1]
        ) if vision_model[1] != "" else models[vision_model[0]](
            temperature=0,
            model=vision_model[0]
        )
        self.reasoning = models[reasoning_model[0]](
            temperature=0.2,
            model=reasoning_model[0],
            api_key=reasoning_model[1]
        ) if reasoning_model[1] != "" else models[reasoning_model[0]](
            temperature=0,
            model=reasoning_model[0]
        )
        self.formatting = models[formatting_model[0]](
            temperature=0.2,
            model=formatting_model[0],
            api_key=formatting_model[1]
        ) if formatting_model[1]  != "" else models[formatting_model[0]](
            temperature=0,
            model=formatting_model[0]
        )

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

    def __init__(self, app_config: StreamHandler, model: Models):
        self.vision_model = model.get_vision_model()
        self.reasoning_model = model.get_reasoning_model()
        self.formatting_model = model.get_formatting_model()
        self.app_configurator = app_config

    @traceable
    def run_full_chain(self, images: List[str]) -> None:
        """Runs the full chain of image analysis and chat models."""
        food_items = set()

        if DEBUG_USE_MOCK_INGREDIENS:
            self.app_configurator.write("Mocking up food ingredients 🐀🔍🧀", True)
            food_items = {'Eggs', 'Yogurt', 'Chocolate bunny', 'Yogurt drink', 'Cheese', 'Sliced meat',
                          'Olives', 'Butter', 'Hummus', 'Cheese slices', 'Sliced cheese', 'Soup',
                          'Potatoes', 'Carrots', 'Green leafy vegetables', 'mushrooms', 'citrus fruits',
                          'eggs', 'tomatoes', 'bread rolls', 'raw chicken drumsticks', 'sauce or broth',
                          'packaged frozen or chilled food', 'strawberries', 'cucumbers', 'red cabbage',
                          'tomato', 'red currants', 'yellow bell peppers', 'green chili peppers',
                          'lettuce', 'cherry tomatoes', 'romaine lettuce', 'mixed greens', 'fennel bulb',
                          'milk', 'watermelon', 'milk cartons', 'green onions'}

        else:

            self.app_configurator.write("Looking for food ingredients 🐀🔍🧀", True)

            for img in images:
                answer = ImageAnswer.parse_raw(self._analyze_image(img))
                for item in answer.answer:
                    food_items.add(item)

            if len(food_items) == 0:
                self.app_configurator.write("Sorry - couldn't find any food ingredients in the images 😔", True)
                return

        self.app_configurator.write("🎉🎉 Found the following food items: 🎉🎉", True)
        ingredients = ", ".join(food_items)
        self._format_ingredients(ingredients)


        if self.app_configurator.session_state.use_recipes_db:
            recipes = self._get_recipes_from_db(ingredients)
        else:
            self.app_configurator.write("🍳🍔 RatatouAIng 3 recipes with these ingredients... 🥕", True)
            recipes = self._get_recipes_from_model(ingredients)
        

        self.app_configurator.write("Almost there! Let me format it nicely, and add some healthy touches 🥦", True)

        # Process each recipe in parallel
        def process_recipe(recipe):
            recipe = recipe.json()
            missing_ingredients = self._get_missing_ingredients(ingredients, recipe)
            reflection = self._get_reflections(recipe)
            return recipe, missing_ingredients, reflection

        with concurrent.futures.ThreadPoolExecutor() as executor:
            recipe_futures = [executor.submit(process_recipe, recipe) for recipe in recipes]
            for future in concurrent.futures.as_completed(recipe_futures):
                recipe, missing_ingredients, reflection = future.result()
                self._format_final_output(recipe, missing_ingredients, reflection)

        self.app_configurator.close()


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
        return self._handle_json(self.app_configurator.stream(self._stream_chain(chain, {}), isjason=True))

    @traceable
    def _format_ingredients(self, ingredients: str) -> str:
        """Formats the ingredients and returns the result."""
        result = self._execute_chain(
            prompts.system_content_formatting_prompt_template,
            prompts.human_ingredients_formatting_prompt_template,
            self.formatting_model,
            StrOutputParser(),
            {"ingredients": ingredients},
            persist = True
        )
        return result

   
    @traceable
    def _get_recipes_from_model(self, ingredients: str) -> List[Recipe]:
        """Gets recipes based on the ingredients and returns a list of Recipe objects."""
        recipes = self._execute_chain(
            prompts.system_ingredients2recipe_prompt_template,
            prompts.human_ingredients2recipe_prompt_template,
            self.reasoning_model,
            JsonOutputParser(pydantic_object=RecipesLists, diff=True),
            {"ingredients_list": ingredients, "recipe_answer_outputformat": prompts.recipeslist_answer_outputformat}, 
            isjason=True
        )

        return RecipesLists.parse_raw(json.dumps(recipes)).recipes_lists
    
        
    @traceable
    def _get_recipe_titles(self, ingredients: str) -> List[str]:
        """Gets recipes based on the ingredients and returns a list of Recipe titles."""
        recipes = self._execute_chain(
            prompts.system_ingredients2recipe_prompt_template,
            prompts.human_ingredients2recipetitles_prompt_template,
            self.reasoning_model,
            JsonOutputParser(pydantic_object=RecipesLists, diff=True),
            {"ingredients_list": ingredients, "recipe_answer_outputformat": prompts.recipeslist_answer_outputformat}, 
            isjason=True
        )
        self.app_configurator.write(recipes, True)
        return recipes


    @traceable
    def _get_recipes_from_db(self, ingredients: str) -> List[Recipe]:
        import requests
       
        recipe_titles=self._get_recipe_titles(ingredients)
        recipes=[]

        excludeCuisine = self.app_configurator.session_state.excludeCuisine
        diet= self.app_configurator.session_state.dietary_preferences
        intolerances   = self.app_configurator.session_state.intolerances

        url = "" #TODO: Add the URL of the recipe database

        headers = { } #TODO: Add the headers for the recipe database

        for recipe in recipe_titles:
            querystring = {"query":recipe['title'],
                           "excludeCuisine":excludeCuisine,
                           "diet":diet,
                           "intolerances":intolerances,
                           "includeIngredients":ingredients,
                           #"excludeIngredients":"eggs",
                           "instructionsRequired":"true",
                           "fillIngredients":"true",
                           "addRecipeInformation":"true",
                           "addRecipeInstructions":"true",
                           "addRecipeNutrition":"true",
                           "offset":"0",
                           "number":"10",
                           "limitLicense":"false",
                           "ranking":"2"}
            self.app_configurator.write(querystring, False)

            response = requests.get(url, headers=headers, params=querystring)
            recipes.append(response.json())

            self.app_configurator.write(response.json(), True)

        return recipes


    @traceable
    def _format_final_output(self, recipe: Recipe, missing_ingredients: str, reflection: str) -> str:
        """Formats the final output and returns the result."""
        return self._execute_chain(
            prompts.system_content_formatting_prompt_template,
            prompts.human_final_formatting_prompt_template,
            self.formatting_model,
            StrOutputParser(),
            {"missing_ingredients": missing_ingredients, "reflection": reflection, "recipe": recipe},
            persist = True
        )
   
    @traceable
    def _get_reflections(self, recipe: Recipe) -> str:
        """Gets the reflections for a recipe and returns the result."""
        return self._execute_chain(
            prompts.system_recipe_reflection_prompt_template,
            prompts.human_recipe_reflection_prompt_template,
            self.reasoning_model,
            StrOutputParser(),
            {"recipe": recipe}
        )
            
    @traceable
    def _get_missing_ingredients(self, ingredients: str, recipe: Recipe) -> str:
        """Gets the missing ingredients for a recipe and returns the result."""
        return self._execute_chain(
            prompts.system_ingredients_gaps_prompt_template,
            prompts.human_ingredients_gaps_prompt_template,
            self.reasoning_model,
            StrOutputParser(),
            {"recipe": recipe, "ingredients_list": ingredients}
        )

    @traceable
    def _get_prompt_template(self, system_template: str, human_template: str) -> ChatPromptTemplate:
        """Gets a prompt template based on the system and human templates."""
        system_message = SystemMessagePromptTemplate.from_template(system_template)
        human_message = HumanMessagePromptTemplate.from_template(human_template)

        return ChatPromptTemplate.from_messages([system_message, human_message])

    @traceable
    def _execute_chain(self, system_template: str, human_template: str, chat: BaseLLM, output_parser: Callable,
                       params_dict: dict, persist: bool = False, isjason: bool = False) -> str:
        """Executes a chain of operations and returns the result."""
        chat_prompt = self._get_prompt_template(system_template, human_template)
        chain = chat_prompt | chat | output_parser

        return self.app_configurator.stream(self._stream_chain(chain, params_dict), persist, isjason)

    @traceable
    def reply(self, user_prompt: str) -> str:
        """Replies to a user prompt and returns the result. Includes the last 10~ messages in the conversation."""
        chat = self.reasoning_model
        messages = [(m["role"], m["content"]) for m in self.app_configurator.messages if m["type"] == "text"][:-1]
        messages.insert(0,("system", prompts.system_chitchat_prompt_template))
        messages.append(("user", user_prompt))

        prompt_template = ChatPromptTemplate.from_messages(messages)
        chain = prompt_template | chat | StrOutputParser()
        return self.app_configurator.stream(self._stream_chain(chain, {}), True)

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
