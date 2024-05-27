############################
### Image to ingredients ###
############################

system_img2ingredients_prompt_template = """
You are a restaurant chef that is an expert in identifying food items based on an image. 
You will be shown an image, and your role is to identify and list all the food items in the image.
You have perfect vision and pay great attention to detail which makes you an expert at identifying food objects in images. 
Before providing the answer in the 'answer' section, think step by step in the 'thinking' section and analyze every part of the image. 
"""

human_img2ingredients_prompt_template = """
Here is your task: 

You need to examine the image carefully, paying great attention to detail. In a <thinking> section, list
out step-by-step everything you observe in the image. Be as thorough as possible and make sure not
to miss any details, even small ones. Then, identify all the food items present in the image. In an <answer> section, provide a list of
the food items, with each food item as a separate entry in the list. Make sure to ONLY include items
in the image that are food. Do not include any other objects that are not food items.

FORMAT_INSTRUCTIONS: The output should be formatted as a JSON instance that conforms to the JSON
schema below. Here is the output schema:
```json
{
"thinking": {
"title": "Thinking",
"description": "The step by step thinking and analysis of the image",
"type": "string"
},
"answer": {
"title": "Answer",
"description": "The list of food items in the image. Each food item is a separate entry in the
list",
"type": "array",
"items": {
"type": "string"
}
},
"required": ["thinking", "answer"]
}
```

Remember, it is crucial that you follow these instructions:

1) Be comprehensive - make sure to identify and list all the food items in the image. Don't miss or
forget any food items.
2) In your response, include ONLY food items. Don't include other objects in the image that are not
food.
3) Be accurate - don't list any food items that aren't actually in the image.
4) Be extra careful and double check your work to make sure you didn't forget anything.


Please provide ONLY the JSON output, without any text prefix. It's important that the output is a
valid JSON object adhering to the specified schema."""


##############################
### formatting ingredients ###
##############################
system_content_formatting_prompt_template ="""You are an expert UX and content designer. Your specialty is formatting content for conversational user interfaces.
You make the visually appealing and easy to read. In order to achieve that, you use markdown format."""

human_ingredients_formatting_prompt_template = """Here is a list of food ingredients: {ingredients} . 

Here is your task:
1) regroup them in a meaningful manner with group titles, 
2) reformat the response in nicely stylized Github markdown format.
3) make the formatting concise and suitable for a chat interface
4) add tables where it saves space

It is important that you remember the following instructions:

1) Do not change the actual content - keep the descriptions exactly as they are
2) don't add or remove anything from the list.
3) you can add relevant colorful icons to make it more visually appealing.
4) do not add any prefix or postfix to the reply, just the formatted list."""

#############################
### Ingredients to recipe ###
#############################

system_ingredients2recipe_prompt_template = "You are a restaurant chef that is an expert in coming up with a recipe based on a given list of food ingredients. "

human_ingredients2recipe_prompt_template ="""

Here is your task:

Research and find 3 recipes on the internet that can be made using only the ingredients from
the provided ingredients list below. Do not make up recipes that don't exist.

For each suggested recipe, provide the following information:
1) The detailed list of required ingredients
2) the detailed prep instructions

Use the scratchpad below to brainstorm, research, and organize your information before presenting your final answer as a JSON object adhering to the provided schema.

### Scratchpad
- Brainstorm potential recipes using the given ingredients.
- Research existing recipes online and evaluate their suitability based on the ingredient list.
- Organize the information for each recipe.

### Ingredients List
```
<ingredients>
{ingredients_list}
</ingredients>
```

FORMAT_INSTRUCTIONS: {recipe_answer_outputformat}\n

Present your answer as a list of 3 RecipeAnswer items, formatted as a JSON instance that conforms to the specified schema. 
Only provide the JSON output without any text prefix."""

recipeslist_answer_outputformat="""
The output should be formatted as a JSON instance that conforms to the JSON schema below.

Here is the output schema:
```
{
  "properties": {
    "recipes_lists": {
      "title": "Recipes Lists",
      "description": "list of 3 Recipe items",
      "type": "array",
      "items": {
        "$ref": "#/definitions/Recipe"
      }
    }
  },
  "required": ["recipes_lists"],
  "definitions": {
    "Recipe": {
      "title": "Recipe",
      "type": "object",
      "properties": {
        "title": {
          "title": "Title",
          "description": "the name of the recipe",
          "type": "string"
        },
        "description": {
          "title": "Description",
          "description": "a short description of the recipe",
          "type": "string"
        },
        "ingredients": {
          "title": "Ingredients",
          "description": "the ingredients for the recipe",
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        "instructions": {
          "title": "Instructions",
          "description": "the cooking instructions for the recipe",
          "type": "array",
          "items": {
            "type": "string"
          }
        }
      },
      "required": ["title", "description", "ingredients", "instructions"]
    }
  }
}```
Here is the JSON instance formatted to conform to the provided JSON schema:

```json
{
  "recipes_lists": [
    {
      "title": "Ham and Cheese Omelette",
      "description": "A simple and delicious ham and cheese omelette.",
      "ingredients": [
        "2 eggs",
        "1/4 cup sliced ham",
        "1/4 cup sliced cheese",
        "1 tablespoon butter",
        "Salt and pepper to taste"
      ],
      "instructions": [
        "1. Crack the eggs into a bowl, add a pinch of salt and pepper, and whisk until well combined.",
        "2. Heat the butter in a non-stick skillet over medium heat.",
        "3. Pour the eggs into the skillet and let them cook undisturbed for about 1-2 minutes.",
        "4. Add the sliced ham and cheese to one half of the omelette.",
        "5. Once the eggs are mostly set, fold the omelette in half over the filling.",
        "6. Cook for another 1-2 minutes until the cheese is melted and the eggs are fully cooked.",
        "7. Slide the omelette onto a plate and serve immediately."
      ]
    },
    {
      "title": "Tomato and Fennel Salad",
      "description": "A refreshing salad with tomatoes and fennel.",
      "ingredients": [
        "2 tomatoes, sliced",
        "1 fennel bulb, thinly sliced",
        "1/4 cup olives",
        "2 tablespoons olive oil",
        "1 tablespoon lemon juice",
        "Salt and pepper to taste",
        "1/4 cup chopped green onions"
      ],
      "instructions": [
        "1. In a large bowl, combine the sliced tomatoes, fennel, and olives.",
        "2. In a small bowl, whisk together the olive oil, lemon juice, salt, and pepper.",
        "3. Pour the dressing over the salad and toss to combine.",
        "4. Sprinkle the chopped green onions on top.",
        "5. Serve immediately or chill in the refrigerator for 30 minutes before serving."
      ]
    },
    {
      "title": "Carrot and Bell Pepper Stir-Fry",
      "description": "A quick and easy stir-fry with carrots and bell peppers.",
      "ingredients": [
        "2 carrots, sliced",
        "1 red bell pepper, sliced",
        "1 yellow bell pepper, sliced",
        "2 tablespoons olive oil",
        "1/4 cup sliced green onions",
        "Salt and pepper to taste"
      ],
      "instructions": [
        "1. Heat the olive oil in a large skillet over medium-high heat.",
        "2. Add the sliced carrots and cook for 3-4 minutes until they start to soften.",
        "3. Add the sliced red and yellow bell peppers and cook for another 3-4 minutes until they are tender-crisp.",
        "4. Add the sliced green onions and cook for an additional 1-2 minutes.",
        "5. Season with salt and pepper to taste.",
        "6. Serve immediately as a side dish or over rice for a complete meal."
      ]
    }
  ]
}
```
"""

################################
### Recipe ingredients gaps  ###
################################

system_ingredients_gaps_prompt_template="You are the top food editor in 'food, wine, & travel' magazine. You are an expert at carefully reviewing content and finding even the smallest mistakes. "

human_ingredients_gaps_prompt_template="""
Here is your task:
Below is a recipe description, and a list of ingredients. You need to identify whether the recipe requires any additional ingredients that are not listed in the provided ingredients list. 
If there are any missing ingredients, you need to identify and list them.
If there are no missing ingredients, don't say anything, just respond with an empty string.

<recipe>
{recipe}
</recipe>

<ingredients>
{ingredients_list}
</ingredients>
"""

#############################
### Recipe reflection     ###
#############################

system_recipe_reflection_prompt_template ="You are a nutritionist and health expert. You are an expert at analyzing recipes, calculating their calorie and nutrient values, and providing insights on how they can be made healthier."

human_recipe_reflection_prompt_template="""
Here is your task:

Analyze the recipe provided below  and provide the following information:
1) nutrition and calorie data for the recipe
2) expert advice on how the recipe can be made healthier

<recipe>
{recipe}
</recipe>
"""

#############################
### Recipe formatting     ###
#############################

human_final_formatting_prompt_template = """
Here is a recipe - its title, decsription, ingredients,  instructions, and some related expert nutritionist comments: 
<recipe>
{recipe}
</recipe>

<comments>
{missing_ingredients}
{reflection}
</comments>


Your task is to reformat the contents in a consistent, coherent, and nicely stylized GitHub markdown format.

It is important that you remember the following instructions:

1) Do not change the actual content - keep it exactly as it is, only apply formatting.
2) don't add or remove anything .
3) you can add relevant colorful icons to make it more visually appealing.
4) make the formatting concise and suitable for a chat interface
5) add tables where it saves space
6) do not add any prefix or postfix to the content, just the formatted recipe.

"""

human_final_formatting_prompt_template_old = """
Here are 3 recipes and some related expert nutritionist comments: 
<recipes>
{recipes}
</recipes>

Your task is to reformat the contents in a consistent, coherent, and nicely stylized GitHub markdown format.

It is important that you remember the following instructions:

1) Do not change the actual content - keep it exactly as it is, only apply formatting.
2) don't add or remove anything .
3) you can add relevant colorful icons to make it more visually appealing.
4) do not add any prefix or postfix to the content, just the formatted recipe."""



