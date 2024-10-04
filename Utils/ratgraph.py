from langgraph.graph import Graph, END
from langgraph.graph.state import StateGraph
from langgraph.graph import MessagesState

# Define the state with messages and other necessary keys
class RecipeState(MessagesState):
    images: list
    text_input: str
    ingredients: list
    preferences: dict
    recipes: list
    selected_recipe: dict

# Define the graph
graph = StateGraph(RecipeState)

# Add nodes
graph.add_node("image_input", lambda state: {"images": capture_images()})
graph.add_node("text_input", lambda state: {"text_input": capture_text()})
graph.add_node("ingredient_recognition", lambda state: {"ingredients": process_image(state["images"])})
graph.add_node("text_analysis", lambda state: {"preferences": analyze_text(state["text_input"])})
graph.add_node("recipe_search", lambda state: {"recipes": search_recipes(state["ingredients"], state["preferences"])})
graph.add_node("recipe_display", lambda state: display_recipes(state["recipes"]))
graph.add_node("user_input", lambda state: {"messages": [("user", get_user_input())]})
graph.add_node("recipe_selection", lambda state: {"selected_recipe": select_recipe(state["recipes"])})
graph.add_node("full_recipe_display", lambda state: display_full_recipe(state["selected_recipe"]))
graph.add_node("feedback_loop", lambda state: {"messages": [("user", get_user_input())]})

# Define edges
graph.set_entry_point("image_input")
graph.add_edge("image_input", "ingredient_recognition")
graph.add_edge("text_input", "text_analysis")
graph.add_edge("ingredient_recognition", "recipe_search")
graph.add_edge("text_analysis", "recipe_search")
graph.add_edge("recipe_search", "recipe_display")
graph.add_edge("recipe_display", "user_input")
graph.add_edge("user_input", "recipe_selection")
graph.add_edge("recipe_selection", "full_recipe_display")
graph.add_edge("full_recipe_display", "feedback_loop")
graph.add_edge("feedback_loop", "recipe_display")  # Loop back to recipe display
graph.add_edge("feedback_loop", END)  # Option to end interaction

# Compile the graph
app = graph.compile()
