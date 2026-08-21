from dotenv import load_dotenv

import ollama
from langsmith import traceable
from typing import Literal


MAX_ITERATIONS = 10
MODEL = "qwen3.5:0.8b"

load_dotenv()

#tools

@traceable(name = "tool")
def get_product_price( product: Literal["laptop", "headphones", "keyboard"]) -> float:
    """ Look up the price of the product in the catalog 
      Before calling this tool, map the user's requested product
    to exactly one of these categories:
    - laptop
    - headphones
    - keyboard
    Examples:
    MacBook Pro -> laptop
    Dell XPS -> laptop
    AirPods -> headphones
    Sony WH-1000XM5 -> headphones
    Logitech MX Keys -> keyboard

    Never pass the user's original product name.
    
    """
    
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product,0)


@traceable(name = "tool")
def get_discount_price(price:float , discount_tier:str) -> float:
    """Apply a discount tier to a price and return the final price.
        Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount= discount_percentages.get(discount_tier , 0)
    return round(price * (1 - discount/100),2)



tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_discount_price",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


# Agent-loop
@traceable(name="Ollama agent loop")
def run_agent(question:str):
    tools_dict = {
        "get_product_price": get_product_price,
        "get_discount_price": get_discount_price,
    }

    
    messages = [
        {
            "role":"system",
            "content":(
                """
    You are a shopping assistant that works with a category-based product catalog.

    The catalog contains EXACTLY these three categories:
    - laptop
    - headphones
    - keyboard

    IMPORTANT:
    The catalog stores prices by CATEGORY, not by specific model.
    Therefore, a category such as "laptop" is sufficient information.
    NEVER ask the user for a specific model if they already said laptop,
    headphones, or keyboard.

    PRODUCT MAPPING RULES:

    1. If the user directly mentions one of the catalog categories,
    use that category immediately.

    Examples:
    "price of laptop" -> product="laptop"
    "price of headphones" -> product="headphones"
    "keyboard with gold discount" -> product="keyboard"

    2. If the user mentions a specific product/model, classify it into
    the closest catalog category.

    Examples:
    "MacBook Pro M3" -> product="laptop"
    "Dell XPS" -> product="laptop"
    "AirPods Pro" -> product="headphones"
    "Sony WH-1000XM5" -> product="headphones"
    "Logitech MX Keys" -> product="keyboard"

    3. When calling get_product_price, the product argument MUST be
    exactly one of:
    - laptop
    - headphones
    - keyboard

    4. NEVER pass a specific model name such as "MacBook Pro" to
    get_product_price.

    5. NEVER ask for a specific model when the user has already provided
    enough information to determine one of the three categories.

    PRICE AND DISCOUNT RULES:

    6. NEVER guess a price. Always call get_product_price.

    7. If a discount is requested, first call get_product_price.

    8. After receiving the price, call get_discount_price with the exact
    price returned by get_product_price.

    9. NEVER calculate the discount yourself.

    10. If no discount tier is specified when a discount is requested,
        ask the user which tier to use.
    """
            )
        },
            {"role": "user", "content": question}, 
        ]
    
    for iterations in range(1,MAX_ITERATIONS + 1):
            print(f"\n--- Iteration {iterations} ---")

            response = ollama_chat_traced(messages=messages)
            ai_message = response.message
            
            tool_calls = ai_message.tool_calls
            if not tool_calls:
                print(f"\nFinal Answer: {ai_message.content}")
                return ai_message.content
            
            tool_call = tool_calls[0]
        # Difference 6: Attribute access (.function.name) instead of dict access (.get("name"))
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            print(f"  [Tool Selected] {tool_name} with args: {tool_args}")
            
            tool_to_use = tools_dict.get(tool_name)
            if tool_to_use is None:
                raise ValueError(f"Tool '{tool_name}' not found")
            
            observation = tool_to_use(**tool_args)
            print(f"  [Tool Result] {observation}")
            messages.append(ai_message)
            messages.append({
                "role": "tool",
                "content": str(observation),
            })
        # print(messages)
    print("ERROR: Max iterations reached without a final answer")
    return None
        
            

if __name__ == "__main__":
    print("Hello langchain bind tools")
    result = run_agent("What is the price of the laptop after applying gold discount ?")
    print(result)



