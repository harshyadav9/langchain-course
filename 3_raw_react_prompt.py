import re
import inspect
from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    price = float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)



tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers (e.g., @traceable adds *, config=None)
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

print(tool_descriptions)
react_prompt = f"""
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

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


