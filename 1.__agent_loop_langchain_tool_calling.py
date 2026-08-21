from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage,SystemMessage,ToolMessage
from langsmith import traceable
from typing import Literal


MAX_ITERATIONS = 10
MODEL = "gpt-5-nano"

load_dotenv()

#tools

@tool
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


@tool
def get_discount_price(price:float , discount_tier:str) -> float:
    """Apply a discount tier to a price and return the final price.
        Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount= discount_percentages.get(discount_tier , 0)
    return round(price * (1 - discount/100),2)


# Agent-loop
@traceable(name = "Langchain agent loop")
def run_agent(question:str):
    tools = [get_product_price , get_discount_price]
    print(tools[0])
    print(get_product_price.args_schema.model_json_schema())
    tools_dict = {t.name:t for t in tools}
    
    llm = init_chat_model(f"openai:{MODEL}",temperature = 0.5)
    llm_with_tools = llm.bind_tools(tools)
    
    messages = [
        SystemMessage(
            content=         """
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
                
               
        ),
        HumanMessage(content=question)
    ]
    for iterations in range(1,MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iterations} ---")

        ai_message= llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")
        
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        
        observation = tool_to_use.invoke(tool_args)
        print(f"  [Tool Result] {observation}")
        messages.append(ai_message)
        messages.append(ToolMessage(
            content=str(observation),tool_call_id=tool_call_id
        ))
        # print(messages)
    print("ERROR: Max iterations reached without a final answer")
    return None
        
            

if __name__ == "__main__":
    print("Hello langchain bind tools")
    result = run_agent("What is the price of the laptop after applying gold discount ?")
    print(result)



