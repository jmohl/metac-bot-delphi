import litellm

litellm.register_model({
  "openrouter/openai/gpt-5": {
    "input_cost_per_token": 0.00000125, "output_cost_per_token": 0.00001
  },
  "openrouter/openai/o4-mini": {
    "input_cost_per_token": 0.00000110, "output_cost_per_token": 0.00000440
  },
})


print("\n" + "="*50)
print("Default LiteLLM Models (containing 'openai' or 'openrouter'):")
print("="*50)

supported_model_names = litellm.model_cost.keys()

# Filter for models containing "openai" or "openrouter"
filtered_models = [
    model for model in supported_model_names 
    if "openai" in model.lower() or "openrouter" in model.lower()
]

for model in filtered_models:
    print(f"  {model}")

# from openai import OpenAI
# import os

# # Initialize OpenAI client
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Get available OpenAI models
# print("\n" + "="*60)
# print("Available OpenAI Models:")
# print("="*60)

# try:
#     models = client.models.list()
#     for model in models.data:
#         print(f"  {model.id}")
# except Exception as e:
#     print(f"Error fetching OpenAI models: {e}")
#     print("Note: You may need to set your OPENAI_API_KEY environment variable")