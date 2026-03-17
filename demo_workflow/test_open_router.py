import os
from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.environ["OPEN_ROUTER_DEV_KEY"],
)

completion = client.chat.completions.create(
  model="anthropic/claude-haiku-4.5",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)

print(completion.choices[0].message.content)
