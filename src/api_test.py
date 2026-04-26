import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI


env_path = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT")

client = AzureOpenAI(
    api_key=api_key,
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

try:
    response = client.chat.completions.create(
        model=deployment,  # deployment name
        messages=[
            {"role": "user", "content": "Say hello in one line"}
        ],
        max_tokens=50
    )

    print("✅ API is working!\n")
    print("Response:")
    print(response.choices[0].message.content)

except Exception as e:
    print("❌ API test failed")
    print("Error:", str(e))