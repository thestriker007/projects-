import asyncio
import os
from openai import AsyncOpenAI
async def main():
    client = AsyncOpenAI(api_key=os.environ["HF_TOKEN"], base_url="https://router.huggingface.co/v1/")
    resp = await client.chat.completions.create(
        model="meta-llama/Llama-3.2-1B-Instruct",
        messages=[{"role": "user", "content": "hi"}],
    )
    print(resp)
asyncio.run(main())
