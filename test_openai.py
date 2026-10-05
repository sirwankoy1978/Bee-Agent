"""Quick smoke test: verify the OpenAI API key works and the agent responds."""
import asyncio, os
from dotenv import load_dotenv
load_dotenv()

from agno.agent import Agent
from agno.models.openai import OpenAIChat

agent = Agent(
    name="test",
    model=OpenAIChat(id=os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini"), api_key=os.getenv("OPENAI_API_KEY")),
    markdown=True,
)

async def main():
    print("Contacting OpenAI...")
    resp = await agent.arun("Say exactly: 'OpenAI connection works perfectly.'", stream=False)
    print("Model reply:", resp.content)

if __name__ == "__main__":
    asyncio.run(main())
