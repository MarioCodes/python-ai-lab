from agent_framework import Agent
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential

import asyncio
import os

"""Description:
    This is an example of how to create an Agent with multi-turn conversation capabilities.

Azure resources needed:
    - Azure Foundry with a deployed 'gpt-5-nano' model

Config:
    - review that the model set here matches THE NAME for your deployed model in foundry
        example: I have deployed in foundry an instance for 'gpt-5.4-nano' which is called 'gpt-5.4-nano-1'
            so the code below has to be set to 'gpt-5.4-nano-1'
    - set FOUNDRY_URL as system var. This is the URL you get when you deploy models in your Foundry
"""

async def _async_main():
    # example: "https://your-foundry-project.services.ai.azure.com/api/projects/your-project-name"
    foundry_url = os.environ["FOUNDRY_URL"]

    client = FoundryChatClient(
        project_endpoint=foundry_url,
        model="gpt-5.4-nano-1",
        credential=AzureCliCredential()
    )

    agent = Agent(
        client=client,
        name="HelloAgent",
        instructions="You are a helpful assistant. Keep your answers brief"
    )

    # create a session to maintain conversation history
    session = agent.create_session()

    # first turn
    result = await agent.run("My name is Mario and I love hiking", session=session)
    print(f"Agent turn 1: {result.text}\n")

    # second turn - agent should remember the information from the first turn
    result = await agent.run("What do you remember about me?", session=session)
    print(f"Agent turn 2: {result.text}")

def main():
    asyncio.run(_async_main())

if __name__ == "__main__":
    main()