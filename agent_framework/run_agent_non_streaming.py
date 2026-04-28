"""Description:
    This is a simple example of how to use the Microsoft Agent Framework to create an agent

Azure resources needed:
    - Azure Foundry with a deployed 'gpt-5-nano' model

Config:
    - review that the model set here matches THE NAME for your deployed model in foundry
        example: I have deployed in foundry an instance for 'gpt-5.4-nano' which is called 'gpt-5.4-nano-1'
            so the code below has to be set to 'gpt-5.4-nano-1'
    - set FOUNDRY_POC_URL as system var. This is the URL you get when you deploy models in your Foundry
"""
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential

import asyncio
import os

async def _async_main():
    # example: "https://your-foundry-project.services.ai.azure.com/api/projects/your-project-name"
    foundry_url = os.environ["FOUNDRY_POC_URL"]

    credential = AzureCliCredential()
    client = FoundryChatClient(
        project_endpoint=foundry_url,
        model="gpt-5.4-nano-1",
        credential=credential
    )

    agent = client.as_agent(
        name="HelloAgent",
        instructions="You are a helpful assistant that provides information about Microsoft Agent Framework. Answer questions concisely and accurately."
    )

    # non-streaming way: get the complete response at once
    result = await agent.run("What's Microsoft Agent Framework?")
    print(f"Agent: {result.text}")

def main():
    asyncio.run(_async_main())

if __name__ == "__main__":
    main()