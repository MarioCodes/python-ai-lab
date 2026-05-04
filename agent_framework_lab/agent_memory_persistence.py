from agent_framework import Agent, AgentSession, SessionContext, ContextProvider
from agent_framework.foundry import FoundryChatClient
from azure.identity import AzureCliCredential

from typing import Any
import asyncio
import os

"""Description:
    Example of how to use the Microsoft Agent Framework to create an agent that can remember information about the user across interactions

Azure resources needed:
    - Azure Foundry with a deployed 'gpt-5-nano' model

Config:
    - review that the model set here matches THE NAME for your deployed model in foundry
        example: I have deployed in foundry an instance for 'gpt-5.4-nano' which is called 'gpt-5.4-nano-1'
            so the code below has to be set to 'gpt-5.4-nano-1'
    - set FOUNDRY_URL as system var. This is the URL you get when you deploy models in your Foundry
"""

class UserMemoryProvider(ContextProvider):
    """A context provider that remembers user info in session state"""

    DEFAULT_SOURCE_ID = "user_memory"

    def __init__(self):
        super().__init__(self.DEFAULT_SOURCE_ID)

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession | None,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Inject personalization instructions based on stored user info"""
        user_name = state.get("user_name")
        if user_name:
            context.extend_instructions(
                self.source_id,
                f"The user's name is {user_name}. Always address them by name'"
            )
        else:
            context.extend_instructions(
                self.source_id,
                "We don't know the user's name yet. Ask for it politely"
            )

    async def after_run(
        self,
        *,
        agent: Any,
        session: AgentSession | None,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """Extract and store user info in session state after each call"""
        for msg in context.input_messages:
            text = msg.text if hasattr(msg, "text") else ""
            if isinstance(text, str) and "my name is" in text.lower():
                state["user_name"] = text.lower().split("my name is")[-1].strip().split()[0].capitalize()

async def _async_main():
    # <create_agent>
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
        instructions="You are a helpful assistant that provides information about Microsoft Agent Framework. Answer questions concisely and accurately.",
        context_providers=[UserMemoryProvider()],
    )
    # </create_agent>

    # <run_with_memory>
    session = agent.create_session()

    # the provider doesn't know the user yet - it will ask for the name
    result = await agent.run("Hello! What's the square root of 16?", session=session)
    print(f"Agent: {result}\n")

    # now we provide the name and the provider stores it in session state
    result = await agent.run("By the way, my name is Mario", session=session)
    print(f"Agent: {result}\n")

    # subsequent calls are personalized. name persist via session state
    result = await agent.run("What's 8 - 20?", session=session)
    print(f"Agent: {result}\n")

    # inspect session state to see what the provider stored
    provider_state = session.state.get("user_memory", {})
    print(f"[Session State] Stored user name: {provider_state.get('user_name')}")
    # </run_with_memory>

def main():
    asyncio.run(_async_main())

if __name__ == "__main__":
    main()