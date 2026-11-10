"""OpenAI tool calling with simulated execution and one tool-call round.

Run: python3 examples/tool_calling.py
Requires openai and OPENAI_API_KEY. Execution makes real OpenAI requests and
may incur costs, but tool results are hardcoded demo data. Tools execute
sequentially. Before connecting real systems, validate model-generated
arguments and enforce authorization independently of the model.
"""

import asyncio
import json

from openai import AsyncOpenAI


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the internal knowledge base for relevant documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_data",
            "description": "Retrieve user account information by user ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The user's ID"},
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fields to retrieve",
                    },
                },
                "required": ["user_id"],
            },
        },
    },
]


async def execute_tool(name: str, args: dict) -> str:
    """Return simulated JSON results; top_k and fields are not applied."""
    if name == "search_knowledge_base":
        return json.dumps([
            {"title": "RAG Guide", "snippet": f"Results for: {args['query']}"},
            {"title": "Best Practices", "snippet": "Use chunking..."},
        ])
    elif name == "get_user_data":
        return json.dumps({
            "user_id": args["user_id"],
            "name": "Ian Saura",
            "plan": "pro",
            "usage_tokens": 150_000,
        })
    return json.dumps({"error": f"Unknown tool: {name}"})


async def chat_with_tools(user_message: str) -> str:
    """Call the model, execute one round of tools, then request a final answer."""
    messages = [
        {"role": "system", "content": "You help users with their account and knowledge base."},
        {"role": "user", "content": user_message},
    ]

    async with AsyncOpenAI() as client:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        choice = response.choices[0]
        if choice.finish_reason != "tool_calls":
            return choice.message.content

        messages.append(choice.message)

        for tool_call in choice.message.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = await execute_tool(tool_call.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        final = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
    return final.choices[0].message.content


async def main():
    answer = await chat_with_tools(
        "Look up user U-1234 and search our docs about their plan limits."
    )
    print(f"Answer: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
