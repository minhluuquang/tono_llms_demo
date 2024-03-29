from langchain.chat_models.openai import ChatOpenAI
from langchain.schema.runnable import Runnable

from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessageGraph
from langchain_core.messages import HumanMessage

import json
from langchain_core.messages import ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool


@tool
def multiply(first_number: int, second_number: int):
    """Multiplies two numbers together."""
    return first_number * second_number


model = ChatOpenAI(temperature=0)
model_with_tools = model.bind(tools=[convert_to_openai_tool(multiply)])

graph = MessageGraph()


def invoke_model(state: list[BaseMessage]):
    return model_with_tools.invoke(state)


graph.add_node("oracle", invoke_model)


def invoke_tool(state: list[BaseMessage]):
    tool_calls = state[-1].additional_kwargs.get("tool_calls", [])
    multiply_call = None

    for tool_call in tool_calls:
        if tool_call.get("function").get("name") == "multiply":
            multiply_call = tool_call

    if multiply_call is None:
        raise Exception("No adder input found.")

    res = multiply.invoke(
        json.loads(multiply_call.get("function").get("arguments"))
    )

    return ToolMessage(
        tool_call_id=multiply_call.get("id"),
        content=res
    )


graph.add_node("multiply", invoke_tool)

graph.add_edge("multiply", END)

graph.set_entry_point("oracle")


def router(state: list[BaseMessage]):
    tool_calls = state[-1].additional_kwargs.get("tool_calls", [])
    if len(tool_calls):
        return "multiply"
    else:
        return "end"


graph.add_conditional_edges("oracle", router, {
    "multiply": "multiply",
    "end": END,
})

runnable = graph.compile()

# result = runnable.invoke(HumanMessage("What is your name?"))
# print(result)


def get_chain() -> Runnable:
    return runnable
