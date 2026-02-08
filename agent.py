from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict, Any, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def build_simple_agent(model: str, system_prompt: str, tools: Sequence[Any], checkpointer = None):
    llm = init_chat_model(model=model)
    llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: AgentState) -> AgentState:

        response = await llm_with_tools.ainvoke(state["messages"])
        # 2. 도구 사용 여부 감지 및 출력
        if response.tool_calls:
            for tool_call in response.tool_calls:
                # 도구 이름과 인자값을 출력
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                print(f"\033[94m > Tool used: {tool_name}\033[0m")
                print(f"   Arguments: {tool_args}\n")

        return {"messages": [response]}
    
    workflow = StateGraph(AgentState)

    workflow.add_node("Agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "Agent")
    workflow.add_conditional_edges(
        "Agent",
        tools_condition,
        {
            "tools": "tools",
            "__end__": END
        }
    )
    workflow.add_edge("tools", "Agent")

    return workflow.compile(checkpointer=checkpointer)
