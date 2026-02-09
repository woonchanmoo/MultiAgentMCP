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
        # 도구 호출 여부 판단을 위해 호출 (스트리밍은 외부 제너레이터가 처리)
        response = await llm_with_tools.ainvoke(state["messages"])
    
        # 여기서 직접 print하지 않고 response만 반환합니다.
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
