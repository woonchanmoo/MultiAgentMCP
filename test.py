from langchain_mcp_adapters.client import MultiServerMCPClient
from agent import build_simple_agent
from langchain_core.messages import HumanMessage, AIMessageChunk, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
import asyncio
import warnings
from prompt import PUBMED_PROMPT
from mcp_client_config import client_config
from typing import AsyncGenerator, Any
warnings.filterwarnings("ignore", category=UserWarning)

async def get_multiline_input(prompt: str) -> str:
    print(f"{prompt} (전송: 빈 줄에서 Enter 입력)")
    lines = []
    
    while True:
        # 각 줄을 받을 때는 strip()을 하지 않고 그대로 받음
        line = await asyncio.to_thread(input, "> ")
        
        # 사용자가 아무것도 치지 않고 엔터만 눌렀을 때 (진짜 빈 줄)
        if line == "": 
            break
        
        lines.append(line)
    
    # 모든 줄을 합친 후, 전체 메시지의 앞뒤 공백만 딱 한 번 제거
    return "\n".join(lines).strip()

async def stream_graph_response(input, graph, config={}):
    async for message_chunk, metadata in graph.astream(
        input=input, stream_mode="messages", config=config
    ):
        # 1. 노드 이름을 몰라도, Agent 노드에서 오는 것만 필터링 (가장 안전)
        if metadata.get("langgraph_node") == "tools":
            continue

        if isinstance(message_chunk, AIMessageChunk):
            # 도구 호출 완료 시 줄바꿈
            if message_chunk.response_metadata.get("finish_reason") == "tool_calls":
                yield "\n\n"

            if message_chunk.tool_call_chunks:
                tool_chunk = message_chunk.tool_call_chunks[0]
                
                # 2. tool_name과 args를 '누적'해서 출력하도록 수정
                if tool_chunk.get("name"):
                    yield f"\033[94m > Tool used: {tool_chunk['name']} \033[0m\n"
                if tool_chunk.get("args"):
                    yield f"\033[90m{tool_chunk['args']}\033[0m"  # 덮어쓰지 않고 이어서 보냄
            else:
                yield message_chunk.content

async def fix_memory_if_broken(graph, config):
    state = await graph.aget_state(config)
    if not state.values or "messages" not in state.values:
        return False

    messages = state.values["messages"]
    if len(messages) < 2: return False

    # 삭제할 메시지 리스트 준비
    to_remove = []
    
    # 1. 마지막 AI의 잘못된 도구 호출 삭제
    last_msg = messages[-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        to_remove.append(RemoveMessage(id=last_msg.id))
        
        # 2. [추가] 그 원인이 된 바로 직전의 Human 메시지도 함께 삭제
        prev_msg = messages[-2]
        if isinstance(prev_msg, HumanMessage):
            print(f"🧹 원인이 된 사용자 요청도 함께 정리합니다: '{prev_msg.content[:20]}...'")
            to_remove.append(RemoveMessage(id=prev_msg.id))

    if to_remove:
        await graph.aupdate_state(config, {"messages": to_remove})
        return True
    return False

async def run_pubmed():

    # Memory Configuration
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "thread_1"}}

    # MCP Server Connection
    try:
        print("CONNECTING MCP SERVER...")
        client = MultiServerMCPClient(client_config)
        # 이 단계에서 서버가 안 뜨면 무한 대기하거나 죽을 수 있습니다.
        tools = await asyncio.wait_for(client.get_tools(), timeout=120.0) 
    except asyncio.TimeoutError:
        print("❌ MCP 서버 연결 타임아웃!")
        return
    except Exception as e:
        print(f"❌ 연결 중 오류 발생: {e}")
        return

    if not tools:
        print("❌ MCP 도구를 로드하지 못했습니다.")
        return
    
    print(f"✅ Loaded {len(tools)} tools.")
    
    # Agent Initialization
    pubmed_agent = build_simple_agent(
        model="gpt-5-nano",
        system_prompt=PUBMED_PROMPT,
        tools=tools,
        checkpointer=memory
    )

    print("\n--- PubMed AI Agent Started ---")
    print("종료하려면 'exit' 또는 'quit'을 입력하세요.")

    # 2. 반복 루프 시작
    while True:
        user_input = await get_multiline_input("\n[User]: ")

        if user_input.lower() in ["exit", "quit"]:
            print("👋 프로그램을 종료합니다.")
            break

        if not user_input:
            continue

        msg = {
            "messages": [HumanMessage(content=user_input)]
        }

        try:
            print("🤖 ...", end="\n", flush=True)
            
            # 통합된 제너레이터 호출
            async for text in stream_graph_response(msg, pubmed_agent, config):
                print(text, end="", flush=True)
            
            print("\n" + "="*50)
            
        except Exception as e:
            # 변수를 미리 초기화해둡니다.
            was_fixed = False
            
            # 1. 도구 호출 메시지와 결과가 짝이 안 맞을 때 (400 에러 등)
            if "tool_calls" in str(e) or "ToolException" in str(type(e).__name__):
                print(f"\n\033[93m🛠️  오류 감지({type(e).__name__}): 메모리 복구 시도...\033[0m")
                was_fixed = await fix_memory_if_broken(pubmed_agent, config)
                
                if was_fixed:
                    print("\033[92m✅ 복구 완료! 이전 에러 메시지가 삭제되었습니다.\033[0m")
                    print("\033[94m💡 팁: 다른 명령을 입력해 주세요.\033[0m")
                    # 💡 [핵심] 여기서 다시 시도하지 않고 'continue'를 통해 루프의 처음(input 단계)으로 점프!
                    continue
                else:
                    print("\n❌ 자동 복구가 불가능한 상태입니다.")
            else:
                # 보안 위반 등 도구 자체의 에러인 경우
                print(f"\n❌ 실행 오류 발생: {e}")

if __name__ == "__main__":
    # 터미널 실행 시에는 아래 두 줄이 없어도 되지만, 노트북 환경 호환성을 위해 유지 가능
    import nest_asyncio
    nest_asyncio.apply()

    try:
        # 우리가 만든 비동기 에이전트 실행 루프
        asyncio.run(run_pubmed())
    except KeyboardInterrupt:
        print("\n강제 종료되었습니다.")