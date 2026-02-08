from langchain_mcp_adapters.client import MultiServerMCPClient
from agent import build_simple_agent
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
import asyncio
import warnings
from prompt import PUBMED_PROMPT
from mcp_client_config import client_config
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
            
            # stream_mode="messages"는 모델이 뱉는 토큰 하나하나를 실시간으로 가져옵니다.
            async for chunk, metadata in pubmed_agent.astream(
                msg, 
                config=config, 
                stream_mode="messages"
            ):
                # 1. AI 메시지이고, 답변 내용(content)이 있는 경우에만 출력
                if isinstance(chunk, AIMessage) and chunk.content:
                    # 실시간으로 한 글자씩/한 문장씩 출력 (줄바꿈 없이)
                    print(chunk.content, end="", flush=True)
            
            print("\n" + "="*50)
            
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(run_pubmed())
    except KeyboardInterrupt:
        print("\n강제 종료되었습니다.")