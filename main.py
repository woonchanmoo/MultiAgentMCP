from langchain_mcp_adapters.client import MultiServerMCPClient
from agent import build_simple_agent
from langchain_core.messages import HumanMessage, AIMessageChunk, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
import asyncio
import warnings
from prompt import BASE_SYSTEM_PROMPT
from config.config import MCP_CONFIG, MCP_FILESYSTEM_DIR
from prompt_toolkit import prompt as pt_prompt

warnings.filterwarnings("ignore", category=UserWarning)

async def get_multiline_input(prompt: str) -> str:
    # \033[96m: Cyan색, \033[1m: Bold, \033[0m: Reset
    guide = "\033[96m\033[1m(전송: Esc 누른 후 Enter)\033[0m"
    print(f"{prompt} {guide}")
    # multiline=True일 때, 전송은 보통 'Esc' 누른 후 'Enter' 또는 'Meta+Enter'
    # 혹은 마우스로 클릭할 수 없는 환경이므로 안내 메시지가 필요합니다.
    user_input = await asyncio.to_thread(
        pt_prompt, 
        "> ", 
        multiline=True,
        prompt_continuation="  " # 줄바꿈 시 앞에 붙는 접두어
    )
    return user_input.strip()

async def stream_graph_response(input, graph, config={}):
    current_tool_args = ""
    last_index = -1  # 현재 출력 중인 도구의 인덱스를 추적
    
    yield "\033[1;32m[AI]:\033[0m "

    async for message_chunk, metadata in graph.astream(
        input=input, stream_mode="messages", config=config
    ):
        if metadata.get("langgraph_node") == "tools":
            continue

        if isinstance(message_chunk, AIMessageChunk):
            # 1. 도구 호출 시작/진행 중
            if message_chunk.tool_call_chunks:
                for chunk in message_chunk.tool_call_chunks:
                    idx = chunk.get("index")
                    
                    # 💡 핵심: 새로운 인덱스가 등장할 때만 이름을 출력합니다.
                    if idx != last_index:
                        if chunk.get("name"):
                            yield f"\n\n\033[94m🛠️  Executing Tool: {chunk['name']}\033[0m\n"
                            last_index = idx  # 출력한 도구의 인덱스를 저장
                    
                    # 인자(args)는 들어오는 대로 바로 출력 (회색)
                    if chunk.get("args"):
                        yield f"\033[90m{chunk['args']}\033[0m"
                        # 나중에 정렬된 출력을 원한다면 여기에 누적만 하세요.
                        current_tool_args += chunk["args"]
            
            # 2. 일반 텍스트 내용 출력
            elif message_chunk.content:
                yield message_chunk.content

            # 3. 마무리 (필요 시)
            if message_chunk.response_metadata.get("finish_reason") == "tool_calls":
                yield "\n"
                last_index = -1 # 초기화

async def fix_memory_if_broken(graph, config, error_type=None):
    state = await graph.aget_state(config)
    if not state.values or "messages" not in state.values:
        return False

    messages = state.values["messages"]
    if not messages: return False
    
    to_remove = []

    # 1. 특정 에러(Recursion)인 경우: HumanMessage까지 거슬러 올라가며 전체 삭제
    if error_type == "RecursionError":
        print("🔄 단계 초과: 관련 문맥을 모두 정리합니다.")
        for msg in reversed(messages):
            to_remove.append(RemoveMessage(id=msg.id))
            if isinstance(msg, HumanMessage): 
                break 

    # 2. 그 외 모든 에러 (도구 에러, API 에러, 일반 예외 등)
    else:
        # 가장 마지막 메시지부터 지우되, HumanMessage를 만날 때까지 지웁니다.
        # 이렇게 하면 '잘못된 도구 호출 AI 메시지'와 '원인이 된 사용자 질문'이 모두 삭제됩니다.
        for msg in reversed(messages):
            to_remove.append(RemoveMessage(id=msg.id))
            if isinstance(msg, HumanMessage):
                break

    if to_remove:
        await graph.aupdate_state(config, {"messages": to_remove}, as_node="Agent")
        return True
    return False

async def run_mcp_agent():

    # Memory Configuration
    memory = MemorySaver()
    config = {
        "configurable": {"thread_id": "thread_1"},
        "recursion_limit": 100} # 50번 이상의 도구 사용 가능

    # MCP Server Connection
    try:
        print("CONNECTING MCP SERVER...")
        client = MultiServerMCPClient(MCP_CONFIG)
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

    system_prompt = f"""
    Your name is Scout and you are an expert data scientist. You help customers manage their data science projects by leveraging the tools available to you. Your goal is to collaborate with the customer in incrementally building their analysis or data modeling project. Version control is a critical aspect of this project, so you must use the git tools to manage the project's version history and maintain a clean, easy to understand commit history.

    <filesystem>
    You have access to a set of tools that allow you to interact with the user's local filesystem. 
    You are only able to access files within the working directory `projects`. 
    The absolute path to this directory is: {MCP_FILESYSTEM_DIR}
    If you try to access a file outside of this directory, you will receive an error.
    Always use absolute paths when specifying files.
    </filesystem>

    {BASE_SYSTEM_PROMPT}

    <tools>
    {tools}
    </tools>

    Assist the customer in all aspects of their data science workflow.
    """
    
    # Agent Initialization
    mcp_agent = build_simple_agent(
        model="gpt-5-nano",
        system_prompt=system_prompt,
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
            print("\n🤖 ...", end="\n\n", flush=True)
            
            # 통합된 제너레이터 호출
            async for text in stream_graph_response(msg, mcp_agent, config):
                print(text, end="", flush=True)
            
            print("\n" + "="*50)
            
        except Exception as e:
                    error_str = str(e)
                    error_name = type(e).__name__
                    
                    # [수정] 어떤 에러가 발생하든 메모리 복구를 시도하도록 통합
                    print(f"\n\033[91m❌ 오류 발생 ({error_name}): 메모리를 정리하고 복구를 시도합니다...\033[0m")
                    
                    # 에러 종류에 따른 타입 지정
                    e_type = "RecursionError" if "Recursion limit" in error_str else "GeneralError"
                    
                    was_fixed = await fix_memory_if_broken(mcp_agent, config, error_type=e_type)
                    
                    if was_fixed:
                        print("\033[92m✅ 메모리 정리 완료. 다음 질문을 입력할 수 있습니다.\033[0m")
                        # 💡 continue를 하면 루프의 처음으로 돌아가 새로운 입력을 기다립니다.
                        continue
                    else:
                        print("\033[93m⚠️ 메모리를 정리할 내용이 없습니다. 계속 진행합니다.\033[0m")

if __name__ == "__main__":
    # 터미널 실행 시에는 아래 두 줄이 없어도 되지만, 노트북 환경 호환성을 위해 유지 가능
    import nest_asyncio
    nest_asyncio.apply()

    try:
        # 우리가 만든 비동기 에이전트 실행 루프
        asyncio.run(run_mcp_agent())
    except KeyboardInterrupt:
        print("\n강제 종료되었습니다.")