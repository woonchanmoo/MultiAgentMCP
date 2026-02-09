from dotenv import load_dotenv
import os

load_dotenv()
MCP_FILESYSTEM_DIR = os.getenv("MCP_FILESYSTEM_DIR")

if __name__ == "__main__":
    print(MCP_FILESYSTEM_DIR)