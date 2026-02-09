from constant import MCP_FILESYSTEM_DIR
from pathlib import Path

abs_path = str(Path(MCP_FILESYSTEM_DIR).resolve())

client_config = {
    "pubmed": {
        "command": "npx",
        "args": ["-y", "@cyanheads/pubmed-mcp-server"],
        "transport": "stdio"
    },
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            abs_path
        ],
        "transport": "stdio"
    },    
}