"""FastMCP server for Claude Code integration - main entry point."""
import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("mcp").setLevel(logging.DEBUG)
logging.getLogger("fastmcp").setLevel(logging.DEBUG)

from mcp_server.code_search_server import CodeSearchServer
from mcp_server.code_search_mcp import CodeSearchMCP
from mcp_server.startup_indexer import run_startup_indexing


def main():
    """Main entry point for the server."""
    import argparse

    parser = argparse.ArgumentParser(description="Code Search MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="Transport protocol to use (default: stdio)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for HTTP transport (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport (default: 8000)"
    )
    parser.add_argument(
        "--auto-index",
        nargs="+",
        metavar="DIR",
        help="Directories to auto-index at startup (overrides config file)"
    )
    parser.add_argument(
        "--no-auto-index",
        action="store_true",
        default=False,
        help="Disable auto-indexing even if config file exists"
    )

    args = parser.parse_args()

    # Create server
    server = CodeSearchServer()

    # Run startup indexing (never blocks server start)
    run_startup_indexing(
        server,
        cli_directories=args.auto_index,
        no_auto_index=args.no_auto_index,
    )

    # Create and run MCP server
    mcp_server = CodeSearchMCP(server)
    mcp_server.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
