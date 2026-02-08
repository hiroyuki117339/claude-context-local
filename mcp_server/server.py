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
from mcp_server.file_watcher import FileWatcher


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
    parser.add_argument(
        "--watch",
        action="store_true",
        default=None,
        help="Watch auto-indexed directories for changes (overrides config)"
    )
    parser.add_argument(
        "--no-watch",
        action="store_true",
        default=False,
        help="Disable file watching even if config has watch: true"
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Debounce interval in seconds for file watcher (default: 5.0)"
    )

    args = parser.parse_args()

    # Resolve --watch / --no-watch into a single override
    watch_override = None
    if args.no_watch:
        watch_override = False
    elif args.watch:
        watch_override = True

    # Create server
    server = CodeSearchServer()

    # Run startup indexing (never blocks server start)
    config, indexed_dirs = run_startup_indexing(
        server,
        cli_directories=args.auto_index,
        no_auto_index=args.no_auto_index,
        watch_override=watch_override,
        debounce_override=args.debounce,
    )

    # Start file watcher if enabled and there are directories to watch
    watcher = None
    if config.watch and indexed_dirs:
        watcher = FileWatcher(server, debounce_seconds=config.debounce_seconds)
        for d in indexed_dirs:
            watcher.add_directory(d.path, d.name, d.incremental)
        watcher.start()

    # Create and run MCP server
    mcp_server = CodeSearchMCP(server)
    try:
        mcp_server.run(transport=args.transport, host=args.host, port=args.port)
    finally:
        if watcher:
            watcher.stop()


if __name__ == "__main__":
    main()
