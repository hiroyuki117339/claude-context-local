"""Startup indexer - loads config and indexes directories at server startup."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from common_utils import get_config_path

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_TEMPLATE = """\
# Claude Code Search - auto index configuration
# Directories listed here will be indexed automatically when the MCP server starts.
#
# auto_index:
#   directories:
#     - path: "/path/to/project1"
#     - path: "/path/to/project2"
#       name: "my-project"     # optional project name
#       incremental: true       # default: true (only reindex changed files)
"""


@dataclass
class DirectoryConfig:
    """Configuration for a single directory to auto-index."""
    path: str
    name: Optional[str] = None
    incremental: bool = True


@dataclass
class AutoIndexConfig:
    """Parsed auto-index configuration."""
    directories: List[DirectoryConfig] = field(default_factory=list)


def load_config(config_path: Optional[Path] = None) -> AutoIndexConfig:
    """Load and validate the auto-index config from YAML.

    Returns an empty config if the file doesn't exist or has no auto_index section.
    """
    if config_path is None:
        config_path = get_config_path()

    if not config_path.exists():
        return AutoIndexConfig()

    try:
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to read config file {config_path}: {e}")
        return AutoIndexConfig()

    if not isinstance(data, dict):
        return AutoIndexConfig()

    auto_index = data.get("auto_index")
    if not isinstance(auto_index, dict):
        return AutoIndexConfig()

    raw_dirs = auto_index.get("directories")
    if not isinstance(raw_dirs, list):
        return AutoIndexConfig()

    directories: List[DirectoryConfig] = []
    for entry in raw_dirs:
        if isinstance(entry, dict) and "path" in entry:
            directories.append(DirectoryConfig(
                path=entry["path"],
                name=entry.get("name"),
                incremental=entry.get("incremental", True),
            ))
        elif isinstance(entry, str):
            directories.append(DirectoryConfig(path=entry))
        else:
            logger.warning(f"Skipping invalid directory entry in config: {entry}")

    return AutoIndexConfig(directories=directories)


def generate_default_config(config_path: Optional[Path] = None) -> None:
    """Write a default (commented-out) config template if no config exists."""
    if config_path is None:
        config_path = get_config_path()

    if config_path.exists():
        return

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(DEFAULT_CONFIG_TEMPLATE)
        logger.info(f"Generated default config template at {config_path}")
    except Exception as e:
        logger.warning(f"Failed to generate default config: {e}")


def run_startup_indexing(
    server,
    cli_directories: Optional[List[str]] = None,
    no_auto_index: bool = False,
) -> None:
    """Run startup indexing based on config file and/or CLI arguments.

    Args:
        server: CodeSearchServer instance.
        cli_directories: Directories specified via --auto-index CLI arg.
        no_auto_index: If True, skip all auto-indexing.
    """
    if no_auto_index:
        logger.info("Auto-indexing disabled via --no-auto-index")
        return

    # Determine directories to index
    if cli_directories:
        # CLI args take precedence over config file
        directories = [DirectoryConfig(path=d) for d in cli_directories]
        logger.info(f"Using {len(directories)} directories from CLI arguments")
    else:
        config = load_config()
        directories = config.directories
        if not directories:
            generate_default_config()
            return
        logger.info(f"Using {len(directories)} directories from config file")

    # Index each directory sequentially
    last_indexed = None
    for dir_config in directories:
        dir_path = Path(dir_config.path).resolve()

        if not dir_path.exists():
            logger.warning(f"Auto-index: directory does not exist, skipping: {dir_path}")
            continue

        if not dir_path.is_dir():
            logger.warning(f"Auto-index: path is not a directory, skipping: {dir_path}")
            continue

        try:
            logger.info(f"Auto-indexing: {dir_path} (incremental={dir_config.incremental})")
            result = server.index_directory(
                directory_path=str(dir_path),
                project_name=dir_config.name,
                incremental=dir_config.incremental,
            )
            logger.info(f"Auto-indexed {dir_path.name}: {result}")
            last_indexed = str(dir_path)
        except Exception as e:
            logger.error(f"Auto-index failed for {dir_path}: {e}")

    # Set the last successfully indexed project as current
    if last_indexed:
        server._current_project = last_indexed
        server._index_manager = None
        server._searcher = None
        logger.info(f"Current project set to: {last_indexed}")
