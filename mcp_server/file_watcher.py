"""File watcher for automatic re-indexing of auto-indexed directories."""

import logging
import threading
from pathlib import Path
from typing import Callable, Dict, Optional, Set

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

# Extensions that trigger re-indexing (matches LANGUAGE_MAP keys)
WATCHED_EXTENSIONS: Set[str] = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.svelte',
    '.go', '.rs', '.java', '.md',
    '.c', '.cpp', '.cc', '.cxx', '.c++', '.cs',
}

# Directories to ignore (matches MultiLanguageChunker.DEFAULT_IGNORED_DIRS)
IGNORED_DIRS: Set[str] = {
    '__pycache__', '.git', '.hg', '.svn',
    '.venv', 'venv', 'env', '.env', '.direnv',
    'node_modules', '.pnpm-store', '.yarn',
    '.pytest_cache', '.mypy_cache', '.ruff_cache', '.pytype', '.ipynb_checkpoints',
    'build', 'dist', 'out', 'public',
    '.next', '.nuxt', '.svelte-kit', '.angular', '.astro', '.vite',
    '.cache', '.parcel-cache', '.turbo',
    'coverage', '.coverage', '.nyc_output',
    '.gradle', '.idea', '.vscode', '.docusaurus', '.vercel',
    '.serverless', '.terraform', '.mvn', '.tox',
    'target', 'bin', 'obj',
}


def _is_in_ignored_dir(path: str) -> bool:
    """Check if the path is inside an ignored directory."""
    parts = Path(path).parts
    return any(part in IGNORED_DIRS for part in parts)


def _is_watched_file(path: str) -> bool:
    """Check if the file has a watched extension."""
    return Path(path).suffix.lower() in WATCHED_EXTENSIONS


class _DebouncedHandler(FileSystemEventHandler):
    """Watchdog event handler that debounces changes and fires a callback."""

    def __init__(self, callback: Callable[[str], None], debounce_seconds: float):
        super().__init__()
        self._callback = callback
        self._debounce_seconds = debounce_seconds
        self._timers: Dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def _schedule(self, watch_path: str) -> None:
        """Schedule a debounced callback for the given watched directory."""
        with self._lock:
            existing = self._timers.get(watch_path)
            if existing is not None:
                existing.cancel()

            timer = threading.Timer(
                self._debounce_seconds,
                self._fire,
                args=(watch_path,),
            )
            timer.daemon = True
            self._timers[watch_path] = timer
            timer.start()

    def _fire(self, watch_path: str) -> None:
        """Execute the callback after debounce period."""
        with self._lock:
            self._timers.pop(watch_path, None)
        try:
            self._callback(watch_path)
        except Exception:
            logger.exception(f"Error in file watcher callback for {watch_path}")

    def _handle_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src = event.src_path
        if _is_in_ignored_dir(src) or not _is_watched_file(src):
            return
        # Find which registered watch root this event belongs to
        # The Observer passes the watch path via the event, but we just
        # need to trigger a re-index – the watch_path is resolved in FileWatcher.
        # We store the root mapping on the handler instance.
        for root in self._watch_roots:
            if src.startswith(root):
                logger.debug(f"File changed: {src} (root: {root})")
                self._schedule(root)
                return

    def on_created(self, event: FileSystemEvent) -> None:
        self._handle_event(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._handle_event(event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._handle_event(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        self._handle_event(event)

    def cancel_all(self) -> None:
        """Cancel all pending timers."""
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()


class FileWatcher:
    """Watches multiple directories and triggers incremental re-indexing on changes."""

    def __init__(self, server, debounce_seconds: float = 5.0):
        """
        Args:
            server: CodeSearchServer instance.
            debounce_seconds: Seconds to wait after last change before re-indexing.
        """
        self._server = server
        self._debounce_seconds = debounce_seconds
        self._index_lock = threading.Lock()
        self._observer: Optional[Observer] = None
        # Mapping: resolved directory path -> DirectoryInfo
        self._directories: Dict[str, "_DirInfo"] = {}
        self._handler = _DebouncedHandler(
            callback=self._on_changes_detected,
            debounce_seconds=debounce_seconds,
        )
        self._handler._watch_roots = []

    def add_directory(self, path: str, project_name: Optional[str] = None, incremental: bool = True) -> None:
        """Register a directory for watching.

        Args:
            path: Directory path to watch.
            project_name: Optional project name for indexing.
            incremental: Whether to use incremental indexing.
        """
        resolved = str(Path(path).resolve())
        self._directories[resolved] = _DirInfo(
            path=resolved,
            project_name=project_name,
            incremental=incremental,
        )
        self._handler._watch_roots.append(resolved)
        logger.info(f"File watcher: registered directory {resolved}")

    def start(self) -> None:
        """Start watching all registered directories."""
        if not self._directories:
            logger.warning("File watcher: no directories registered, not starting")
            return

        self._observer = Observer()
        for resolved_path in self._directories:
            if Path(resolved_path).exists():
                self._observer.schedule(self._handler, resolved_path, recursive=True)
                logger.info(f"File watcher: watching {resolved_path}")
            else:
                logger.warning(f"File watcher: directory does not exist, skipping: {resolved_path}")

        self._observer.daemon = True
        self._observer.start()
        logger.info(
            f"File watcher started: {len(self._directories)} directories, "
            f"debounce={self._debounce_seconds}s"
        )

    def stop(self) -> None:
        """Stop watching and clean up."""
        self._handler.cancel_all()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        logger.info("File watcher stopped")

    def _on_changes_detected(self, watch_path: str) -> None:
        """Called (from timer thread) when changes settle for a directory."""
        dir_info = self._directories.get(watch_path)
        if dir_info is None:
            return

        logger.info(f"File watcher: changes detected in {watch_path}, re-indexing...")

        with self._index_lock:
            try:
                result = self._server.index_directory(
                    directory_path=dir_info.path,
                    project_name=dir_info.project_name,
                    incremental=dir_info.incremental,
                )
                logger.info(f"File watcher: re-indexed {Path(watch_path).name}: {result}")
            except Exception:
                logger.exception(f"File watcher: re-index failed for {watch_path}")


class _DirInfo:
    """Simple container for watched directory metadata."""
    __slots__ = ('path', 'project_name', 'incremental')

    def __init__(self, path: str, project_name: Optional[str], incremental: bool):
        self.path = path
        self.project_name = project_name
        self.incremental = incremental
