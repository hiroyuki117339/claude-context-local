"""Incremental indexing using Merkle tree change detection."""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from merkle.change_detector import ChangeDetector, FileChanges
from merkle.merkle_dag import MerkleDAG
from merkle.snapshot_manager import SnapshotManager
from chunking.multi_language_chunker import MultiLanguageChunker
from embeddings.embedder import CodeEmbedder
from search.indexer import CodeIndexManager as Indexer

logger = logging.getLogger(__name__)


def _mem_mb() -> str:
    """Get current process RSS in MB (best-effort)."""
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS returns bytes, Linux returns KB
        if os.uname().sysname == "Darwin":
            return f"{rss / 1024 / 1024:.0f}MB"
        return f"{rss / 1024:.0f}MB"
    except Exception:
        return "?MB"


@dataclass
class IncrementalIndexResult:
    """Result of incremental indexing operation."""
    
    files_added: int
    files_removed: int
    files_modified: int
    chunks_added: int
    chunks_removed: int
    time_taken: float
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'files_added': self.files_added,
            'files_removed': self.files_removed,
            'files_modified': self.files_modified,
            'chunks_added': self.chunks_added,
            'chunks_removed': self.chunks_removed,
            'time_taken': self.time_taken,
            'success': self.success,
            'error': self.error
        }


class IncrementalIndexer:
    """Handles incremental indexing of code changes."""
    
    def __init__(
        self,
        indexer: Optional[Indexer] = None,
        embedder: Optional[CodeEmbedder] = None,
        chunker: Optional[MultiLanguageChunker] = None,
        snapshot_manager: Optional[SnapshotManager] = None
    ):
        """Initialize incremental indexer.
        
        Args:
            indexer: Indexer instance
            embedder: Embedder instance
            chunker: Code chunker instance
            snapshot_manager: Snapshot manager instance
        """
        self.indexer = indexer or Indexer()
        self.embedder = embedder or CodeEmbedder()
        self.chunker = chunker or MultiLanguageChunker()
        self.snapshot_manager = snapshot_manager or SnapshotManager()
        self.change_detector = ChangeDetector(self.snapshot_manager)
    
    def detect_changes(self, project_path: str) -> Tuple[FileChanges, MerkleDAG]:
        """Detect changes in project since last snapshot.
        
        Args:
            project_path: Path to project
            
        Returns:
            Tuple of (FileChanges, current MerkleDAG)
        """
        return self.change_detector.detect_changes_from_snapshot(project_path)
    
    def incremental_index(
        self,
        project_path: str,
        project_name: Optional[str] = None,
        force_full: bool = False
    ) -> IncrementalIndexResult:
        """Perform incremental indexing of a project.
        
        Args:
            project_path: Path to project
            project_name: Optional project name
            force_full: Force full reindex even if snapshot exists
            
        Returns:
            IncrementalIndexResult with statistics
        """
        start_time = time.time()
        project_path = str(Path(project_path).resolve())
        
        if not project_name:
            project_name = Path(project_path).name
        
        try:
            # Check if we should do full index
            has_snap = self.snapshot_manager.has_snapshot(project_path)
            logger.info(f"[incremental_index] project={project_name} force_full={force_full} has_snapshot={has_snap} (RSS={_mem_mb()})")
            if force_full or not has_snap:
                logger.info(f"[incremental_index] -> full index for {project_name}")
                return self._full_index(project_path, project_name, start_time)

            # Detect changes
            t_detect = time.time()
            logger.info(f"[incremental_index] Detecting changes in {project_name}")
            changes, current_dag = self.detect_changes(project_path)
            logger.info(f"[incremental_index] Change detection took {time.time()-t_detect:.2f}s (RSS={_mem_mb()})")

            if not changes.has_changes():
                logger.info(f"[incremental_index] No changes detected in {project_name}")
                return IncrementalIndexResult(
                    files_added=0,
                    files_removed=0,
                    files_modified=0,
                    chunks_added=0,
                    chunks_removed=0,
                    time_taken=time.time() - start_time,
                    success=True
                )
            
            # Log changes
            logger.info(
                f"Changes detected - Added: {len(changes.added)}, "
                f"Removed: {len(changes.removed)}, Modified: {len(changes.modified)}"
            )
            
            # Process changes
            chunks_removed = self._remove_old_chunks(changes, project_name)
            chunks_added = self._add_new_chunks(changes, project_path, project_name)
            
            # Update snapshot
            self.snapshot_manager.save_snapshot(current_dag, {
                'project_name': project_name,
                'incremental_update': True,
                'files_added': len(changes.added),
                'files_removed': len(changes.removed),
                'files_modified': len(changes.modified)
            })
            
            # Update index
            self.indexer.save_index()
            
            return IncrementalIndexResult(
                files_added=len(changes.added),
                files_removed=len(changes.removed),
                files_modified=len(changes.modified),
                chunks_added=chunks_added,
                chunks_removed=chunks_removed,
                time_taken=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Incremental indexing failed: {e}")
            return IncrementalIndexResult(
                files_added=0,
                files_removed=0,
                files_modified=0,
                chunks_added=0,
                chunks_removed=0,
                time_taken=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def _full_index(
        self,
        project_path: str,
        project_name: str,
        start_time: float
    ) -> IncrementalIndexResult:
        """Perform full indexing of a project.
        
        Args:
            project_path: Path to project
            project_name: Project name
            start_time: Start time for timing
            
        Returns:
            IncrementalIndexResult
        """
        try:
            # Clear existing index
            self.indexer.clear_index()

            # Step 1: Build DAG
            t1 = time.time()
            logger.info(f"[full_index] Step 1/4: Building MerkleDAG for {project_path} (RSS={_mem_mb()})")
            dag = MerkleDAG(project_path)
            dag.build()
            all_files = dag.get_all_files()
            logger.info(f"[full_index] Step 1 done: {len(all_files)} files found in {time.time()-t1:.2f}s (RSS={_mem_mb()})")

            # Step 2: Filter & chunk
            t2 = time.time()
            supported_files = [f for f in all_files if self.chunker.is_supported(f)]
            logger.info(f"[full_index] Step 2/4: Chunking {len(supported_files)} supported files (of {len(all_files)} total)")

            all_chunks = []
            for i, file_path in enumerate(supported_files):
                full_path = Path(project_path) / file_path
                try:
                    chunks = self.chunker.chunk_file(str(full_path))
                    if chunks:
                        all_chunks.extend(chunks)
                except Exception as e:
                    logger.warning(f"Failed to chunk {file_path}: {e}")
                if (i + 1) % 100 == 0:
                    logger.info(f"[full_index]   chunked {i+1}/{len(supported_files)} files, {len(all_chunks)} chunks so far (RSS={_mem_mb()})")
            logger.info(f"[full_index] Step 2 done: {len(all_chunks)} chunks from {len(supported_files)} files in {time.time()-t2:.2f}s (RSS={_mem_mb()})")

            # Step 3+4: Embed in batches and add to index incrementally
            t3 = time.time()
            batch_size = 128
            chunks_added = 0
            total_batches = (len(all_chunks) + batch_size - 1) // batch_size
            logger.info(f"[full_index] Step 3/4: Embed + index {len(all_chunks)} chunks in {total_batches} batches (RSS={_mem_mb()})")

            for bi in range(0, len(all_chunks), batch_size):
                batch = all_chunks[bi:bi + batch_size]
                try:
                    batch_results = self.embedder.embed_chunks(batch, batch_size=len(batch))
                    for chunk, emb_result in zip(batch, batch_results):
                        emb_result.metadata['project_name'] = project_name
                        emb_result.metadata['content'] = chunk.content
                    self.indexer.add_embeddings(batch_results)
                    chunks_added += len(batch_results)
                except Exception as e:
                    logger.error(f"[full_index] Batch embed/index failed at {bi}: {e}", exc_info=True)

                batch_num = bi // batch_size + 1
                if batch_num % 10 == 0 or batch_num == total_batches:
                    logger.info(f"[full_index]   batch {batch_num}/{total_batches} ({chunks_added} chunks, {time.time()-t3:.1f}s, RSS={_mem_mb()})")

            # Free chunk list
            del all_chunks
            logger.info(f"[full_index] Step 3 done: {chunks_added} embedded+indexed in {time.time()-t3:.2f}s (RSS={_mem_mb()})")

            # Step 4: Save
            t5 = time.time()
            logger.info("[full_index] Step 4/4: Saving snapshot and index")
            self.snapshot_manager.save_snapshot(dag, {
                'project_name': project_name,
                'full_index': True,
                'total_files': len(all_files),
                'supported_files': len(supported_files),
                'chunks_indexed': chunks_added
            })
            self.indexer.save_index()
            logger.info(f"[full_index] Step 4 done in {time.time()-t5:.2f}s")
            logger.info(f"[full_index] COMPLETE: {chunks_added} chunks indexed in {time.time()-start_time:.2f}s total (RSS={_mem_mb()})")
            
            return IncrementalIndexResult(
                files_added=len(supported_files),
                files_removed=0,
                files_modified=0,
                chunks_added=chunks_added,
                chunks_removed=0,
                time_taken=time.time() - start_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Full indexing failed: {e}")
            return IncrementalIndexResult(
                files_added=0,
                files_removed=0,
                files_modified=0,
                chunks_added=0,
                chunks_removed=0,
                time_taken=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def _remove_old_chunks(self, changes: FileChanges, project_name: str) -> int:
        """Remove chunks for deleted and modified files.
        
        Args:
            changes: File changes
            project_name: Project name
            
        Returns:
            Number of chunks removed
        """
        files_to_remove = self.change_detector.get_files_to_remove(changes)
        chunks_removed = 0
        
        for file_path in files_to_remove:
            # Remove from metadata
            removed = self.indexer.remove_file_chunks(file_path, project_name)
            chunks_removed += removed
            logger.debug(f"Removed {removed} chunks from {file_path}")
        
        return chunks_removed
    
    def _add_new_chunks(
        self,
        changes: FileChanges,
        project_path: str,
        project_name: str
    ) -> int:
        """Add chunks for new and modified files.
        
        Args:
            changes: File changes
            project_path: Project root path
            project_name: Project name
            
        Returns:
            Number of chunks added
        """
        files_to_index = self.change_detector.get_files_to_reindex(changes)

        # Filter supported files
        supported_files = [f for f in files_to_index if self.chunker.is_supported(f)]
        logger.info(f"[_add_new_chunks] {len(supported_files)} files to chunk/embed (RSS={_mem_mb()})")

        # Collect all chunks first, then embed in a single pass
        chunks_to_embed = []
        for file_path in supported_files:
            full_path = Path(project_path) / file_path
            try:
                chunks = self.chunker.chunk_file(str(full_path))
                if chunks:
                    chunks_to_embed.extend(chunks)
            except Exception as e:
                logger.warning(f"Failed to chunk {file_path}: {e}")
        logger.info(f"[_add_new_chunks] {len(chunks_to_embed)} chunks to embed (RSS={_mem_mb()})")

        # Embed in batches and add to index incrementally
        batch_size = 128
        chunks_added = 0
        t_embed = time.time()
        total_batches = (len(chunks_to_embed) + batch_size - 1) // batch_size if chunks_to_embed else 0

        for bi in range(0, len(chunks_to_embed), batch_size):
            batch = chunks_to_embed[bi:bi + batch_size]
            try:
                batch_results = self.embedder.embed_chunks(batch, batch_size=len(batch))
                for chunk, emb_result in zip(batch, batch_results):
                    emb_result.metadata['project_name'] = project_name
                    emb_result.metadata['content'] = chunk.content
                self.indexer.add_embeddings(batch_results)
                chunks_added += len(batch_results)
            except Exception as e:
                logger.error(f"[_add_new_chunks] Batch failed at {bi}: {e}", exc_info=True)

            batch_num = bi // batch_size + 1
            if batch_num % 10 == 0 or batch_num == total_batches:
                logger.info(f"[_add_new_chunks] batch {batch_num}/{total_batches} ({chunks_added} chunks, {time.time()-t_embed:.1f}s, RSS={_mem_mb()})")

        del chunks_to_embed
        logger.info(f"[_add_new_chunks] DONE: {chunks_added} chunks in {time.time()-t_embed:.1f}s (RSS={_mem_mb()})")
        return chunks_added
    
    
    def get_indexing_stats(self, project_path: str) -> Optional[Dict]:
        """Get indexing statistics for a project.
        
        Args:
            project_path: Path to project
            
        Returns:
            Dictionary with statistics or None
        """
        metadata = self.snapshot_manager.load_metadata(project_path)
        if not metadata:
            return None
        
        # Add current index stats
        metadata['current_chunks'] = self.indexer.get_index_size()
        metadata['snapshot_age'] = self.snapshot_manager.get_snapshot_age(project_path)
        
        return metadata
    
    def needs_reindex(self, project_path: str, max_age_minutes: float = 5) -> bool:
        """Check if a project needs reindexing.
        
        Args:
            project_path: Path to project
            max_age_minutes: Maximum age of snapshot in minutes (default 5)
            
        Returns:
            True if reindex is needed
        """
        # No snapshot means needs index
        if not self.snapshot_manager.has_snapshot(project_path):
            return True
        
        # Check snapshot age (convert minutes to seconds)
        age = self.snapshot_manager.get_snapshot_age(project_path)
        if age and age > max_age_minutes * 60:
            return True
        
        # Quick check for changes
        return self.change_detector.quick_check(project_path)
    
    def auto_reindex_if_needed(self, project_path: str, project_name: Optional[str] = None, 
                              max_age_minutes: float = 5) -> IncrementalIndexResult:
        """Automatically reindex if the index is stale.
        
        Args:
            project_path: Path to project
            project_name: Optional project name
            max_age_minutes: Maximum age before auto-reindex (default 5 minutes)
            
        Returns:
            IncrementalIndexResult with statistics
        """
        import time
        start_time = time.time()
        
        if self.needs_reindex(project_path, max_age_minutes):
            logger.info(f"Auto-reindexing {project_path} (index older than {max_age_minutes} minutes)")
            return self.incremental_index(project_path, project_name)
        else:
            logger.debug(f"Index for {project_path} is fresh, skipping reindex")
            return IncrementalIndexResult(
                files_added=0,
                files_removed=0,
                files_modified=0,
                chunks_added=0,
                chunks_removed=0,
                time_taken=time.time() - start_time,
                success=True
            )
