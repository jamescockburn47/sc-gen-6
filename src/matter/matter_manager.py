"""Matter manager for multi-case workspace handling."""

import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.matter.matter_config import MatterConfig

logger = logging.getLogger(__name__)


class MatterManager:
    """
    Manages multiple matter workspaces.
    
    Each matter has isolated:
    - Document storage
    - Vector database (Chroma)
    - BM25 keyword index
    - Case graph
    
    Only one matter can be active at a time.
    Services are lazy-loaded when accessed.
    """
    
    def __init__(self, base_path: Path | str):
        """Initialize matter manager.
        
        Args:
            base_path: Base path for all data (typically 'data/')
        """
        self.base_path = Path(base_path)
        self.matters_path = self.base_path / "matters"
        self.matters_path.mkdir(parents=True, exist_ok=True)
        
        # Global settings path
        self.global_path = self.base_path / "global"
        self.global_path.mkdir(parents=True, exist_ok=True)
        
        # Current matter
        self.current_matter: Optional[MatterConfig] = None
        self._loaded_services: dict[str, Any] = {}
        
        # Check for legacy single-matter data and migrate if needed
        self._check_legacy_migration()
    
    def _check_legacy_migration(self) -> None:
        """Check for legacy single-matter layout and offer migration."""
        # Legacy paths (pre-multi-matter)
        legacy_chroma = self.base_path / "chroma_db"
        legacy_bm25 = self.base_path / "bm25_index"
        legacy_docs = self.base_path / "documents"
        
        if legacy_chroma.exists() or legacy_bm25.exists() or legacy_docs.exists():
            # Check if we already have matters
            if not any(self.matters_path.iterdir()):
                logger.info("Legacy single-matter data detected. Will migrate on first matter creation.")
    
    def list_matters(self) -> list[MatterConfig]:
        """List all available matters.
        
        Returns:
            List of MatterConfig, sorted by last accessed (most recent first)
        """
        matters = []
        
        for matter_dir in self.matters_path.iterdir():
            if matter_dir.is_dir():
                config_path = matter_dir / "matter.json"
                if config_path.exists():
                    try:
                        matter = MatterConfig.load(config_path)
                        matters.append(matter)
                    except Exception as e:
                        logger.warning(f"Failed to load matter config from {config_path}: {e}")
        
        return sorted(matters, key=lambda m: m.last_accessed, reverse=True)
    
    def get_matter(self, matter_id: str) -> Optional[MatterConfig]:
        """Get matter by ID.
        
        Args:
            matter_id: Matter ID
            
        Returns:
            MatterConfig or None if not found
        """
        matter_path = self.matters_path / matter_id / "matter.json"
        if matter_path.exists():
            return MatterConfig.load(matter_path)
        return None
    
    def create_matter(
        self,
        name: str,
        reference: str = "",
        client: str = "",
        parties: Optional[dict[str, str]] = None,
        notes: str = "",
        migrate_legacy: bool = False,
    ) -> MatterConfig:
        """Create a new matter.
        
        Args:
            name: Matter name (e.g., "Smith v Acme Ltd")
            reference: Case reference number
            client: Client name
            parties: Dict of party roles to names
            notes: User notes
            migrate_legacy: If True, migrate legacy single-matter data
            
        Returns:
            Created MatterConfig
        """
        # Generate folder-safe ID from name
        safe_name = re.sub(r'[^\w\s-]', '', name.lower())
        safe_name = re.sub(r'[\s]+', '-', safe_name)
        matter_id = f"{safe_name[:20]}-{datetime.now().strftime('%Y%m%d')}"
        
        # Ensure unique
        base_id = matter_id
        counter = 1
        while (self.matters_path / matter_id).exists():
            matter_id = f"{base_id}-{counter}"
            counter += 1
        
        # Create directory structure
        matter_dir = self.matters_path / matter_id
        matter_dir.mkdir(parents=True)
        (matter_dir / "documents").mkdir()
        (matter_dir / "chroma_db").mkdir()
        (matter_dir / "bm25_index").mkdir()
        (matter_dir / "case_graph").mkdir()
        (matter_dir / "exports").mkdir()
        
        # Create config
        config = MatterConfig(
            id=matter_id,
            name=name,
            reference=reference,
            client=client,
            parties=parties or {},
            notes=notes,
            base_path=str(matter_dir),
        )
        config.save(matter_dir / "matter.json")
        
        logger.info(f"Created matter: {name} ({matter_id})")
        
        # Migrate legacy data if requested
        if migrate_legacy:
            self._migrate_legacy_data(config)
        
        return config
    
    def _migrate_legacy_data(self, matter: MatterConfig) -> None:
        """Migrate legacy single-matter data to new matter.
        
        Args:
            matter: Target matter config
        """
        matter_dir = Path(matter.base_path)
        
        # Migrate Chroma
        legacy_chroma = self.base_path / "chroma_db"
        if legacy_chroma.exists():
            logger.info("Migrating legacy Chroma database...")
            shutil.copytree(legacy_chroma, matter_dir / "chroma_db", dirs_exist_ok=True)
        
        # Migrate BM25
        legacy_bm25 = self.base_path / "bm25_index"
        if legacy_bm25.exists():
            logger.info("Migrating legacy BM25 index...")
            shutil.copytree(legacy_bm25, matter_dir / "bm25_index", dirs_exist_ok=True)
        
        # Migrate documents
        legacy_docs = self.base_path / "documents"
        if legacy_docs.exists():
            logger.info("Migrating legacy documents...")
            shutil.copytree(legacy_docs, matter_dir / "documents", dirs_exist_ok=True)
        
        logger.info("Legacy data migration complete")
    
    def delete_matter(self, matter_id: str, confirm: bool = False) -> bool:
        """Delete a matter and all its data.
        
        Args:
            matter_id: Matter ID to delete
            confirm: Must be True to actually delete
            
        Returns:
            True if deleted, False otherwise
        """
        if not confirm:
            logger.warning("Delete not confirmed - set confirm=True to delete")
            return False
        
        matter_dir = self.matters_path / matter_id
        if not matter_dir.exists():
            logger.warning(f"Matter not found: {matter_id}")
            return False
        
        # Unload if current
        if self.current_matter and self.current_matter.id == matter_id:
            self._unload_current()
        
        # Delete directory
        shutil.rmtree(matter_dir)
        logger.info(f"Deleted matter: {matter_id}")
        return True
    
    def switch_matter(self, matter_id: str) -> MatterConfig:
        """Switch to a different matter.
        
        Args:
            matter_id: Matter ID to switch to
            
        Returns:
            MatterConfig for the switched matter
            
        Raises:
            ValueError: If matter not found
        """
        matter = self.get_matter(matter_id)
        if not matter:
            raise ValueError(f"Matter not found: {matter_id}")
        
        # Unload current matter services
        self._unload_current()
        
        # Update last accessed
        matter.last_accessed = datetime.now()
        matter.save(self.matters_path / matter_id / "matter.json")
        
        self.current_matter = matter
        logger.info(f"Switched to matter: {matter.display_name}")
        
        return matter
    
    def _unload_current(self) -> None:
        """Unload current matter services to free resources."""
        if "vector_store" in self._loaded_services:
            # Chroma handles persistence automatically
            del self._loaded_services["vector_store"]
        
        if "bm25_index" in self._loaded_services:
            # BM25 should save on unload
            try:
                self._loaded_services["bm25_index"].save()
            except Exception:
                pass
            del self._loaded_services["bm25_index"]
        
        if "case_graph" in self._loaded_services:
            # Save graph
            try:
                self._loaded_services["case_graph"].save()
            except Exception:
                pass
            del self._loaded_services["case_graph"]
        
        self._loaded_services = {}
        self.current_matter = None
    
    def get_vector_store(self):
        """Get vector store for current matter.
        
        Returns:
            VectorStore instance
            
        Raises:
            RuntimeError: If no matter is active
        """
        if not self.current_matter:
            raise RuntimeError("No matter selected")
        
        if "vector_store" not in self._loaded_services:
            from src.retrieval.vector_store import VectorStore
            from src.config_loader import get_settings
            
            settings = get_settings()
            # Override persist path for current matter
            self._loaded_services["vector_store"] = VectorStore(
                settings=settings,
                persist_path=Path(self.current_matter.base_path) / "chroma_db"
            )
        
        return self._loaded_services["vector_store"]
    
    def get_bm25_index(self):
        """Get BM25 index for current matter.
        
        Returns:
            BM25Index instance
            
        Raises:
            RuntimeError: If no matter is active
        """
        if not self.current_matter:
            raise RuntimeError("No matter selected")
        
        if "bm25_index" not in self._loaded_services:
            from src.retrieval.bm25_index import BM25Index
            from src.config_loader import get_settings
            
            settings = get_settings()
            # Override index path for current matter
            self._loaded_services["bm25_index"] = BM25Index(
                settings=settings,
                index_path=Path(self.current_matter.base_path) / "bm25_index"
            )
        
        return self._loaded_services["bm25_index"]
    
    def get_case_graph(self):
        """Get case graph for current matter.
        
        Returns:
            CaseGraph instance
            
        Raises:
            RuntimeError: If no matter is active
        """
        if not self.current_matter:
            raise RuntimeError("No matter selected")
        
        if "case_graph" not in self._loaded_services:
            from src.graph.case_graph import CaseGraph
            
            self._loaded_services["case_graph"] = CaseGraph(
                path=Path(self.current_matter.base_path) / "case_graph"
            )
        
        return self._loaded_services["case_graph"]
    
    def get_documents_path(self) -> Path:
        """Get documents path for current matter.
        
        Returns:
            Path to documents directory
            
        Raises:
            RuntimeError: If no matter is active
        """
        if not self.current_matter:
            raise RuntimeError("No matter selected")
        
        return Path(self.current_matter.base_path) / "documents"
    
    def update_matter_stats(self) -> None:
        """Update stats for current matter."""
        if not self.current_matter:
            return
        
        try:
            # Get vector store stats
            vs = self.get_vector_store()
            vs_stats = vs.stats()
            self.current_matter.chunk_count = vs_stats.get("total_chunks", 0)
            self.current_matter.document_count = vs_stats.get("document_count", 0)
            
            # Get graph stats
            graph = self.get_case_graph()
            graph_stats = graph.stats()
            self.current_matter.entity_count = graph_stats.get("entity_count", 0)
            
            # Save updated config
            self.current_matter.save(
                self.matters_path / self.current_matter.id / "matter.json"
            )
        except Exception as e:
            logger.warning(f"Failed to update matter stats: {e}")
    
    def export_matter(self, matter_id: str, export_path: Path) -> Path:
        """Export a matter to a zip file.
        
        Args:
            matter_id: Matter ID to export
            export_path: Directory to save export
            
        Returns:
            Path to exported zip file
        """
        matter = self.get_matter(matter_id)
        if not matter:
            raise ValueError(f"Matter not found: {matter_id}")
        
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Create zip
        zip_name = f"{matter_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        zip_path = shutil.make_archive(
            str(export_path / zip_name),
            'zip',
            self.matters_path / matter_id
        )
        
        logger.info(f"Exported matter to: {zip_path}")
        return Path(zip_path)
    
    def import_matter(self, zip_path: Path) -> MatterConfig:
        """Import a matter from a zip file.
        
        Args:
            zip_path: Path to zip file
            
        Returns:
            Imported MatterConfig
        """
        import zipfile
        
        # Extract to temp location
        temp_dir = self.matters_path / f"_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        temp_dir.mkdir()
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(temp_dir)
            
            # Find matter.json
            config_path = temp_dir / "matter.json"
            if not config_path.exists():
                # Check subdirectory
                for subdir in temp_dir.iterdir():
                    if subdir.is_dir() and (subdir / "matter.json").exists():
                        config_path = subdir / "matter.json"
                        temp_dir = subdir
                        break
            
            if not config_path.exists():
                raise ValueError("No matter.json found in archive")
            
            # Load config
            matter = MatterConfig.load(config_path)
            
            # Generate new ID to avoid conflicts
            original_id = matter.id
            matter.id = f"{matter.id}-imported-{datetime.now().strftime('%H%M%S')}"
            
            # Move to final location
            final_dir = self.matters_path / matter.id
            shutil.move(str(temp_dir), str(final_dir))
            
            # Update paths
            matter.base_path = str(final_dir)
            matter.last_accessed = datetime.now()
            matter.save(final_dir / "matter.json")
            
            logger.info(f"Imported matter: {matter.display_name} (was: {original_id})")
            return matter
            
        except Exception as e:
            # Cleanup on failure
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise



