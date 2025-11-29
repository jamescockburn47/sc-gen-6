"""Background task manager for overnight/long-running generation tasks.

Manages model selection, task execution, and progress tracking for
background tasks like case graph generation, timeline generation, and
case overview generation.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from src.config_loader import Settings, get_settings

logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    """Manages background/overnight tasks with optimal model selection."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize background task manager.
        
        Args:
            settings: Settings instance. If None, loads from config.
        """
        self.settings = settings or get_settings()
        self._current_task: Optional[str] = None
        self._cancel_requested = False
    
    def get_optimal_model(self, task_type: Optional[str] = None) -> str:
        """Select the largest available model for background tasks.
        
        Args:
            task_type: Type of task (for future task-specific model selection)
            
        Returns:
            Model name to use
        """
        config = self.settings.background_tasks
        
        if config.model_selection == "specific":
            return config.specific_model
        
        if config.model_selection == "default":
            return self.settings.models.llm.default
        
        # largest_available: check model_priority list
        for model_name in config.model_priority:
            if self._is_model_available(model_name):
                logger.info(f"Selected optimal model for background task: {model_name}")
                return model_name
        
        # Fallback to default if none available
        logger.warning("No priority models available, falling back to default")
        return self.settings.models.llm.default
    
    def _is_model_available(self, model_name: str) -> bool:
        """Check if a model is available.
        
        Args:
            model_name: Model name to check
            
        Returns:
            True if model is available
        """
        # For llama.cpp models, check if file exists
        if "gpt-oss" in model_name.lower() or "gguf" in model_name.lower():
            from pathlib import Path
            from src.llm.model_presets import get_model_presets
            
            presets = get_model_presets()
            for preset in presets:
                if preset.model_name == model_name:
                    model_path = Path(preset.path)
                    
                    # Check if path exists
                    if model_path.exists():
                        return True
                    
                    # For multi-file models (e.g., *-00001-of-00002.gguf)
                    # Check if the first file exists
                    if "-00001-of-" in str(model_path):
                        return model_path.exists()
                    
                    # Check if it's a pattern that needs the first file
                    parent = model_path.parent
                    if parent.exists():
                        # Look for any file matching the model name
                        pattern = model_path.stem.split("-00")[0] + "*.gguf"
                        matching_files = list(parent.glob(pattern))
                        if matching_files:
                            logger.info(f"Found {len(matching_files)} files for {model_name}")
                            return True
                    
                    return False
            return False
        
        # For Ollama models, check via API
        try:
            import requests
            ollama_host = self.settings.models.ollama.host
            response = requests.get(f"{ollama_host}/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                return any(model_name in name for name in model_names)
        except Exception as e:
            logger.debug(f"Error checking Ollama model availability: {e}")
        
        return False
    
    def should_use_background_model(self, task_type: str) -> bool:
        """Check if a task should use the background model.
        
        Args:
            task_type: Type of task
            
        Returns:
            True if task should use background model
        """
        return task_type in self.settings.background_tasks.enabled_for
    
    def run_task(
        self,
        task_type: str,
        task_function: Callable,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        incremental: bool = False,
        **kwargs
    ) -> Any:
        """Execute a background task with optimal model.
        
        Args:
            task_type: Type of task (e.g., "case_graph_generation")
            task_function: Function to execute
            progress_callback: Optional callback(message, current, total)
            incremental: If True, only process new/changed items
            **kwargs: Additional arguments for task_function
            
        Returns:
            Result from task_function
        """
        self._current_task = task_type
        self._cancel_requested = False
        
        try:
            # Select optimal model if task should use background model
            if self.should_use_background_model(task_type):
                model = self.get_optimal_model(task_type)
                kwargs["model"] = model
                logger.info(f"Running {task_type} with model: {model}")
            
            # Add progress callback
            if progress_callback:
                kwargs["progress_callback"] = progress_callback
            
            # Add incremental flag
            kwargs["incremental"] = incremental
            
            # Execute task
            result = task_function(**kwargs)
            
            logger.info(f"Completed {task_type}")
            return result
            
        except Exception as e:
            logger.error(f"Error in background task {task_type}: {e}", exc_info=True)
            raise
        finally:
            self._current_task = None
            self._cancel_requested = False
    
    def run_all_tasks(
        self,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        incremental: bool = False,
    ) -> dict[str, Any]:
        """Run all enabled background tasks in sequence.
        
        Args:
            progress_callback: Optional callback(message, current, total)
            incremental: If True, only process new/changed items
            
        Returns:
            Dictionary with results from each task
        """
        results = {}
        enabled_tasks = self.settings.background_tasks.enabled_for
        total_tasks = len(enabled_tasks)
        
        for idx, task_type in enumerate(enabled_tasks):
            if self._cancel_requested:
                logger.info("All tasks cancelled by user")
                break
            
            if progress_callback:
                progress_callback(f"Running {task_type}", idx, total_tasks)
            
            try:
                # Import and run the appropriate generator
                if task_type == "case_graph_generation":
                    from src.graph.graph_generator import CaseGraphGenerator
                    generator = CaseGraphGenerator(settings=self.settings)
                    result = self.run_task(
                        task_type,
                        generator.generate_full_graph,
                        progress_callback,
                        incremental
                    )
                    results[task_type] = result
                
                elif task_type == "timeline_generation":
                    from src.graph.timeline_generator import TimelineGenerator
                    generator = TimelineGenerator(settings=self.settings)
                    result = self.run_task(
                        task_type,
                        generator.generate_full_timeline,
                        progress_callback,
                        incremental
                    )
                    results[task_type] = result
                
                elif task_type == "case_overview_generation":
                    from src.generation.case_overview_generator import CaseOverviewGenerator
                    generator = CaseOverviewGenerator(settings=self.settings)
                    result = self.run_task(
                        task_type,
                        generator.generate_overview,
                        progress_callback,
                        incremental
                    )
                    results[task_type] = result
                
                elif task_type == "document_renaming":
                    from src.generation.document_renamer import DocumentRenamer
                    generator = DocumentRenamer(settings=self.settings)
                    result = self.run_task(
                        task_type,
                        generator.rename_all_documents,
                        progress_callback,
                        incremental
                    )
                    results[task_type] = result
                
            except Exception as e:
                logger.error(f"Error in {task_type}: {e}", exc_info=True)
                results[task_type] = {"error": str(e)}
        
        return results
    
    def cancel(self):
        """Request cancellation of current task."""
        self._cancel_requested = True
        logger.info(f"Cancellation requested for task: {self._current_task}")
    
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancel_requested
