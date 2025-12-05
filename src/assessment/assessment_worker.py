import asyncio
from PySide6.QtCore import QThread, Signal
from .assessment_models import AssessmentPayload, EvaluationResult
from .cloud_evaluator import CloudEvaluator
from .assessment_db import AssessmentDB
from src.config.api_key_manager import APIKeyManager

class AssessmentWorker(QThread):
    """
    Worker thread to run cloud quality assessment in the background.
    """
    finished = Signal(object)  # Emits EvaluationResult
    error = Signal(str)
    
    def __init__(self, payload: AssessmentPayload):
        super().__init__()
        self.payload = payload
        self.api_key_manager = APIKeyManager()
        self.db = AssessmentDB()
        
    def run(self):
        """Run the assessment."""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Determine provider from config (or default to openai)
            # For now, we'll fetch the provider from the payload config if present, 
            # or default to what's in the settings (which we need to access).
            # Since we don't have easy access to settings here, we'll use the APIKeyManager 
            # to check which keys are available or pass the provider in the payload/init.
            
            # Ideally, the provider should be passed in. 
            # For this implementation, we'll assume a default or check config.
            from src.config_loader import get_settings
            settings = get_settings()
            provider = getattr(settings.quality, "provider", "openai")
            model_name = getattr(settings.quality, "model", "gpt-5.1-instant")
            
            api_key = self.api_key_manager.get_key(provider)
            if not api_key:
                self.error.emit(f"No API key found for provider: {provider}")
                return

            evaluator = CloudEvaluator(provider=provider, model_name=model_name)
            
            # Run evaluation
            result = loop.run_until_complete(
                evaluator.evaluate(self.payload)
            )
            
            loop.close()
            
            if result:
                # Save to DB
                self.db.save_result(result)
                self.finished.emit(result)
            else:
                self.error.emit("Evaluation returned no result")
                
        except Exception as e:
            self.error.emit(f"Assessment failed: {str(e)}")
