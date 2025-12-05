from typing import List, Dict, Any
from .assessment_models import EvaluationResult

class SuggestionParser:
    """
    Parses evaluation results into actionable suggestions.
    """
    
    @staticmethod
    def parse(result: EvaluationResult) -> List[Dict[str, Any]]:
        """
        Parse evaluation result into a list of structured suggestion dictionaries.
        """
        parsed_suggestions = []
        
        # Parse general suggestions
        for suggestion in result.suggestions:
            parsed_suggestions.append({
                "type": "general",
                "description": suggestion,
                "actionable": False
            })
            
        # Parse prompt improvements
        for improvement in result.prompt_improvements:
            parsed_suggestions.append({
                "type": "prompt_edit",
                "description": improvement,
                "actionable": True,
                "target": "system_prompt",
                "value": improvement
            })
            
        # Parse config changes
        for key, value in result.config_changes.items():
            parsed_suggestions.append({
                "type": "config_change",
                "description": f"Change {key} to {value}",
                "actionable": True,
                "target": key,
                "value": value
            })
            
        return parsed_suggestions
