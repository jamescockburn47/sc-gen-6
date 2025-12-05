import asyncio
import json
import os
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))

from src.assessment.assessment_models import AssessmentPayload, EvaluationResult
from src.assessment.assessment_db import AssessmentDB
from src.assessment.suggestion_parser import SuggestionParser
from src.assessment.suggestion_applicator import SuggestionApplicator
from src.assessment.cloud_evaluator import CloudEvaluator

async def run_verification():
    print("Starting Verification of AI Quality Assessment System...")
    
    # 1. Setup Mock Data
    print("\n[1] Creating Mock Assessment Payload...")
    payload = AssessmentPayload(
        query="What is the capital of France?",
        retrieved_chunks=[{"text": "Paris is the capital of France.", "score": 0.9}],
        generated_answer="The capital of France is Paris.",
        model_used="gpt-4",
        system_prompt="You are a helpful assistant.",
        generation_config={"temperature": 0.7},
        diagnostics={"token_count": 10},
        timestamp=datetime.now()
    )
    print("Payload created successfully.")
    
    # 2. Mock Cloud Evaluator
    print("\n[2] Testing Cloud Evaluator (Mocked)...")
    
    mock_response = {
        "scores": {
            "answer_quality": 9.5,
            "retrieval_quality": 9.0,
            "prompt_effectiveness": 8.0,
            "generation_quality": 9.0
        },
        "suggestions": ["Consider lowering temperature for more deterministic results."],
        "prompt_improvements": ["You are a helpful and precise assistant."],
        "config_changes": {"temperature": 0.3},
        "overall_rating": 9.2,
        "critique": "Excellent answer, accurate and concise."
    }
    
    # Patch the _evaluate_openai method to avoid real API calls
    with patch.object(CloudEvaluator, '_evaluate_openai', new_callable=AsyncMock) as mock_eval:
        mock_eval.return_value = EvaluationResult(
            provider="openai",
            timestamp=datetime.now(),
            scores=mock_response["scores"],
            suggestions=mock_response["suggestions"],
            prompt_improvements=mock_response["prompt_improvements"],
            config_changes=mock_response["config_changes"],
            overall_rating=mock_response["overall_rating"],
            raw_response=json.dumps(mock_response)
        )
        
        # Mock APIKeyManager to return a fake key
        mock_key_manager = MagicMock()
        mock_key_manager.get_key.return_value = "fake-api-key"
        
        evaluator = CloudEvaluator(api_key_manager=mock_key_manager, provider="openai", model_name="gpt-5.1-instant")
        result = await evaluator.evaluate(payload)
        
        if result and result.overall_rating == 9.2:
            print("Cloud Evaluator returned expected result.")
        else:
            print(f"Cloud Evaluator failed. Result: {result}")
            return

    # 3. Test Database Storage
    print("\n[3] Testing Database Storage...")
    db = AssessmentDB(db_path="test_assessment.db")
    assessment_id = db.save_result(result)
    print(f"Saved assessment with ID: {assessment_id}")
    
    assessments = db.get_recent_assessments(limit=1)
    if assessments and assessments[0]['id'] == assessment_id:
        print("Successfully retrieved assessment from DB.")
    else:
        print("Failed to retrieve assessment from DB.")
        return
        
    suggestions = db.get_pending_suggestions()
    print(f"Found {len(suggestions)} pending suggestions.")
    
    # 4. Test Suggestion Parsing
    print("\n[4] Testing Suggestion Parser...")
    parsed_suggestions = SuggestionParser.parse(result)
    print(f"Parsed {len(parsed_suggestions)} suggestions.")
    for s in parsed_suggestions:
        print(f" - [{s['type']}] {s['description']}")
        
    # 5. Test Suggestion Applicator
    print("\n[5] Testing Suggestion Applicator...")
    applicator = SuggestionApplicator(config_manager=MagicMock())
    
    # Test applying a config change
    config_sugg = next((s for s in parsed_suggestions if s['type'] == 'config_change'), None)
    if config_sugg:
        success = applicator.apply(config_sugg)
        if success:
            print("Successfully applied config change suggestion.")
        else:
            print("Failed to apply suggestion.")
            
    # Cleanup
    if os.path.exists("test_assessment.db"):
        try:
            os.remove("test_assessment.db")
            print("\nCleaned up test database.")
        except PermissionError:
            print("\nCould not remove test database (file might be in use).")
        
    print("\nVerification Complete: SUCCESS")

if __name__ == "__main__":
    asyncio.run(run_verification())
