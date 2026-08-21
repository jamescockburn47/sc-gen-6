import json
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
import aiohttp

from loguru import logger

from ..config.api_key_manager import APIKeyManager
from .assessment_models import AssessmentPayload, EvaluationResult
from .evaluation_prompts import EVALUATION_SYSTEM_PROMPT, EVALUATION_USER_PROMPT_TEMPLATE

# Anonymisation gateway (optional — graceful if not configured)
try:
    from ..anonymisation.gateway import CloudExportGateway, ExportBlockedError
    ANON_AVAILABLE = True
except ImportError:
    ANON_AVAILABLE = False


class CloudEvaluator:
    """
    Handles communication with cloud AI providers for quality assessment.

    When an anonymisation gateway is attached, all outbound data is
    automatically anonymised before reaching the cloud API, and
    responses are de-anonymised before being returned to the caller.
    """
    
    def __init__(
        self,
        api_key_manager: Optional[APIKeyManager] = None,
        provider: str = "openai",
        model_name: str = "gpt-5.1-instant",
        anonymisation_gateway: Optional['CloudExportGateway'] = None,
    ):
        self.api_key_manager = api_key_manager or APIKeyManager()
        self.provider = provider
        self.model = model_name
        self._gateway = anonymisation_gateway
        
    def set_anonymisation_gateway(self, gateway: 'CloudExportGateway') -> None:
        """Attach or replace the anonymisation gateway.

        Args:
            gateway: CloudExportGateway instance to use for all exports.
        """
        self._gateway = gateway
        logger.info("CloudEvaluator: anonymisation gateway attached")

    async def evaluate(self, payload: AssessmentPayload) -> Optional[EvaluationResult]:
        """
        Send assessment payload to cloud provider for evaluation.

        If an anonymisation gateway is attached, all PII in the payload
        is pseudonymised before transmission and the response is
        de-anonymised before return.
        
        Args:
            payload: The data to evaluate
            
        Returns:
            EvaluationResult or None if evaluation failed
        """
        if not payload.generated_answer or not payload.generated_answer.strip():
            print("Skipping evaluation: Empty answer")
            return None
            
        api_key = self.api_key_manager.get_key(self.provider)
        if not api_key:
            print(f"Skipping evaluation: No API key for {self.provider}")
            return None

        # --- Anonymise outbound data if gateway is configured ---
        if self._gateway:
            try:
                anon_payload = self._gateway.export_qa(
                    query=payload.query,
                    answer=payload.generated_answer,
                    chunks=payload.retrieved_chunks,
                )
                # Replace payload fields with anonymised versions
                payload_query = anon_payload.anonymised_query
                payload_answer = anon_payload.anonymised_text
                payload_chunks = anon_payload.anonymised_chunks
                logger.info(
                    "CloudEvaluator: payload anonymised via gateway "
                    f"(tokens={len(anon_payload.token_legend)})"
                )
            except ExportBlockedError as e:
                logger.error(f"CloudEvaluator: export blocked — {e.reason}")
                print(f"Export blocked by anonymisation gateway: {e.reason}")
                return None
        else:
            payload_query = payload.query
            payload_answer = payload.generated_answer
            payload_chunks = payload.retrieved_chunks

        # Prepare chunks string (limit to top 10 to respect context)
        top_chunks = payload_chunks[:10]
        chunks_str = "\n\n".join([
            f"Chunk {i+1} (Score: {c.get('score', 0):.2f}):\n{c.get('text', '')[:1000]}..." 
            for i, c in enumerate(top_chunks)
        ])

        # Format prompt
        user_prompt = EVALUATION_USER_PROMPT_TEMPLATE.format(
            query=payload_query,
            num_chunks=len(top_chunks),
            chunks=chunks_str,
            answer=payload_answer,
            model=payload.model_used,
            system_prompt=payload.system_prompt,
            config=json.dumps(payload.generation_config),
            diagnostics=json.dumps(payload.diagnostics)
        )

        try:
            if self.provider == "openai":
                result = await self._evaluate_openai(api_key, user_prompt)
                # --- De-anonymise response if gateway is configured ---
                if result and self._gateway:
                    result.raw_response = self._gateway.import_response(result.raw_response)
                    result.suggestions = [
                        self._gateway.import_response(s) for s in result.suggestions
                    ]
                    result.prompt_improvements = [
                        self._gateway.import_response(p) for p in result.prompt_improvements
                    ]
                return result
            # Add other providers here as needed
            else:
                print(f"Provider {self.provider} not implemented")
                return None
                
        except Exception as e:
            print(f"Cloud evaluation failed: {e}")
            return None

    async def _evaluate_openai(self, api_key: str, user_prompt: str) -> Optional[EvaluationResult]:
        """Evaluate using OpenAI API."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.3  # Low temp for consistent evaluation
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"OpenAI API error: {response.status} - {error_text}")
                    return None
                    
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                
                try:
                    eval_data = json.loads(content)
                    
                    # Map JSON response to EvaluationResult
                    return EvaluationResult(
                        provider=self.provider,
                        timestamp=datetime.now(),
                        scores=eval_data.get("scores", {}),
                        suggestions=eval_data.get("suggestions", []),
                        prompt_improvements=eval_data.get("prompt_improvements", []),
                        config_changes=eval_data.get("config_changes", {}),
                        overall_rating=eval_data.get("overall_rating", 0.0),
                        raw_response=content
                    )
                except json.JSONDecodeError:
                    print("Failed to parse JSON response from OpenAI")
                    return None
