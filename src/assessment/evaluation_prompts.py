"""
Prompts for cloud-based quality assessment of RAG outputs.
"""

EVALUATION_SYSTEM_PROMPT = """You are an expert AI Quality Assurance System. 
Your goal is to evaluate the quality of Retrieval-Augmented Generation (RAG) outputs.
You must provide critical, constructive, and structured feedback.
Focus on:
1. Accuracy and faithfulness to retrieved context
2. Retrieval quality (relevance, coverage, redundancy)
3. Prompt effectiveness
4. Actionable improvements for the system configuration

You must output your response in valid JSON format matching the requested schema.
"""

EVALUATION_USER_PROMPT_TEMPLATE = """
Analyze this AI-generated answer and provide detailed feedback.

QUERY: {query}

RETRIEVED CHUNKS (Top {num_chunks}):
{chunks}

GENERATED ANSWER:
{answer}

SYSTEM CONFIGURATION:
- Model: {model}
- System Prompt: {system_prompt}
- Generation Config: {config}

DIAGNOSTICS:
{diagnostics}

Provide a structured evaluation in JSON format with the following fields:
1. "scores": Object with float values (1-10) for:
   - "answer_quality": Accuracy, completeness, clarity
   - "retrieval_quality": Relevance of chunks to query
   - "prompt_effectiveness": How well the system prompt guided the model
   - "generation_quality": Technical quality (length, formatting, etc.)

2. "suggestions": List of strings describing high-level improvements.

3. "prompt_improvements": List of strings with specific system prompt edits.

4. "config_changes": Object with suggested parameter changes (e.g., {{"temperature": 0.5}}).

5. "overall_rating": Float (1-10) summary score.

6. "critique": String with detailed qualitative analysis.

Ensure your feedback is realistic given the provided context chunks. 
If the chunks are insufficient, blame the retrieval, not the generation.
"""
