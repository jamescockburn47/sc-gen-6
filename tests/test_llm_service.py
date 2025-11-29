"""Tests for LLM service."""

from unittest.mock import MagicMock

import pytest

from src.config.llm_config import LLMConfig
from src.generation.llm_service import LLMService


@pytest.fixture
def mock_llm_config(monkeypatch) -> LLMConfig:
    """Mock environment-driven LLM config."""
    config = LLMConfig(
        provider="lmstudio",
        base_url="http://localhost:1234/v1",
        api_key="lm-studio",
        model_name="qwen3:32b-instruct",
    )
    monkeypatch.setattr("src.generation.llm_service.load_llm_config", lambda: config)
    return config


@pytest.fixture
def mock_llm_client(monkeypatch, mock_llm_config):
    """Provide a mock LLM client returned by the factory."""
    client = MagicMock()
    client.list_models.return_value = [
        "qwen3:32b-instruct",
        "qwen3:14b-instruct",
    ]

    def factory(config):
        return client

    monkeypatch.setattr("src.generation.llm_service.get_llm_client", factory)
    return client


@pytest.fixture
def llm_service(mock_llm_client):
    """Create LLM service with mocked client."""
    return LLMService()


def test_llm_service_initialization(llm_service: LLMService, mock_llm_client):
    """Test LLM service wires up the client."""
    assert llm_service is not None
    assert llm_service.llm_client is mock_llm_client


def test_get_default_model(llm_service: LLMService):
    """Test getting default model."""
    model = llm_service.get_default_model()
    assert isinstance(model, str)
    assert model == "qwen3:32b-instruct"


def test_get_available_models(llm_service: LLMService, mock_llm_client):
    """Test getting available models."""
    mock_llm_client.list_models.return_value = ["model-a", "model-b"]
    models = llm_service.get_available_models()
    assert isinstance(models, list)
    assert models == ["model-a", "model-b"]


def test_generate_non_streaming(llm_service: LLMService, mock_llm_client, mock_llm_config):
    """Test non-streaming generation."""
    mock_llm_client.generate_chat_completion.return_value = "Test answer"

    result = llm_service.generate("Test query", stream=False)

    assert result == "Test answer"
    mock_llm_client.generate_chat_completion.assert_called_once()
    kwargs = mock_llm_client.generate_chat_completion.call_args.kwargs
    assert kwargs.get("response_format") == {"type": "text"}
    assert kwargs.get("stop") == ["<|channel|>"]


def test_generate_non_streaming_llama_skips_stop(monkeypatch, mock_llm_client):
    """Ensure llama provider omits response_format and stop."""
    llama_config = LLMConfig(
        provider="llama_cpp",
        base_url="http://127.0.0.1:8000/v1",
        api_key="local-llama",
        model_name="gpt-oss-20b",
    )
    monkeypatch.setattr("src.generation.llm_service.load_llm_config", lambda: llama_config)
    llama_service = LLMService()
    llama_service.llm_client = mock_llm_client
    mock_llm_client.generate_chat_completion.return_value = "<|channel|>final<|message|>Answer"

    result = llama_service.generate("Test query", stream=False)

    kwargs = mock_llm_client.generate_chat_completion.call_args.kwargs
    assert "response_format" not in kwargs
    assert "stop" not in kwargs
    assert result == "Answer"


def test_generate_with_custom_model(llm_service: LLMService, mock_llm_client):
    """Test generation with custom model."""
    mock_llm_client.generate_chat_completion.return_value = "Answer"

    llm_service.generate("query", model="qwen3:14b-instruct")

    call_args = mock_llm_client.generate_chat_completion.call_args
    assert call_args.kwargs["model"] == "qwen3:14b-instruct"


def test_generate_with_context(llm_service: LLMService, mock_llm_client):
    """Test generate_with_context composes prompt correctly."""
    mock_llm_client.generate_chat_completion.return_value = "Answer with citations"

    chunks = [
        {
            "text": "Sample text from document.",
            "metadata": {
                "file_name": "test.pdf",
                "page_number": 1,
                "paragraph_number": 1,
            },
        }
    ]

    result = llm_service.generate_with_context("Test query", chunks, stream=False)

    assert result == "Answer with citations"
    call_args = mock_llm_client.generate_chat_completion.call_args
    messages = call_args.kwargs["messages"]
    user_message = messages[-1]["content"]
    assert "Sample text from document" in user_message
    assert "test.pdf" in user_message


def test_generate_with_context_empty_chunks(llm_service: LLMService):
    """Test generate_with_context raises error for empty chunks."""
    with pytest.raises(ValueError, match="without context chunks"):
        llm_service.generate_with_context("query", [])


def test_generate_stream(llm_service: LLMService, mock_llm_client):
    """Test streaming generation."""
    mock_llm_client.stream_chat_completion.return_value = iter(["Hello", " world", "!"])

    tokens_received = []

    def callback(token: str):
        tokens_received.append(token)

    result = llm_service.generate_stream("query", callback=callback)

    assert result == "Hello world!"
    assert tokens_received == ["Hello", " world", "!"]


def test_generate_stream_llama_filters(monkeypatch, mock_llm_client):
    """Ensure llama streams skip analysis channel."""
    llama_config = LLMConfig(
        provider="llama_cpp",
        base_url="http://127.0.0.1:8000/v1",
        api_key="local-llama",
        model_name="gpt-oss-20b",
    )
    monkeypatch.setattr("src.generation.llm_service.load_llm_config", lambda: llama_config)
    llama_service = LLMService()
    llama_service.llm_client = mock_llm_client
    mock_llm_client.stream_chat_completion.return_value = iter(
        [
            "<|channel|>analysis<|message|>thinking...",
            "<|channel|>final<|message|>Answer",
            " continues",
        ]
    )

    tokens_received = []

    def callback(token: str):
        tokens_received.append(token)

    result = llama_service.generate_stream("query", callback=callback)

    assert result == "Answer continues"
    assert tokens_received == ["Answer", " continues"]


def test_generate_stream_llama_passthrough(monkeypatch, mock_llm_client):
    """Ensure llama stream without markers passes through."""
    llama_config = LLMConfig(
        provider="llama_cpp",
        base_url="http://127.0.0.1:8000/v1",
        api_key="local-llama",
        model_name="gpt-oss-20b",
    )
    monkeypatch.setattr("src.generation.llm_service.load_llm_config", lambda: llama_config)
    llama_service = LLMService()
    llama_service.llm_client = mock_llm_client
    mock_llm_client.stream_chat_completion.return_value = iter(["Plain ", "text"])

    result = llama_service.generate_stream("query")

    assert result == "Plain text"


def test_acceptance_criteria_smoke_test(llm_service: LLMService, mock_llm_client):
    """Test acceptance criteria: smoke test returns text."""
    mock_llm_client.generate_chat_completion.return_value = "Test response"

    result = llm_service.generate("test query")

    assert isinstance(result, str)
    assert len(result) > 0
    assert result == "Test response"

    print("✓ Acceptance criteria PASSED: smoke test returns text")


def test_acceptance_criteria_model_switching(llm_service: LLMService, mock_llm_client):
    """Test acceptance criteria: switching model uses registry key."""
    mock_llm_client.generate_chat_completion.return_value = "Answer"

    # Generate with default model
    llm_service.generate("query1")
    call1 = mock_llm_client.generate_chat_completion.call_args
    default_model = call1.kwargs["model"]

    # Generate with different model
    llm_service.generate("query2", model="qwen3:14b-instruct")
    call2 = mock_llm_client.generate_chat_completion.call_args
    custom_model = call2.kwargs["model"]

    assert custom_model == "qwen3:14b-instruct"
    assert custom_model != default_model

    print("✓ Acceptance criteria PASSED: switching model uses registry key")


def test_check_model_available(llm_service: LLMService, mock_llm_client):
    """Test checking model availability."""
    mock_llm_client.list_models.return_value = ["qwen3:32b-instruct", "qwen3:14b-instruct"]

    assert llm_service.check_model_available("qwen3:32b-instruct") is True
    assert llm_service.check_model_available("nonexistent-model") is False


def test_get_refusal_message(llm_service: LLMService):
    """Test getting refusal message."""
    message = llm_service.get_refusal_message()
    assert isinstance(message, str)
    assert len(message) > 0
    assert "cannot answer" in message.lower() or "sufficient" in message.lower()




