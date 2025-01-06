"""Tests for individual prompts."""
import pytest
import traceback
from report_gen_eval.evaluator import get_model_response, ModelProvider, get_model
from report_gen_eval.prompts import (
    CHECK_RELEVANCE_SYSTEM, CHECK_RELEVANCE_USER,
    CHECK_NEGATIVE_SYSTEM, CHECK_NEGATIVE_USER,
    REQUIRES_CITATION_SYSTEM, REQUIRES_CITATION_USER,
    FIRST_INSTANCE_SYSTEM, FIRST_INSTANCE_USER,
    NUGGET_AGREEMENT_SYSTEM, NUGGET_AGREEMENT_USER
)

class TestError(Exception):
    """Custom error for test failures."""
    def __init__(self, message: str, context: str = None):
        self.message = message
        self.context = context
        self.traceback = traceback.format_exc()
        super().__init__(self.full_message)
    
    @property
    def full_message(self) -> str:
        """Get the full error message including context and stack trace."""
        msg = []
        if self.context:
            msg.append(f"Context: {self.context}")
        msg.append(f"Error: {self.message}")
        msg.append("\nStack trace:")
        msg.append(self.traceback)
        return "\n".join(msg)

@pytest.fixture
def default_provider():
    """Default model provider for tests."""
    return ModelProvider.TOGETHER

def validate_response(response: str, expected: str, context: str):
    """Validate model response with detailed error message."""
    response = response.strip()
    if response not in ["YES", "NO"]:
        raise TestError(
            f"Invalid response format. Expected 'YES' or 'NO', got: '{response}'",
            context=context
        )
    if response != expected:
        raise TestError(
            f"Unexpected response. Expected: {expected}, Got: {response}",
            context=context
        )

def test_model_providers():
    """Test model provider selection."""
    try:
        # Test OpenAI
        model = get_model(ModelProvider.OPENAI)
        assert "openai" in str(model).lower(), "OpenAI model not properly initialized"
        
        # Test Anthropic
        model = get_model(ModelProvider.ANTHROPIC)
        assert "anthropic" in str(model).lower(), "Anthropic model not properly initialized"
        
        # Test Together
        model = get_model(ModelProvider.TOGETHER)
        assert "together" in str(model).lower(), "Together model not properly initialized"
        
        # Test invalid provider
        with pytest.raises(ValueError, match="Unsupported model provider"):
            get_model("invalid_provider")
    except Exception as e:
        raise TestError(str(e), context="model provider initialization")

def test_check_relevance(relevance_examples, default_provider):
    """Test citation relevance prompt."""
    for i, example in enumerate(relevance_examples):
        context = f"relevance example {i + 1}: {example['text']}"
        try:
            response = get_model_response(
                CHECK_RELEVANCE_SYSTEM,
                CHECK_RELEVANCE_USER.format(
                    sentence=example["text"],
                    citation_content=example["citation"]
                ),
                provider=default_provider
            )
            validate_response(response, example["expected"], context)
        except Exception as e:
            if isinstance(e, TestError):
                raise
            raise TestError(str(e), context=context)

def test_check_negative(negative_examples, default_provider):
    """Test negative assertion detection prompt."""
    for i, example in enumerate(negative_examples):
        context = f"negative example {i + 1}: {example['text']}"
        try:
            response = get_model_response(
                CHECK_NEGATIVE_SYSTEM,
                CHECK_NEGATIVE_USER.format(sentence=example["text"]),
                provider=default_provider
            )
            validate_response(response, example["expected"], context)
        except Exception as e:
            if isinstance(e, TestError):
                raise
            raise TestError(str(e), context=context)

def test_requires_citation(citation_requirement_examples, default_provider):
    """Test citation requirement prompt."""
    for i, example in enumerate(citation_requirement_examples):
        context = f"citation requirement example {i + 1}: {example['text']}"
        try:
            response = get_model_response(
                REQUIRES_CITATION_SYSTEM,
                REQUIRES_CITATION_USER.format(sentence=example["text"]),
                provider=default_provider
            )
            validate_response(response, example["expected"], context)
        except Exception as e:
            if isinstance(e, TestError):
                raise
            raise TestError(str(e), context=context)

def test_first_instance(first_instance_examples, default_provider):
    """Test first instance detection prompt."""
    for i, example in enumerate(first_instance_examples):
        context = f"first instance example {i + 1}: {example['text']}"
        try:
            response = get_model_response(
                FIRST_INSTANCE_SYSTEM,
                FIRST_INSTANCE_USER.format(
                    sentence=example["text"],
                    previous_sentences="\n".join(example["previous"])
                ),
                provider=default_provider
            )
            validate_response(response, example["expected"], context)
        except Exception as e:
            if isinstance(e, TestError):
                raise
            raise TestError(str(e), context=context)

def test_nugget_agreement(nugget_examples, default_provider):
    """Test nugget agreement prompt."""
    for i, example in enumerate(nugget_examples):
        context = f"nugget agreement example {i + 1}: {example['text']}"
        try:
            response = get_model_response(
                NUGGET_AGREEMENT_SYSTEM,
                NUGGET_AGREEMENT_USER.format(
                    sentence=example["text"],
                    nugget=example["nugget"]
                ),
                provider=default_provider
            )
            validate_response(response, example["expected"], context)
        except Exception as e:
            if isinstance(e, TestError):
                raise
            raise TestError(str(e), context=context) 