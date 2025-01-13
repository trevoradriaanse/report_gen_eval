"""Tests for individual prompts."""

import pytest
import traceback

from report_gen_eval.evaluator import get_model_response, evaluate_report, ModelProvider
from report_gen_eval.prompts import (
    CHECK_RELEVANCE_SYSTEM,
    CHECK_RELEVANCE_USER,
    CHECK_NEGATIVE_SYSTEM,
    CHECK_NEGATIVE_USER,
    REQUIRES_CITATION_SYSTEM,
    REQUIRES_CITATION_USER,
    FIRST_INSTANCE_SYSTEM,
    FIRST_INSTANCE_USER,
    NUGGET_AGREEMENT_SYSTEM,
    NUGGET_AGREEMENT_USER,
)
import json
import tempfile
from typing import List, Dict


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
            context=context,
        )
    if response != expected:
        raise TestError(
            f"Unexpected response. Expected: {expected}, Got: {response}",
            context=context,
        )


def test_model_providers():
    """Test model provider selection."""
    try:
        # Test OpenAI
        response = get_model_response("test", "test", provider=ModelProvider.OPENAI)
        assert response is not None, "OpenAI model response failed"

        # Test Anthropic
        # response = get_model_response("test", "test", provider=ModelProvider.ANTHROPIC)
        assert response is not None, "Anthropic model response failed"

        # Test Together
        response = get_model_response("test", "test", provider=ModelProvider.TOGETHER)
        assert response is not None, "Together model response failed"

        # Test Hugging Face
        response = get_model_response(
            "test", "test", provider=ModelProvider.HUGGINGFACE
        )
        assert response is not None, "Hugging Face model response failed"

        # Test invalid provider
        with pytest.raises(ValueError, match="Unsupported model provider"):
            get_model_response("test", "test", provider="invalid_provider")
    except Exception as e:
        raise TestError(str(e), context="model provider initialization")


def test_check_relevance(relevance_examples, default_provider):
    """Test citation relevance prompt.  Matches 'cited document is relevant' diamond"""
    for i, example in enumerate(relevance_examples):
        context = f"relevance example {i + 1}: {example['text']}"
        try:
            response = get_model_response(
                CHECK_RELEVANCE_SYSTEM,
                CHECK_RELEVANCE_USER.format(
                    sentence=example["text"], citation_content=example["citation"]
                ),
                provider=default_provider,
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
                provider=default_provider,
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
                provider=default_provider,
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
                    previous_sentences="\n".join(example["previous"]),
                ),
                provider=default_provider,
            )
            validate_response(response, example["expected"], context)
        except Exception as e:
            if isinstance(e, TestError):
                raise
            raise TestError(str(e), context=context)


@pytest.fixture
def nugget_examples() -> List[Dict]:
    """Example sentences with nuggets to test agreement."""
    return [
        {
            "text": "The model achieved 95% accuracy.",
            "question_text": "What was the model's accuracy?",
            "gold_answers": ["95%"],
            "expected": "YES",
        },
        {
            "text": "Deep learning has transformed AI.",
            "question_text": "What has changed AI?",
            "gold_answers": ["neural networks", "deep learning"],
            "expected": "YES",
        },
        {
            "text": "The study found no correlation.",
            "question_text": "What correlation was found?",
            "gold_answers": ["A strong correlation"],
            "expected": "NO",
        },
    ]


def test_nugget_agreement(nugget_examples):
    """Test nugget agreement prompt."""
    for example in nugget_examples:
        for answer in example["gold_answers"]:
            # Format nugget as question-answer pair
            nugget_text = f"Question: {example['question_text']}\nAnswer: {answer}"
            response = get_model_response(
                NUGGET_AGREEMENT_SYSTEM,
                NUGGET_AGREEMENT_USER.format(
                    sentence=example["text"], nugget_question=example['question_text'], nugget_answer=answer
                ),
            )
            assert response == example["expected"]


@pytest.mark.skip("no test pkl")
def test_evaluate_report():
    """Test report evaluation with nuggets."""
    report = {
        "request_id": "300",
        "run_id": "test",
        "collection_ids": ["test"],
        "sentences": [
            {
                "text": "The suicide rate increased by 3.7% in 2020.",
                "citations": ["doc1"],
            }
        ],
    }

    # Create a temporary nuggets file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as f:
        json.dump(
            {
                "query_id": "300",
                "test_collection": "rus_2024",
                "query_text": "Test query",
                "hash": 1111,
                "items": [
                    {
                        "query_id": "300",
                        "info": {"importance": "vital", "used": False},
                        "question_id": "300_test",
                        "question_text": "How much did suicides rise by in 2020?",
                        "gold_answers": ["3.7%"],
                    }
                ],
            },
            f,
        )
        f.flush()

        results = evaluate_report(report, nuggets_file=f.name)
        assert results["metrics"]["unique_nuggets_matched"] == 1
