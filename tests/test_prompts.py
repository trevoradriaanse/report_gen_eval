"""Tests for individual prompts."""
import pytest
from report_gen_eval.evaluator import get_model_response
from report_gen_eval.prompts import (
    CHECK_RELEVANCE_SYSTEM, CHECK_RELEVANCE_USER,
    CHECK_NEGATIVE_SYSTEM, CHECK_NEGATIVE_USER,
    REQUIRES_CITATION_SYSTEM, REQUIRES_CITATION_USER,
    FIRST_INSTANCE_SYSTEM, FIRST_INSTANCE_USER,
    NUGGET_AGREEMENT_SYSTEM, NUGGET_AGREEMENT_USER
)

def test_check_relevance(relevance_examples):
    """Test citation relevance prompt."""
    for example in relevance_examples:
        response = get_model_response(
            CHECK_RELEVANCE_SYSTEM,
            CHECK_RELEVANCE_USER.format(
                sentence=example["text"],
                citation_content=example["citation"]
            )
        )
        assert response.strip() == example["expected"]

def test_check_negative(negative_examples):
    """Test negative assertion detection prompt."""
    for example in negative_examples:
        response = get_model_response(
            CHECK_NEGATIVE_SYSTEM,
            CHECK_NEGATIVE_USER.format(sentence=example["text"])
        )
        assert response.strip() == example["expected"]

def test_requires_citation(citation_requirement_examples):
    """Test citation requirement prompt."""
    for example in citation_requirement_examples:
        response = get_model_response(
            REQUIRES_CITATION_SYSTEM,
            REQUIRES_CITATION_USER.format(sentence=example["text"])
        )
        assert response.strip() == example["expected"]

def test_first_instance(first_instance_examples):
    """Test first instance detection prompt."""
    for example in first_instance_examples:
        response = get_model_response(
            FIRST_INSTANCE_SYSTEM,
            FIRST_INSTANCE_USER.format(
                sentence=example["text"],
                previous_sentences="\n".join(example["previous"])
            )
        )
        assert response.strip() == example["expected"]

def test_nugget_agreement(nugget_examples):
    """Test nugget agreement prompt."""
    for example in nugget_examples:
        response = get_model_response(
            NUGGET_AGREEMENT_SYSTEM,
            NUGGET_AGREEMENT_USER.format(
                sentence=example["text"],
                nugget=example["nugget"]
            )
        )
        assert response.strip() == example["expected"] 