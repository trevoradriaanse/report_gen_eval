"""Test fixtures for report_gen_eval."""
import pytest
from typing import Dict, List

@pytest.fixture
def citation_examples() -> List[Dict]:
    """Example sentences with citations."""
    return [
        {
            "text": "Recent studies have shown that AI models can exhibit biases.",
            "expected": "YES"
        },
        {
            "text": "Deep learning has transformed AI.",
            "expected": "NO"
        },
        {
            "text": "去年新冠感染率有所上升",
            "expected": "YES"
        }
    ]

@pytest.fixture
def relevance_examples() -> List[Dict]:
    """Example sentences with citations and their relevance."""
    return [
        {
            "text": "Deep learning models can exhibit bias [1].",
            "citation": "A study of bias in deep neural networks across different domains.",
            "expected": "YES"
        },
        {
            "text": "The model achieved 95% accuracy.",
            "citation": "A review of climate change impacts on agriculture.",
            "expected": "NO"
        },
        {
            "text": "LLMs show emergent abilities.",
            "citation": "Investigation of emergent capabilities in large language models.",
            "expected": "YES"
        }
    ]

@pytest.fixture
def negative_examples() -> List[Dict]:
    """Example sentences with negative assertions."""
    return [
        {
            "text": "The study found no correlation between variables.",
            "expected": "YES"
        },
        {
            "text": "AI models show impressive performance.",
            "expected": "NO"
        },
        {
            "text": "研究表明该方法不有效。",
            "expected": "YES"
        }
    ]

@pytest.fixture
def citation_requirement_examples() -> List[Dict]:
    """Example sentences that may require citations."""
    return [
        {
            "text": "Studies have shown that exercise improves health.",
            "expected": "YES"
        },
        {
            "text": "The sky appears blue on clear days.",
            "expected": "NO"
        },
        {
            "text": "Recent research indicates a 15% increase in adoption.",
            "expected": "YES"
        }
    ]

@pytest.fixture
def first_instance_examples() -> List[Dict]:
    """Example sentences to test for first instance detection."""
    return [
        {
            "text": "AI models can show bias.",
            "previous": [],
            "expected": "YES"
        },
        {
            "text": "The models exhibit systematic bias in gender recognition.",
            "previous": ["AI models can show bias in various ways."],
            "expected": "YES"
        },
        {
            "text": "The models show significant bias.",
            "previous": ["AI models can show bias.", "This bias affects performance."],
            "expected": "NO"
        }
    ]

@pytest.fixture
def nugget_examples() -> List[Dict]:
    """Example sentences with nuggets to test agreement."""
    return [
        {
            "text": "The model achieved 95% accuracy.",
            "nugget": "The model's accuracy was 95%",
            "expected": "YES"
        },
        {
            "text": "Deep learning has transformed AI.",
            "nugget": "AI has been changed by neural networks",
            "expected": "YES"
        },
        {
            "text": "The study found no correlation.",
            "nugget": "A strong correlation was found",
            "expected": "NO"
        }
    ] 