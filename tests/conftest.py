"""Test fixtures for report_gen_eval."""

import pytest
from typing import Dict, List


@pytest.fixture
def citation_examples() -> List[Dict]:
    """Example sentences with citations."""
    return [
        {
            "text": "Recent studies have shown that AI models can exhibit biases.",
            "expected": "YES",
        },
        {"text": "Deep learning has transformed AI.", "expected": "NO"},
        {"text": "去年新冠感染率有所上升", "expected": "YES"},
    ]


@pytest.fixture
def relevance_examples() -> List[Dict]:
    """Example sentences with citations and their relevance."""
    return [
        {
            "text": "The new transformer architecture achieved a 25% improvement in translation quality [37e79a54-0f8c-4ebd-bf5b-a1a7408b2b06].",
            "citation": "Experimental results showing the modified transformer model improved BLEU scores by 25.3% across 12 language pairs.",
            "expected": "YES",
        },
        {
            "text": "Climate change has accelerated glacial melting in the Arctic [37e79a54-0f8c-4ebd-bf5b-a1a7408b2b06].",
            "citation": "Economic analysis of renewable energy adoption rates in Nordic countries.",
            "expected": "NO",
        },
        {
            "text": "The study revealed significant gender bias in hiring practices [37e79a54-0f8c-4ebd-bf5b-a1a7408b2b06].",
            "citation": "Analysis of 1000+ hiring decisions showing 35% lower callback rates for female candidates with identical qualifications.",
            "expected": "YES",
        },
    ]


@pytest.fixture
def negative_examples() -> List[Dict]:
    """Example sentences with negative assertions."""
    return [
        {
            "text": "The researchers failed to establish a causal relationship between the variables.",
            "expected": "YES",
        },
        {
            "text": "The proposed method was unable to handle edge cases in high-dimensional data.",
            "expected": "YES",
        },
        {
            "text": "Contrary to previous findings, the treatment showed no significant improvement over the placebo.",
            "expected": "YES",
        },
    ]


@pytest.fixture
def citation_requirement_examples() -> List[Dict]:
    """Example sentences that may require citations."""
    return [
        {
            "text": "Recent clinical trials have shown a 45% reduction in tumor size using the novel treatment protocol.",
            "expected": "YES",
        },
        {
            "text": "The experiment was conducted in a controlled laboratory setting at room temperature.",
            "expected": "NO",
        },
        {
            "text": "The proposed algorithm outperforms state-of-the-art methods by 12% on standard benchmarks.",
            "expected": "YES",
        },
    ]


@pytest.fixture
def first_instance_examples() -> List[Dict]:
    """Example sentences to test for first instance detection."""
    return [
        {
            "text": "Deep neural networks exhibit systematic biases in facial recognition tasks.",
            "previous": [],
            "expected": "YES",
        },
        {
            "text": "The facial recognition system shows particularly strong bias against minority groups.",
            "previous": [
                "Deep neural networks exhibit systematic biases in facial recognition tasks."
            ],
            "expected": "YES",
        },
        {
            "text": "These biases in neural networks affect various recognition tasks.",
            "previous": [
                "Deep neural networks exhibit systematic biases in facial recognition tasks.",
                "The facial recognition system shows particularly strong bias against minority groups.",
            ],
            "expected": "NO",
        },
    ]


@pytest.fixture
def nugget_examples() -> List[Dict]:
    """Example sentences with nuggets to test agreement."""
    return [
        {
            "text": "The transformer-based model achieved 94.3% accuracy on the GLUE benchmark.",
            "nugget": "The model's performance on GLUE was 94.3% accurate",
            "expected": "YES",
        },
        {
            "text": "The novel treatment reduced mortality rates by 35% in long-term follow-up studies.",
            "nugget": "Treatment effectiveness showed 35% reduction in death rates",
            "expected": "YES",
        },
        {
            "text": "Researchers found no significant correlation between diet and cancer risk.",
            "nugget": "A strong link was found between dietary habits and cancer incidence",
            "expected": "NO",
        },
    ]
