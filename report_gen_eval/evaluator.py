"""
Report Generation Evaluator

This module implements the evaluation framework from "On the Evaluation of Machine-Generated Reports"
(Mayfield et al., SIGIR 2024). It systematically evaluates AI-generated reports based on:
1. Citation usage
2. Factual accuracy
3. Information coverage

The evaluation follows these rules for each sentence:

For Sentences With Citations:
1. Check if cited documents support the claim:
   - Penalize (-1) if any document doesn't support
   - Continue to step 2 if all documents support
2. Check nugget matching:
   - Reward (+1) for each nugget correctly answered
   - Ignore (0) if no nuggets matched

For Sentences Without Citations:
1. For negative statements ("X is not true"):
   - Reward (+1) if a nugget confirms
   - Penalize (-1) if no nugget supports
2. For statements requiring citations:
   - Penalize (-1) if first occurrence
   - Ignore (0) if previously cited
3. For statements not requiring citations:
   - Ignore (0)

The framework calculates two primary metrics:
- Recall = (Unique nuggets correctly reported) / (Total nuggets)
- Precision = (Rewarded sentences) / (Total scored sentences)
"""

import os
import traceback
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union
import json
from tqdm import tqdm
import logging

# Import utility functions
from .utils import (
    ModelProvider,
    get_model_response,
    modify_model_response,
    get_text_from_id_fast,
    batch_model_responses,
    load_jsonl,
    save_jsonl,
)

# Import all prompts
from .prompts import (
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

# Configure logging
logger = logging.getLogger(__name__)

load_dotenv()


def check_citations_relevance(
    sentence: str,
    citation_texts: List[str],
    provider: str = ModelProvider.TOGETHER,
    model_name: str = None,
) -> bool:
    """Check if all citations are relevant to a sentence.

    This function is part of the citation validation step in the evaluation framework.
    It checks whether each cited document actually supports the claim made in the sentence.

    Args:
        sentence: The sentence to check
        citation_texts: List of citation texts to check against
        provider: The model provider to use
        model_name: Optional specific model name

    Returns:
        True if all citations are relevant, False if any citation is irrelevant
    """
    user_prompts = [
        CHECK_RELEVANCE_USER.format(sentence=sentence, citation_content=doc_text)
        for doc_text in citation_texts
    ]

    responses = batch_model_responses(
        CHECK_RELEVANCE_SYSTEM, user_prompts, provider, model_name
    )
    return all(response == "YES" for response in responses)


def check_nugget_matches(
    sentence: str,
    nuggets: List[Dict[str, Any]],
    provider: str = ModelProvider.TOGETHER,
    model_name: str = None,
) -> List[Dict[str, Any]]:
    """Check which nuggets match a sentence.

    This function is part of the information coverage evaluation step.
    It determines which evaluation nuggets (question-answer pairs) are correctly
    addressed by the sentence.

    Args:
        sentence: The sentence to check
        nuggets: List of nuggets to check against
        provider: The model provider to use
        model_name: Optional specific model name

    Returns:
        List of matched nugget dictionaries, each containing:
        - question_text: The nugget question
        - matched_answer: The specific answer that matched
        - importance: The importance level of the nugget

    Note:
        A sentence can match multiple nuggets, but each nugget is only counted once
        even if multiple gold answers match.
    """
    user_prompts = []
    nugget_map = []  # Keep track of which prompt corresponds to which nugget/answer

    for nugget in nuggets:
        for answer in nugget["gold_answers"]:
            user_prompts.append(
                NUGGET_AGREEMENT_USER.format(
                    sentence=sentence,
                    nugget_question=nugget["question_text"],
                    nugget_answer=answer,
                )
            )
            nugget_map.append(
                {
                    "question_text": nugget["question_text"],
                    "matched_answer": answer,
                    "importance": nugget["info"]["importance"],
                }
            )

    if not user_prompts:
        return []

    responses = batch_model_responses(
        NUGGET_AGREEMENT_SYSTEM, user_prompts, provider, model_name
    )

    matched_nuggets = []
    for response, nugget_info in zip(responses, nugget_map):
        if response == "YES":
            # Only add if not already matched, don't want duplicates
            nugget_key = (nugget_info["question_text"], nugget_info["matched_answer"])
            if nugget_key not in [
                (n["question_text"], n["matched_answer"]) for n in matched_nuggets
            ]:
                matched_nuggets.append(nugget_info)

    return matched_nuggets


def evaluate_sentence(
    sentence: str,
    citation_content: Optional[List[str]] = None,
    previous_sentences: Optional[List[str]] = None,
    nuggets: Optional[List[Dict[str, Any]]] = None,
    provider: str = ModelProvider.TOGETHER,
    model_name: str = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single sentence according to the evaluation framework.

    The evaluation follows a decision tree based on whether the sentence has citations:
    1. For sentences with citations:
       - First checks if citations support the claim
       - Then checks for nugget matches if citations are relevant
    2. For sentences without citations:
       - Handles negative statements differently
       - Checks if citations are required
       - Considers if it's the first instance of a claim

    Args:
        sentence: The sentence to evaluate
        citation_content: List of citation texts
        previous_sentences: List of sentences that came before this one (for first instance checking)
        nuggets: List of nuggets to check against
        provider: The model provider to use
        model_name: Optional specific model name
        verbose: Whether to log debug information

    Returns:
        Dictionary containing:
        - sentence: The original sentence
        - evaluation_path: List of steps taken in evaluation
        - matched_nuggets: List of nuggets that were matched
        - score: The final score
        - citation_details: Information about citations and their relevance
        - evaluation_details: Details about the evaluation process
    """
    if verbose:
        logger.debug(f"Evaluating sentence: {sentence[:100]}...")

    results = {
        "sentence": sentence,
        "evaluation_path": [],
        "matched_nuggets": [],
        "score": 0,
        "citation_details": {
            "has_citations": False,
            "citation_texts": [],
            "citation_relevance": None,
        },
        "evaluation_details": {
            "is_negative": None,
            "requires_citation": None,
            "is_first_instance": None,
            "model_responses": [],
        },
    }

    # Step 1: Check for citations
    has_citations = citation_content is not None and len(citation_content) > 0
    if verbose:
        logger.debug(f"Has citations: {has_citations}")
        if has_citations:
            logger.debug(f"Citations: {citation_content}")

    results["citation_details"]["has_citations"] = has_citations
    results["evaluation_path"].append(1)

    if has_citations:
        # Process sentences with citations
        citation_texts = citation_content
        results["citation_details"]["citation_texts"] = citation_texts

        # Check if citations support the claim
        all_citations_relevant = check_citations_relevance(
            sentence, citation_texts, provider, model_name
        )

        if not all_citations_relevant:
            results["citation_details"]["citation_relevance"] = "NOT_RELEVANT"
            results["evaluation_details"]["model_responses"].append(
                {
                    "type": "citation_relevance",
                    "response": "NO",
                    "context": {"num_citations": len(citation_texts)},
                }
            )
            results["score"] = -1  # Penalize if any document doesn't support the claim
            return results

        results["citation_details"]["citation_relevance"] = "RELEVANT"
        results["evaluation_details"]["model_responses"].append(
            {
                "type": "citation_relevance",
                "response": "YES",
                "context": {"num_citations": len(citation_texts)},
            }
        )

        # Step 2: Batch check all nugget matches
        if nuggets:
            matched_nuggets = check_nugget_matches(
                sentence, nuggets, provider, model_name
            )
            results["matched_nuggets"] = matched_nuggets
            if matched_nuggets:
                results["score"] = len(
                    matched_nuggets
                )  # Reward for each matched nugget
            else:
                results["score"] = 0  # Ignore if no nuggets are matched
    else:
        # Process sentences without citations
        is_negative = get_model_response(
            CHECK_NEGATIVE_SYSTEM,
            CHECK_NEGATIVE_USER.format(sentence=sentence),
            provider=provider,
            model_name=model_name,
        )
        is_negative = modify_model_response(is_negative)
        results["evaluation_details"]["is_negative"] = is_negative == "YES"
        results["evaluation_details"]["model_responses"].append(
            {"type": "check_negative", "response": is_negative}
        )

        if is_negative == "YES":
            # For negative statements, batch check all nugget matches
            if nuggets:
                matched_nuggets = check_nugget_matches(
                    sentence, nuggets, provider, model_name
                )
                results["matched_nuggets"] = matched_nuggets
                if matched_nuggets:
                    results["score"] = (
                        1  # Reward if any nugget confirms negative statement
                    )
                else:
                    results[
                        "score"
                    ] = -1  # Penalize if no nugget supports negative claim
        else:
            # For non-negative statements without citations
            requires_cite = get_model_response(
                REQUIRES_CITATION_SYSTEM,
                REQUIRES_CITATION_USER.format(sentence=sentence),
                provider=provider,
                model_name=model_name,
            )
            requires_cite = modify_model_response(requires_cite)
            results["evaluation_details"]["requires_citation"] = requires_cite == "YES"
            results["evaluation_details"]["model_responses"].append(
                {"type": "requires_citation", "response": requires_cite}
            )

            if requires_cite == "YES":
                # Check if it's the first instance (this must be sequential)
                if previous_sentences:
                    is_first = get_model_response(
                        FIRST_INSTANCE_SYSTEM,
                        FIRST_INSTANCE_USER.format(
                            sentence=sentence,
                            previous_sentences="\n".join(previous_sentences),
                        ),
                        provider=provider,
                        model_name=model_name,
                    )
                    is_first = modify_model_response(is_first)
                    results["evaluation_details"]["is_first_instance"] = (
                        is_first == "YES"
                    )
                    results["evaluation_details"]["model_responses"].append(
                        {
                            "type": "first_instance",
                            "response": is_first,
                            "context": {
                                "num_previous_sentences": len(previous_sentences)
                            },
                        }
                    )
                    results["score"] = (
                        -1 if is_first == "YES" else 0
                    )  # Penalize first occurrence, ignore repeats
                else:
                    results["evaluation_details"]["is_first_instance"] = True
                    results["score"] = -1  # Penalize first occurrence
            else:
                results["score"] = 0  # Ignore statements not requiring citations

    if verbose:
        logger.debug(f"Sentence evaluation complete. Score: {results['score']}")
    return results


def evaluate_report(
    report: Dict[str, Any],
    nuggets_file: str = "example_nuggets.jsonl",
    provider: str = ModelProvider.TOGETHER,
    model_name: str = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Evaluate an entire report according to the evaluation framework.

    Processes each sentence in the report sequentially (required for first instance checking)
    and calculates overall metrics.

    Args:
        report: The report to evaluate, containing sentences and metadata
        nuggets_file: Path to the JSONL file containing evaluation nuggets
        provider: The model provider to use
        model_name: Optional specific model name
        verbose: Whether to log debug information

    Returns:
        Dictionary containing:
        - request_id: The original request ID
        - run_id: The run ID
        - collection_ids: The collection IDs
        - sentence_results: List of results for each sentence
        - metrics: Overall metrics including recall and precision
        - citation_documents: Map of citation texts to their indices
    """
    if verbose:
        logger.info(
            f"Starting evaluation of report {report.get('request_id', 'unknown')}"
        )

    results = []
    unique_nuggets_matched = set()
    total_sentences = 0
    rewarded_sentences = 0
    penalized_sentences = 0
    citation_documents = {}  # Store all citation texts

    # Load nuggets if file is provided
    nuggets = None
    if nuggets_file and os.path.exists(nuggets_file):
        if verbose:
            logger.info(f"Loading nuggets from {nuggets_file}")
        try:
            with open(nuggets_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    nuggets_data = json.loads(line)
                    if str(nuggets_data["query_id"]) == str(report["request_id"]):
                        if verbose:
                            logger.info(
                                f"Found matching nuggets for report {report['request_id']}"
                            )
                            logger.info(f"Loaded {len(nuggets_data['items'])} nuggets")
                        nuggets = nuggets_data["items"]
                        break
                if nuggets is None and verbose:
                    logger.warning(
                        f"No matching nuggets found for report {report['request_id']}"
                    )
        except Exception as e:
            if verbose:
                logger.error(f"Failed to load nuggets: {str(e)}")
            raise

    # Extract all sentences
    sentences = report["sentences"]
    all_sentence_texts = [s["text"] for s in sentences]
    if verbose:
        logger.info(f"Processing {len(sentences)} sentences")

    # Process sentences sequentially (required for first instance checking)
    for i, sentence_data in enumerate(sentences):
        if verbose:
            logger.info(f"Processing sentence {i+1}/{len(sentences)}")
        try:
            # Extract citation texts from the sentence data
            citation_texts = []
            if "citations" in sentence_data and sentence_data["citations"]:
                if isinstance(sentence_data["citations"], list):
                    for doc_id in sentence_data["citations"]:
                        # For each document ID, get the text from all possible collections
                        for collection_id in report["collection_ids"]:
                            title, text = get_text_from_id_fast(doc_id, collection_id)
                            if title is not None and text is not None:
                                doc_text = f"Title: {title}\n\nContent: {text}"
                                citation_texts.append(doc_text)
                                break  # Found the document, no need to check other collections
                    # assert we found all the citations
                    assert (
                        len(citation_texts) == len(sentence_data["citations"])
                    ), f"Expected {len(sentence_data['citations'])} citations, but found {len(citation_texts)}"
                else:
                    logger.warning(f"Unexpected citation format in sentence {i+1}")

            result = evaluate_sentence(
                sentence=sentence_data["text"],
                citation_content=citation_texts if citation_texts else None,
                previous_sentences=all_sentence_texts[:i] if i > 0 else None,
                nuggets=nuggets,
                provider=provider,
                model_name=model_name,
                verbose=verbose,
            )

            # Update metrics
            if result["matched_nuggets"]:
                if verbose:
                    logger.debug(
                        f"Sentence {i+1} matched {len(result['matched_nuggets'])} nuggets"
                    )
                for nugget in result["matched_nuggets"]:
                    unique_nuggets_matched.add(
                        (nugget["question_text"], nugget["matched_answer"])
                    )

            if result["score"] != 0:
                total_sentences += 1
                if result["score"] > 0:
                    rewarded_sentences += 1
                elif result["score"] < 0:
                    penalized_sentences += 1

            # Store citation texts if present
            if result["citation_details"]["citation_texts"]:
                citation_indices = []  # Track which citations were used for this sentence
                for text in result["citation_details"]["citation_texts"]:
                    # Find existing citation or create new one
                    citation_key = None
                    for key, existing_text in citation_documents.items():
                        if text == existing_text:
                            citation_key = key
                            break

                    if citation_key is None:
                        citation_key = f"citation_{len(citation_documents)}"
                        citation_documents[citation_key] = text

                    citation_indices.append(citation_key)

                # Store the citation indices with the result
                result["citation_indices"] = citation_indices

            result["sentence_index"] = i
            results.append(result)

        except Exception as e:
            if verbose:
                logger.error(f"Error processing sentence {i+1}: {str(e)}")
                # add traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
            results.append(
                {
                    "sentence": sentence_data["text"],
                    "sentence_index": i,
                    "score": 0,
                    "error": str(e),
                    "citation_details": {
                        "has_citations": bool(citation_texts),
                        "citation_texts": citation_texts if citation_texts else [],
                    },
                }
            )

    # Calculate overall metrics
    total_nuggets = (
        sum(len(nugget["gold_answers"]) for nugget in nuggets) if nuggets else 0
    )
    recall = len(unique_nuggets_matched) / total_nuggets if total_nuggets > 0 else 0
    precision = rewarded_sentences / total_sentences if total_sentences > 0 else 0

    if verbose:
        logger.info(
            f"Evaluation complete. Metrics: recall={recall:.2f}, precision={precision:.2f}"
        )
        logger.info(f"Matched {len(unique_nuggets_matched)}/{total_nuggets} nuggets")
        logger.info(
            f"Sentences: {rewarded_sentences} rewarded, {penalized_sentences} penalized, {total_sentences} total"
        )

    return {
        "request_id": report["request_id"],
        "run_id": report["run_id"],
        "collection_ids": report["collection_ids"],
        "sentence_results": results,
        "metrics": {
            "recall": recall,
            "precision": precision,
            "unique_nuggets_matched": len(unique_nuggets_matched),
            "total_nuggets": total_nuggets,
            "rewarded_sentences": rewarded_sentences,
            "penalized_sentences": penalized_sentences,
            "total_evaluated_sentences": total_sentences,
        },
        "citation_documents": citation_documents,
    }
