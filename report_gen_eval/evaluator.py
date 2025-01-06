import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union
import json
from tqdm import tqdm
from langchain_together import ChatTogether
from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage
import sys

# Import all prompts
from .prompts import (
    CHECK_RELEVANCE_SYSTEM, CHECK_RELEVANCE_USER,
    CHECK_NEGATIVE_SYSTEM, CHECK_NEGATIVE_USER,
    REQUIRES_CITATION_SYSTEM, REQUIRES_CITATION_USER,
    FIRST_INSTANCE_SYSTEM, FIRST_INSTANCE_USER,
    NUGGET_AGREEMENT_SYSTEM, NUGGET_AGREEMENT_USER
)

load_dotenv()

class ModelProvider:
    """Model provider configuration."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"

def get_model(provider: str = ModelProvider.TOGETHER, model_name: str = None) -> Any:
    """Get the appropriate model based on provider."""
    if provider == ModelProvider.OPENAI:
        return ChatOpenAI(
            model_name=model_name or "gpt-4o-2024-11-20",
            temperature=0,
            max_tokens=10  # Increased to safely handle YES/NO responses
        )
    elif provider == ModelProvider.ANTHROPIC:
        return ChatAnthropic(
            model_name=model_name or "claude-3-5-sonnet-20241022",
            temperature=0,
            max_tokens=10
        )
    elif provider == ModelProvider.TOGETHER:
        return ChatTogether(
            model=model_name or "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            temperature=0,
            max_tokens=10
        )
    else:
        raise ValueError(f"Unsupported model provider: {provider}")

def get_model_response(system_prompt: str, user_prompt: str, 
                      provider: str = ModelProvider.TOGETHER,
                      model_name: str = None) -> str:
    """Get response from the specified model."""
    try:
        model = get_model(provider, model_name)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = model.invoke(messages)
        response_text = response.content.strip().upper()
        
        if response_text not in ["YES", "NO"]:
            raise ValueError(f"Invalid model response: {response_text}. Expected YES or NO.")
            
        return response_text
    except Exception as e:
        raise RuntimeError(f"Model response error: {str(e)}")

def evaluate_sentence(sentence: str, citation_content: Optional[Union[List[str], Dict[str, str]]] = None, 
                     previous_sentences: Optional[List[str]] = None, 
                     nuggets: Optional[List[Dict[str, Any]]] = None,
                     provider: str = ModelProvider.TOGETHER,
                     model_name: str = None) -> Dict[str, Any]:
    """Evaluate a sentence through all steps of the workflow."""
    results = {
        "sentence": sentence,
        "evaluation_path": [],
        "matched_nuggets": [],
        "score": 0
    }
    
    # Step 1: Check for citations
    has_citations = citation_content is not None and len(citation_content) > 0
    results["has_citations"] = has_citations
    results["evaluation_path"].append(1)
    
    if has_citations:
        # Handle both array and dictionary formats
        citation_ids = citation_content if isinstance(citation_content, list) else list(citation_content.keys())
        
        # For now, since we don't have the actual content, we'll assume citations are relevant
        # This should be updated once you have access to the citation contents
        all_citations_relevant = True
        results["citation_relevance"] = "ASSUMED_RELEVANT"
        
        # If all citations are relevant, check nugget matching
        if all_citations_relevant:
            matched_any_nugget = False
            if nuggets:
                # For each citation, check if it matches any nuggets
                for citation_id in citation_ids:
                    for nugget in nuggets:
                        # Format nugget as question-answer pair for comparison
                        nugget_text = f"Question: {nugget['question']}\nAnswer: {nugget['answer']}"
                        # Check if the current citation matches the nugget's documents
                        if citation_id in nugget['documents_with_nugget']:
                            nugget_agrees = get_model_response(
                                NUGGET_AGREEMENT_SYSTEM,
                                NUGGET_AGREEMENT_USER.format(
                                    sentence=sentence,
                                    nugget=nugget_text
                                ),
                                provider=provider,
                                model_name=model_name
                            )
                            if nugget_agrees == "YES":
                                # Only add if not already matched
                                if nugget not in results["matched_nuggets"]:
                                    results["matched_nuggets"].append(nugget)
                                    results["score"] += 1  # Can get multiple rewards
                                matched_any_nugget = True
            
            # If citations are relevant but don't match any nuggets, ignore (0)
            if not matched_any_nugget:
                results["score"] = 0
    else:
        # No citations path
        is_negative = get_model_response(
            CHECK_NEGATIVE_SYSTEM,
            CHECK_NEGATIVE_USER.format(sentence=sentence),
            provider=provider,
            model_name=model_name
        )
        results["is_negative"] = is_negative == "YES"
        
        if is_negative == "YES":
            # Check nugget agreement for negative assertions
            if nuggets:
                for nugget in nuggets:
                    nugget_text = f"Question: {nugget['question']}\nAnswer: {nugget['answer']}"
                    nugget_agrees = get_model_response(
                        NUGGET_AGREEMENT_SYSTEM,
                        NUGGET_AGREEMENT_USER.format(
                            sentence=sentence,
                            nugget=nugget_text
                        ),
                        provider=provider,
                        model_name=model_name
                    )
                    if nugget_agrees == "YES":
                        results["matched_nuggets"].append(nugget)
                        results["score"] += 1
                    else:
                        results["score"] = -1  # Penalize if no nugget agrees
                        break
            else:
                results["score"] = -1
        else:
            # Check if citation is required
            requires_cite = get_model_response(
                REQUIRES_CITATION_SYSTEM,
                REQUIRES_CITATION_USER.format(sentence=sentence),
                provider=provider,
                model_name=model_name
            )
            results["requires_citation"] = requires_cite == "YES"
            
            if requires_cite == "YES":
                # Check if it's the first instance
                if previous_sentences:
                    is_first = get_model_response(
                        FIRST_INSTANCE_SYSTEM,
                        FIRST_INSTANCE_USER.format(
                            sentence=sentence,
                            previous_sentences="\n".join(previous_sentences)
                        ),
                        provider=provider,
                        model_name=model_name
                    )
                    results["is_first_instance"] = is_first == "YES"
                    results["score"] = -1 if is_first == "YES" else 0
                else:
                    results["is_first_instance"] = True
                    results["score"] = -1
            else:
                results["score"] = 0
    
    return results

def evaluate_report(report: Dict[str, Any], nuggets_file: str = "example_nuggets.jsonl",
                   provider: str = ModelProvider.TOGETHER,
                   model_name: str = None) -> Dict[str, Any]:
    """Evaluate an entire report."""
    results = []
    unique_nuggets_matched = set()  # Using tuple of (question, answer) as unique identifier
    total_sentences = 0
    rewarded_sentences = 0
    penalized_sentences = 0
    
    # Load nuggets if file is provided
    nuggets = None
    if nuggets_file and os.path.exists(nuggets_file):
        try:
            with open(nuggets_file, 'r') as f:
                nuggets_data = json.load(f)
                if str(nuggets_data['request_id']) == str(report['request_id']):
                    nuggets = nuggets_data['nuggets']
                    
                    # Validate nugget format
                    for nugget in nuggets:
                        if not isinstance(nugget, dict) or 'question' not in nugget or 'answer' not in nugget:
                            raise ValueError("Invalid nugget format")
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to load nuggets: {str(e)}")
    
    # Extract all sentences for previous sentence context
    sentences = report['sentences']
    all_sentence_texts = [s['text'] for s in sentences]
    
    # Process sentences in chunks to manage memory
    chunk_size = 50  # Process 50 sentences at a time
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i:i + chunk_size]
        
        for j, sentence_data in enumerate(chunk):
            global_idx = i + j
            previous_sentences = all_sentence_texts[:i] if i > 0 else None
            
            try:
                result = evaluate_sentence(
                    sentence=sentence_data['text'],
                    citation_content=sentence_data.get('citations'),
                    previous_sentences=previous_sentences,
                    nuggets=nuggets,
                    provider=provider,
                    model_name=model_name
                )
                
                # Track unique nuggets and update scores
                if result["matched_nuggets"]:
                    for nugget in result["matched_nuggets"]:
                        unique_nuggets_matched.add((nugget['question'], nugget['answer']))
                
                # Update sentence counts
                if result["score"] != 0:
                    total_sentences += 1
                    if result["score"] > 0:
                        rewarded_sentences += 1
                    elif result["score"] < 0:
                        penalized_sentences += 1
                
                # Add metadata
                result['sentence_index'] = global_idx
                result['citations'] = sentence_data.get('citations', [])
                
                results.append(result)
            except Exception as e:
                print(f"Error processing sentence {global_idx}: {str(e)}", file=sys.stderr)
                results.append({
                    "sentence": sentence_data['text'],
                    "score": 0,
                    "error": str(e)
                })
    
    # Calculate final metrics
    total_nuggets = len(nuggets) if nuggets else 0
    recall = len(unique_nuggets_matched) / total_nuggets if total_nuggets > 0 else 0
    precision = rewarded_sentences / total_sentences if total_sentences > 0 else 0
    
    return {
        "request_id": report['request_id'],
        "run_id": report['run_id'],
        "collection_ids": report['collection_ids'],
        "sentence_results": results,
        "metrics": {
            "recall": recall,
            "precision": precision,
            "unique_nuggets_matched": len(unique_nuggets_matched),
            "total_nuggets": total_nuggets,
            "rewarded_sentences": rewarded_sentences,
            "penalized_sentences": penalized_sentences,
            "total_evaluated_sentences": total_sentences
        }
    } 