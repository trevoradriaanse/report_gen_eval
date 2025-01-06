import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import json
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI, ChatAnthropic, ChatTogetherAI
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
        return ChatTogetherAI(
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

def evaluate_sentence(sentence: str, citation_content: Optional[Dict[str, str]] = None, 
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
        # Step 2: Check citation relevance for each citation
        all_citations_relevant = True
        for citation_id, content in citation_content.items():
            relevance = get_model_response(
                CHECK_RELEVANCE_SYSTEM,
                CHECK_RELEVANCE_USER.format(
                    sentence=sentence,
                    citation_content=content
                ),
                provider=provider,
                model_name=model_name
            )
            results[f"citation_{citation_id}_relevant"] = relevance == "YES"
            if relevance == "NO":
                results["score"] = -1
                all_citations_relevant = False
                return results
        
        # If all citations are relevant, check nugget matching
        if all_citations_relevant:
            matched_any_nugget = False
            if nuggets:
                for nugget in nuggets:
                    # Format nugget as question-answer pair for comparison
                    nugget_text = f"Question: {nugget['question']}\nAnswer: {nugget['answer']}"
                    # Check if any of the cited documents match the nugget's documents
                    cited_docs = set(citation_content.keys())
                    nugget_docs = set(nugget['documents_with_nugget'])
                    if cited_docs.intersection(nugget_docs):
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
                results["score"] = 0  # Ignore sentences not requiring citation
    
    return results

def evaluate_report(sentences: List[str], citation_contents: Optional[Dict[str, Dict[str, str]]] = None,
                   request_id: str = None, nuggets_file: str = "example_nuggets.jsonl") -> Dict[str, Any]:
    """Evaluate an entire report."""
    results = []
    unique_nuggets_matched = set()  # Using tuple of (question, answer) as unique identifier
    total_sentences = 0
    rewarded_sentences = 0
    penalized_sentences = 0
    
    # Load nuggets for this request_id
    nuggets = None
    if request_id and nuggets_file and os.path.exists(nuggets_file):
        try:
            with open(nuggets_file, 'r') as f:
                nuggets_data = json.load(f)
                if str(nuggets_data['request_id']) == str(request_id):
                    nuggets = nuggets_data['nuggets']
                    
                    # Validate nugget format
                    for nugget in nuggets:
                        if not isinstance(nugget, dict) or 'question' not in nugget or 'answer' not in nugget:
                            raise ValueError("Invalid nugget format")
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to load nuggets: {str(e)}")
    
    # Process sentences in chunks to manage memory
    chunk_size = 50  # Process 50 sentences at a time
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i:i + chunk_size]
        
        for j, sentence in enumerate(chunk):
            global_idx = i + j
            previous_sentences = sentences[max(0, global_idx-10):global_idx] if global_idx > 0 else None
            
            try:
                result = evaluate_sentence(
                    sentence=sentence,
                    citation_content=citation_contents.get(str(global_idx)) if citation_contents else None,
                    previous_sentences=previous_sentences,
                    nuggets=nuggets
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
                
                results.append(result)
            except Exception as e:
                print(f"Error processing sentence {global_idx}: {str(e)}", file=sys.stderr)
                results.append({
                    "sentence": sentence,
                    "score": 0,
                    "error": str(e)
                })
    
    # Calculate final metrics
    total_nuggets = len(nuggets) if nuggets else 0
    recall = len(unique_nuggets_matched) / total_nuggets if total_nuggets > 0 else 0
    precision = rewarded_sentences / total_sentences if total_sentences > 0 else 0
    
    return {
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