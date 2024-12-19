import os
import together
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import json
from tqdm import tqdm

# Import all prompts
from .prompts import (
    check_relevance,
    check_negative,
    requires_citation,
    first_instance,
    nugget_agreement
)

load_dotenv()

# Initialize Together AI
together.api_key = os.getenv("TOGETHER_API_KEY")
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

def get_model_response(system_prompt: str, user_prompt: str) -> str:
    """Get response from Together AI model."""
    client = together.Together()
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=10,
        temperature=0,
        top_k=1,
        top_p=1,
        repetition_penalty=1
    )
    
    return response.choices[0].message.content.strip()

def evaluate_sentence(sentence: str, citation_content: Optional[Dict[str, str]] = None, 
                     previous_sentences: Optional[List[str]] = None, 
                     nuggets: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
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
                check_relevance.SYSTEM_PROMPT,
                check_relevance.USER_PROMPT.format(
                    sentence=sentence,
                    citation_content=content
                )
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
                            nugget_agreement.SYSTEM_PROMPT,
                            nugget_agreement.USER_PROMPT.format(
                                sentence=sentence,
                                nugget=nugget_text
                            )
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
            check_negative.SYSTEM_PROMPT,
            check_negative.USER_PROMPT.format(sentence=sentence)
        )
        results["is_negative"] = is_negative == "YES"
        
        if is_negative == "YES":
            # Check nugget agreement for negative assertions
            if nuggets:
                for nugget in nuggets:
                    nugget_text = f"Question: {nugget['question']}\nAnswer: {nugget['answer']}"
                    nugget_agrees = get_model_response(
                        nugget_agreement.SYSTEM_PROMPT,
                        nugget_agreement.USER_PROMPT.format(
                            sentence=sentence,
                            nugget=nugget_text
                        )
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
                requires_citation.SYSTEM_PROMPT,
                requires_citation.USER_PROMPT.format(sentence=sentence)
            )
            results["requires_citation"] = requires_cite == "YES"
            
            if requires_cite == "YES":
                # Check if it's the first instance
                if previous_sentences:
                    is_first = get_model_response(
                        first_instance.SYSTEM_PROMPT,
                        first_instance.USER_PROMPT.format(
                            sentence=sentence,
                            previous_sentences="\n".join(previous_sentences)
                        )
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
    unique_nuggets_matched = set()
    total_sentences = 0
    rewarded_sentences = 0
    penalized_sentences = 0
    
    # Load nuggets for this request_id
    nuggets = None
    if request_id and nuggets_file:
        with open(nuggets_file, 'r') as f:
            nuggets_data = json.load(f)
            if str(nuggets_data['request_id']) == str(request_id):
                nuggets = nuggets_data['nuggets']
    
    for i, sentence in enumerate(tqdm(sentences)):
        previous_sentences = sentences[:i] if i > 0 else None
        citation_content = citation_contents.get(str(i)) if citation_contents else None
        
        result = evaluate_sentence(
            sentence=sentence,
            citation_content=citation_content,
            previous_sentences=previous_sentences,
            nuggets=nuggets
        )
        
        # Track unique nuggets for recall
        if result["matched_nuggets"]:
            # Use question-answer pair as unique identifier for nugget
            unique_nuggets_matched.update(
                (nugget['question'], nugget['answer']) 
                for nugget in result["matched_nuggets"]
            )
        
        # Track sentences for precision
        if result["score"] != 0:  # Don't count ignored sentences
            total_sentences += 1
            if result["score"] > 0:
                rewarded_sentences += 1
            elif result["score"] < 0:
                penalized_sentences += 1
        
        results.append(result)
    
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
            "total_scored_sentences": total_sentences,
            "penalized_sentences": penalized_sentences
        }
    }

if __name__ == "__main__":
    # Example usage
    sentences = [
        "Recent studies have shown that AI models can exhibit biases [1].",
        "This phenomenon has been extensively documented in the literature (Smith et al., 2023).",
        "Machine learning systems require careful evaluation."
    ]
    
    citation_contents = {
        "0": {"1": "A comprehensive study of AI model biases across different domains..."},
        "1": {"Smith2023": "Analysis of bias patterns in large language models..."}
    }
    
    # Example request_id matching the nuggets file
    request_id = "300"
    
    results = evaluate_report(sentences, citation_contents, request_id)
    print(json.dumps(results, indent=2)) 