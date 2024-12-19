import os
import together
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from .prompts import (
    HAS_CITATIONS_SYSTEM, HAS_CITATIONS_USER,
    CHECK_RELEVANCE_SYSTEM, CHECK_RELEVANCE_USER,
    CHECK_NEGATIVE_SYSTEM, CHECK_NEGATIVE_USER,
    REQUIRES_CITATION_SYSTEM, REQUIRES_CITATION_USER,
    FIRST_INSTANCE_SYSTEM, FIRST_INSTANCE_USER,
    NUGGET_AGREEMENT_SYSTEM, NUGGET_AGREEMENT_USER
)

load_dotenv()

# Initialize Together AI
together.api_key = os.getenv("TOGETHER_API_KEY")
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
client = together.Together()


def get_model_response(system_prompt: str, user_prompt: str) -> str:
    """Get response from Together AI model."""    
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        model=MODEL,
        max_tokens=10,
        temperature=0,
        top_k=1,
        top_p=1,
        repetition_penalty=1,
        stop=['</s>']
    )
    
    return response.choices[0].message.content


def evaluate_sentence(
    sentence: str,
    citation_content: Optional[Dict[str, str]] = None,
    previous_sentences: Optional[List[str]] = None,
    nugget: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate a sentence through all steps of the workflow."""
    results = {"sentence": sentence, "evaluation_path": []}
    
    # Step 1: Check for citations
    has_citation = get_model_response(
        HAS_CITATIONS_SYSTEM,
        HAS_CITATIONS_USER.format(sentence=sentence)
    )
    results["has_citations"] = has_citation == "YES"
    results["evaluation_path"].append(1)
    
    if has_citation == "YES" and citation_content:
        # Step 2: Check citation relevance
        for citation_id, content in citation_content.items():
            relevance = get_model_response(
                CHECK_RELEVANCE_SYSTEM,
                CHECK_RELEVANCE_USER.format(
                    sentence=sentence,
                    citation_content=content
                )
            )
            results[f"citation_{citation_id}_relevant"] = relevance == "YES"
            if relevance == "NO":
                results["score"] = -1
                return results
    
    # Step 3: Check for negative assertion
    is_negative = get_model_response(
        CHECK_NEGATIVE_SYSTEM,
        CHECK_NEGATIVE_USER.format(sentence=sentence)
    )
    results["is_negative"] = is_negative == "YES"
    results["evaluation_path"].append(3)
    
    if is_negative == "YES":
        # Step 4: Check if nugget agrees
        if nugget:
            nugget_agrees = get_model_response(
                NUGGET_AGREEMENT_SYSTEM,
                NUGGET_AGREEMENT_USER.format(
                    sentence=sentence,
                    nugget=nugget
                )
            )
            results["nugget_agrees"] = nugget_agrees == "YES"
            results["score"] = 1 if nugget_agrees == "YES" else -1
            return results
        else:
            results["score"] = -1
            return results
    
    # Step 5: Check if citation is required
    requires_cite = get_model_response(
        REQUIRES_CITATION_SYSTEM,
        REQUIRES_CITATION_USER.format(sentence=sentence)
    )
    results["requires_citation"] = requires_cite == "YES"
    
    if requires_cite == "NO":
        results["score"] = 0
        return results
    
    # Step 6: Check if it's the first instance
    if previous_sentences:
        is_first = get_model_response(
            FIRST_INSTANCE_SYSTEM,
            FIRST_INSTANCE_USER.format(
                sentence=sentence,
                previous_sentences="\n".join(previous_sentences)
            )
        )
        results["is_first_instance"] = is_first == "YES"
        results["score"] = -1 if is_first == "NO" else 0
    else:
        results["is_first_instance"] = True
        results["score"] = 0
    
    return results

def evaluate_report(
    sentences: List[str],
    citation_contents: Optional[Dict[str, Dict[str, str]]] = None,
    nuggets: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """Evaluate an entire report."""
    results = []
    for i, sentence in enumerate(sentences):
        previous_sentences = sentences[:i] if i > 0 else None
        citation_content = citation_contents.get(str(i)) if citation_contents else None
        nugget = nuggets.get(str(i)) if nuggets else None
        
        result = evaluate_sentence(
            sentence=sentence,
            citation_content=citation_content,
            previous_sentences=previous_sentences,
            nugget=nugget
        )
        results.append(result)
    
    return results 