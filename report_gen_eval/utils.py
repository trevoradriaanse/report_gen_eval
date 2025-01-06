"""Utility functions for the report generation evaluator.

This module provides general utility functions used across the codebase, including:
1. Model management and interaction
2. File I/O operations
3. Document lookup functionality
4. Error handling and retry logic
"""

import logging
import os
import time
from random import uniform
import pickle
import json
from typing import Any, List, Optional, Dict, Tuple
from pathlib import Path
from langchain.schema import SystemMessage, HumanMessage
from langchain_together import ChatTogether
from langchain_community.chat_models import ChatOpenAI, ChatAnthropic

logger = logging.getLogger(__name__)

class ModelProvider:
    """Model provider configuration.
    
    Supported providers:
    - OPENAI: OpenAI's models (e.g., GPT-4)
    - ANTHROPIC: Anthropic's models (e.g., Claude)
    - TOGETHER: Together.ai's models (e.g., Llama)
    """
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"

def get_model(provider: str = ModelProvider.TOGETHER, model_name: str = None) -> Any:
    """Get the appropriate model based on provider.
    
    Args:
        provider: The model provider to use (openai, anthropic, or together)
        model_name: Optional specific model name to use
        
    Returns:
        A configured LangChain chat model instance
        
    Raises:
        ValueError: If an unsupported provider is specified
    """
    if provider == ModelProvider.OPENAI:
        return ChatOpenAI(
            model_name=model_name or "gpt-4o-2024-11-20",
            temperature=0,
            max_tokens=10
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
                      model_name: str = None,
                      max_retries: int = 3,
                      base_delay: float = 2.0) -> str:
    """Get a single YES/NO response from the specified model with retry logic.
    
    Args:
        system_prompt: The system prompt to use
        user_prompt: The user prompt to use
        provider: The model provider to use
        model_name: Optional specific model name
        max_retries: Maximum number of retries on failure
        base_delay: Base delay between retries (uses exponential backoff)
        
    Returns:
        The model's response as "YES" or "NO"
        
    Raises:
        RuntimeError: If max retries exceeded or invalid response received
    """
    for attempt in range(max_retries):
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
            if "429" in str(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + uniform(0, 0.1)
                logger.warning(f"Rate limit hit. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
            elif attempt < max_retries - 1:
                delay = base_delay + uniform(0, 0.1)
                logger.warning(f"Error: {str(e)}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
            raise RuntimeError(f"Model response error after {max_retries} attempts: {str(e)}")

def get_text_from_id_fast(docid: str, collection: str) -> Tuple[Optional[str], Optional[str]]:
    """Get document text from ID using cached lookup.
    
    Args:
        docid: The document ID to look up
        collection: The collection ID (e.g., 'neuclir/1/zh')
        
    Returns:
        A tuple of (title, text) if found, (None, None) if not found
        
    Note:
        Uses an in-memory cache to avoid repeated disk reads.
        The cache is shared across all instances using the function.
    """
    collection = collection.replace('zho', 'zh').replace('fas', 'fa').replace('rus', 'ru')
    mapping_file = f"neuclir-docs-lookup/doc_mapping_{collection.replace('/', '_')}.pkl"
    
    # Load mapping if not already loaded
    if not hasattr(get_text_from_id_fast, 'cache'):
        get_text_from_id_fast.cache = {}
    
    if collection not in get_text_from_id_fast.cache:
        with open(mapping_file, 'rb') as f:
            get_text_from_id_fast.cache[collection] = pickle.load(f)
    
    doc = get_text_from_id_fast.cache[collection].get(docid)
    if doc:
        return doc['title'], doc['text']
    return None, None

def batch_model_responses(system_prompt: str, user_prompts: List[str],
                         provider: str = ModelProvider.TOGETHER,
                         model_name: str = None,
                         max_retries: int = 3,
                         base_delay: float = 2.0) -> List[str]:
    """Get multiple YES/NO responses from the model in batches.
    
    Processes prompts in batches to avoid rate limits while maintaining efficiency.
    
    Args:
        system_prompt: The system prompt to use for all queries
        user_prompts: List of user prompts to process
        provider: The model provider to use
        model_name: Optional specific model name
        max_retries: Maximum number of retries on failure
        base_delay: Base delay between retries (uses exponential backoff)
        
    Returns:
        List of "YES"/"NO" responses matching the input prompts order
        
    Note:
        Uses batching to reduce API calls and includes rate limit handling.
        A small delay is added between batches to avoid overwhelming the API.
    """
    for attempt in range(max_retries):
        try:
            model = get_model(provider, model_name)
            responses = []
            
            # Process in batches of 10 to avoid rate limits
            batch_size = 10
            for i in range(0, len(user_prompts), batch_size):
                batch = user_prompts[i:i + batch_size]
                batch_responses = []
                
                for user_prompt in batch:
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt)
                    ]
                    response = model.invoke(messages)
                    response_text = response.content.strip().upper()
                    
                    if response_text not in ["YES", "NO"]:
                        raise ValueError(f"Invalid model response: {response_text}. Expected YES or NO.")
                    
                    batch_responses.append(response_text)
                
                responses.extend(batch_responses)
                if i + batch_size < len(user_prompts):  # If not the last batch
                    time.sleep(0.5)  # Small delay between batches
                    
            return responses
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + uniform(0, 0.1)
                logger.warning(f"Rate limit hit. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
            elif attempt < max_retries - 1:
                delay = base_delay + uniform(0, 0.1)
                logger.warning(f"Error: {str(e)}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                continue
            raise RuntimeError(f"Model response error after {max_retries} attempts: {str(e)}")

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file to load
        
    Returns:
        List of dictionaries, one per line in the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], file_path: str):
    """Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path where the file should be saved
        
    Note:
        Creates parent directories if they don't exist.
        Uses UTF-8 encoding and preserves non-ASCII characters.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n') 