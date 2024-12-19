"""Command line interface for report evaluation."""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

from .evaluate_report import evaluate_sentence

def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], file_path: str):
    """Save data as JSONL."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def process_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single report."""
    results = []
    sentences = report['sentences']
    
    # Extract all sentences for previous sentence context
    all_sentence_texts = [s['text'] for s in sentences]
    
    for i, sentence_data in enumerate(sentences):
        # Get previous sentences for context
        previous_sentences = all_sentence_texts[:i] if i > 0 else None
        
        # Process the sentence
        result = evaluate_sentence(
            sentence=sentence_data['text'],
            citation_content=sentence_data.get('citations'),
            previous_sentences=previous_sentences,
            nugget=sentence_data.get('nugget')
        )
        
        # Add metadata
        result['sentence_index'] = i
        result['citations'] = sentence_data.get('citations', [])
        
        results.append(result)
    
    return {
        'request_id': report['request_id'],
        'run_id': report['run_id'],
        'collection_ids': report['collection_ids'],
        'results': results
    }

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description='Evaluate report sentences from JSONL file')
    parser.add_argument('input_file', type=str, help='Path to input JSONL file')
    parser.add_argument('output_dir', type=str, help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of reports to process in parallel')
    args = parser.parse_args()
    
    # Load input data
    print(f"Loading data from {args.input_file}")
    reports = load_jsonl(args.input_file)
    
    # Process reports
    print(f"Processing {len(reports)} reports...")
    results = []
    for report in tqdm(reports):
        result = process_report(report)
        results.append(result)
    
    # Save results
    output_file = Path(args.output_dir) / f"results_{Path(args.input_file).stem}.jsonl"
    print(f"Saving results to {output_file}")
    save_jsonl(results, str(output_file))

if __name__ == '__main__':
    main() 