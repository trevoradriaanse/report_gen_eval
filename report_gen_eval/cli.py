"""Command line interface for report evaluation."""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from .evaluator import evaluate_sentence, ModelProvider

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

def process_report(report: Dict[str, Any], nuggets_file: str = None, 
                  model_provider: str = ModelProvider.TOGETHER,
                  model_name: str = None) -> Dict[str, Any]:
    """Process a single report."""
    # Validate required fields
    required_fields = ['request_id', 'run_id', 'collection_ids', 'sentences']
    missing_fields = [field for field in required_fields if field not in report]
    if missing_fields:
        raise ValueError(f"Missing required fields in report: {', '.join(missing_fields)}")
    
    results = []
    sentences = report['sentences']
    
    # Validate sentences structure
    if not isinstance(sentences, list):
        raise ValueError("'sentences' must be a list")
    
    for i, sentence in enumerate(sentences):
        if not isinstance(sentence, dict) or 'text' not in sentence:
            raise ValueError(f"Invalid sentence structure at index {i}: missing 'text' field")
        
        # Validate citations if present
        citations = sentence.get('citations', [])
        if citations and not isinstance(citations, (dict, list)):
            raise ValueError(f"Invalid citations format at sentence index {i}: must be a dictionary or list")
    
    # Load nuggets if file is provided
    nuggets = None
    if nuggets_file and os.path.exists(nuggets_file):
        try:
            with open(nuggets_file, 'r') as f:
                nuggets_data = json.load(f)
                if not isinstance(nuggets_data, dict) or 'request_id' not in nuggets_data or 'nuggets' not in nuggets_data:
                    raise ValueError("Invalid nuggets file format")
                if str(nuggets_data['request_id']) == str(report['request_id']):
                    nuggets = nuggets_data['nuggets']
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse nuggets file: {str(e)}")
    
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
            nuggets=nuggets,
            provider=model_provider,
            model_name=model_name
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
    parser.add_argument('nuggets_file', type=str, help='Path to nuggets file containing evaluation criteria')
    parser.add_argument('output_dir', type=str, help='Directory to save results')
    parser.add_argument('-b', '--batch-size', type=int, default=10,
                      help='Number of reports to process in parallel (default: 10)')
    parser.add_argument('-p', '--model-provider', type=str, 
                      choices=['openai', 'anthropic', 'together'],
                      default='together', 
                      help='Model provider to use (default: together)')
    parser.add_argument('-m', '--model-name', type=str,
                      help='Specific model name to use (defaults to provider-specific default)')
    args = parser.parse_args()
    
    try:
        # Validate input files exist
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        if not os.path.exists(args.nuggets_file):
            raise FileNotFoundError(f"Nuggets file not found: {args.nuggets_file}")
        
        # Validate output directory
        try:
            os.makedirs(args.output_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create output directory {args.output_dir}: {str(e)}")
        
        # Load input data
        print(f"Loading data from {args.input_file}")
        try:
            reports = load_jsonl(args.input_file)
            if not reports:
                raise ValueError("Input file contains no reports")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse input file {args.input_file}: {str(e)}")
        
        # Validate batch size
        if args.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        if args.batch_size > len(reports):
            print(f"Warning: Batch size {args.batch_size} is larger than number of reports {len(reports)}. Using {len(reports)} workers.")
            args.batch_size = len(reports)
        
        # Process reports in batches
        print(f"Processing {len(reports)} reports in batches of {args.batch_size}...")
        results = []
        failed_reports = []
        
        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            # Submit all reports for processing
            future_to_report = {
                executor.submit(
                    process_report,
                    report,
                    nuggets_file=args.nuggets_file,
                    model_provider=args.model_provider,
                    model_name=args.model_name
                ): report for report in reports
            }
            
            # Process completed reports with progress bar
            with tqdm(total=len(reports), desc="Processing reports") as pbar:
                for future in as_completed(future_to_report):
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        report = future_to_report[future]
                        report_id = report.get('request_id', 'unknown')
                        error_msg = f"Error processing report {report_id}:"
                        print(f"\nError: {error_msg}")
                        print(traceback.format_exc())  # Print full stack trace
                        failed_reports.append({
                            'report_id': report_id,
                            'error': str(e),
                            'traceback': traceback.format_exc(),
                            'report_data': report
                        })
                        pbar.update(1)
        
        # Report summary
        print(f"\nProcessing complete:")
        print(f"- Successfully processed: {len(results)} reports")
        if failed_reports:
            print(f"- Failed to process: {len(failed_reports)} reports")
            
            # Save failed reports for debugging
            failed_file = Path(args.output_dir) / f"failed_reports_{Path(args.input_file).stem}.jsonl"
            print(f"Saving failed reports to {failed_file}")
            save_jsonl(failed_reports, str(failed_file))
        
        # Save successful results
        if results:
            output_file = Path(args.output_dir) / f"results_{Path(args.input_file).stem}.jsonl"
            print(f"Saving successful results to {output_file}")
            save_jsonl(results, str(output_file))
        else:
            print("No successful results to save")
        
        # Exit with error if any reports failed
        if failed_reports:
            sys.exit(1)
            
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        print("\nStack trace:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 