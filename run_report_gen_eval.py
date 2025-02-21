import argparse
import json
import logging
import os
import traceback
import sys

from concurrent.futures import ThreadPoolExecutor, as_completed
from report_gen_eval import evaluate_report, ModelProvider
from report_gen_eval.utils import load_jsonl
from tqdm import tqdm
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# suppress INFO messages from  OpenAI
logging.getLogger("openai").setLevel(logging.WARNING)

# suppress INFO messages from Hugging Face
logging.getLogger("transformers").setLevel(logging.WARNING)


def run_report_gen_eval(input_file):
    try:
        reports = load_jsonl(input_file)
        if not reports:
            raise ValueError("Input file contains no reports")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse input file {input_file}: {str(e)}")
    return reports


def process_report(
    report: Dict[str, Any],
    nuggets_file: str = None,
    provider: str = ModelProvider.TOGETHER,
    model_name: str = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Process a single report with error handling.

    Args:
        report: The report to process
        nuggets_file: Path to the nuggets file
        provider: The model provider to use
        model_name: Optional specific model name
        verbose: Whether to enable verbose logging

    Returns:
        The evaluation results or None if processing failed

    Note:
        This function wraps evaluate_report with additional validation
        and error handling suitable for batch processing.
    """
    try:
        report_id = report.get("request_id", "unknown")
        if verbose:
            logger.info(f"Processing report {report_id}")

        # Validate required fields
        required_fields = ["request_id", "run_id", "collection_ids", "sentences"]
        missing_fields = [field for field in required_fields if field not in report]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in report: {', '.join(missing_fields)}"
            )

        # Process the report with evaluate_report
        if verbose:
            logger.info(f"Starting evaluation for report {report_id}")
        result = evaluate_report(
            report,
            nuggets_file=nuggets_file,
            provider=provider,
            model_name=model_name,
            verbose=verbose,
        )

        if result is None:
            raise ValueError("Failed to evaluate report")

        if verbose:
            logger.info(f"Completed processing report {report_id}")
        return result
    except Exception as e:
        if verbose:
            logger.error(
                f"Error processing report {report.get('request_id', 'unknown')}: {str(e)}"
            )
            logger.debug("Stack trace:", exc_info=True)
        return None


def main():
    """Main entry point for the CLI.

    This function:
    1. Parses command line arguments
    2. Validates input files and settings
    3. Processes reports in parallel batches
    4. Saves results and handles errors
    """
    parser = argparse.ArgumentParser(
        description="Evaluate report sentences from JSONL file"
    )
    parser.add_argument("input_file", type=str, help="Path to input JSONL file")
    parser.add_argument(
        "nuggets_file",
        type=str,
        help="Path to nuggets file containing evaluation criteria",
    )
    parser.add_argument("output_dir", type=str, help="Directory to save results")
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=10,
        help="Number of reports to process in parallel (default: 10)",
    )
    parser.add_argument(
        "-p",
        "--model-provider",
        type=str,
        choices=["openai", "anthropic", "together", "huggingface"],
        default="together",
        help="Model provider to use (default: together)",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        help="Specific model name to use (defaults to provider-specific default)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args()

    # Set logging level based on verbosity
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    try:
        # Validate input files exist
        if args.verbose:
            logger.info("Validating input files...")
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        if not os.path.exists(args.nuggets_file):
            raise FileNotFoundError(f"Nuggets file not found: {args.nuggets_file}")

        # Load input data
        if args.verbose:
            logger.info(f"Loading data from {args.input_file}")
        try:
            reports = load_jsonl(args.input_file)
            if not reports:
                raise ValueError("Input file contains no reports")
            if args.verbose:
                logger.info(f"Loaded {len(reports)} reports")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse input file {args.input_file}: {str(e)}")

        # Validate batch size
        if args.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        if args.batch_size > len(reports):
            if args.verbose:
                logger.warning(
                    f"Batch size {args.batch_size} is larger than number of reports {len(reports)}. Using {len(reports)} workers."
                )
            args.batch_size = len(reports)

        # Process reports in batches
        if args.verbose:
            logger.info(
                f"Processing {len(reports)} reports in batches of {args.batch_size}..."
            )
        results = []
        failed_reports = []

        with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
            future_to_report = {
                executor.submit(
                    process_report,
                    report,
                    nuggets_file=args.nuggets_file,
                    provider=args.model_provider,
                    model_name=args.model_name,
                    verbose=args.verbose,
                ): report
                for report in reports
            }

            with tqdm(
                total=len(reports), desc="Processing reports", disable=not args.verbose
            ) as pbar:
                for future in as_completed(future_to_report):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                        else:
                            report = future_to_report[future]
                            failed_reports.append(
                                {
                                    "report_id": report.get("request_id", "unknown"),
                                    "error": "Evaluation returned None",
                                    "report_data": report,
                                }
                            )
                        pbar.update(1)
                    except Exception as e:
                        report = future_to_report[future]
                        report_id = report.get("request_id", "unknown")
                        error_msg = f"Error processing report {report_id}: {str(e)}"
                        if args.verbose:
                            logger.error(error_msg)
                            logger.debug("Stack trace:", exc_info=True)
                        failed_reports.append(
                            {
                                "report_id": report_id,
                                "error": str(e),
                                "traceback": traceback.format_exc(),
                                "report_data": report,
                            }
                        )
                        pbar.update(1)

        # Report summary
        if args.verbose:
            logger.info("\nProcessing complete:")
            logger.info(f"- Successfully processed: {len(results)} reports")
        if failed_reports:
            if args.verbose:
                logger.warning(f"- Failed to process: {len(failed_reports)} reports")

        # Save successful results
        summary_statistics = {}
        if results:
            for result in results:
                request_id = result["request_id"]
                summary_statistics[request_id] = {
                    "metrics": {
                        "precision": result["metrics"]["precision"],
                        "recall": result["metrics"]["recall"],
                    }
                }
            with open(args.output_dir, "w") as fp:
                json.dump(summary_statistics, fp, indent=4)

        elif args.verbose:
            logger.warning("No successful results to save")

        # Exit with error if any reports failed
        if failed_reports:
            sys.exit(1)

    except Exception as e:
        if args.verbose:
            logger.error(f"\nError: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
