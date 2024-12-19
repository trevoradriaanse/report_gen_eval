# Report Generation Evaluator

This repository implements the evaluation framework from ["On the Evaluation of Machine-Generated Reports"](https://arxiv.org/abs/2405.00982) (Mayfield et al., SIGIR 2024). It provides tools for systematically evaluating AI-generated reports based on citation usage, factual accuracy, and information coverage.

## Evaluation Framework

For each sentence in a report, the framework applies the following evaluation steps:

### For Sentences Without Citations:

1. If the sentence contains a negative statement ("X is not true"):
   - Reward (+) if a nugget confirms this statement
   - Penalize (-) if no nugget supports this claim
   
2. For statements requiring citations:
   - Penalize (-) if it's the first occurrence of the claim
   - Ignore (0) if the claim was previously cited
   
3. For statements not requiring citations (e.g., introductory text):
   - Ignore (0)

### For Sentences With Citations:

For each citation in the sentence:
1. Verify if the cited document supports the claim:
   - Penalize (-) if the document doesn't support the claim
   - Ignore (0) if the document supports the claim but isn't relevant to any nugget
   - Reward (+) if the sentence answers a nugget question and cites the correct document

### Scoring Metrics

The framework calculates two primary metrics:

- **Recall** = (Number of different nuggets correctly reported) / (Total number of nuggets)
- **Precision** = (Number of rewarded sentences) / (Total sentences - Ignored sentences)

Important Notes:
- A single sentence can receive multiple rewards for answering multiple nuggets
- Repeated nuggets only count once for recall
- Any penalized sentence (-) counts against precision
- Ignored sentences (0) don't affect the scores

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/orionweller/report_gen_eval.git
```

Or install in development mode:

```bash
git clone https://github.com/orionweller/report_gen_eval.git
cd report_gen_eval
pip install -e .
```

## Configuration

Create a `.env` file with your Together AI API key:
```bash
TOGETHER_API_KEY=your_api_key_here
```

## Usage

### Command Line Interface

```bash
report-eval example_input.jsonl results/ --batch-size 10
```

### Input Format

The input JSONL file should follow this structure:
```json
{
  "request_id": "123",
  "run_id": "example-run",
  "collection_ids": ["collection1"],
  "sentences": [
    {
      "text": "Example sentence with citation [1].",
      "citations": ["citation_id_1"]
    }
  ]
}
```

### Python API

```python
from report_gen_eval import evaluate_report

sentences = [
    "Recent studies have shown that AI models can exhibit biases [1].",
    "This phenomenon has been extensively documented (Smith et al., 2023)."
]

citation_contents = {
    "1": "A comprehensive study of AI model biases...",
    "Smith2023": "Analysis of bias patterns in language models..."
}

nuggets = {
    "1": "AI models can show biases in their outputs",
    "2": "Bias in AI systems is well-documented"
}

results = evaluate_report(sentences, citation_contents, nuggets)
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

Or run specific tests:
```bash
python -m pytest tests/test_prompts.py -k test_has_citations -v
```

## Citation

If you use this evaluation framework in your research, please cite:
```bibtex
@inproceedings{Mayfield2024OnTE,
  title={On the Evaluation of Machine-Generated Reports},
  author={James Mayfield and Eugene Yang and Dawn J Lawrie and Sean MacAvaney and Paul McNamee and Douglas W. Oard and Luca Soldaini and Ian Soboroff and Orion Weller and Efsun Kayi and Kate Sanders and Marc Mason and Noah Hibbler},
  booktitle={Annual International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:269502216}
}
AND
@TODO
```

## License

[Add License Information]
