# Report Generation Evaluator
### This repository is in progress

This repository implements the evaluation framework from ["On the Evaluation of Machine-Generated Reports"](https://arxiv.org/abs/2405.00982) (Mayfield et al., SIGIR 2024). It provides tools for systematically evaluating AI-generated reports based on citation usage, factual accuracy, and information coverage.

## Installation

### Option 1: Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver. To set up the environment using UV:

1. Install UV (if not already installed):
```bash
curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh | sh
```

2. Clone and set up the environment:
```bash
git clone https://github.com/orionweller/report_gen_eval.git
cd report_gen_eval
chmod +x uv_install.sh
./uv_install.sh
```

3. Activate the environment:
```bash
source env/bin/activate
```

4. Install in development mode:
```bash
uv pip install -e .
```

### Option 2: Using pip

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

The evaluator uses LangChain to interact with various LLM providers. Currently supported providers:
- Together AI (default)
- OpenAI
- Anthropic

Create a `.env` file with your API key(s):
```bash
TOGETHER_API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_key_here  # Optional
ANTHROPIC_API_KEY=your_anthropic_key_here  # Optional
```

## Usage

### Command Line Interface

Basic usage:
```bash
report-eval tests/assets/example_input.jsonl tests/assets/example_nuggets.jsonl results/ --batch-size 1
```

With specific model provider:
```bash
report-eval tests/assets/example_input_one_only.jsonl tests/assets/example_nuggets.jsonl results/ -p openai -m gpt-4o-2024-11-20
```

### Input Format

The input JSONL file should contain report entries with this structure:
```json
{
  "request_id": "300",
  "run_id": "example-run",
  "collection_ids": ["collection1"],
  "sentences": [
    {
      "text": "Japan experienced a significant increase in suicide rates during the COVID-19 pandemic.",
      "citations": ["56b44b0f-fd8d-4d81-bae9-7f8d80e6b745"]
    }
  ]
}
```

Note: The citations field is an array of citation IDs. The evaluator will assume the citations are relevant since the actual citation content is not provided.

The nuggets file should contain evaluation criteria in this format:
```json
{
  "request_id": 300,
  "nuggets": [
    {
      "question": "How much did suicides rise by in 2020?",
      "answer": "3.7%",
      "documents_with_nugget": ["56b44b0f-fd8d-4d81-bae9-7f8d80e6b745"]
    }
  ]
}
```

### Python API

```python
from report_gen_eval import evaluate_report, ModelProvider

# Example report evaluation
sentences = [
    {
        "text": "Japan experienced a significant increase in suicide rates during the COVID-19 pandemic.",
        "citations": ["56b44b0f-fd8d-4d81-bae9-7f8d80e6b745"]
    },
    {
        "text": "In 2020, suicides rose by 3.7% to 20,919, the first increase in 11 years.",
        "citations": ["56b44b0f-fd8d-4d81-bae9-7f8d80e6b745"]
    }
]

report = {
    "request_id": "300",
    "run_id": "example-run",
    "collection_ids": ["collection1"],
    "sentences": sentences
}

# Evaluate with specific model provider
results = evaluate_report(
    report,
    nuggets_file="tests/assets/example_nuggets.jsonl",
    provider=ModelProvider.TOGETHER,
    model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo"
)
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

Or run specific tests:
```bash
python -m pytest tests/test_prompts.py -k test_check_relevance -v
```

## Evaluation Framework

For each sentence in a report, the framework applies the following evaluation steps:

### For Sentences Without Citations:

1. If the sentence contains a negative statement ("X is not true"):
   - Reward (+1) if a nugget confirms this statement
   - Penalize (-1) if no nugget supports this claim
   
2. For statements requiring citations:
   - Penalize (-1) if it's the first occurrence of the claim
   - Ignore (0) if the claim was previously cited
   
3. For statements not requiring citations (e.g., introductory text):
   - Ignore (0)

### For Sentences With Citations:

1. Check if each cited document supports the claim:
   - Penalize (-1) if any document doesn't support the claim
   - Continue to step 2 if all documents support the claim

2. Check nugget matching:
   - Reward (+1) for each nugget the sentence correctly answers
   - Ignore (0) if no nuggets are matched

### Scoring Metrics

The framework calculates two primary metrics:

- **Recall** = (Number of unique nuggets correctly reported) / (Total number of nuggets)
- **Precision** = (Number of rewarded sentences) / (Total scored sentences)

Important Notes:
- A sentence can receive multiple rewards for matching multiple nuggets
- Each unique nugget counts only once for recall
- Ignored sentences (score=0) don't affect precision
- Any penalized sentence (-1) counts against precision

## Citation

If you use this evaluation framework in your research, please cite both:
This implementation:
```bibtex
TODO
```
and the original framework:
```bibtex
@inproceedings{Mayfield2024OnTE,
  title={On the Evaluation of Machine-Generated Reports},
  author={James Mayfield and Eugene Yang and Dawn J Lawrie and Sean MacAvaney and Paul McNamee and Douglas W. Oard and Luca Soldaini and Ian Soboroff and Orion Weller and Efsun Kayi and Kate Sanders and Marc Mason and Noah Hibbler},
  booktitle={Annual International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:269502216}
}
```