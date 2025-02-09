"""Prompt for checking if a sentence contains negative assertions."""

SYSTEM_PROMPT = """You are an expert at identifying negative assertions in text.
Your task is to determine if a sentence contains any negative claims, findings, or results.
Consider:
1. Direct negations (no, not, never)
2. Negative findings (failed to, unable to)
3. Contradictions or opposing results
4. Absence of effects or relationships
5. Limitations or shortcomings
6. Negative comparisons

Respond with ONLY 'YES' or 'NO'."""

USER_PROMPT = """Does this sentence contain any negative assertions?

Examples:
1. Sentence: The study found no correlation between diet and cancer risk.
Answer (YES/NO): YES

2. Sentence: The treatment significantly improved patient outcomes.
Answer (YES/NO): NO

3. Sentence: The model failed to generalize to new datasets.
Answer (YES/NO): YES

4. Sentence: Researchers were unable to replicate the original findings.
Answer (YES/NO): YES

5. Sentence: The experiment demonstrated strong evidence for the hypothesis.
Answer (YES/NO): NO

6. Sentence: The proposed method showed limitations in handling edge cases.
Answer (YES/NO): YES

7. Sentence: The control group performed worse than the treatment group.
Answer (YES/NO): YES

8. Sentence: The results contradicted previous research in this area.
Answer (YES/NO): YES

Sentence: {sentence}

Answer (YES/NO):"""
