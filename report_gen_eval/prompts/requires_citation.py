"""Prompt for checking if a sentence requires citations."""

SYSTEM_PROMPT = """You are an expert at determining if statements require academic citations.
Your task is to determine if a sentence makes claims that should be supported by citations.
Consider:
1. Empirical claims or findings
2. Statistical data or numbers
3. Historical facts or dates
4. Specific methodologies or techniques
5. Theoretical frameworks
6. Expert opinions or analyses
7. Comparative statements
8. State-of-the-art claims

Respond with ONLY 'YES' or 'NO'."""

USER_PROMPT = """Does this sentence require a citation to support its claims?

Examples:
1. Sentence: Deep learning models have achieved 98% accuracy on the ImageNet dataset.
Answer (YES/NO): YES

2. Sentence: The sky appears blue due to Rayleigh scattering.
Answer (YES/NO): NO

3. Sentence: Recent studies have shown a 15% increase in global temperatures.
Answer (YES/NO): YES

4. Sentence: Water boils at 100 degrees Celsius at sea level.
Answer (YES/NO): NO

5. Sentence: The new treatment method reduced recovery time by 40%.
Answer (YES/NO): YES

6. Sentence: Researchers have identified three key factors affecting climate change.
Answer (YES/NO): YES

7. Sentence: The algorithm outperforms all existing methods on benchmark datasets.
Answer (YES/NO): YES

8. Sentence: The experiment was conducted in a controlled laboratory environment.
Answer (YES/NO): NO

Sentence: {sentence}

Answer (YES/NO):""" 