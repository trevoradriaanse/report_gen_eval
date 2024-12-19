"""Prompt for checking if a sentence agrees with an information nugget."""

SYSTEM_PROMPT = """You are an expert at determining if statements agree with given information.
Your task is to determine if a sentence's claims align with a provided information nugget.
Consider:
1. Core meaning and implications
2. Factual consistency
3. Semantic equivalence
4. Logical entailment
5. Scope of claims
6. Contextual meaning
7. Direct vs indirect agreement
8. Quantitative precision

Respond with ONLY 'YES' or 'NO'."""

USER_PROMPT = """Does the sentence agree with the information nugget?

Examples:
1. Sentence: The model achieved 95% accuracy on the test set.
Nugget: The model's performance was 95% accurate.
Answer (YES/NO): YES

2. Sentence: The study found no correlation between the variables.
Nugget: The variables showed strong correlation.
Answer (YES/NO): NO

3. Sentence: Climate change has led to rising global temperatures.
Nugget: Global warming is causing temperature increases worldwide.
Answer (YES/NO): YES

4. Sentence: The treatment reduced symptoms in 80% of patients.
Nugget: The treatment was effective for most patients.
Answer (YES/NO): YES

5. Sentence: The algorithm failed to generalize to new datasets.
Nugget: The method worked well on unseen data.
Answer (YES/NO): NO

6. Sentence: Neural networks require significant computational resources.
Nugget: Deep learning models need substantial computing power.
Answer (YES/NO): YES

7. Sentence: The experiment showed inconclusive results.
Nugget: The study provided clear evidence for the hypothesis.
Answer (YES/NO): NO

8. Sentence: The new method improved efficiency by 40%.
Nugget: The approach increased performance by 40%.
Answer (YES/NO): YES

Sentence: {sentence}
Nugget: {nugget}

Answer (YES/NO):""" 