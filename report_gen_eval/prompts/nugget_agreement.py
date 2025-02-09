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
Question: What was the model's accuracy?
Answer: 95%
Answer (YES/NO): YES

2. Sentence: The study found no correlation between the variables.
Question: What was the relationship between the variables?
Answer: Strong correlation
Answer (YES/NO): NO

3. Sentence: Climate change has led to rising global temperatures.
Question: What is causing temperature increases worldwide?
Answer: Global warming
Answer (YES/NO): YES

4. Sentence: The treatment reduced symptoms in 80% of patients.
Question: How effective was the treatment?
Answer: It worked for most patients
Answer (YES/NO): YES

5. Sentence: The algorithm failed to generalize to new datasets.
Question: How did the method perform on unseen data?
Answer: It worked well
Answer (YES/NO): NO

6. Sentence: Neural networks require significant computational resources.
Question: What do deep learning models need?
Answer: Substantial computing power
Answer (YES/NO): YES

7. Sentence: The experiment showed inconclusive results.
Question: What did the study show about the hypothesis?
Answer: Clear evidence supporting it
Answer (YES/NO): NO

8. Sentence: The new method improved efficiency by 40%.
Question: How much did the approach increase performance?
Answer: 40%
Answer (YES/NO): YES

Sentence: {sentence}
Question: {nugget_question}
Answer: {nugget_answer}

Answer (YES/NO):"""
