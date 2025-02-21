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
1. Sentence: Blue Origin's New Shepard rocket successfully completed its first crewed flight in 2021.
Answer (YES/NO): YES

2. Sentence: The movie The Titanic is thus one of the most important works in modern film.
Answer (YES/NO): NO

3. Sentence: Recent studies have shown a 15% increase in global temperatures.
Answer (YES/NO): YES

4. Sentence: Planting trees is one way children can play a role in positive climate action.
Answer (YES/NO): NO

5. Sentence: Taylor Swift attended the 2024 Golden Globe Awards and received nominations for her concert film Taylor Swift: The Eras Tour.
Answer (YES/NO): YES

6. Sentence: Policy researchers have concluded that immigration to the United States will increase despite anti-immigrant rhetoric.
Answer (YES/NO): YES

7. Sentence: The winner of the New York Marathon is an excellent runner.
Answer (YES/NO): NO

8. Sentence: Novickok was used on the Skripals in England in 2018.
Answer (YES/NO): YES

Sentence: {sentence}

Answer (YES/NO):"""

USER_PROMPT_SHORT = "Does this sentence require a citation to support its claims? \nSentence: {sentence} \nAnswer (YES/NO):"
