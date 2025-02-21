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
1. Sentence: Air quality is not influenced by pollution.
Answer (YES/NO): YES

2. Sentence: Lady Gaga has had a substantial impact on the music industry.
Answer (YES/NO): NO

3. Sentence: Pesticides are not dangerous to human health.
Answer (YES/NO): YES

4. Sentence: Solar power has failed to provide a reliable solution for energy storage.
Answer (YES/NO): YES

5. Sentence: The environmental study showed promising results for new irrigation techniques.
Answer (YES/NO): NO

6. Sentence: Luxury dog breeds are expected to increase in popularity over the next 10 years.
Answer (YES/NO): NO

7. Sentence: Fans failed to act responsibly after their football team lost the game.
Answer (YES/NO): YES

8. Sentence: African nations have successfully implemented large-scale infrastructure projects.
Answer (YES/NO): YES

Sentence: {sentence}

Answer (YES/NO):"""
