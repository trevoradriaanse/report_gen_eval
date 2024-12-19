"""Prompt for checking if a sentence is the first instance of a claim."""

SYSTEM_PROMPT = """You are an expert at identifying novel claims in text.
Your task is to determine if a sentence is the first instance of a claim in a sequence of sentences.
Consider:
1. Core claim or finding being presented
2. Previous mentions of similar claims
3. Specificity vs. generality of claims
4. Variations or elaborations of previous claims
5. Novel aspects vs. restatements
6. Context and scope of claims

Respond with ONLY 'YES' or 'NO'."""

USER_PROMPT = """Is this sentence the first instance of its main claim in the text?

Examples:
1. Sentence: Deep learning models can exhibit bias in their predictions.
Previous: []
Answer (YES/NO): YES

2. Sentence: The model showed significant gender bias in occupation classification.
Previous: ["Deep learning models can exhibit bias in their predictions."]
Answer (YES/NO): YES

3. Sentence: Neural networks demonstrate various forms of bias.
Previous: ["Deep learning models can exhibit bias in their predictions.", "The model showed significant gender bias in occupation classification."]
Answer (YES/NO): NO

4. Sentence: The experiment yielded promising results for cancer treatment.
Previous: ["Clinical trials are ongoing.", "Patients showed improved outcomes."]
Answer (YES/NO): YES

5. Sentence: The treatment was effective in reducing tumor size.
Previous: ["The experiment yielded promising results for cancer treatment."]
Answer (YES/NO): YES

6. Sentence: The study showed positive outcomes from the treatment.
Previous: ["The experiment yielded promising results for cancer treatment.", "The treatment was effective in reducing tumor size."]
Answer (YES/NO): NO

7. Sentence: Climate change has accelerated in recent decades.
Previous: ["Global temperatures are rising.", "Sea levels have increased."]
Answer (YES/NO): YES

8. Sentence: The rate of global warming has increased since 2000.
Previous: ["Climate change has accelerated in recent decades.", "Arctic ice is melting faster than expected."]
Answer (YES/NO): NO

Sentence: {sentence}
Previous Sentences:
{previous_sentences}

Answer (YES/NO):""" 