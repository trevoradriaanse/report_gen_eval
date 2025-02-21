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
1. Sentence: The new vaccine showed a 90% efficacy rate in clinical trials.
Question: What was the vaccine's effectiveness?
Answer: 90%
Do the sentence's claims agree with the nugget question answer (YES/NO): YES

2. Sentence: The research indicates a 30% increase in global temperatures over the last century partly largely due to pollution.
Question: Why did global temperatures increase by 30%?
Answer: Increased government spending
Do the sentence's claims agree with the nugget question answer (YES/NO): NO

3. Sentence: Stratolaunch's airplane is known for its massive size and impressive wingspan.
Question: What is notable about Stratolaunch's airplane?
Answer: Its size and wingspan
Do the sentence's claims agree with the nugget question answer (YES/NO): YES

4. Sentence: Winter the Dolphin lost her tail due to becoming entangled in a crab trap and became a symbol for resilience.
Question: How did Winter the Dolphin lose her tail?
Answer: Becoming entangled in a crab trap
Do the sentence's claims agree with the nugget question answer (YES/NO): YES

5. Sentence: Maya Angelou won the Nobel Prize in Literature in 1995.
Question: When did Maya Angelou win the Nobel Prize in Literature?
Answer: Maya Angelou never won the Nobel Prize in Literature 
Do the sentence's claims agree with the nugget question answer (YES/NO): NO

6. Sentence: Rising temperatures have had no significant effect on strawberry farming.
Question: How has the climate affected strawberry farming?
Answer: Climate change has negatively impacted strawberry farming, including yield reductions
Do the sentence's claims agree with the nugget question answer (YES/NO): YES

7. Sentence: Refugees crossing the English Channel usually take large boats that are not typically safe or properly equipped.
Question: How do refugees typically attempt to move from France to England?
Answer: Refugees cross the English Channel safely using high-quality boats
Do the sentence's claims agree with the nugget question answer (YES/NO): NO

8. Sentence: The discovery of water on Mars has opened up possibilities for past life on the planet.
Question: What has the discovery of water on Mars suggested about its history?
Answer: The possibility of past life
Do the sentence's claims agree with the nugget question answer (YES/NO): YES

Sentence: {sentence}
Question: {nugget_question}
Do the sentence's claims agree with the nugget question answer (YES/NO): {nugget_answer}

Answer (YES/NO):"""
