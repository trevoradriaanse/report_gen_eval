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
1. Sentence: Solar power adoption has led to significant cost reductions in energy production.
Previous: []
Answer (YES/NO): YES

2. Sentence: China's Belt and Road Initiative aims to enhance global trade connections.
Previous: ["China has made major investments in its domestic economy in recent years."]
Answer (YES/NO): YES

3. Sentence: The United States signed the Paris Agreement to combat climate change in 2016.
Previous: ["Climate change activists were thrilled by the signing of the Paris Agreement."]
Answer (YES/NO): NO

4. Sentence: The United States has imposed sanctions on Russia in response to its actions in Ukraine.
Previous: ["U.S. sanctions have wounded the Russian economy.", "Sanctions for the War in Ukraine have negatively impacted the Russian economy."]
Answer (YES/NO): NO

5. Sentence: Key artists like Botticelli, Michelangelo, and Leonardo da Vinci were supported by the Medici family during the Renaissance, shaping the course of art history.
Previous: ["Art history studies in the United States focus on both classical and contemporary visionaries."]
Answer (YES/NO): YES

6. Sentence: The United States has recently seen an increase in the number of immigrant families seeking citizenship through various legal channels.
Previous: ["Critics argue that the process of acquiring U.S. citizenship through family reunification may strain social services."]
Answer (YES/NO): NO

7. Sentence: Climate change negatively impacts crop yields.
Previous: ["Global temperatures are rising.", "Sea levels have increased."]
Answer (YES/NO): YES

8. Sentence: The deer's fur color was confirmed to be caused by a rare genetic trait called albinism.
Previous: ["Albinism can leave animals like deer with a distinctive white fur."]
Answer (YES/NO): NO

Sentence: {sentence}
Previous Sentences:
{previous_sentences}

Answer (YES/NO):"""
