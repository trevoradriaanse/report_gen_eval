"""Prompt for checking if a citation is relevant to the sentence."""

SYSTEM_PROMPT = """You are an expert at determining if a citation's content is relevant to a sentence.
Your task is to determine if the cited content supports or is directly relevant to the claims made in the sentence.
Consider:
1. Topic alignment between sentence and citation
2. Specific claims and their support in the citation
3. Temporal relevance (if applicable)
4. Methodological relevance (if applicable)
5. Scope alignment (broad vs specific claims)

Respond with ONLY 'YES' or 'NO'."""

USER_PROMPT = """Is the citation content relevant to the sentence?

Examples:
1. Sentence: Deep learning models have achieved superhuman performance in image recognition tasks [5709e6d9-8945-4025-af43-9fb6ca0a7f11].
Citation: A comprehensive study of deep neural networks showing 99% accuracy on ImageNet, surpassing human performance of 95%.
Answer (YES/NO): YES

2. Sentence: Climate change has led to rising sea levels [8e45c80f-f63b-4eca-9976-79185811cd7d].
Citation: A study of migratory patterns in Arctic birds during winter months.
Answer (YES/NO): NO

3. Sentence: The new algorithm improved efficiency by 40% [56b44b0f-fd8d-4d81-bae9-7f8d80e6b745].
Citation: Experimental results showing the algorithm reduced computation time by 40% compared to baseline methods.
Answer (YES/NO): YES

4. Sentence: Recent studies have shown concerning levels of bias in AI systems.
Citation: Analysis of gender and racial bias in large language models, revealing systematic prejudices in model outputs.
Answer (YES/NO): YES

5. Sentence: The treatment showed promising results in clinical trials [8eac052b-6764-4092-af4f-63acd7ea8c71].
Citation: A market analysis of pharmaceutical companies' stock prices in 2023.
Answer (YES/NO): NO

6. Sentence: According to recent research, meditation can reduce stress levels.
Citation: A meta-analysis of 50 studies examining the effects of meditation on cortisol levels and self-reported stress.
Answer (YES/NO): YES

7. Sentence: The experiment revealed unexpected quantum effects [a9f4ae31-e2fc-45f5-b064-87d94c1cc059].
Citation: A theoretical framework for quantum mechanics without experimental validation.
Answer (YES/NO): NO

8. Sentence: Urban air pollution has increased by 15% since 2010 [8eac052b-6764-4092-af4f-63acd7ea8c71].
Citation: Environmental data showing a 15.3% rise in urban air pollutant concentrations between 2010 and 2023.
Answer (YES/NO): YES

Sentence: {sentence}
Citation: {citation_content}

Answer (YES/NO):"""
