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
1. Sentence: California courts have ruled on liability claims related to wildfires caused by utility companies [5709e6d9-8945-4025-af43-9fb6ca0a7f11].
Citation: Lawsuits filed against utility companies in California criticized the companies for their role in causing wildfires and discussed damages and settlements to be paid out to victims.
Answer (YES/NO): YES

2. Sentence: The development of electric vehicles has reduced carbon emissions in urban areas [8e45c80f-f63b-4eca-9976-79185811cd7d].
Citation: Although electic cars are increasingly popular, there is a continued role for gas-powered cars.
Answer (YES/NO): NO

3. Sentence: Trophy hunting of lions has been linked to declines in certain lion populations in Africa [56b44b0f-fd8d-4d81-bae9-7f8d80e6b745].
Citation: A study clearly demonstrated the correlation between trophy hunting practices and the decline of lion populations in specific African regions.
Answer (YES/NO): YES

4. Sentence: The SolarWinds cyberattack has been described as one of the most significant espionage campaigns in recent history [c5a3a72e-b4e8-4b72-bd65-8e2b5bdb015f].
Citation: An analysis of the SolarWinds cyberattack illustrated how the attackers used software vulnerabilities to infiltrate major organizations and compromise data.
Answer (YES/NO): YES

5. Sentence: The long-term use of statins can lead to serious side effects, including liver damage [8eac052b-6764-4092-af4f-63acd7ea8c71].
Citation: Statins have been shown to demonstrate cholesterol-lowering properties.
Answer (YES/NO): NO

6. Sentence: Studies show that excessive use of social media is linked to increased anxiety and depression among teenagers [9a75cd2b-2f99-4b9b-9fe5-28797f229e01].
Citation: Research has examined the impact of social media on adolescent mental health, highlighting a significant correlation between heavy social media use and higher levels of anxiety and depression.
Answer (YES/NO): YES

7. Sentence: Satellite communications have significantly improved our ability to gather data for weather forecasting [a9f4ae31-e2fc-45f5-b064-87d94c1cc059].
Citation: Commercial space tourism is expected to become more poular in the coming decades.
Answer (YES/NO): NO

8. Sentence: FEMA has been pivotal in providing disaster relief funding to affected regions during and after natural disasters, such as hurricanes [8eac052b-6764-4092-af4f-63acd7ea8c71].
Citation: A report on FEMA's disaster relief efforts detailed funding distribution and support provided to hurricane-affected areas.
Answer (YES/NO): YES

Sentence: {sentence}
Citation: {citation_content}

Answer (YES/NO):"""
