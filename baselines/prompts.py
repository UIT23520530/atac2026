"""Prompt templates for baseline sentiment analysis experiments."""


def get_zero_shot_prompt(text: str, target: str, domain: str = "general") -> str:
    """Generate zero-shot prompt: simple instruction without examples."""
    return f"""You are an expert in implicit sentiment analysis.

Task: Determine the implicit sentiment polarity expressed about a {domain}.
The sentiment MUST be one of the following:
- positive
- negative
- neutral

Aspect/Target: "{target}"
Text: "{text}"

Respond with ONLY one label: positive, negative, or neutral.
"""


def get_few_shot_laptop_prompt(text: str, target: str) -> str:
    """Generate few-shot prompt for laptop domain with examples."""
    return f"""You are an expert in implicit sentiment analysis.

Task: Determine the implicit sentiment polarity expressed about a laptop.
The sentiment MUST be one of the following:
- positive
- negative
- neutral

Below are annotated examples of implicit sentiment analysis in the laptop review domain.
Each example includes the text, the target (aspect), and the correct sentiment.

Example 1:
Text: "The laptop battery lasts all day, which is amazing for productivity."
Target: "battery life"
Sentiment: Positive

Example 2:
Text: "The keyboard feels terrible to type on."
Target: "keyboard"
Sentiment: Negative

Example 3:
Text: "For the price, you could do worse, though there are certainly better options."
Target: "value for money"
Sentiment: Neutral

Now analyze the following input.
Your job is to determine the sentiment (Positive, Negative, Neutral) expressed about the target.
Respond ONLY with one label.

Text: "{text}"
Target: "{target}"
Sentiment:
"""


def get_few_shot_restaurant_prompt(text: str, target: str) -> str:
    """Generate few-shot prompt for restaurant domain with examples."""
    return f"""You are an expert in implicit sentiment analysis.

Task: Determine the implicit sentiment polarity expressed about a restaurant.
The sentiment MUST be one of the following:
- positive
- negative
- neutral

Below are annotated examples of implicit sentiment analysis in the restaurant/food review domain.
Each example includes the text, the target (aspect), and the correct sentiment label.

Example 1:
Text: "The food was absolutely delicious, with perfect seasoning and excellent presentation."
Target: "food quality"
Sentiment: Positive

Example 2:
Text: "The restaurant meets basic standards - clean environment, food arrives warm."
Target: "overall experience"
Sentiment: Neutral

Example 3:
Text: "Service was terrible, waiters ignored us and orders took over an hour."
Target: "service"
Sentiment: Negative

Now analyze the following input.
Your job is to determine the sentiment (Positive, Negative, Neutral) expressed about the target.
Respond ONLY with one label.

Text: "{text}"
Target: "{target}"
Sentiment:
"""


def get_few_shot_prompt(text: str, target: str, domain: str = "general") -> str:
    """Generate few-shot prompt: instruction with domain-specific examples."""
    if domain == "restaurant":
        return get_few_shot_restaurant_prompt(text, target)
    else:
        return get_few_shot_laptop_prompt(text, target)


def get_cot_prompt(text: str, target: str, domain: str = "general") -> str:
    """Generate Chain-of-Thought prompt: step-by-step reasoning instruction."""
    return f"""You are an expert in implicit sentiment analysis.

Task: Determine the implicit sentiment polarity expressed about a {domain}.
The sentiment MUST be one of the following:
- positive
- negative
- neutral

Instructions:
1. Carefully analyze the text and identify any implicit opinions, evaluations, or pragmatic cues related to the target.
2. Consider contextual signals such as expectations, sarcasm, understatement, contrast, or culturally implied judgments.
3. Reason step by step to infer the underlying sentiment toward the target.
4. Do NOT output your reasoning process.
5. Respond ONLY with one label: positive, negative, or neutral.

Now analyze the following input.

Text: "{text}"
Target: "{target}"

Sentiment:
"""
