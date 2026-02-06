"""Sentiment Analysis Module using LLM."""

from openai import AsyncOpenAI
from typing import Dict, Any


class SentimentAnalyzer:
    """Analyze market sentiment using LLM."""

    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)

    async def analyze(self, symbol: str) -> Dict[str, Any]:
        """Analyze sentiment for a symbol."""
        prompt = f"""Analyze the current market sentiment for {symbol} stock.

Consider:
1. Recent news and developments
2. Market trends
3. Sector performance
4. General market conditions

Provide a sentiment analysis in JSON format:
{{
    "score": <float from -1.0 (very bearish) to 1.0 (very bullish)>,
    "confidence": <float from 0 to 1>,
    "summary": "<brief explanation>",
    "key_factors": ["<factor1>", "<factor2>"]
}}

Respond ONLY with the JSON, no other text."""

        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )

        try:
            import json
            content = response.choices[0].message.content
            return json.loads(content)
        except (json.JSONDecodeError, IndexError):
            return {
                "score": 0.0,
                "confidence": 0.0,
                "summary": "Unable to analyze sentiment",
                "key_factors": [],
            }
