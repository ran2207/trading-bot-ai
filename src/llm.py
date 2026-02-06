"""LLM Advisory Module."""

from openai import AsyncOpenAI
from typing import Dict, Any
import json


class LLMAdvisor:
    """LLM-powered market analysis and advice."""

    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)

    async def analyze_market(
        self,
        symbol: str,
        technicals: Dict[str, Any],
        sentiment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get LLM analysis of market conditions."""
        prompt = f"""You are an expert financial analyst. Analyze the following data for {symbol}:

TECHNICAL ANALYSIS:
- Current Price: ${technicals.get('price', 'N/A'):.2f}
- RSI: {technicals.get('rsi', 'N/A'):.2f}
- MACD: {technicals.get('macd', 'N/A'):.4f}
- Trend: {technicals.get('trend', 'N/A')}
- Signals: {technicals.get('signals', {})}

SENTIMENT ANALYSIS:
- Score: {sentiment.get('score', 0):.2f} (-1 bearish to +1 bullish)
- Key Factors: {sentiment.get('key_factors', [])}
- Summary: {sentiment.get('summary', 'N/A')}

Based on this analysis, provide your recommendation in JSON format:
{{
    "recommendation": "BUY" | "SELL" | "HOLD",
    "confidence": <float 0-1>,
    "reasoning": "<brief explanation>",
    "risks": ["<risk1>", "<risk2>"],
    "price_target": <float or null>,
    "stop_loss": <float or null>
}}

Respond ONLY with the JSON, no other text."""

        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )

        try:
            content = response.choices[0].message.content
            return json.loads(content)
        except (json.JSONDecodeError, IndexError):
            return {
                "recommendation": "HOLD",
                "confidence": 0.0,
                "reasoning": "Unable to analyze",
                "risks": [],
                "price_target": None,
                "stop_loss": None,
            }
