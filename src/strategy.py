"""Trading Strategy Module."""

from typing import Dict, Any


class TradingStrategy:
    """Generate trading signals based on analysis."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = {
            "sentiment_strong": 0.5,
            "sentiment_weak": -0.5,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
        }

    def generate_signal(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signal from analysis."""
        technicals = analysis["technicals"]
        sentiment = analysis["sentiment"]
        llm = analysis["llm_analysis"]
        
        # Score components
        tech_score = self._score_technicals(technicals)
        sent_score = sentiment.get("score", 0) * 0.3
        llm_score = self._score_llm(llm)
        
        total_score = tech_score + sent_score + llm_score
        
        # Generate signal
        if total_score > 0.5:
            action = "BUY"
            confidence = min(total_score, 1.0)
        elif total_score < -0.5:
            action = "SELL"
            confidence = min(abs(total_score), 1.0)
        else:
            action = "HOLD"
            confidence = 1.0 - abs(total_score)
        
        return {
            "action": action,
            "confidence": confidence,
            "score": total_score,
            "components": {
                "technical": tech_score,
                "sentiment": sent_score,
                "llm": llm_score,
            },
        }

    def _score_technicals(self, technicals: Dict[str, Any]) -> float:
        """Score technical indicators."""
        score = 0.0
        signals = technicals.get("signals", {})
        
        # RSI
        if signals.get("rsi") == "OVERSOLD":
            score += 0.2
        elif signals.get("rsi") == "OVERBOUGHT":
            score -= 0.2
        
        # MACD
        if signals.get("macd") == "BULLISH":
            score += 0.15
        elif signals.get("macd") == "BEARISH":
            score -= 0.15
        
        # Moving Averages
        if signals.get("ma_cross") == "BULLISH":
            score += 0.15
        elif signals.get("ma_cross") == "BEARISH":
            score -= 0.15
        
        # Trend
        trend = technicals.get("trend", "SIDEWAYS")
        if trend == "STRONG_UPTREND":
            score += 0.2
        elif trend == "UPTREND":
            score += 0.1
        elif trend == "STRONG_DOWNTREND":
            score -= 0.2
        elif trend == "DOWNTREND":
            score -= 0.1
        
        return score

    def _score_llm(self, llm: Dict[str, Any]) -> float:
        """Score LLM analysis."""
        recommendation = llm.get("recommendation", "HOLD").upper()
        confidence = llm.get("confidence", 0.5)
        
        if recommendation == "BUY":
            return 0.3 * confidence
        elif recommendation == "SELL":
            return -0.3 * confidence
        return 0.0
