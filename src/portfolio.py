"""Portfolio Management Module with Risk Management."""

from typing import Dict, Any, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manage portfolio and position sizing with risk management rules."""

    def __init__(self, client, config: Dict[str, Any]):
        self.client = client
        self.config = config
        
        # Risk management parameters
        self.max_position_pct = config.get("max_position_pct", 0.10)  # Max 10% per position
        self.stop_loss_pct = config.get("stop_loss_pct", 0.02)        # 2% stop loss
        self.take_profit_pct = config.get("take_profit_pct", 0.05)    # 5% take profit
        self.max_total_risk_pct = config.get("max_total_risk_pct", 0.30)  # Max 30% total risk
        
        # Cache for account info
        self._account_cache: Optional[Dict] = None
        self._positions_cache: Dict[str, Dict] = {}

    async def refresh_cache(self):
        """Refresh cached account and position data."""
        try:
            self._account_cache = await self.client.get_account()
            positions = await self.client.get_positions()
            self._positions_cache = {p["symbol"]: p for p in positions}
        except Exception as e:
            logger.error(f"Error refreshing cache: {e}")

    async def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        if not self._account_cache:
            await self.refresh_cache()
        return float(self._account_cache.get("equity", 0))

    async def get_available_cash(self) -> float:
        """Get available cash for trading."""
        if not self._account_cache:
            await self.refresh_cache()
        return float(self._account_cache.get("cash", 0))

    async def get_buying_power(self) -> float:
        """Get buying power (including margin if applicable)."""
        if not self._account_cache:
            await self.refresh_cache()
        return float(self._account_cache.get("buying_power", 0))

    async def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for a symbol."""
        if not self._positions_cache:
            await self.refresh_cache()
        return self._positions_cache.get(symbol)

    async def get_position_value(self, symbol: str) -> float:
        """Get current position value for a symbol."""
        position = await self.get_position(symbol)
        if position:
            return float(position.get("market_value", 0))
        return 0.0

    async def get_all_positions(self) -> Dict[str, Dict]:
        """Get all current positions."""
        if not self._positions_cache:
            await self.refresh_cache()
        return self._positions_cache

    def calculate_position_size(
        self, 
        symbol: str, 
        signal: Dict[str, Any], 
        current_price: float = None
    ) -> int:
        """
        Calculate position size based on risk management rules.
        
        Uses Kelly Criterion-inspired sizing adjusted by signal confidence.
        
        Args:
            symbol: Stock symbol
            signal: Trading signal with action and confidence
            current_price: Current stock price
        
        Returns:
            Number of shares to trade
        """
        if not current_price or current_price <= 0:
            return 0
        
        # Get cached values (synchronous access to cached data)
        if not self._account_cache:
            # If no cache, return 0 (caller should ensure cache is populated)
            logger.warning("Account cache not populated, returning 0")
            return 0
        
        portfolio_value = float(self._account_cache.get("equity", 0))
        cash = float(self._account_cache.get("cash", 0))
        
        # Get current position value
        position = self._positions_cache.get(symbol)
        current_position_value = float(position.get("market_value", 0)) if position else 0.0
        current_position_qty = int(position.get("qty", 0)) if position else 0
        
        # Calculate max position value based on risk rules
        max_position_value = portfolio_value * self.max_position_pct
        
        # Adjust by confidence (lower confidence = smaller position)
        confidence = signal.get("confidence", 0.5)
        adjusted_max = max_position_value * confidence
        
        if signal["action"] == "BUY":
            # How much more can we add to position?
            available_for_position = adjusted_max - current_position_value
            
            # Can't exceed available cash
            available_cash = min(cash, available_for_position)
            
            if available_cash <= 0:
                logger.debug(f"No room to add to {symbol} position")
                return 0
            
            # Calculate shares (round down)
            shares = int(available_cash / current_price)
            
            # Ensure minimum position value is sensible
            min_value = portfolio_value * 0.01  # At least 1% of portfolio
            if shares * current_price < min_value:
                shares = int(min_value / current_price)
            
            # Final check against cash
            if shares * current_price > cash:
                shares = int(cash / current_price)
            
            logger.debug(f"BUY signal for {symbol}: {shares} shares @ ${current_price:.2f}")
            return max(0, shares)
        
        elif signal["action"] == "SELL":
            # Can only sell what we have
            if current_position_qty <= 0:
                logger.debug(f"No position to sell for {symbol}")
                return 0
            
            # Sell based on confidence (higher confidence = sell more)
            shares_to_sell = int(current_position_qty * confidence)
            
            # Minimum: sell at least 1 share if we have any
            shares_to_sell = max(1, shares_to_sell)
            
            # Don't exceed what we have
            shares_to_sell = min(shares_to_sell, current_position_qty)
            
            logger.debug(f"SELL signal for {symbol}: {shares_to_sell} shares")
            return shares_to_sell
        
        return 0

    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """
        Calculate stop loss price for risk management.
        
        Args:
            entry_price: Position entry price
            side: "buy" for long positions, "sell" for short
        
        Returns:
            Stop loss price
        """
        if side.lower() == "buy":
            return entry_price * (1 - self.stop_loss_pct)
        else:
            return entry_price * (1 + self.stop_loss_pct)

    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """
        Calculate take profit price.
        
        Args:
            entry_price: Position entry price
            side: "buy" for long positions, "sell" for short
        
        Returns:
            Take profit price
        """
        if side.lower() == "buy":
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)

    def calculate_risk_reward_ratio(
        self, 
        entry_price: float, 
        stop_loss: float, 
        take_profit: float
    ) -> float:
        """
        Calculate risk/reward ratio for a trade.
        
        Returns:
            Risk/reward ratio (higher is better)
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return 0.0
        
        return reward / risk

    async def check_portfolio_risk(self) -> Dict[str, Any]:
        """
        Analyze current portfolio risk exposure.
        
        Returns:
            Dictionary with risk metrics
        """
        await self.refresh_cache()
        
        portfolio_value = float(self._account_cache.get("equity", 0))
        cash = float(self._account_cache.get("cash", 0))
        
        positions = []
        total_exposure = 0.0
        
        for symbol, position in self._positions_cache.items():
            market_value = float(position.get("market_value", 0))
            unrealized_pl = float(position.get("unrealized_pl", 0))
            unrealized_plpc = float(position.get("unrealized_plpc", 0)) * 100
            
            exposure_pct = (market_value / portfolio_value * 100) if portfolio_value > 0 else 0
            total_exposure += market_value
            
            positions.append({
                "symbol": symbol,
                "market_value": market_value,
                "exposure_pct": exposure_pct,
                "unrealized_pl": unrealized_pl,
                "unrealized_plpc": unrealized_plpc,
            })
        
        # Sort by exposure
        positions.sort(key=lambda x: x["exposure_pct"], reverse=True)
        
        exposure_pct = (total_exposure / portfolio_value * 100) if portfolio_value > 0 else 0
        
        return {
            "portfolio_value": portfolio_value,
            "cash": cash,
            "total_exposure": total_exposure,
            "exposure_pct": exposure_pct,
            "position_count": len(positions),
            "positions": positions,
            "risk_level": self._assess_risk_level(exposure_pct),
        }

    def _assess_risk_level(self, exposure_pct: float) -> str:
        """Assess overall risk level."""
        if exposure_pct < 30:
            return "LOW"
        elif exposure_pct < 60:
            return "MODERATE"
        elif exposure_pct < 80:
            return "HIGH"
        else:
            return "VERY_HIGH"


class RiskManager:
    """Additional risk management utilities."""
    
    @staticmethod
    def kelly_criterion(win_rate: float, win_loss_ratio: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Args:
            win_rate: Probability of winning (0-1)
            win_loss_ratio: Average win / Average loss
        
        Returns:
            Optimal fraction of portfolio to risk (0-1)
        """
        if win_loss_ratio <= 0:
            return 0.0
        
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Half-Kelly is more conservative and commonly used
        half_kelly = kelly / 2
        
        # Clamp to reasonable bounds
        return max(0, min(0.25, half_kelly))

    @staticmethod
    def calculate_var(returns: list, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: List of historical returns
            confidence: Confidence level (default 95%)
        
        Returns:
            VaR as a percentage
        """
        import numpy as np
        
        if not returns:
            return 0.0
        
        returns_array = np.array(returns)
        var = np.percentile(returns_array, (1 - confidence) * 100)
        
        return abs(var)
