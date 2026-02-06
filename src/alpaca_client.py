"""Alpaca Markets API Client."""

import aiohttp
from typing import Optional


class AlpacaClient:
    """Async client for Alpaca Trading API."""

    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.data_url = "https://data.alpaca.markets"
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }

    async def _request(self, method: str, url: str, **kwargs) -> dict:
        """Make authenticated request."""
        async with aiohttp.ClientSession(headers=self.headers) as session:
            async with session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()

    async def get_account(self) -> dict:
        """Get account information."""
        return await self._request("GET", f"{self.base_url}/v2/account")

    async def get_clock(self) -> dict:
        """Get market clock status."""
        return await self._request("GET", f"{self.base_url}/v2/clock")

    async def get_positions(self) -> list:
        """Get all open positions."""
        return await self._request("GET", f"{self.base_url}/v2/positions")

    async def get_position(self, symbol: str) -> Optional[dict]:
        """Get position for a symbol."""
        try:
            return await self._request(
                "GET", f"{self.base_url}/v2/positions/{symbol}"
            )
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                return None
            raise

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1D",
        limit: int = 100,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> list:
        """Get historical bar data."""
        params = {"timeframe": timeframe, "limit": limit}
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        url = f"{self.data_url}/v2/stocks/{symbol}/bars"
        response = await self._request("GET", url, params=params)
        return response.get("bars", [])

    async def get_latest_quote(self, symbol: str) -> dict:
        """Get latest quote for symbol."""
        url = f"{self.data_url}/v2/stocks/{symbol}/quotes/latest"
        return await self._request("GET", url)

    async def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        type: str = "market",
        time_in_force: str = "day",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> dict:
        """Submit a new order."""
        order_data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": type,
            "time_in_force": time_in_force,
        }
        if limit_price:
            order_data["limit_price"] = str(limit_price)
        if stop_price:
            order_data["stop_price"] = str(stop_price)

        return await self._request(
            "POST", f"{self.base_url}/v2/orders", json=order_data
        )

    async def cancel_order(self, order_id: str) -> None:
        """Cancel an order."""
        await self._request("DELETE", f"{self.base_url}/v2/orders/{order_id}")

    async def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        await self._request("DELETE", f"{self.base_url}/v2/orders")
