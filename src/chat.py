"""Natural Language Interface for Trading Bot."""

import asyncio
import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

from openai import AsyncOpenAI


class TradingChatInterface:
    """
    Natural language interface for querying positions, getting market analysis,
    and controlling the trading bot through conversation.
    """

    def __init__(self, api_key: str, trading_bot=None):
        self.client = AsyncOpenAI(api_key=api_key)
        self.trading_bot = trading_bot
        self.conversation_history: List[Dict[str, str]] = []
        
        # System prompt defining the assistant's capabilities
        self.system_prompt = """You are an AI trading assistant for a stock trading bot. You have access to:

1. Portfolio information (positions, P&L, cash balance)
2. Market analysis tools (technical indicators, sentiment analysis)
3. Trading capabilities (buy/sell orders on paper trading)
4. Historical data and backtesting results

When the user asks about their portfolio or positions, respond with the actual data provided.
When asked to analyze a stock, provide technical and sentiment insights.
When asked to execute trades, confirm the action before proceeding.

Always be helpful, precise with numbers, and warn about risks when appropriate.
Format monetary values with $ and appropriate decimal places.
Format percentages with % symbol.

If you need to take an action, respond with a JSON command block like:
```json
{"action": "get_positions"}
```

Available actions:
- get_positions: Get all current positions
- get_account: Get account balance and equity
- analyze_symbol: Analyze a stock (requires "symbol" parameter)
- get_quote: Get current price for a symbol
- place_order: Place an order (requires "symbol", "side", "quantity")
- get_history: Get recent trade history
"""

    async def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trading action and return results."""
        action_type = action.get("action", "")
        
        if not self.trading_bot:
            return {"error": "Trading bot not initialized"}
        
        try:
            if action_type == "get_positions":
                positions = await self.trading_bot.client.get_positions()
                return {"positions": positions}
            
            elif action_type == "get_account":
                account = await self.trading_bot.client.get_account()
                return {
                    "equity": float(account.get("equity", 0)),
                    "cash": float(account.get("cash", 0)),
                    "buying_power": float(account.get("buying_power", 0)),
                    "portfolio_value": float(account.get("portfolio_value", 0))
                }
            
            elif action_type == "analyze_symbol":
                symbol = action.get("symbol", "").upper()
                if not symbol:
                    return {"error": "Symbol required"}
                analysis = await self.trading_bot.analyze_symbol(symbol)
                return analysis
            
            elif action_type == "get_quote":
                symbol = action.get("symbol", "").upper()
                if not symbol:
                    return {"error": "Symbol required"}
                quote = await self.trading_bot.client.get_latest_quote(symbol)
                return quote
            
            elif action_type == "place_order":
                symbol = action.get("symbol", "").upper()
                side = action.get("side", "").lower()
                quantity = int(action.get("quantity", 0))
                
                if not all([symbol, side, quantity]):
                    return {"error": "Symbol, side, and quantity required"}
                
                order = await self.trading_bot.client.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    type="market"
                )
                return {"order": order, "status": "submitted"}
            
            elif action_type == "get_history":
                # Return recent trades from bot's history
                return {"trades": getattr(self.trading_bot, 'trade_history', [])}
            
            else:
                return {"error": f"Unknown action: {action_type}"}
        
        except Exception as e:
            return {"error": str(e)}

    async def _parse_and_execute_commands(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse response for JSON commands and execute them."""
        # Look for JSON command blocks
        json_pattern = r'```json\s*(\{[^`]+\})\s*```'
        matches = re.findall(json_pattern, response)
        
        results = []
        for match in matches:
            try:
                command = json.loads(match)
                if "action" in command:
                    result = await self._execute_action(command)
                    results.append(result)
            except json.JSONDecodeError:
                continue
        
        return results if results else None

    async def chat(self, user_message: str) -> str:
        """
        Process a natural language message and return a response.
        May execute trading actions based on the conversation.
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Build messages for API call
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history[-10:])  # Keep last 10 messages
        
        # First, get LLM's interpretation and any commands
        response = await self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        assistant_response = response.choices[0].message.content
        
        # Check if response contains commands to execute
        action_results = await self._parse_and_execute_commands(assistant_response)
        
        if action_results:
            # If there were actions, get a follow-up response with the data
            data_message = f"Here is the data from the executed actions: {json.dumps(action_results, indent=2)}"
            
            follow_up_messages = messages + [
                {"role": "assistant", "content": assistant_response},
                {"role": "system", "content": data_message},
                {"role": "user", "content": "Now provide a natural language summary of this data for the user."}
            ]
            
            follow_up = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=follow_up_messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            final_response = follow_up.choices[0].message.content
        else:
            final_response = assistant_response
        
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": final_response
        })
        
        return final_response

    async def quick_query(self, query_type: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a quick query without full conversation.
        Useful for programmatic access.
        """
        action = {"action": query_type, **kwargs}
        return await self._execute_action(action)

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


class ChatCLI:
    """Command-line interface for the trading chat assistant."""
    
    def __init__(self, chat_interface: TradingChatInterface):
        self.chat = chat_interface
        self.running = True
    
    async def run(self):
        """Run the interactive chat CLI."""
        print("\n" + "=" * 60)
        print("🤖 Trading Bot AI - Chat Interface")
        print("=" * 60)
        print("Type your questions about your portfolio, market analysis,")
        print("or trading commands. Type 'quit' or 'exit' to leave.")
        print("=" * 60 + "\n")
        
        while self.running:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye! Happy trading!")
                    break
                
                if user_input.lower() == 'clear':
                    self.chat.clear_history()
                    print("💭 Conversation history cleared.\n")
                    continue
                
                if user_input.lower() == 'help':
                    self._print_help()
                    continue
                
                print("\n🤔 Thinking...")
                response = await self.chat.chat(user_input)
                print(f"\n🤖 Assistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    def _print_help(self):
        """Print help message."""
        print("""
📚 Available Commands:
----------------------
• "Show my positions" - View current holdings
• "What's my account balance?" - Check cash and equity
• "Analyze AAPL" - Get technical analysis for a stock
• "What's the price of TSLA?" - Get current quote
• "Buy 10 shares of MSFT" - Place a buy order (paper trading)
• "Sell 5 GOOGL" - Place a sell order
• "Show my recent trades" - View trade history
• "clear" - Clear conversation history
• "quit" or "exit" - Exit the chat

💡 Tips:
• You can ask questions naturally, like talking to a financial advisor
• Always confirm before placing actual trades
• This is paper trading mode - no real money involved
""")


# Example standalone usage
async def main():
    """Run standalone chat interface (for testing)."""
    import os
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Create chat interface without trading bot (limited functionality)
    chat_interface = TradingChatInterface(api_key=api_key)
    cli = ChatCLI(chat_interface)
    await cli.run()


if __name__ == "__main__":
    asyncio.run(main())
