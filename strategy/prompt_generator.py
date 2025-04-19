import json
from typing import Dict, Any, List, Optional
import pandas as pd
import os
from datetime import datetime


class PromptGenerator:
    @staticmethod
    def generate_strategy_prompt(
            market_data: Dict[str, Any],
            historical_trades: Optional[List[Dict[str, Any]]] = None,
            onchain_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a GPT prompt for trading strategy based on market data,
        onchain data, and historical trades

        Args:
            market_data (Dict): Current market data
            historical_trades (Optional[List]): Historical trade records
            onchain_data (Optional[Dict]): Onchain indicators data

        Returns:
            str: Prompt string to send to GPT
        """
        # Get primary timeframe info (default: 4H)
        primary_timeframe = market_data.get('primary_timeframe', '4H')

        prompt = f"""
    Please analyze the current market conditions and suggest a trading strategy for BTCUSDT.

    Basic Market Information:
    - Current Price: {market_data.get('price', 0)}
    - Timestamp: {market_data.get('timestamp', 'No information')}
    - Primary Timeframe: {primary_timeframe}
    """

        # Add indicators data for multiple timeframes
        timeframes_data = market_data.get('timeframes', {})

        # Add data for each timeframe
        for timeframe, tf_data in timeframes_data.items():
            prompt += f"\n{timeframe} Timeframe Technical Indicators:\n"

            # Add indicator data if available
            indicators = tf_data.get('indicators', {})
            if indicators:
                for indicator_name, indicator_data in indicators.items():
                    # Display indicator name and value in a concise format
                    if isinstance(indicator_data, dict) and 'signal' in indicator_data:
                        prompt += f"- {indicator_name}: Signal = {indicator_data['signal']}"
                        if 'value' in indicator_data:
                            prompt += f", Value = {indicator_data['value']}"
                        prompt += "\n"
            else:
                prompt += "- No indicator data available\n"

        # Add onchain data if available
        if onchain_data and 'indicators' in onchain_data:
            prompt += "\nOnchain Indicators Analysis (Long-Term Market Perspective):\n"
            prompt += "Onchain indicators are crucial for understanding long-term market trends and macro cycle positioning. These metrics are more valuable for identifying market cycles and long-term investment strategies rather than short-term price fluctuations.\n\n"

            for indicator_name, indicator_data in onchain_data['indicators'].items():
                if isinstance(indicator_data, dict):
                    # For MVRV Z-Score or similar structured indicators
                    if 'signal' in indicator_data:
                        prompt += f"- {indicator_name}: Signal = {indicator_data['signal']}"

                        if 'market_phase' in indicator_data:
                            prompt += f", Market Phase = {indicator_data['market_phase']}"

                        if 'mvrv_ratio' in indicator_data and indicator_data['mvrv_ratio'] is not None:
                            prompt += f", MVRV Ratio = {indicator_data['mvrv_ratio']:.2f}"

                        if 'z_scores' in indicator_data and indicator_data['z_scores'] is not None:
                            z_scores = indicator_data['z_scores']
                            if 'historical' in z_scores and z_scores['historical'] is not None:
                                prompt += f", Historical Z-Score = {z_scores['historical']:.2f}"

                        prompt += "\n"

                        # Add detailed description if available
                        if 'description' in indicator_data:
                            prompt += f"  Description: {indicator_data['description']}\n"

                        # Add cycle position if available
                        if 'cycle_position' in indicator_data:
                            prompt += f"  Cycle Position: {indicator_data['cycle_position']}\n"

                    # For other types of indicators with different structures
                    else:
                        prompt += f"- {indicator_name}: "
                        for key, value in indicator_data.items():
                            if not isinstance(value, (dict, list)) and key not in ['error']:
                                prompt += f"{key} = {value}, "
                        prompt = prompt.rstrip(', ') + '\n'

        # Add historical trade information
        if historical_trades and len(historical_trades) > 0:
            prompt += "\nRecent Trade History:\n"
            for i, trade in enumerate(historical_trades[-3:]):  # Show only the last 3 trades
                action = trade.get('action', 'UNKNOWN')
                entry_price = trade.get('entry_price', 0)
                result = trade.get('result', 'No information')

                prompt += f"#{i + 1}: {action} @ {entry_price:.2f}, Result: {result}\n"

        # Add response requirements
        prompt += """
    Multi-Timeframe and Onchain Analysis Request:
    - Consider both technical indicators (4H and 1D timeframes) and onchain data in your analysis.
    - Evaluate if short-term (4H), long-term (1D) trends, and onchain fundamentals align.
    - Consider market cycle positioning based on onchain indicators like MVRV Z-Score.
    - Use onchain metrics as a long-term strategic guide, not for short-term timing.
    - Explain your strategy when different timeframes or data sources show conflicting signals.

    Long-Term Perspective on Onchain Data:
    - Onchain indicators reflect blockchain's fundamental health and adoption.
    - These metrics are most valuable for identifying major market cycle phases (accumulation, expansion, euphoria, capitulation).
    - Prioritize onchain data for strategic positioning rather than day-to-day trading decisions.
    - Align short-term technical trades with the broader market cycle identified by onchain metrics.
    - Consider divergence between onchain fundamentals and price as potential market inefficiency.

    Requirements:
    1. Respond in JSON format (e.g., {"action": "BUY/SELL/HOLD", "confidence": 0-100, "reasoning": "explanation", "stop_loss": price, "take_profit": price})
    2. The action must be one of: BUY, SELL, or HOLD.
    3. Confidence should be expressed as a number between 0-100.
    4. Clearly present the rationale for your strategy in the reasoning field.
    5. If you suggest entry (BUY or SELL), you must propose stop_loss and take_profit prices.
    6. Position size is proportional to confidence and will not exceed 5%.
    7. Consider the market conditions, technical indicators, and onchain data comprehensively.
    8. Include your interpretation of the market cycle position in your reasoning.
    9. Explain how onchain metrics influence your long-term outlook while technical indicators guide short-term execution.

    Please respond in JSON format only.
    """

        return prompt

    @staticmethod
    def parse_gpt_response(gpt_response: str) -> Dict[str, Any]:
        """
        Parses the GPT response into JSON.

        Args:
            gpt_response (str): GPT's response string

        Returns:
            Dict: Parsed strategy JSON
        """
        try:
            # Simple parsing logic to extract just the JSON part
            if '{' in gpt_response and '}' in gpt_response:
                start = gpt_response.find('{')
                end = gpt_response.rfind('}') + 1
                json_str = gpt_response[start:end]
                parsed_data = json.loads(json_str)
            else:
                parsed_data = json.loads(gpt_response)

            # Verify required fields
            if 'action' not in parsed_data:
                parsed_data['action'] = 'HOLD'
                parsed_data['reasoning'] = parsed_data.get('reasoning', '') + " (Default HOLD action applied)"

            if 'confidence' not in parsed_data:
                parsed_data['confidence'] = 0

            # Normalize action
            if parsed_data['action'].upper() not in ['BUY', 'SELL', 'HOLD']:
                parsed_data['action'] = 'HOLD'
                parsed_data['reasoning'] = parsed_data.get('reasoning', '') + " (Invalid action, HOLD applied)"

            parsed_data['action'] = parsed_data['action'].upper()

            return parsed_data

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}. Original response: {gpt_response[:100]}...")
            return {
                "action": "HOLD",
                "confidence": 0,
                "reasoning": "JSON parsing failed"
            }

    @staticmethod
    def load_historical_trades(max_trades: int = 5) -> List[Dict[str, Any]]:
        """
        Load historical trade records.

        Args:
            max_trades (int): Maximum number of trades to load

        Returns:
            List[Dict[str, Any]]: List of historical trades
        """
        try:
            if not os.path.exists('trades.xlsx'):
                return []

            df = pd.read_excel('trades.xlsx')

            # Filter by trade type
            entry_trades = df[df['type'] == 'ENTRY'].sort_values('timestamp', ascending=False).head(max_trades)
            exit_trades = df[df['type'] == 'EXIT']

            historical_trades = []

            for _, entry in entry_trades.iterrows():
                # Find matching exit for this entry
                matching_exit = exit_trades[
                    (exit_trades['round'] == entry['round']) &
                    (exit_trades['strategy_id'] == entry['strategy_id'])
                    ]

                trade_result = "In progress"
                profit_loss = 0

                if not matching_exit.empty:
                    exit_price = matching_exit.iloc[0]['exit_price']
                    if entry['symbol'].endswith('USDT'):  # USDT pair
                        if entry['action'] == 'BUY':
                            profit_loss = (exit_price - entry['entry_price']) / entry['entry_price'] * 100
                        else:  # SELL
                            profit_loss = (entry['entry_price'] - exit_price) / entry['entry_price'] * 100

                    trade_result = f"{'Profit' if profit_loss > 0 else 'Loss'} ({profit_loss:.2f}%)"

                historical_trades.append({
                    "action": entry['action'] if 'action' in entry else 'BUY',
                    "entry_price": entry['entry_price'],
                    "timestamp": entry['timestamp'],
                    "confidence": 0,  # This information might not be in Excel
                    "symbol": entry['symbol'],
                    "strategy_id": entry['strategy_id'],
                    "result": trade_result,
                    "profit_loss": profit_loss
                })

            return historical_trades

        except Exception as e:
            print(f"Error loading historical trades: {e}")
            return []