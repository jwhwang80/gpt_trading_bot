import json
from typing import Dict, Any, List
from datetime import datetime


class ReflectionPromptGenerator:
    @staticmethod
    def build_reflection_prompt(
            gpt_strategy: Dict[str, Any],
            entry_price: float = 0,
            exit_price: float = 0,
            profit_percentage: float = 0,
            market_cycle_info: str = None
    ) -> str:
        """
        Generate a prompt requesting GPT to reflect on the trading strategy.

        Args:
            gpt_strategy (Dict): The original strategy suggested by GPT
            entry_price (float): Entry price
            exit_price (float): Exit price
            profit_percentage (float): Profit percentage
            market_cycle_info (str): Additional market cycle information from onchain data

        Returns:
            str: Reflection prompt
        """
        # Prepare simplified strategy information (not too complex)
        strategy_info = {
            "action": gpt_strategy.get("action", "UNKNOWN"),
            "confidence": gpt_strategy.get("confidence", 0),
            "reasoning": gpt_strategy.get("reasoning", "No explanation"),
            "stop_loss": gpt_strategy.get("stop_loss", 0),
            "take_profit": gpt_strategy.get("take_profit", 0),
        }

        # Add market cycle info if available
        market_cycle_section = ""
        if market_cycle_info:
            market_cycle_section = f"""
    Market Cycle Information:
    {market_cycle_info}
    """

        prompt_template = f"""
    Please conduct an in-depth reflection on the recently executed trading strategy. This reflection is important for future strategy improvements.

    Original Strategy:
    {json.dumps(strategy_info, indent=2, ensure_ascii=False)}

    Trade Results:
    - Entry Price: {entry_price}
    - Exit Price: {exit_price}
    - Profit/Loss: {profit_percentage:.2f}%
    - Trade Type: {"Long" if gpt_strategy.get("action") == "BUY" else "Short" if gpt_strategy.get("action") == "SELL" else "Hold"}
    {market_cycle_section}

    Analysis Perspectives:
    1. Detailed analysis of factors contributing to the strategy's success/failure
    2. Identification of differences between the outcome and the initial strategy
    3. Suggestions for improvements in similar market conditions in the future
    4. Recommendations for improvements in risk management
    5. Evaluation of market cycle positioning and onchain data interpretation (if available)
    6. Respond in JSON format (follow the structure below)

    Response JSON Structure:
    {{
        "overall_assessment": "Success/Partial Success/Failure",
        "key_insights": ["Insight 1", "Insight 2", "Insight 3"],
        "improvement_recommendations": ["Recommendation 1", "Recommendation 2", "Recommendation 3"],
        "confidence_in_original_strategy": 0-100,
        "risk_management_score": 0-100,
        "market_cycle_evaluation": "Your assessment of market cycle position and its impact on the strategy",
        "future_considerations": "Points to consider when executing similar strategies in the future"
    }}

    Please respond in JSON format only.
    """

        return prompt_template

    @staticmethod
    def parse_reflection_response(reflection_response: str) -> Dict[str, Any]:
        """
        Parses GPT's reflection response.

        Args:
            reflection_response (str): GPT's reflection response string

        Returns:
            Dict: Parsed reflection result
        """
        try:
            # Handle truncated responses
            if '{' in reflection_response and '}' not in reflection_response:
                print("Response was truncated. Recovering partial response.")
                # Return a default structure for partial responses
                return {
                    "overall_assessment": "Partial Response",
                    "key_insights": ["Response was truncated and could not be fully parsed."],
                    "improvement_recommendations": ["Try again to get a complete response."],
                    "confidence_in_original_strategy": 50,
                    "reflection_date": datetime.now().isoformat()
                }

            # Extract just the JSON part
            if '{' in reflection_response and '}' in reflection_response:
                start = reflection_response.find('{')
                end = reflection_response.rfind('}') + 1
                json_str = reflection_response[start:end]
                parsed_data = json.loads(json_str)
            else:
                parsed_data = json.loads(reflection_response)

            # Check required fields
            required_fields = [
                "overall_assessment",
                "key_insights",
                "improvement_recommendations",
                "confidence_in_original_strategy"
            ]

            for field in required_fields:
                if field not in parsed_data:
                    if field == "overall_assessment":
                        parsed_data[field] = "Assessment Failed"
                    elif field in ["key_insights", "improvement_recommendations"]:
                        parsed_data[field] = ["Parsing Error"]
                    else:
                        parsed_data[field] = 0

            # Add date
            parsed_data["reflection_date"] = datetime.now().isoformat()

            return parsed_data

        except json.JSONDecodeError as e:
            print(f"Reflection response JSON parsing failed: {e}. Original response: {reflection_response[:100]}...")
            return {
                "overall_assessment": "Analysis Failed",
                "key_insights": ["JSON Parsing Error"],
                "improvement_recommendations": [],
                "confidence_in_original_strategy": 0,
                "reflection_date": datetime.now().isoformat()
            }