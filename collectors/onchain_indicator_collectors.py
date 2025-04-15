import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

from collectors.base_indicator import BaseIndicator
# Import indicator classes from relative path
from collectors import onchain_indicators


class OnchainIndicatorCalculator:
    """
    Calculator to compute and manage various onchain indicators
    """

    def __init__(self, output_dir: str = "output"):
        """
        Initialize onchain indicator calculator

        Args:
            output_dir (str): Directory to store output data
        """
        self.output_dir = output_dir
        self.indicators = {}
        self.latest_data = {}

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Register default indicators
        self.register_default_indicators()

    def register_indicator(self, indicator: BaseIndicator) -> None:
        """
        Register a new onchain indicator.

        Args:
            indicator: Indicator object (inherits from BaseIndicator)
        """
        self.indicators[indicator.name] = indicator
        print(f"Onchain indicator registered: {indicator.name} - {indicator.description}")

    def register_default_indicators(self) -> None:
        """Automatically register all onchain indicators."""
        # Import all classes from onchain_indicators module
        import inspect

        # Find all classes that inherit from OnchainIndicatorBase
        indicator_classes_list = []
        for name, obj in inspect.getmembers(onchain_indicators):
            # Class that inherits from OnchainIndicatorBase or BaseIndicator
            if (inspect.isclass(obj) and
                (issubclass(obj, BaseIndicator) or
                 (hasattr(obj, '__module__') and obj.__module__ == 'collectors.onchain_indicators')) and
                obj is not BaseIndicator and
                obj is not onchain_indicators.OnchainIndicatorBase):
                indicator_classes_list.append(obj)

        print(f"Found {len(indicator_classes_list)} onchain indicator classes.")

        # Create and register instances of each indicator class
        for indicator_class in indicator_classes_list:
            try:
                # Output class name
                class_name = indicator_class.__name__
                print(f"Registering onchain indicator: {class_name}")

                # Create instance with default parameters
                indicator = indicator_class()

                # Register the instance
                self.register_indicator(indicator)

            except Exception as e:
                print(f"Error registering onchain indicator {indicator_class.__name__}: {e}")

        print(f"Successfully registered {len(self.indicators)} onchain indicators.")

    def calculate_indicator(self, df: pd.DataFrame, indicator_name: str) -> pd.DataFrame:
        """
        Calculate a specific onchain indicator.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            indicator_name (str): Name of the indicator to calculate

        Returns:
            pd.DataFrame: DataFrame with calculated indicator added
        """
        if indicator_name not in self.indicators:
            raise ValueError(f"Indicator not registered: {indicator_name}")

        return self.indicators[indicator_name].calculate(df)

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all registered onchain indicators.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data

        Returns:
            pd.DataFrame: DataFrame with all indicators added
        """
        result_df = df.copy()

        for indicator_name, indicator in self.indicators.items():
            try:
                print(f"Calculating onchain indicator: {indicator_name}...")
                result_df = indicator.calculate(result_df)
            except Exception as e:
                print(f"Error calculating {indicator_name}: {e}")

        return result_df

    def interpret_indicators(self, row: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Interpret current values of each onchain indicator.

        Args:
            row (pd.Series): Data series containing indicator values

        Returns:
            Dict[str, Dict[str, Any]]: Interpretation results for each indicator
        """
        interpretations = {}

        for indicator_name, indicator in self.indicators.items():
            try:
                interpretations[indicator_name] = indicator.interpret(row)
            except Exception as e:
                print(f"Error interpreting {indicator_name}: {e}")
                interpretations[indicator_name] = {'error': str(e)}

        return interpretations

    def create_empty_dataframe(self) -> pd.DataFrame:
        """
        Create an empty dataframe with current timestamp for onchain data.

        Returns:
            pd.DataFrame: Empty dataframe with timestamp index
        """
        df = pd.DataFrame(index=[datetime.now()])
        return df

    def calculate_onchain_indicators(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Calculate all onchain indicators and their interpretations.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: DataFrame with calculated indicators and interpretation results
        """
        # Create an empty dataframe with current timestamp
        df = self.create_empty_dataframe()

        print(f"Calculating onchain indicators...")

        # Calculate all indicators
        df_with_indicators = self.calculate_all_indicators(df)

        # Interpret the most recent data
        latest_row = df_with_indicators.iloc[-1]
        interpretations = self.interpret_indicators(latest_row)

        # Format the results
        result = {
            'timestamp': df_with_indicators.index[-1],
            'indicators': interpretations
        }

        print(f"Onchain indicators calculation completed")

        # Store the latest data for later use
        self.latest_data = result

        return df_with_indicators, result

    def get_registered_indicators(self) -> Dict[str, Dict[str, str]]:
        """
        Get information about all registered indicators.

        Returns:
            Dict[str, Dict[str, str]]: Indicator names and descriptions
        """
        return {name: indicator.get_info() for name, indicator in self.indicators.items()}

    def save_indicators_to_csv(self, df_with_indicators: pd.DataFrame) -> str:
        """
        Save calculated indicators to CSV file.

        Args:
            df_with_indicators (pd.DataFrame): DataFrame with calculated indicators

        Returns:
            str: Path to saved CSV file
        """
        # Include current time in filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save data
        filename = f"{self.output_dir}/onchain_indicators_{timestamp}.csv"
        df_with_indicators.to_csv(filename)
        print(f"Saved onchain indicators to: {filename}")

        return filename

    def get_latest_onchain_data(self) -> Dict[str, Any]:
        """
        Get the latest calculated onchain data.

        Returns:
            Dict[str, Any]: Latest onchain data or empty dict if not calculated yet
        """
        return self.latest_data if self.latest_data else {}


# Example usage when run as a standalone module
if __name__ == "__main__":
    # Initialize calculator
    calculator = OnchainIndicatorCalculator()

    # Check registered indicators
    print("Registered onchain indicators:")
    for name, info in calculator.get_registered_indicators().items():
        print(f"- {name}: {info['description']}")

    # Calculate indicators
    df, results = calculator.calculate_onchain_indicators()

    # Save results
    calculator.save_indicators_to_csv(df)

    # Print interpretations
    for indicator_name, interpretation in results['indicators'].items():
        print(f"\n{indicator_name}:")
        for key, value in interpretation.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value}")