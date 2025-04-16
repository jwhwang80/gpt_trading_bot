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

    def __init__(self, output_dir: str = "output/onchain_data"):
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

    def calculate_indicator(self, indicator_name: str, update_csv: bool = True) -> Dict[str, Any]:
        """
        Calculate a specific onchain indicator and optionally update its CSV file.

        Args:
            indicator_name (str): Name of the indicator to calculate
            update_csv (bool): Whether to update the CSV file

        Returns:
            Dict[str, Any]: Indicator data
        """
        if indicator_name not in self.indicators:
            raise ValueError(f"Indicator not registered: {indicator_name}")

        indicator = self.indicators[indicator_name]

        # Load existing data
        historical_data = None
        csv_path = os.path.join(self.output_dir, f"{indicator_name.lower()}_data.csv")

        if hasattr(indicator, 'load_historical_data'):
            historical_data = indicator.load_historical_data()
        elif os.path.exists(csv_path):
            try:
                historical_data = pd.read_csv(csv_path)
                if 'date' in historical_data.columns:
                    historical_data['date'] = pd.to_datetime(historical_data['date'])
                    historical_data = historical_data.sort_values('date', ascending=False)
            except Exception as e:
                print(f"Error loading CSV for {indicator_name}: {e}")

        # Check if today's data already exists
        today = datetime.now().date()
        today_data_exists = False

        if historical_data is not None and not historical_data.empty and 'date' in historical_data.columns:
            latest_date = pd.to_datetime(historical_data['date'].iloc[0]).date()
            today_data_exists = (latest_date == today)

        # Create a dummy DataFrame for indicator calculation
        dummy_df = pd.DataFrame(index=[datetime.now()])

        if today_data_exists and not update_csv:
            print(f"{indicator_name}: Today's data already exists. Skipping API call.")
            # Extract latest data from historical data
            latest_row = historical_data.iloc[0]
            # Convert to dictionary for consistent format with API responses
            latest_data = latest_row.to_dict()
        else:
            # Calculate indicator (will make API call)
            print(f"{indicator_name}: Fetching latest data from API...")
            dummy_df = indicator.calculate(dummy_df)

            # Get the latest result
            if indicator.column_name in dummy_df.columns:
                latest_value = dummy_df[indicator.column_name].iloc[-1]
                latest_data = {
                    'date': datetime.now(),
                    indicator.column_name: latest_value
                }
            else:
                print(f"Warning: No data returned for {indicator_name}")
                latest_data = {'date': datetime.now()}

        # Store latest data for later use
        self.latest_data[indicator_name] = latest_data

        return latest_data

    def calculate_all_indicators(self, update_csv: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Calculate all registered onchain indicators.

        Args:
            update_csv (bool): Whether to update CSV files

        Returns:
            Dict[str, Dict[str, Any]]: All indicator data
        """
        results = {}

        for indicator_name in self.indicators:
            try:
                print(f"Calculating onchain indicator: {indicator_name}...")
                indicator_data = self.calculate_indicator(indicator_name, update_csv)
                results[indicator_name] = indicator_data
            except Exception as e:
                print(f"Error calculating {indicator_name}: {e}")
                results[indicator_name] = {'error': str(e)}

        return results

    def interpret_indicators(self) -> Dict[str, Dict[str, Any]]:
        """
        Interpret current values of each onchain indicator.

        Returns:
            Dict[str, Dict[str, Any]]: Interpretation results for each indicator
        """
        interpretations = {}

        for indicator_name, indicator in self.indicators.items():
            try:
                # Create a row-like structure from latest data
                if indicator_name in self.latest_data:
                    data = self.latest_data[indicator_name]
                    row = pd.Series(data)
                    interpretations[indicator_name] = indicator.interpret(row)
                else:
                    interpretations[indicator_name] = {'error': 'No data available'}
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

    def calculate_onchain_indicators(self, force_update: bool = False) -> Tuple[
        Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """
        Calculate all onchain indicators and their interpretations.

        Args:
            force_update (bool): Whether to force update all indicators even if today's data exists

        Returns:
            Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]: Raw indicator data and interpretation results
        """
        print(f"Calculating onchain indicators...")

        # Calculate all indicators (updates CSV files)
        indicators_data = self.calculate_all_indicators(update_csv=force_update)

        # Interpret the data
        interpretations = self.interpret_indicators()

        # Format the results
        result = {
            'timestamp': datetime.now().isoformat(),
            'indicators': interpretations
        }

        print(f"Onchain indicators calculation completed")

        # Store the latest data for later use
        self.latest_data = indicators_data

        return indicators_data, result

    def get_registered_indicators(self) -> Dict[str, Dict[str, str]]:
        """
        Get information about all registered indicators.

        Returns:
            Dict[str, Dict[str, str]]: Indicator names and descriptions
        """
        return {name: indicator.get_info() for name, indicator in self.indicators.items()}

    def list_indicator_csv_files(self) -> List[str]:
        """
        List all indicator CSV files in the output directory.

        Returns:
            List[str]: List of CSV file paths
        """
        csv_files = []
        for indicator_name in self.indicators:
            csv_path = os.path.join(self.output_dir, f"{indicator_name.lower()}_data.csv")
            if os.path.exists(csv_path):
                csv_files.append(csv_path)
        return csv_files

    def get_latest_onchain_data(self) -> Dict[str, Any]:
        """
        Get the latest calculated onchain data.

        Returns:
            Dict[str, Any]: Latest onchain data or empty dict if not calculated yet
        """
        return self.latest_data if self.latest_data else {}

    def load_indicator_from_csv(self, indicator_name: str) -> Optional[pd.DataFrame]:
        """
        Load indicator data from its CSV file.

        Args:
            indicator_name (str): The name of the indicator

        Returns:
            Optional[pd.DataFrame]: DataFrame with indicator data or None if file doesn't exist
        """
        if indicator_name not in self.indicators:
            print(f"Unknown indicator: {indicator_name}")
            return None

        csv_path = os.path.join(self.output_dir, f"{indicator_name.lower()}_data.csv")

        if not os.path.exists(csv_path):
            print(f"CSV file not found for {indicator_name}")
            return None

        try:
            df = pd.read_csv(csv_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date', ascending=False)
            return df
        except Exception as e:
            print(f"Error loading {indicator_name} CSV: {e}")
            return None


# Example usage when run as a standalone module
if __name__ == "__main__":
    # Initialize calculator
    calculator = OnchainIndicatorCalculator()

    # Check registered indicators
    print("Registered onchain indicators:")
    for name, info in calculator.get_registered_indicators().items():
        print(f"- {name}: {info['description']}")

    # Calculate indicators
    indicators_data, results = calculator.calculate_onchain_indicators()

    # Print CSV file paths
    print("\nOnchain indicator CSV files:")
    for csv_file in calculator.list_indicator_csv_files():
        print(f"- {csv_file}")

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