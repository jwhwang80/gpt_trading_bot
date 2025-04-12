import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
# Import indicator classes from relative path
from collectors import indicator_classes


class TechnicalIndicatorCalculator:
    """
    Calculator to compute and manage various technical indicators
    """

    def __init__(self, csv_file_path: Optional[str] = None, symbol: str = "BTCUSDT"):
        """
        Initialize technical indicator calculator

        Args:
            csv_file_path (Optional[str]): Path to CSV file with 1-minute data
            symbol (str): Cryptocurrency symbol
        """
        self.csv_file_path = csv_file_path or f"{symbol}_1m_3000day_data.csv"
        self.symbol = symbol
        self.df_1m = None
        self.df_4h = None
        self.df_1d = None
        self.indicators = {}

        # Register default indicators
        self.register_default_indicators()

    def register_indicator(self, indicator):
        """
        Register a new technical indicator.

        Args:
            indicator: Indicator object (inherits from BaseIndicator)
        """
        self.indicators[indicator.name] = indicator
        print(f"Indicator registered: {indicator.name} - {indicator.description}")

    def register_default_indicators(self) -> None:
        """Automatically register all technical indicators."""
        # Import all classes from indicator_classes module
        import inspect

        # Find all classes that inherit from BaseIndicator
        indicator_classes_list = []
        for name, obj in inspect.getmembers(indicator_classes):
            # Class that inherits from BaseIndicator but is not BaseIndicator itself
            if (inspect.isclass(obj) and
                    issubclass(obj, indicator_classes.BaseIndicator) and
                    obj is not indicator_classes.BaseIndicator):
                indicator_classes_list.append(obj)

        print(f"Found {len(indicator_classes_list)} indicator classes.")

        # Create and register instances of each indicator class
        for indicator_class in indicator_classes_list:
            try:
                # Output class name
                class_name = indicator_class.__name__
                print(f"Registering: {class_name}")

                # Create instance with default parameters
                indicator = indicator_class()

                # Register the instance
                self.register_indicator(indicator)

            except Exception as e:
                print(f"Error registering indicator {indicator_class.__name__}: {e}")

        print(f"Successfully registered {len(self.indicators)} indicators.")

    def load_data(self, days: int = 120) -> None:
        """
        Load 1-minute data and preprocess it.
        Load only the specified number of recent days.

        Args:
            days (int): Number of recent days to load (default: 120 days)
        """
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(f"Data file not found: {self.csv_file_path}")

        # Load CSV file
        print(f"Loading 1-minute data: {self.csv_file_path}")
        df = pd.read_csv(self.csv_file_path)

        # Convert 'Time' column to datetime
        df['Time'] = pd.to_datetime(df['Time'])

        # Sort by time (ascending)
        df = df.sort_values(by='Time', ascending=True)

        # Filter only recent days
        end_date = df['Time'].max()
        start_date = end_date - pd.Timedelta(days=days)
        df = df[df['Time'] >= start_date]

        print(f"Filtered recent {days} days of data: {start_date} ~ {end_date}")
        print(f"Number of data points after filtering: {len(df)}")

        # Set index
        df.set_index('Time', inplace=True)

        # Store filtered data
        self.df_1m = df

        print(f"1-minute data loading complete: {len(self.df_1m)} data points")

        # Resample to 4-hour and daily timeframes
        self._resample_data()

    def _resample_data(self) -> None:
        """Resample 1-minute data to 4-hour and daily timeframes."""
        if self.df_1m is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Resample to 4-hour timeframe
        print("Resampling to 4-hour timeframe...")
        self.df_4h = self.df_1m.resample('4H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        self.df_4h.dropna(inplace=True)
        print(f"4-hour resampling complete: {len(self.df_4h)} candles")

        # Resample to daily timeframe
        print("Resampling to daily timeframe...")
        self.df_1d = self.df_1m.resample('1D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        self.df_1d.dropna(inplace=True)
        print(f"Daily resampling complete: {len(self.df_1d)} candles")

    def calculate_indicator(self, df: pd.DataFrame, indicator_name: str) -> pd.DataFrame:
        """
        Calculate a specific indicator.

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
        Calculate all registered indicators.

        Args:
            df (pd.DataFrame): DataFrame with OHLCV data

        Returns:
            pd.DataFrame: DataFrame with all indicators added
        """
        result_df = df.copy()

        for indicator_name, indicator in self.indicators.items():
            try:
                print(f"Calculating {indicator_name}...")
                result_df = indicator.calculate(result_df)
            except Exception as e:
                print(f"Error calculating {indicator_name}: {e}")

        return result_df

    def interpret_indicators(self, row: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Interpret current values of each indicator.

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

    def calculate_technical_indicators(self, timeframe: str = '4H', periods: int = 120) -> Tuple[
        pd.DataFrame, Dict[str, Any]]:
        """
        Calculate technical indicators for a specific timeframe.

        Args:
            timeframe (str): Timeframe ('4H' or '1D')
            periods (int): Number of recent candles to use

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: DataFrame with calculated indicators and interpretation results
        """
        if timeframe == '4H':
            df = self.df_4h.copy()
        elif timeframe == '1D':
            df = self.df_1d.copy()
        else:
            raise ValueError("Unsupported timeframe. Only '4H' or '1D' are allowed.")

        # Use only recent candles
        df = df.iloc[-periods:]

        if len(df) < periods:
            print(f"Warning: Only {len(df)} candles available, less than the requested {periods} candles.")

        print(f"Calculating technical indicators for {timeframe} timeframe (candles: {len(df)})...")

        # Calculate all indicators
        df_with_indicators = self.calculate_all_indicators(df)

        # Interpret the most recent data
        latest_row = df_with_indicators.iloc[-1]
        interpretations = self.interpret_indicators(latest_row)

        # Format the results
        result = {
            'timeframe': timeframe,
            'timestamp': df_with_indicators.index[-1],
            'close_price': latest_row['Close'],
            'indicators': interpretations
        }

        print(f"Technical indicators calculation completed for {timeframe} timeframe")

        return df_with_indicators, result

    def calculate_all_timeframes(self) -> Dict[str, Dict[str, Any]]:
        """Calculate all technical indicators for 4-hour and daily timeframes."""
        if self.df_1m is None:
            self.load_data()

        # Calculate 4-hour indicators
        df_4h, interpretation_4h = self.calculate_technical_indicators('4H', 120)

        # Calculate daily indicators
        df_1d, interpretation_1d = self.calculate_technical_indicators('1D', 120)

        return {
            '4H': {
                'data': df_4h,
                'interpretation': interpretation_4h
            },
            '1D': {
                'data': df_1d,
                'interpretation': interpretation_1d
            }
        }

    def save_indicators_to_csv(self, results: Dict[str, Dict[str, Any]], output_dir: str = 'output') -> None:
        """
        Save calculated indicators to CSV files.

        Args:
            results (dict): Results from calculate_all_timeframes()
            output_dir (str): Output directory
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Include current time in filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save data for each timeframe
        for timeframe in results:
            df = results[timeframe]['data']
            filename = f"{output_dir}/{self.symbol}_{timeframe}_{timestamp}.csv"
            df.to_csv(filename)
            print(f"Saved {timeframe} indicators to: {filename}")

        # Save interpretation results
        with open(f"{output_dir}/interpretation_{timestamp}.txt", 'w') as f:
            f.write(f"Technical Indicator Interpretation ({timestamp})\n\n")

            for timeframe in results:
                f.write(f"==== {timeframe} Timeframe ====\n")
                interp = results[timeframe]['interpretation']
                f.write(f"Time: {interp['timestamp']}\n")
                f.write(f"Close Price: {interp['close_price']}\n\n")

                for indicator_name, indicator_data in interp['indicators'].items():
                    f.write(f"{indicator_name}:\n")
                    for key, value in indicator_data.items():
                        if isinstance(value, (float, np.float64)):
                            f.write(f"  {key}: {value:.2f}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                    f.write("\n")

                f.write("\n\n")

        print(f"Saved interpretation results to: {output_dir}/interpretation_{timestamp}.txt")

    def get_registered_indicators(self) -> Dict[str, Dict[str, str]]:
        """
        Get information about all registered indicators.

        Returns:
            Dict[str, Dict[str, str]]: Indicator names and descriptions
        """
        return {name: indicator.get_info() for name, indicator in self.indicators.items()}


# Example usage when run as a standalone module
if __name__ == "__main__":
    # Initialize calculator
    calculator = TechnicalIndicatorCalculator('../BTCUSDT_1m_3000day_data.csv')

    # Check registered indicators
    print("Registered indicators:")
    for name, info in calculator.get_registered_indicators().items():
        print(f"- {name}: {info['description']}")

    # Load data
    calculator.load_data()

    # Calculate indicators for all timeframes
    results = calculator.calculate_all_timeframes()
    print(results)
    # Save results
    calculator.save_indicators_to_csv(results)