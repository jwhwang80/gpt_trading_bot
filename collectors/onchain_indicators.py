import pandas as pd
from typing import Dict, Any, List, Optional

from collectors.base_indicator import BaseIndicator
from collectors.mvrv_calculator import MVRVZScoreCalculator


class MVRVIndicator(BaseIndicator):
    """
    MVRV Z-Score indicator class

    An on-chain indicator based on the difference between Market Value and Realized Value
    divided by the standard deviation.
    """

    def __init__(self, csv_file: str = "output/mvrv_data.csv"):
        """
        Initialize MVRV indicator

        Args:
            csv_file (str): Path to CSV file containing MVRV data
        """
        super().__init__(
            name="MVRV_Z_Score",
            description="Market Value to Realized Value Z-Score (On-chain market cycle indicator)"
        )
        self.csv_file = csv_file
        self.calculator = MVRVZScoreCalculator(csv_file)

        # Define column names
        self.column_z_score_1y = "MVRV_Z_Score_1y"
        self.column_z_score_4y = "MVRV_Z_Score_4y"
        self.column_z_score_hist = "MVRV_Z_Score_Historical"
        self.column_mvrv_ratio = "MVRV_Ratio"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add MVRV indicator to dataframe (latest values only)

        Note: MVRV is calculated separately through a calculator, so we're just
        adding the latest values to all rows.

        Args:
            df (pd.DataFrame): OHLCV dataframe

        Returns:
            pd.DataFrame: Dataframe with MVRV indicators added
        """
        df_copy = df.copy()

        try:
            # Get latest MVRV data
            self.calculator.update_market_and_realized_cap()
            self.calculator.calculate_mvrv_data()
            self.calculator.save_to_csv()

            # Load MVRV data
            mvrv_df = pd.read_csv(self.csv_file)

            # Convert date format
            mvrv_df['date'] = pd.to_datetime(mvrv_df['date'])

            # Get latest data only (first row)
            latest_mvrv = mvrv_df.iloc[0]  # Assuming data is sorted in descending date order

            # Add the same latest MVRV values to all rows
            df_copy[self.column_mvrv_ratio] = latest_mvrv['mvrv_ratio']
            df_copy[self.column_z_score_1y] = latest_mvrv['mvrv_z_score_1y']
            df_copy[self.column_z_score_4y] = latest_mvrv['mvrv_z_score_4y']
            df_copy[self.column_z_score_hist] = latest_mvrv['mvrv_z_score_historical']

            print(
                f"Latest MVRV indicators added: MVRV Ratio = {latest_mvrv['mvrv_ratio']:.2f}, Historical Z-Score = {latest_mvrv['mvrv_z_score_historical']:.2f}")

        except Exception as e:
            print(f"Error calculating MVRV indicators: {e}")
            # If error occurs, add NaN columns
            df_copy[self.column_mvrv_ratio] = float('nan')
            df_copy[self.column_z_score_1y] = float('nan')
            df_copy[self.column_z_score_4y] = float('nan')
            df_copy[self.column_z_score_hist] = float('nan')

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """
        Interpret MVRV Z-Score values

        Args:
            row (pd.Series): Data series containing indicator values

        Returns:
            Dict[str, Any]: Indicator interpretation results
        """
        try:
            # Get Z-Score values
            mvrv_ratio = row.get(self.column_mvrv_ratio, None)
            z_score_1y = row.get(self.column_z_score_1y, None)
            z_score_4y = row.get(self.column_z_score_4y, None)
            z_score_hist = row.get(self.column_z_score_hist, None)

            # Check for NaN values
            if pd.isna(z_score_hist) or pd.isna(z_score_4y) or pd.isna(z_score_1y) or pd.isna(mvrv_ratio):
                return {
                    'mvrv_ratio': None,
                    'z_scores': {
                        '1y': None,
                        '4y': None,
                        'historical': None
                    },
                    'signal': 'UNKNOWN',
                    'market_phase': 'UNKNOWN',
                    'description': 'Insufficient data available'
                }

            # Interpret signal (based on historical Z-Score)
            if z_score_hist > 7:
                signal = 'EXTREME_OVERVALUED'
                description = 'Extremely overvalued state, likely approaching the peak of the Bitcoin market cycle.'
                market_phase = 'BUBBLE_TOP'
            elif z_score_hist > 3:
                signal = 'OVERVALUED'
                description = 'Overvalued state, market is overheated. High risk zone.'
                market_phase = 'EUPHORIA'
            elif z_score_hist > 1:
                signal = 'SLIGHTLY_OVERVALUED'
                description = 'Slightly overvalued state, possible short-term correction ahead.'
                market_phase = 'OPTIMISM'
            elif z_score_hist < -0.5:
                signal = 'UNDERVALUED'
                description = 'Undervalued state, favorable for long-term investment.'
                market_phase = 'FEAR'
            elif z_score_hist < -1:
                signal = 'OPPORTUNITY_TO_BUY'
                description = 'Significantly undervalued state, historically a good buying opportunity.'
                market_phase = 'CAPITULATION'
            else:
                signal = 'NEUTRAL'
                description = 'Close to fair value.'
                market_phase = 'NEUTRAL'

            # Check for timeframe divergence
            timeframe_signals = []
            if z_score_1y > 0 and z_score_4y < 0:
                timeframe_signals.append("Short-term (1y) overvalued, long-term (4y) undervalued")
            elif z_score_1y < 0 and z_score_4y > 0:
                timeframe_signals.append("Short-term (1y) undervalued, long-term (4y) overvalued")

            # Estimate market cycle position
            cycle_position = "Unknown"
            if mvrv_ratio > 3.0:
                cycle_position = "Top 10% (Late cycle)"
            elif mvrv_ratio > 2.0:
                cycle_position = "Top 25% (Mid-to-late cycle)"
            elif mvrv_ratio < 1.0:
                cycle_position = "Bottom 25% (Early cycle)"
            else:
                cycle_position = "Mid-range (Mid cycle)"

            return {
                'mvrv_ratio': mvrv_ratio,
                'z_scores': {
                    '1y': z_score_1y,
                    '4y': z_score_4y,
                    'historical': z_score_hist
                },
                'signal': signal,
                'market_phase': market_phase,
                'cycle_position': cycle_position,
                'timeframe_divergence': timeframe_signals,
                'description': description
            }

        except Exception as e:
            print(f"Error interpreting MVRV indicator: {e}")
            return {
                'error': str(e),
                'signal': 'ERROR'
            }

    def get_columns(self) -> List[str]:
        """Return list of column names"""
        return [
            self.column_mvrv_ratio,
            self.column_z_score_1y,
            self.column_z_score_4y,
            self.column_z_score_hist
        ]

# 추가 온체인 지표 클래스는 여기에 구현