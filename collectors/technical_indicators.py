import pandas as pd
import numpy as np
import talib
from typing import Dict, Any, List

from collectors.base_indicator import BaseIndicator


class RSIIndicator(BaseIndicator):
    """
    Relative Strength Index (RSI) indicator
    """

    def __init__(self, period: int = 14):
        super().__init__(
            name=f"RSI_{period}",
            description=f"Relative Strength Index (Period: {period})"
        )
        self.period = period
        self.column_name = f"RSI_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI"""
        df_copy = df.copy()
        df_copy[self.column_name] = talib.RSI(df_copy['Close'].values, timeperiod=self.period)
        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret RSI values"""
        value = row[self.column_name]
        if pd.isna(value):
            return {
                'value': None,
                'signal': 'UNKNOWN'
            }

        if value > 70:
            signal = 'OVERBOUGHT'
        elif value < 30:
            signal = 'OVERSOLD'
        else:
            signal = 'NEUTRAL'

        return {
            'value': value,
            'signal': signal,
            'strength': min(100, max(0, abs(value - 50) * 2))  # Strength between 0-100
        }

    def get_columns(self) -> List[str]:
        return [self.column_name]


class MACDIndicator(BaseIndicator):
    """
    Moving Average Convergence Divergence (MACD) indicator
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(
            name=f"MACD_{fast_period}_{slow_period}_{signal_period}",
            description=f"Moving Average Convergence Divergence (Fast Period: {fast_period}, Slow Period: {slow_period}, Signal Period: {signal_period})"
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.column_macd = "MACD"
        self.column_signal = "MACD_Signal"
        self.column_hist = "MACD_Hist"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD"""
        df_copy = df.copy()
        macd, signal, hist = talib.MACD(
            df_copy['Close'].values,
            fastperiod=self.fast_period,
            slowperiod=self.slow_period,
            signalperiod=self.signal_period
        )
        df_copy[self.column_macd] = macd
        df_copy[self.column_signal] = signal
        df_copy[self.column_hist] = hist
        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret MACD values"""
        macd_value = row[self.column_macd]
        signal_value = row[self.column_signal]
        hist_value = row[self.column_hist]

        if pd.isna(macd_value) or pd.isna(signal_value):
            return {
                'macd': None,
                'signal': None,
                'histogram': None,
                'trend_signal': 'UNKNOWN'
            }

        # Histogram trend detection (additional analysis)
        hist_trend = 'FLAT'

        # MACD line crosses above signal line: strong buy signal
        if macd_value > signal_value and hist_value > 0:
            trend_signal = 'STRONG_BULLISH'
        # MACD line is above signal line: weak buy signal
        elif macd_value > signal_value:
            trend_signal = 'BULLISH'
        # MACD line crosses below signal line: strong sell signal
        elif macd_value < signal_value and hist_value < 0:
            trend_signal = 'STRONG_BEARISH'
        # MACD line is below signal line: weak sell signal
        else:
            trend_signal = 'BEARISH'

        # Zero line analysis
        if macd_value > 0 and signal_value > 0:
            zero_line = 'ABOVE_ZERO'
        elif macd_value < 0 and signal_value < 0:
            zero_line = 'BELOW_ZERO'
        else:
            zero_line = 'MIXED'

        return {
            'macd': macd_value,
            'signal': signal_value,
            'histogram': hist_value,
            'trend_signal': trend_signal,
            'zero_line': zero_line
        }

    def get_columns(self) -> List[str]:
        return [self.column_macd, self.column_signal, self.column_hist]


class StochasticIndicator(BaseIndicator):
    """
    Stochastic Oscillator indicator
    """

    def __init__(self, fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3):
        super().__init__(
            name=f"Stochastic_{fastk_period}_{slowk_period}_{slowd_period}",
            description=f"Stochastic Oscillator (K Period: {fastk_period}, K Slow: {slowk_period}, D Slow: {slowd_period})"
        )
        self.fastk_period = fastk_period
        self.slowk_period = slowk_period
        self.slowd_period = slowd_period
        self.column_k = "Stochastic_K"
        self.column_d = "Stochastic_D"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        df_copy = df.copy()
        slowk, slowd = talib.STOCH(
            df_copy['High'].values,
            df_copy['Low'].values,
            df_copy['Close'].values,
            fastk_period=self.fastk_period,
            slowk_period=self.slowk_period,
            slowk_matype=0,
            slowd_period=self.slowd_period,
            slowd_matype=0
        )
        df_copy[self.column_k] = slowk
        df_copy[self.column_d] = slowd
        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret Stochastic values"""
        k_value = row[self.column_k]
        d_value = row[self.column_d]

        if pd.isna(k_value) or pd.isna(d_value):
            return {
                'k': None,
                'd': None,
                'signal': 'UNKNOWN'
            }

        if k_value > 80 and d_value > 80:
            signal = 'OVERBOUGHT'
        elif k_value < 20 and d_value < 20:
            signal = 'OVERSOLD'
        elif k_value > d_value and k_value < 80 and d_value < 80:
            signal = 'BULLISH_CROSSOVER'
        elif k_value < d_value and k_value > 20 and d_value > 20:
            signal = 'BEARISH_CROSSOVER'
        else:
            signal = 'NEUTRAL'

        return {
            'k': k_value,
            'd': d_value,
            'signal': signal
        }

    def get_columns(self) -> List[str]:
        return [self.column_k, self.column_d]


class BollingerBandsIndicator(BaseIndicator):
    """
    Bollinger Bands indicator
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(
            name=f"BB_{period}_{std_dev}",
            description=f"Bollinger Bands (Period: {period}, Std Dev: {std_dev})"
        )
        self.period = period
        self.std_dev = std_dev
        self.column_upper = "BB_Upper"
        self.column_middle = "BB_Middle"
        self.column_lower = "BB_Lower"
        self.column_width = "BB_Width"
        self.column_pct_b = "BB_Pct_B"  # %B indicator added

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df_copy = df.copy()
        upper, middle, lower = talib.BBANDS(
            df_copy['Close'].values,
            timeperiod=self.period,
            nbdevup=self.std_dev,
            nbdevdn=self.std_dev,
            matype=0
        )
        df_copy[self.column_upper] = upper
        df_copy[self.column_middle] = middle
        df_copy[self.column_lower] = lower
        df_copy[self.column_width] = (upper - lower) / middle * 100

        # Calculate %B (position of price within the bands as a value between 0-1)
        df_copy[self.column_pct_b] = (df_copy['Close'] - lower) / (upper - lower)

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret Bollinger Bands values"""
        close = row['Close']
        upper = row[self.column_upper]
        middle = row[self.column_middle]
        lower = row[self.column_lower]
        width = row[self.column_width]
        pct_b = row[self.column_pct_b]

        if pd.isna(upper) or pd.isna(middle) or pd.isna(lower):
            return {
                'upper': None,
                'middle': None,
                'lower': None,
                'width': None,
                'pct_b': None,
                'position': 'UNKNOWN',
                'signal': 'UNKNOWN'
            }

        # Price position analysis
        if close > upper:
            position = 'ABOVE_UPPER'
            signal = 'OVERBOUGHT'
        elif close < lower:
            position = 'BELOW_LOWER'
            signal = 'OVERSOLD'
        elif close > middle:
            position = 'ABOVE_MIDDLE'
            signal = 'NEUTRAL_BULLISH'
        else:
            position = 'BELOW_MIDDLE'
            signal = 'NEUTRAL_BEARISH'

        # Band width interpretation (narrow bands suggest potential expansion, wide bands suggest potential contraction)
        if width < 10:  # arbitrary threshold
            band_signal = 'SQUEEZE'
        elif width > 40:  # arbitrary threshold
            band_signal = 'EXPANSION'
        else:
            band_signal = 'NORMAL'

        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': width,
            'pct_b': pct_b,
            'position': position,
            'signal': signal,
            'band_signal': band_signal
        }

    def get_columns(self) -> List[str]:
        return [self.column_upper, self.column_middle, self.column_lower, self.column_width, self.column_pct_b]


class ATRIndicator(BaseIndicator):
    """
    Average True Range (ATR) indicator
    """

    def __init__(self, period: int = 14):
        super().__init__(
            name=f"ATR_{period}",
            description=f"Average True Range (Period: {period})"
        )
        self.period = period
        self.column_name = f"ATR_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR"""
        df_copy = df.copy()
        df_copy[self.column_name] = talib.ATR(
            df_copy['High'].values,
            df_copy['Low'].values,
            df_copy['Close'].values,
            timeperiod=self.period
        )
        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret ATR values"""
        value = row[self.column_name]
        close = row['Close']

        if pd.isna(value) or pd.isna(close):
            return {
                'value': None,
                'percent_of_price': None,
                'volatility': 'UNKNOWN'
            }

        # Calculate ATR as percentage of price
        percent_of_price = (value / close) * 100

        # Define volatility categories (arbitrary thresholds)
        if percent_of_price < 1:
            volatility = 'LOW'
        elif percent_of_price < 3:
            volatility = 'MEDIUM'
        else:
            volatility = 'HIGH'

        return {
            'value': value,
            'percent_of_price': percent_of_price,
            'volatility': volatility
        }

    def get_columns(self) -> List[str]:
        return [self.column_name]


class IchimokuIndicator(BaseIndicator):
    """
    Ichimoku Cloud indicator
    """

    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52):
        super().__init__(
            name=f"Ichimoku_{tenkan_period}_{kijun_period}_{senkou_b_period}",
            description=f"Ichimoku Cloud (Tenkan Period: {tenkan_period}, Kijun Period: {kijun_period}, Senkou B Period: {senkou_b_period})"
        )
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period

        self.column_tenkan = "Ichimoku_Tenkan"
        self.column_kijun = "Ichimoku_Kijun"
        self.column_senkou_a = "Ichimoku_Senkou_A"
        self.column_senkou_b = "Ichimoku_Senkou_B"
        self.column_chikou = "Ichimoku_Chikou"

    def _calc_donchian(self, high_series: pd.Series, low_series: pd.Series, period: int) -> pd.Series:
        """Calculate Donchian Channel (basic Ichimoku calculation)"""
        return (high_series.rolling(period).max() + low_series.rolling(period).min()) / 2

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud"""
        df_copy = df.copy()

        # Tenkan-sen (Conversion Line) - Short-term trend
        df_copy[self.column_tenkan] = self._calc_donchian(
            df_copy['High'], df_copy['Low'], self.tenkan_period
        )

        # Kijun-sen (Base Line) - Medium-term trend
        df_copy[self.column_kijun] = self._calc_donchian(
            df_copy['High'], df_copy['Low'], self.kijun_period
        )

        # Senkou Span A (Leading Span A) - First cloud boundary
        df_copy[self.column_senkou_a] = ((df_copy[self.column_tenkan] + df_copy[self.column_kijun]) / 2).shift(
            self.kijun_period)

        # Senkou Span B (Leading Span B) - Second cloud boundary
        df_copy[self.column_senkou_b] = self._calc_donchian(
            df_copy['High'], df_copy['Low'], self.senkou_b_period
        ).shift(self.kijun_period)

        # Chikou Span (Lagging Span) - Confirmation line
        df_copy[self.column_chikou] = df_copy['Close'].shift(-self.kijun_period)

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret Ichimoku Cloud values"""
        close = row['Close']
        tenkan = row[self.column_tenkan]
        kijun = row[self.column_kijun]
        senkou_a = row[self.column_senkou_a]
        senkou_b = row[self.column_senkou_b]

        if pd.isna(tenkan) or pd.isna(kijun) or pd.isna(senkou_a) or pd.isna(senkou_b):
            return {
                'tenkan': None,
                'kijun': None,
                'senkou_a': None,
                'senkou_b': None,
                'cloud_position': 'UNKNOWN',
                'tenkan_kijun_signal': 'UNKNOWN',
                'signal': 'UNKNOWN'
            }

        # Cloud position
        if close > max(senkou_a, senkou_b):
            cloud_position = 'ABOVE_CLOUD'
        elif close < min(senkou_a, senkou_b):
            cloud_position = 'BELOW_CLOUD'
        else:
            cloud_position = 'IN_CLOUD'

        # Tenkan/Kijun signal
        if tenkan > kijun:
            tenkan_kijun_signal = 'BULLISH'
        elif tenkan < kijun:
            tenkan_kijun_signal = 'BEARISH'
        else:
            tenkan_kijun_signal = 'NEUTRAL'

        # Cloud color (green when Senkou A > Senkou B, red otherwise)
        cloud_color = 'GREEN' if senkou_a > senkou_b else 'RED'

        # Overall signal
        if cloud_position == 'ABOVE_CLOUD' and tenkan_kijun_signal == 'BULLISH':
            signal = 'STRONG_BULLISH'
        elif cloud_position == 'BELOW_CLOUD' and tenkan_kijun_signal == 'BEARISH':
            signal = 'STRONG_BEARISH'
        elif cloud_position == 'ABOVE_CLOUD':
            signal = 'BULLISH'
        elif cloud_position == 'BELOW_CLOUD':
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'

        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'cloud_position': cloud_position,
            'cloud_color': cloud_color,
            'tenkan_kijun_signal': tenkan_kijun_signal,
            'signal': signal
        }

    def get_columns(self) -> List[str]:
        return [
            self.column_tenkan,
            self.column_kijun,
            self.column_senkou_a,
            self.column_senkou_b,
            self.column_chikou
        ]

class FibonacciIndicator(BaseIndicator):
    """
    Fibonacci Retracement indicator
    """

    def __init__(self, period: int = 100):
        super().__init__(
            name=f"Fibonacci_{period}",
            description=f"Fibonacci Retracement (Period: {period})"
        )
        self.period = period
        self.fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        self.columns = [f"Fib_{level}" for level in self.fib_levels]

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Fibonacci Retracement"""
        df_copy = df.copy()

        # Find recent major high and low
        for i in range(self.period, len(df_copy)):
            window = df_copy.iloc[i - self.period:i]

            # Major high
            high = window['High'].max()
            # Major low
            low = window['Low'].min()

            # Calculate Fibonacci levels
            for level, column in zip(self.fib_levels, self.columns):
                df_copy.loc[df_copy.index[i], column] = high - (high - low) * level

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret Fibonacci Retracement values"""
        close = row['Close']

        # Collect level values
        levels = {}
        for level, column in zip(self.fib_levels, self.columns):
            levels[f"{level:.3f}"] = row[column]

        # Check if values are valid
        if all(pd.isna(v) for v in levels.values()):
            return {
                'levels': levels,
                'current_zone': 'UNKNOWN',
                'support': None,
                'resistance': None
            }

        # Find current Fibonacci zone
        sorted_levels = sorted([(float(k), v) for k, v in levels.items() if not pd.isna(v)], key=lambda x: x[1])

        current_zone = None
        support = None
        resistance = None

        for i in range(len(sorted_levels) - 1):
            lower_level, lower_value = sorted_levels[i]
            upper_level, upper_value = sorted_levels[i + 1]

            if lower_value <= close <= upper_value:
                current_zone = f"{lower_level}-{upper_level}"
                support = lower_value
                resistance = upper_value
                break

        # Handle cases where price is below lowest level or above highest level
        if current_zone is None:
            if close < sorted_levels[0][1]:
                current_zone = f"Below_{sorted_levels[0][0]}"
                resistance = sorted_levels[0][1]
            elif close > sorted_levels[-1][1]:
                current_zone = f"Above_{sorted_levels[-1][0]}"
                support = sorted_levels[-1][1]

        return {
            'levels': levels,
            'current_zone': current_zone,
            'support': support,
            'resistance': resistance
        }

    def get_columns(self) -> List[str]:
        return self.columns


class PivotPointsIndicator(BaseIndicator):
    """
    Pivot Points indicator
    """

    def __init__(self, method: str = 'standard'):
        super().__init__(
            name=f"PivotPoints_{method}",
            description=f"Pivot Points (Method: {method})"
        )
        self.method = method
        self.column_pivot = "PP"
        self.column_s1 = "S1"
        self.column_s2 = "S2"
        self.column_s3 = "S3"
        self.column_r1 = "R1"
        self.column_r2 = "R2"
        self.column_r3 = "R3"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Pivot Points"""
        df_copy = df.copy()

        # Calculate pivot points using previous day's data
        prev_high = df_copy['High'].shift(1)
        prev_low = df_copy['Low'].shift(1)
        prev_close = df_copy['Close'].shift(1)

        if self.method == 'standard':
            # Standard pivot points
            pivot = (prev_high + prev_low + prev_close) / 3
            s1 = (2 * pivot) - prev_high
            r1 = (2 * pivot) - prev_low
            s2 = pivot - (prev_high - prev_low)
            r2 = pivot + (prev_high - prev_low)
            s3 = s1 - (prev_high - prev_low)
            r3 = r1 + (prev_high - prev_low)

        elif self.method == 'fibonacci':
            # Fibonacci pivot points
            pivot = (prev_high + prev_low + prev_close) / 3
            r1 = pivot + 0.382 * (prev_high - prev_low)
            s1 = pivot - 0.382 * (prev_high - prev_low)
            r2 = pivot + 0.618 * (prev_high - prev_low)
            s2 = pivot - 0.618 * (prev_high - prev_low)
            r3 = pivot + 1.0 * (prev_high - prev_low)
            s3 = pivot - 1.0 * (prev_high - prev_low)

        else:
            # Default: standard method
            pivot = (prev_high + prev_low + prev_close) / 3
            s1 = (2 * pivot) - prev_high
            r1 = (2 * pivot) - prev_low
            s2 = pivot - (prev_high - prev_low)
            r2 = pivot + (prev_high - prev_low)
            s3 = s1 - (prev_high - prev_low)
            r3 = r1 + (prev_high - prev_low)

        df_copy[self.column_pivot] = pivot
        df_copy[self.column_s1] = s1
        df_copy[self.column_s2] = s2
        df_copy[self.column_s3] = s3
        df_copy[self.column_r1] = r1
        df_copy[self.column_r2] = r2
        df_copy[self.column_r3] = r3

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret Pivot Points values"""
        close = row['Close']
        pivot = row[self.column_pivot]
        s1 = row[self.column_s1]
        s2 = row[self.column_s2]
        s3 = row[self.column_s3]
        r1 = row[self.column_r1]
        r2 = row[self.column_r2]
        r3 = row[self.column_r3]

        if pd.isna(pivot):
            return {
                'pivot': None,
                'supports': None,
                'resistances': None,
                'position': 'UNKNOWN',
                'signal': 'UNKNOWN'
            }

        # Check price position
        if close > r3:
            position = 'ABOVE_R3'
            signal = 'VERY_BULLISH'
            support = r3
            resistance = None
        elif close > r2:
            position = 'BETWEEN_R2_R3'
            signal = 'STRONG_BULLISH'
            support = r2
            resistance = r3
        elif close > r1:
            position = 'BETWEEN_R1_R2'
            signal = 'BULLISH'
            support = r1
            resistance = r2
        elif close > pivot:
            position = 'BETWEEN_PP_R1'
            signal = 'WEAK_BULLISH'
            support = pivot
            resistance = r1
        elif close > s1:
            position = 'BETWEEN_S1_PP'
            signal = 'WEAK_BEARISH'
            support = s1
            resistance = pivot
        elif close > s2:
            position = 'BETWEEN_S2_S1'
            signal = 'BEARISH'
            support = s2
            resistance = s1
        elif close > s3:
            position = 'BETWEEN_S3_S2'
            signal = 'STRONG_BEARISH'
            support = s3
            resistance = s2
        else:
            position = 'BELOW_S3'
            signal = 'VERY_BEARISH'
            support = None
            resistance = s3

        return {
            'pivot': pivot,
            'supports': [s1, s2, s3],
            'resistances': [r1, r2, r3],
            'position': position,
            'signal': signal,
            'nearest_support': support,
            'nearest_resistance': resistance
        }

    def get_columns(self) -> List[str]:
        return [
            self.column_pivot,
            self.column_s1,
            self.column_s2,
            self.column_s3,
            self.column_r1,
            self.column_r2,
            self.column_r3
        ]


class CMFIndicator(BaseIndicator):
    """
    Chaikin Money Flow (CMF) indicator
    """

    def __init__(self, period: int = 20):
        super().__init__(
            name=f"CMF_{period}",
            description=f"Chaikin Money Flow (Period: {period})"
        )
        self.period = period
        self.column_cmf = f"CMF_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Chaikin Money Flow"""
        df_copy = df.copy()

        # Money Flow Multiplier
        mfm = ((df_copy['Close'] - df_copy['Low']) - (df_copy['High'] - df_copy['Close'])) / (
                    df_copy['High'] - df_copy['Low'])

        # Money Flow Volume
        mfv = mfm * df_copy['Volume']

        # Chaikin Money Flow
        df_copy[self.column_cmf] = mfv.rolling(window=self.period).sum() / df_copy['Volume'].rolling(
            window=self.period).sum()

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret Chaikin Money Flow values"""
        cmf = row[self.column_cmf]

        if pd.isna(cmf):
            return {
                'value': None,
                'signal': 'UNKNOWN'
            }

        # Interpret CMF values
        if cmf > 0.25:
            signal = 'STRONG_BULLISH'
        elif cmf > 0.1:
            signal = 'BULLISH'
        elif cmf > 0:
            signal = 'WEAK_BULLISH'
        elif cmf > -0.1:
            signal = 'WEAK_BEARISH'
        elif cmf > -0.25:
            signal = 'BEARISH'
        else:
            signal = 'STRONG_BEARISH'

        return {
            'value': cmf,
            'signal': signal
        }

    def get_columns(self) -> List[str]:
        return [self.column_cmf]


class ElderRayIndicator(BaseIndicator):
    """
    Elder Ray indicator
    """

    def __init__(self, period: int = 13):
        super().__init__(
            name=f"ElderRay_{period}",
            description=f"Elder Ray Indicator (Period: {period})"
        )
        self.period = period
        self.column_bull_power = "Bull_Power"
        self.column_bear_power = "Bear_Power"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Elder Ray"""
        df_copy = df.copy()

        # Calculate EMA
        ema = talib.EMA(df_copy['Close'].values, timeperiod=self.period)

        # Bull Power = High - EMA
        df_copy[self.column_bull_power] = df_copy['High'] - ema

        # Bear Power = Low - EMA
        df_copy[self.column_bear_power] = df_copy['Low'] - ema

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret Elder Ray values"""
        bull_power = row[self.column_bull_power]
        bear_power = row[self.column_bear_power]

        if pd.isna(bull_power) or pd.isna(bear_power):
            return {
                'bull_power': None,
                'bear_power': None,
                'signal': 'UNKNOWN'
            }

        # Interpret Elder Ray signals
        if bull_power > 0 and bear_power < 0:
            if bull_power > abs(bear_power):
                signal = 'BULLISH'
            else:
                signal = 'NEUTRAL'
        elif bull_power > 0 and bear_power >= 0:
            signal = 'STRONG_BULLISH'
        elif bull_power <= 0 and bear_power < 0:
            signal = 'STRONG_BEARISH'
        else:
            signal = 'BEARISH'

        # Calculate strength
        strength = (bull_power - bear_power) / abs(bear_power) if bear_power != 0 else bull_power

        return {
            'bull_power': bull_power,
            'bear_power': bear_power,
            'signal': signal,
            'strength': strength
        }

    def get_columns(self) -> List[str]:
        return [self.column_bull_power, self.column_bear_power]


class HeikinAshiIndicator(BaseIndicator):
    """
    Heikin-Ashi candlestick indicator
    """

    def __init__(self):
        super().__init__(
            name="HeikinAshi",
            description="Heikin-Ashi Candlesticks (Shows smoother trends than traditional candlesticks)"
        )
        self.column_ha_open = "HA_Open"
        self.column_ha_high = "HA_High"
        self.column_ha_low = "HA_Low"
        self.column_ha_close = "HA_Close"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Heikin-Ashi candlesticks"""
        df_copy = df.copy()

        # Initialize first candle same as regular candle
        df_copy[self.column_ha_close] = (df_copy['Open'] + df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 4
        df_copy[self.column_ha_open] = df_copy['Open'].copy()

        # Calculate remaining candles
        for i in range(1, len(df_copy)):
            prev_ha_open = df_copy[self.column_ha_open].iloc[i - 1]
            prev_ha_close = df_copy[self.column_ha_close].iloc[i - 1]

            # Current candle info
            current_open = df_copy['Open'].iloc[i]
            current_high = df_copy['High'].iloc[i]
            current_low = df_copy['Low'].iloc[i]
            current_close = df_copy['Close'].iloc[i]

            # Heikin-Ashi formula
            ha_open = (prev_ha_open + prev_ha_close) / 2
            ha_close = (current_open + current_high + current_low + current_close) / 4

            # Update values
            df_copy[self.column_ha_open].iloc[i] = ha_open
            df_copy[self.column_ha_close].iloc[i] = ha_close

        # Calculate HA_High and HA_Low
        df_copy[self.column_ha_high] = df_copy[[self.column_ha_open, self.column_ha_close, 'High']].max(axis=1)
        df_copy[self.column_ha_low] = df_copy[[self.column_ha_open, self.column_ha_close, 'Low']].min(axis=1)

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret Heikin-Ashi candlestick values"""
        ha_open = row[self.column_ha_open]
        ha_high = row[self.column_ha_high]
        ha_low = row[self.column_ha_low]
        ha_close = row[self.column_ha_close]

        if pd.isna(ha_open) or pd.isna(ha_close):
            return {
                'open': None,
                'high': None,
                'low': None,
                'close': None,
                'candle_type': 'UNKNOWN',
                'signal': 'UNKNOWN'
            }

        # Determine candle type
        if ha_close > ha_open:
            candle_type = 'BULLISH'
            body_size = (ha_close - ha_open) / ha_open * 100 if ha_open != 0 else 0
        else:
            candle_type = 'BEARISH'
            body_size = (ha_open - ha_close) / ha_open * 100 if ha_open != 0 else 0

        # Calculate upper and lower shadows
        upper_shadow = (ha_high - max(ha_open, ha_close)) / ha_close * 100 if ha_close != 0 else 0
        lower_shadow = (min(ha_open, ha_close) - ha_low) / ha_close * 100 if ha_close != 0 and min(ha_open,
                                                                                                   ha_close) > ha_low else 0

        # Interpret signal
        if candle_type == 'BULLISH':
            if body_size > 1.0:  # arbitrary threshold
                if lower_shadow < 0.1:  # almost no lower shadow
                    signal = 'STRONG_BULLISH'
                else:
                    signal = 'BULLISH'
            else:
                signal = 'WEAK_BULLISH'
        else:  # 'BEARISH'
            if body_size > 1.0:  # arbitrary threshold
                if upper_shadow < 0.1:  # almost no upper shadow
                    signal = 'STRONG_BEARISH'
                else:
                    signal = 'BEARISH'
            else:
                signal = 'WEAK_BEARISH'

        # Trend continuation (can't determine from a single candle)
        trend_continuation = 'UNKNOWN'

        return {
            'open': ha_open,
            'high': ha_high,
            'low': ha_low,
            'close': ha_close,
            'candle_type': candle_type,
            'body_size_percent': body_size,
            'upper_shadow_percent': upper_shadow,
            'lower_shadow_percent': lower_shadow,
            'signal': signal,
            'trend_continuation': trend_continuation
        }

    def get_columns(self) -> List[str]:
        return [
            self.column_ha_open,
            self.column_ha_high,
            self.column_ha_low,
            self.column_ha_close
        ]

class AdxIndicator(BaseIndicator):
    """
    Average Directional Index (ADX) indicator - Measures trend strength
    """

    def __init__(self, period: int = 14):
        super().__init__(
            name=f"ADX_{period}",
            description=f"Average Directional Index (Period: {period})"
        )
        self.period = period
        self.column_adx = f"ADX_{period}"
        self.column_plus_di = f"PLUS_DI_{period}"
        self.column_minus_di = f"MINUS_DI_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADX"""
        df_copy = df.copy()

        adx = talib.ADX(
            df_copy['High'].values,
            df_copy['Low'].values,
            df_copy['Close'].values,
            timeperiod=self.period
        )
        plus_di = talib.PLUS_DI(
            df_copy['High'].values,
            df_copy['Low'].values,
            df_copy['Close'].values,
            timeperiod=self.period
        )
        minus_di = talib.MINUS_DI(
            df_copy['High'].values,
            df_copy['Low'].values,
            df_copy['Close'].values,
            timeperiod=self.period
        )

        df_copy[self.column_adx] = adx
        df_copy[self.column_plus_di] = plus_di
        df_copy[self.column_minus_di] = minus_di

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret ADX values"""
        adx = row[self.column_adx]
        plus_di = row[self.column_plus_di]
        minus_di = row[self.column_minus_di]

        if pd.isna(adx) or pd.isna(plus_di) or pd.isna(minus_di):
            return {
                'adx': None,
                'plus_di': None,
                'minus_di': None,
                'trend_strength': 'UNKNOWN',
                'trend_direction': 'UNKNOWN'
            }

        # Evaluate trend strength
        if adx < 20:
            trend_strength = 'WEAK'
        elif adx < 40:
            trend_strength = 'MODERATE'
        elif adx < 60:
            trend_strength = 'STRONG'
        else:
            trend_strength = 'VERY_STRONG'

        # Trend direction
        if plus_di > minus_di:
            trend_direction = 'BULLISH'
            di_spread = plus_di - minus_di
        else:
            trend_direction = 'BEARISH'
            di_spread = minus_di - plus_di

        # DI spread significance
        if di_spread > 10:
            direction_strength = 'STRONG'
        elif di_spread > 5:
            direction_strength = 'MODERATE'
        else:
            direction_strength = 'WEAK'

        # Combined signal
        if trend_strength in ['STRONG', 'VERY_STRONG'] and direction_strength in ['STRONG', 'MODERATE']:
            if trend_direction == 'BULLISH':
                signal = 'STRONG_BULLISH'
            else:
                signal = 'STRONG_BEARISH'
        elif trend_strength == 'WEAK':
            signal = 'NEUTRAL'
        else:
            if trend_direction == 'BULLISH':
                signal = 'BULLISH'
            else:
                signal = 'BEARISH'

        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'direction_strength': direction_strength,
            'signal': signal
        }

    def get_columns(self) -> List[str]:
        return [self.column_adx, self.column_plus_di, self.column_minus_di]


class OBVIndicator(BaseIndicator):
    """
    On-Balance Volume (OBV) indicator - Analyzes volume and price relationship
    """

    def __init__(self):
        super().__init__(
            name="OBV",
            description="On-Balance Volume (Volume-based price momentum indicator)"
        )
        self.column_obv = "OBV"
        self.column_obv_ema = "OBV_EMA_20"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate OBV"""
        df_copy = df.copy()

        df_copy[self.column_obv] = talib.OBV(
            df_copy['Close'].values,
            df_copy['Volume'].values
        )

        # Add OBV EMA for trend confirmation
        df_copy[self.column_obv_ema] = talib.EMA(
            df_copy[self.column_obv].values,
            timeperiod=20
        )

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret OBV values"""
        obv = row[self.column_obv]
        obv_ema = row[self.column_obv_ema]

        if pd.isna(obv) or pd.isna(obv_ema):
            return {
                'obv': None,
                'obv_ema': None,
                'signal': 'UNKNOWN'
            }

        # Compare OBV and OBV EMA
        if obv > obv_ema:
            signal = 'BULLISH'  # Volume-based momentum is rising
        elif obv < obv_ema:
            signal = 'BEARISH'  # Volume-based momentum is falling
        else:
            signal = 'NEUTRAL'

        return {
            'obv': obv,
            'obv_ema': obv_ema,
            'signal': signal
        }

    def get_columns(self) -> List[str]:
        return [self.column_obv, self.column_obv_ema]


class AwesomeOscillatorIndicator(BaseIndicator):
    """
    Awesome Oscillator (AO) indicator - Measures market momentum
    """

    def __init__(self, fast_period: int = 5, slow_period: int = 34):
        super().__init__(
            name=f"AO_{fast_period}_{slow_period}",
            description=f"Awesome Oscillator (Fast Period: {fast_period}, Slow Period: {slow_period})"
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.column_ao = "AO"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Awesome Oscillator"""
        df_copy = df.copy()

        # Median Price (HL/2)
        median_price = (df_copy['High'] + df_copy['Low']) / 2

        # Calculate fast SMA and slow SMA
        fast_sma = median_price.rolling(window=self.fast_period).mean()
        slow_sma = median_price.rolling(window=self.slow_period).mean()

        # Calculate AO
        df_copy[self.column_ao] = fast_sma - slow_sma

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret AO values"""
        ao = row[self.column_ao]

        if pd.isna(ao):
            return {
                'value': None,
                'signal': 'UNKNOWN'
            }

        # Interpret AO signal
        if ao > 0:
            if ao > 0.5:  # arbitrary threshold
                signal = 'STRONG_BULLISH'
            else:
                signal = 'BULLISH'
        elif ao < 0:
            if ao < -0.5:  # arbitrary threshold
                signal = 'STRONG_BEARISH'
            else:
                signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'

        return {
            'value': ao,
            'signal': signal
        }

    def get_columns(self) -> List[str]:
        return [self.column_ao]


class MFIIndicator(BaseIndicator):
    """
    Money Flow Index (MFI) indicator - Combines price and volume for momentum analysis
    """

    def __init__(self, period: int = 14):
        super().__init__(
            name=f"MFI_{period}",
            description=f"Money Flow Index (Period: {period})"
        )
        self.period = period
        self.column_mfi = f"MFI_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MFI"""
        df_copy = df.copy()

        df_copy[self.column_mfi] = talib.MFI(
            df_copy['High'].values,
            df_copy['Low'].values,
            df_copy['Close'].values,
            df_copy['Volume'].values,
            timeperiod=self.period
        )

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret MFI values"""
        mfi = row[self.column_mfi]

        if pd.isna(mfi):
            return {
                'value': None,
                'signal': 'UNKNOWN'
            }

        # Interpret MFI signal
        if mfi > 80:
            signal = 'OVERBOUGHT'
        elif mfi < 20:
            signal = 'OVERSOLD'
        elif mfi > 50:
            signal = 'BULLISH'
        elif mfi < 50:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'

        # Calculate strength (distance from 50 = stronger signal)
        strength = abs(mfi - 50) * 2

        return {
            'value': mfi,
            'signal': signal,
            'strength': strength
        }

    def get_columns(self) -> List[str]:
        return [self.column_mfi]


class WilliamsRIndicator(BaseIndicator):
    """
    Williams %R indicator - Measures overbought/oversold conditions
    """

    def __init__(self, period: int = 14):
        super().__init__(
            name=f"WilliamsR_{period}",
            description=f"Williams %R (Period: {period})"
        )
        self.period = period
        self.column_willr = f"WilliamsR_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Williams %R"""
        df_copy = df.copy()

        df_copy[self.column_willr] = talib.WILLR(
            df_copy['High'].values,
            df_copy['Low'].values,
            df_copy['Close'].values,
            timeperiod=self.period
        )

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret Williams %R values"""
        willr = row[self.column_willr]

        if pd.isna(willr):
            return {
                'value': None,
                'signal': 'UNKNOWN'
            }

        # Interpret Williams %R signal (range -100 to 0)
        if willr > -20:
            signal = 'OVERBOUGHT'
        elif willr < -80:
            signal = 'OVERSOLD'
        elif willr > -50:
            signal = 'BULLISH'
        else:
            signal = 'BEARISH'

        return {
            'value': willr,
            'signal': signal
        }

    def get_columns(self) -> List[str]:
        return [self.column_willr]


class ChandelierExitIndicator(BaseIndicator):
    """
    Chandelier Exit indicator - Volatility-based trailing stop
    """

    def __init__(self, period: int = 22, multiplier: float = 3.0):
        super().__init__(
            name=f"ChandelierExit_{period}_{multiplier}",
            description=f"Chandelier Exit (Period: {period}, Multiplier: {multiplier})"
        )
        self.period = period
        self.multiplier = multiplier
        self.column_ce_long = "CE_Long"
        self.column_ce_short = "CE_Short"
        self.column_signal = "CE_Signal"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Chandelier Exit"""
        df_copy = df.copy()

        # Calculate ATR
        atr = talib.ATR(
            df_copy['High'].values,
            df_copy['Low'].values,
            df_copy['Close'].values,
            timeperiod=self.period
        )

        # Calculate highest high and lowest low
        high_max = df_copy['High'].rolling(window=self.period).max()
        low_min = df_copy['Low'].rolling(window=self.period).min()

        # Long position exit (Highest High - ATR * multiplier)
        df_copy[self.column_ce_long] = high_max - (atr * self.multiplier)

        # Short position exit (Lowest Low + ATR * multiplier)
        df_copy[self.column_ce_short] = low_min + (atr * self.multiplier)

        # Calculate signal (based on close price relative to long/short exits)
        signals = []
        for i in range(len(df_copy)):
            if pd.isna(df_copy[self.column_ce_long].iloc[i]) or pd.isna(df_copy[self.column_ce_short].iloc[i]):
                signals.append(np.nan)
            elif df_copy['Close'].iloc[i] > df_copy[self.column_ce_long].iloc[i]:
                signals.append(1)  # Maintain long position
            elif df_copy['Close'].iloc[i] < df_copy[self.column_ce_short].iloc[i]:
                signals.append(-1)  # Maintain short position
            else:
                signals.append(0)  # Neutral

        df_copy[self.column_signal] = signals

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret Chandelier Exit values"""
        close = row['Close']
        ce_long = row[self.column_ce_long]
        ce_short = row[self.column_ce_short]
        ce_signal = row[self.column_signal]

        if pd.isna(ce_long) or pd.isna(ce_short) or pd.isna(ce_signal):
            return {
                'long_exit': None,
                'short_exit': None,
                'signal': 'UNKNOWN'
            }

        # Interpret signal
        if ce_signal == 1:
            signal = 'BULLISH'
            stop_loss = ce_long
        elif ce_signal == -1:
            signal = 'BEARISH'
            stop_loss = ce_short
        else:
            signal = 'NEUTRAL'
            stop_loss = None

        # Calculate distance from price to exit level
        if signal == 'BULLISH':
            distance = (close - ce_long) / close * 100
        elif signal == 'BEARISH':
            distance = (ce_short - close) / close * 100
        else:
            distance = 0

        return {
            'long_exit': ce_long,
            'short_exit': ce_short,
            'signal': signal,
            'stop_loss': stop_loss,
            'distance_percent': distance
        }

    def get_columns(self) -> List[str]:
        return [self.column_ce_long, self.column_ce_short, self.column_signal]


class SupertrendIndicator(BaseIndicator):
    """
    Supertrend indicator
    """

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        super().__init__(
            name=f"Supertrend_{period}_{multiplier}",
            description=f"Supertrend (Period: {period}, Multiplier: {multiplier})"
        )
        self.period = period
        self.multiplier = multiplier
        self.column_supertrend = "Supertrend"
        self.column_direction = "Supertrend_Direction"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Supertrend"""
        df_copy = df.copy()

        # Calculate ATR
        atr = talib.ATR(
            df_copy['High'].values,
            df_copy['Low'].values,
            df_copy['Close'].values,
            timeperiod=self.period
        )

        # Calculate basic bands
        hl2 = (df_copy['High'] + df_copy['Low']) / 2
        upper_band = hl2 + (self.multiplier * atr)
        lower_band = hl2 - (self.multiplier * atr)

        # Calculate Supertrend
        supertrend = pd.Series(np.nan, index=df_copy.index)
        direction = pd.Series(np.nan, index=df_copy.index)

        # Find first valid index
        start_idx = self.period

        # Initialize first value
        if start_idx < len(df_copy):
            supertrend.iloc[start_idx] = upper_band.iloc[start_idx]
            direction.iloc[start_idx] = 1 if df_copy['Close'].iloc[start_idx] <= upper_band.iloc[start_idx] else -1

        # Calculate rest of the values
        for i in range(start_idx + 1, len(df_copy)):
            curr_close = df_copy['Close'].iloc[i]
            curr_upper = upper_band.iloc[i]
            curr_lower = lower_band.iloc[i]
            prev_supertrend = supertrend.iloc[i - 1]
            prev_direction = direction.iloc[i - 1]

            # If previous direction was up (1)
            if prev_direction == 1:
                # Calculate supertrend value
                curr_supertrend = max(curr_lower, prev_supertrend)
                # Determine direction
                curr_direction = 1 if curr_close <= curr_supertrend else -1
            # If previous direction was down (-1)
            else:
                # Calculate supertrend value
                curr_supertrend = min(curr_upper, prev_supertrend)
                # Determine direction
                curr_direction = -1 if curr_close >= curr_supertrend else 1

            supertrend.iloc[i] = curr_supertrend
            direction.iloc[i] = curr_direction

        df_copy[self.column_supertrend] = supertrend
        df_copy[self.column_direction] = direction

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret Supertrend values"""
        close = row['Close']
        supertrend = row[self.column_supertrend]
        direction = row[self.column_direction]

        if pd.isna(supertrend) or pd.isna(direction):
            return {
                'value': None,
                'direction': None,
                'signal': 'UNKNOWN'
            }

        # Interpret direction
        if direction == -1:  # Price is above supertrend line
            signal = 'BULLISH'
        else:  # Price is below supertrend line
            signal = 'BEARISH'

        # Calculate distance from price to supertrend line
        distance = abs(close - supertrend) / close * 100

        return {
            'value': supertrend,
            'direction': direction,
            'distance_percent': distance,
            'signal': signal
        }

    def get_columns(self) -> List[str]:
        return [self.column_supertrend, self.column_direction]


class EMAIndicator(BaseIndicator):
    """
    Exponential Moving Average (EMA) indicator
    """

    def __init__(self, periods: List[int] = [9, 20, 50, 100, 200]):
        name_parts = [str(p) for p in periods]
        super().__init__(
            name=f"EMA_{'_'.join(name_parts)}",
            description=f"Exponential Moving Average (Periods: {', '.join(name_parts)})"
        )
        self.periods = periods
        self.columns = [f"EMA_{period}" for period in self.periods]

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Exponential Moving Averages"""
        df_copy = df.copy()

        for period, column in zip(self.periods, self.columns):
            df_copy[column] = talib.EMA(df_copy['Close'].values, timeperiod=period)

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret EMA values"""
        close = row['Close']
        ema_values = {column: row[column] for column in self.columns}

        if any(pd.isna(value) for value in ema_values.values()):
            return {
                'values': ema_values,
                'signal': 'UNKNOWN'
            }

        # Compare short-term EMA with long-term EMA
        short_term = ema_values[self.columns[0]]  # First (shortest) period
        mid_term = ema_values[self.columns[len(self.columns) // 2]]  # Middle period
        long_term = ema_values[self.columns[-1]]  # Last (longest) period

        # Price above all EMAs with short>mid>long = strong uptrend
        if (close > short_term > mid_term > long_term):
            trend = 'STRONG_BULLISH'
        # Price above all EMAs = uptrend
        elif (close > short_term and close > long_term):
            trend = 'BULLISH'
        # Price below all EMAs with short<mid<long = strong downtrend
        elif (close < short_term < mid_term < long_term):
            trend = 'STRONG_BEARISH'
        # Price below all EMAs = downtrend
        elif (close < short_term and close < long_term):
            trend = 'BEARISH'
        # Short-term EMA above long-term EMA = weak uptrend
        elif (short_term > long_term):
            trend = 'WEAK_BULLISH'
        # Short-term EMA below long-term EMA = weak downtrend
        elif (short_term < long_term):
            trend = 'WEAK_BEARISH'
        # Other cases
        else:
            trend = 'NEUTRAL'

        # Check for aligned EMAs
        aligned = True
        for i in range(len(self.columns) - 1):
            if ema_values[self.columns[i]] < ema_values[self.columns[i + 1]]:
                aligned = False
                break

        alignment = 'ALIGNED_BULLISH' if aligned else 'NOT_ALIGNED'

        # Check for golden cross/death cross (approximate since we only have current data)
        cross_signal = 'NO_CROSS'
        if abs(short_term - long_term) / long_term < 0.002:  # Within 0.2% = possible cross
            if short_term > long_term:
                cross_signal = 'POTENTIAL_GOLDEN_CROSS'
            else:
                cross_signal = 'POTENTIAL_DEATH_CROSS'

        return {
            'values': ema_values,
            'short_term': short_term,
            'mid_term': mid_term,
            'long_term': long_term,
            'trend': trend,
            'alignment': alignment,
            'cross_signal': cross_signal
        }

    def get_columns(self) -> List[str]:
        return self.columns


class SqueezeIndicator(BaseIndicator):
    """
    Squeeze Momentum indicator - Combines Bollinger Bands and Keltner Channels to analyze compression and momentum
    """

    def __init__(self, bb_length: int = 20, kc_length: int = 20, kc_mult: float = 1.5, bb_mult: float = 2.0,
                 mom_length: int = 12):
        super().__init__(
            name=f"SqueezeMomentum_{bb_length}_{kc_length}_{kc_mult}_{mom_length}",
            description=f"Squeeze Momentum (Bollinger Period: {bb_length}, Keltner Period: {kc_length}, Keltner Mult: {kc_mult}, Momentum Period: {mom_length})"
        )
        self.bb_length = bb_length
        self.kc_length = kc_length
        self.kc_mult = kc_mult
        self.bb_mult = bb_mult
        self.mom_length = mom_length

        self.column_squeeze = "Squeeze"
        self.column_momentum = "Momentum"
        self.column_momentum_hist = "Momentum_Hist"
        self.column_squeeze_state = "Squeeze_State"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Squeeze Momentum"""
        df_copy = df.copy()

        # Calculate Bollinger Bands
        basis = df_copy['Close'].rolling(window=self.bb_length).mean()
        dev = df_copy['Close'].rolling(window=self.bb_length).std()

        bb_upper = basis + (dev * self.bb_mult)
        bb_lower = basis - (dev * self.bb_mult)

        # Calculate Keltner Channel
        tp = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3
        atr = talib.ATR(df_copy['High'].values, df_copy['Low'].values, df_copy['Close'].values,
                        timeperiod=self.kc_length)

        kc_upper = basis + (atr * self.kc_mult)
        kc_lower = basis - (atr * self.kc_mult)

        # Calculate squeeze state (True when Bollinger Bands are inside Keltner Channel)
        df_copy[self.column_squeeze] = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(int)

        # Calculate squeeze state as string
        df_copy[self.column_squeeze_state] = np.where(df_copy[self.column_squeeze] == 1, "ON", "OFF")

        # Calculate momentum (linear regression slope based on Donchian Channel midpoint)
        highest_high = df_copy['High'].rolling(window=self.mom_length).max()
        lowest_low = df_copy['Low'].rolling(window=self.mom_length).min()

        mid = (highest_high + lowest_low) / 2

        # Linear regression slope calculation function
        def linreg(x, y):
            if len(x) != len(y) or len(x) < 2:
                return 0

            x_mean = np.mean(x)
            y_mean = np.mean(y)

            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))

            if denominator == 0:
                return 0

            return numerator / denominator

        # Calculate momentum (linear regression slope)
        momentum = np.zeros(len(df_copy))

        for i in range(self.mom_length, len(df_copy)):
            momentum[i] = linreg(
                list(range(self.mom_length)),
                df_copy['Close'].iloc[i - self.mom_length + 1:i + 1].values - mid.iloc[
                                                                              i - self.mom_length + 1:i + 1].values
            )

        df_copy[self.column_momentum] = momentum

        # Calculate momentum histogram state (positive/negative)
        df_copy[self.column_momentum_hist] = np.where(
            df_copy[self.column_momentum] >= 0,
            np.where(df_copy[self.column_momentum] > df_copy[self.column_momentum].shift(1), "INC_POS", "DEC_POS"),
            np.where(df_copy[self.column_momentum] < df_copy[self.column_momentum].shift(1), "DEC_NEG", "INC_NEG")
        )

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """Interpret Squeeze Momentum values"""
        squeeze = row[self.column_squeeze]
        momentum = row[self.column_momentum]
        momentum_hist = row[self.column_momentum_hist]
        squeeze_state = row[self.column_squeeze_state]

        if pd.isna(squeeze) or pd.isna(momentum):
            return {
                'squeeze': None,
                'momentum': None,
                'momentum_hist': None,
                'squeeze_state': None,
                'signal': 'UNKNOWN'
            }

        # Interpret market state
        if squeeze_state == "ON":
            if momentum > 0 and momentum_hist == "INC_POS":
                signal = "POTENTIAL_BULLISH_BREAKOUT"
                description = "Strong bullish momentum in squeeze state. Preparing for bullish breakout."
            elif momentum > 0:
                signal = "SQUEEZE_BULLISH"
                description = "Bullish momentum in squeeze state. Energy accumulating."
            elif momentum < 0 and momentum_hist == "DEC_NEG":
                signal = "POTENTIAL_BEARISH_BREAKOUT"
                description = "Strong bearish momentum in squeeze state. Preparing for bearish breakout."
            elif momentum < 0:
                signal = "SQUEEZE_BEARISH"
                description = "Bearish momentum in squeeze state. Energy accumulating."
            else:
                signal = "SQUEEZE_NEUTRAL"
                description = "Squeeze state with neutral momentum."
        else:  # Squeeze OFF state
            if momentum > 0 and momentum_hist == "INC_POS":
                signal = "STRONG_BULLISH"
                description = "Strong bullish momentum after squeeze release. Uptrend in progress."
            elif momentum > 0 and momentum_hist == "DEC_POS":
                signal = "BULLISH_WEAKENING"
                description = "Bullish momentum weakening after squeeze."
            elif momentum < 0 and momentum_hist == "DEC_NEG":
                signal = "STRONG_BEARISH"
                description = "Strong bearish momentum after squeeze release. Downtrend in progress."
            elif momentum < 0 and momentum_hist == "INC_NEG":
                signal = "BEARISH_WEAKENING"
                description = "Bearish momentum weakening after squeeze."
            else:
                signal = "NEUTRAL"
                description = "Weak momentum after squeeze."

        # Calculate momentum strength
        momentum_strength = abs(momentum)

        # Define strength category
        if momentum_strength > 0.5:  # Adjustable threshold
            strength = "STRONG"
        elif momentum_strength > 0.2:  # Adjustable threshold
            strength = "MODERATE"
        else:
            strength = "WEAK"

        return {
            'squeeze': squeeze,
            'momentum': momentum,
            'momentum_hist': momentum_hist,
            'squeeze_state': squeeze_state,
            'signal': signal,
            'description': description,
            'momentum_strength': momentum_strength,
            'strength_category': strength
        }

    def get_columns(self) -> List[str]:
        return [self.column_squeeze, self.column_momentum, self.column_momentum_hist, self.column_squeeze_state]
