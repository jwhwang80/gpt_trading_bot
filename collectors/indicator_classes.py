import pandas as pd
import numpy as np
import talib
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import math


class BaseIndicator(ABC):
    """
    기술적 지표 계산을 위한 기본 클래스
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터프레임에서 지표를 계산합니다.

        Args:
            df (pd.DataFrame): OHLCV 데이터가 포함된 데이터프레임

        Returns:
            pd.DataFrame: 계산된 지표가 추가된 데이터프레임
        """
        pass

    @abstractmethod
    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """
        지표의 현재 값을 해석합니다.

        Args:
            row (pd.Series): 지표 값을 포함하는 데이터 시리즈

        Returns:
            Dict[str, Any]: 지표 해석 결과
        """
        pass

    def get_columns(self) -> List[str]:
        """
        이 지표가 생성하는 열 이름 목록을 반환합니다.

        Returns:
            List[str]: 열 이름 목록
        """
        return []

    def get_info(self) -> Dict[str, str]:
        """
        지표에 대한 정보를 반환합니다.

        Returns:
            Dict[str, str]: 지표 정보
        """
        return {
            'name': self.name,
            'description': self.description
        }


class RSIIndicator(BaseIndicator):
    """
    상대강도지수(RSI) 지표
    """

    def __init__(self, period: int = 14):
        super().__init__(
            name=f"RSI_{period}",
            description=f"상대강도지수 (기간: {period})"
        )
        self.period = period
        self.column_name = f"RSI_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI 계산"""
        df_copy = df.copy()
        df_copy[self.column_name] = talib.RSI(df_copy['Close'].values, timeperiod=self.period)
        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """RSI 값 해석"""
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
            'strength': min(100, max(0, abs(value - 50) * 2))  # 0-100 사이의 강도
        }

    def get_columns(self) -> List[str]:
        return [self.column_name]


class MACDIndicator(BaseIndicator):
    """
    이동평균수렴발산(MACD) 지표
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(
            name=f"MACD_{fast_period}_{slow_period}_{signal_period}",
            description=f"이동평균수렴발산 (빠른 기간: {fast_period}, 느린 기간: {slow_period}, 시그널 기간: {signal_period})"
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.column_macd = "MACD"
        self.column_signal = "MACD_Signal"
        self.column_hist = "MACD_Hist"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD 계산"""
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
        """MACD 값 해석"""
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

        # 히스토그램 변화 감지 (추가 분석)
        hist_trend = 'FLAT'

        # MACD 라인이 시그널 라인을 상향 돌파: 강한 매수 신호
        if macd_value > signal_value and hist_value > 0:
            trend_signal = 'STRONG_BULLISH'
        # MACD 라인이 시그널 라인보다 위에 있음: 약한 매수 신호
        elif macd_value > signal_value:
            trend_signal = 'BULLISH'
        # MACD 라인이 시그널 라인을 하향 돌파: 강한 매도 신호
        elif macd_value < signal_value and hist_value < 0:
            trend_signal = 'STRONG_BEARISH'
        # MACD 라인이 시그널 라인보다 아래에 있음: 약한 매도 신호
        else:
            trend_signal = 'BEARISH'

        # 제로 라인 기준 분석
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
    스토캐스틱 지표
    """

    def __init__(self, fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3):
        super().__init__(
            name=f"Stochastic_{fastk_period}_{slowk_period}_{slowd_period}",
            description=f"스토캐스틱 (K기간: {fastk_period}, K슬로우: {slowk_period}, D슬로우: {slowd_period})"
        )
        self.fastk_period = fastk_period
        self.slowk_period = slowk_period
        self.slowd_period = slowd_period
        self.column_k = "Stochastic_K"
        self.column_d = "Stochastic_D"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """스토캐스틱 계산"""
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
        """스토캐스틱 값 해석"""
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
    볼린저 밴드 지표
    """

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        super().__init__(
            name=f"BB_{period}_{std_dev}",
            description=f"볼린저 밴드 (기간: {period}, 표준편차: {std_dev})"
        )
        self.period = period
        self.std_dev = std_dev
        self.column_upper = "BB_Upper"
        self.column_middle = "BB_Middle"
        self.column_lower = "BB_Lower"
        self.column_width = "BB_Width"
        self.column_pct_b = "BB_Pct_B"  # %B 지표 추가

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """볼린저 밴드 계산"""
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

        # %B 계산 (가격이 밴드 내에서 어디에 위치하는지 나타내는 0-1 사이의 값)
        df_copy[self.column_pct_b] = (df_copy['Close'] - lower) / (upper - lower)

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """볼린저 밴드 값 해석"""
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

        # 가격 위치 파악
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

        # 밴드 폭 해석 (밴드 폭이 좁으면 확장 가능성, 넓으면 수축 가능성)
        if width < 10:  # 임의의 임계값
            band_signal = 'SQUEEZE'
        elif width > 40:  # 임의의 임계값
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
    평균진폭(ATR) 지표
    """

    def __init__(self, period: int = 14):
        super().__init__(
            name=f"ATR_{period}",
            description=f"평균진폭지표 (기간: {period})"
        )
        self.period = period
        self.column_name = f"ATR_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR 계산"""
        df_copy = df.copy()
        df_copy[self.column_name] = talib.ATR(
            df_copy['High'].values,
            df_copy['Low'].values,
            df_copy['Close'].values,
            timeperiod=self.period
        )
        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """ATR 값 해석"""
        value = row[self.column_name]
        close = row['Close']

        if pd.isna(value) or pd.isna(close):
            return {
                'value': None,
                'percent_of_price': None,
                'volatility': 'UNKNOWN'
            }

        # ATR을 가격 대비 백분율로 계산
        percent_of_price = (value / close) * 100

        # 변동성 범주 정의 (임의의 임계값)
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
    일목균형표 지표
    """

    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26, senkou_b_period: int = 52):
        super().__init__(
            name=f"Ichimoku_{tenkan_period}_{kijun_period}_{senkou_b_period}",
            description=f"일목균형표 (전환선: {tenkan_period}, 기준선: {kijun_period}, 선행스팬B: {senkou_b_period})"
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
        """도치안 채널 계산 (일목균형표 기본 계산)"""
        return (high_series.rolling(period).max() + low_series.rolling(period).min()) / 2

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """일목균형표 계산"""
        df_copy = df.copy()

        # 전환선 (Tenkan-sen) - 단기 추세
        df_copy[self.column_tenkan] = self._calc_donchian(
            df_copy['High'], df_copy['Low'], self.tenkan_period
        )

        # 기준선 (Kijun-sen) - 중기 추세
        df_copy[self.column_kijun] = self._calc_donchian(
            df_copy['High'], df_copy['Low'], self.kijun_period
        )

        # 선행스팬A (Senkou Span A) - 구름의 상단
        df_copy[self.column_senkou_a] = ((df_copy[self.column_tenkan] + df_copy[self.column_kijun]) / 2).shift(
            self.kijun_period)

        # 선행스팬B (Senkou Span B) - 구름의 하단
        df_copy[self.column_senkou_b] = self._calc_donchian(
            df_copy['High'], df_copy['Low'], self.senkou_b_period
        ).shift(self.kijun_period)

        # 후행스팬 (Chikou Span) - 확인선
        df_copy[self.column_chikou] = df_copy['Close'].shift(-self.kijun_period)

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """일목균형표 값 해석"""
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

        # 구름 위치
        if close > max(senkou_a, senkou_b):
            cloud_position = 'ABOVE_CLOUD'
        elif close < min(senkou_a, senkou_b):
            cloud_position = 'BELOW_CLOUD'
        else:
            cloud_position = 'IN_CLOUD'

        # 전환선/기준선 시그널
        if tenkan > kijun:
            tenkan_kijun_signal = 'BULLISH'
        elif tenkan < kijun:
            tenkan_kijun_signal = 'BEARISH'
        else:
            tenkan_kijun_signal = 'NEUTRAL'

        # 구름 색상 (선행스팬A > 선행스팬B일 때 녹색, 반대면 적색)
        cloud_color = 'GREEN' if senkou_a > senkou_b else 'RED'

        # 종합 시그널
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
    피보나치 리트레이스먼트(Fibonacci Retracement) 지표
    """

    def __init__(self, period: int = 100):
        super().__init__(
            name=f"Fibonacci_{period}",
            description=f"피보나치 리트레이스먼트 (기간: {period})"
        )
        self.period = period
        self.fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        self.columns = [f"Fib_{level}" for level in self.fib_levels]

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """피보나치 리트레이스먼트 계산"""
        df_copy = df.copy()

        # 최근 주요 고점과 저점 찾기
        for i in range(self.period, len(df_copy)):
            window = df_copy.iloc[i - self.period:i]

            # 주요 고점
            high = window['High'].max()
            # 주요 저점
            low = window['Low'].min()

            # 피보나치 레벨 계산
            for level, column in zip(self.fib_levels, self.columns):
                df_copy.loc[df_copy.index[i], column] = high - (high - low) * level

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """피보나치 리트레이스먼트 값 해석"""
        close = row['Close']

        # 레벨 값 수집
        levels = {}
        for level, column in zip(self.fib_levels, self.columns):
            levels[f"{level:.3f}"] = row[column]

        # 값이 유효한지 확인
        if all(pd.isna(v) for v in levels.values()):
            return {
                'levels': levels,
                'current_zone': 'UNKNOWN',
                'support': None,
                'resistance': None
            }

        # 현재 위치한 피보나치 구간 찾기
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

        # 가장 낮은 레벨보다 낮거나 가장 높은 레벨보다 높은 경우
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
    피벗 포인트(Pivot Points) 지표
    """

    def __init__(self, method: str = 'standard'):
        super().__init__(
            name=f"PivotPoints_{method}",
            description=f"피벗 포인트 (방식: {method})"
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
        """피벗 포인트 계산"""
        df_copy = df.copy()

        # 전날 데이터로 피벗 포인트 계산
        prev_high = df_copy['High'].shift(1)
        prev_low = df_copy['Low'].shift(1)
        prev_close = df_copy['Close'].shift(1)

        if self.method == 'standard':
            # 표준 피벗 포인트
            pivot = (prev_high + prev_low + prev_close) / 3
            s1 = (2 * pivot) - prev_high
            r1 = (2 * pivot) - prev_low
            s2 = pivot - (prev_high - prev_low)
            r2 = pivot + (prev_high - prev_low)
            s3 = s1 - (prev_high - prev_low)
            r3 = r1 + (prev_high - prev_low)

        elif self.method == 'fibonacci':
            # 피보나치 피벗 포인트
            pivot = (prev_high + prev_low + prev_close) / 3
            r1 = pivot + 0.382 * (prev_high - prev_low)
            s1 = pivot - 0.382 * (prev_high - prev_low)
            r2 = pivot + 0.618 * (prev_high - prev_low)
            s2 = pivot - 0.618 * (prev_high - prev_low)
            r3 = pivot + 1.0 * (prev_high - prev_low)
            s3 = pivot - 1.0 * (prev_high - prev_low)

        else:
            # 기본값: 표준 방식
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
        """피벗 포인트 값 해석"""
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

        # 가격 위치 확인
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
    차이크 머니 플로우(Chaikin Money Flow) 지표
    """

    def __init__(self, period: int = 20):
        super().__init__(
            name=f"CMF_{period}",
            description=f"차이크 머니 플로우 (기간: {period})"
        )
        self.period = period
        self.column_cmf = f"CMF_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """차이크 머니 플로우 계산"""
        df_copy = df.copy()

        # 머니 플로우 멀티플라이어 (Money Flow Multiplier)
        mfm = ((df_copy['Close'] - df_copy['Low']) - (df_copy['High'] - df_copy['Close'])) / (
                    df_copy['High'] - df_copy['Low'])

        # 머니 플로우 볼륨 (Money Flow Volume)
        mfv = mfm * df_copy['Volume']

        # 차이크 머니 플로우
        df_copy[self.column_cmf] = mfv.rolling(window=self.period).sum() / df_copy['Volume'].rolling(
            window=self.period).sum()

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """차이크 머니 플로우 값 해석"""
        cmf = row[self.column_cmf]

        if pd.isna(cmf):
            return {
                'value': None,
                'signal': 'UNKNOWN'
            }

        # CMF 값 해석
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
    엘더 레이(Elder Ray) 지표
    """

    def __init__(self, period: int = 13):
        super().__init__(
            name=f"ElderRay_{period}",
            description=f"엘더 레이 지표 (기간: {period})"
        )
        self.period = period
        self.column_bull_power = "Bull_Power"
        self.column_bear_power = "Bear_Power"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """엘더 레이 계산"""
        df_copy = df.copy()

        # EMA 계산
        ema = talib.EMA(df_copy['Close'].values, timeperiod=self.period)

        # 강세력 (Bull Power) = 고가 - EMA
        df_copy[self.column_bull_power] = df_copy['High'] - ema

        # 약세력 (Bear Power) = 저가 - EMA
        df_copy[self.column_bear_power] = df_copy['Low'] - ema

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """엘더 레이 값 해석"""
        bull_power = row[self.column_bull_power]
        bear_power = row[self.column_bear_power]

        if pd.isna(bull_power) or pd.isna(bear_power):
            return {
                'bull_power': None,
                'bear_power': None,
                'signal': 'UNKNOWN'
            }

        # 엘더 레이 신호 해석
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

        # 강도 계산
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
    헤이킨아시(Heikin-Ashi) 캔들스틱 지표
    """

    def __init__(self):
        super().__init__(
            name="HeikinAshi",
            description="헤이킨아시 캔들스틱 (트렌드를 부드럽게 보여주는 캔들스틱 기법)"
        )
        self.column_ha_open = "HA_Open"
        self.column_ha_high = "HA_High"
        self.column_ha_low = "HA_Low"
        self.column_ha_close = "HA_Close"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """헤이킨아시 캔들스틱 계산"""
        df_copy = df.copy()

        # 첫 번째 캔들은 일반 캔들과 동일하게 설정
        df_copy[self.column_ha_close] = (df_copy['Open'] + df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 4
        df_copy[self.column_ha_open] = df_copy['Open'].copy()

        # 두 번째 캔들부터 계산
        for i in range(1, len(df_copy)):
            prev_ha_open = df_copy[self.column_ha_open].iloc[i - 1]
            prev_ha_close = df_copy[self.column_ha_close].iloc[i - 1]

            # 현재 캔들 정보
            current_open = df_copy['Open'].iloc[i]
            current_high = df_copy['High'].iloc[i]
            current_low = df_copy['Low'].iloc[i]
            current_close = df_copy['Close'].iloc[i]

            # 헤이킨아시 계산식
            ha_open = (prev_ha_open + prev_ha_close) / 2
            ha_close = (current_open + current_high + current_low + current_close) / 4

            # 값 업데이트
            df_copy[self.column_ha_open].iloc[i] = ha_open
            df_copy[self.column_ha_close].iloc[i] = ha_close

        # HA_High와 HA_Low 계산
        df_copy[self.column_ha_high] = df_copy[[self.column_ha_open, self.column_ha_close, 'High']].max(axis=1)
        df_copy[self.column_ha_low] = df_copy[[self.column_ha_open, self.column_ha_close, 'Low']].min(axis=1)

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """헤이킨아시 캔들스틱 해석"""
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

        # 캔들 유형 파악
        if ha_close > ha_open:
            candle_type = 'BULLISH'
            body_size = (ha_close - ha_open) / ha_open * 100 if ha_open != 0 else 0
        else:
            candle_type = 'BEARISH'
            body_size = (ha_open - ha_close) / ha_open * 100 if ha_open != 0 else 0

        # 위꼬리와 아래꼬리 계산
        upper_shadow = (ha_high - max(ha_open, ha_close)) / ha_close * 100 if ha_close != 0 else 0
        lower_shadow = (min(ha_open, ha_close) - ha_low) / ha_close * 100 if ha_close != 0 and min(ha_open,
                                                                                                   ha_close) > ha_low else 0

        # 신호 해석
        if candle_type == 'BULLISH':
            if body_size > 1.0:  # 임의의 임계값
                if lower_shadow < 0.1:  # 거의 아래꼬리가 없음
                    signal = 'STRONG_BULLISH'
                else:
                    signal = 'BULLISH'
            else:
                signal = 'WEAK_BULLISH'
        else:  # 'BEARISH'
            if body_size > 1.0:  # 임의의 임계값
                if upper_shadow < 0.1:  # 거의 위꼬리가 없음
                    signal = 'STRONG_BEARISH'
                else:
                    signal = 'BEARISH'
            else:
                signal = 'WEAK_BEARISH'

        # 추세 지속성 판단 (연속된 같은 색 캔들)
        trend_continuation = 'UNKNOWN'  # 단일 캔들로는 판단 불가

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
    방향성 지수(ADX) 지표 - 추세의 강도를 측정
    """

    def __init__(self, period: int = 14):
        super().__init__(
            name=f"ADX_{period}",
            description=f"방향성 지수 (기간: {period})"
        )
        self.period = period
        self.column_adx = f"ADX_{period}"
        self.column_plus_di = f"PLUS_DI_{period}"
        self.column_minus_di = f"MINUS_DI_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """ADX 계산"""
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
        """ADX 값 해석"""
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

        # 추세 강도 평가
        if adx < 20:
            trend_strength = 'WEAK'
        elif adx < 40:
            trend_strength = 'MODERATE'
        elif adx < 60:
            trend_strength = 'STRONG'
        else:
            trend_strength = 'VERY_STRONG'

        # 추세 방향
        if plus_di > minus_di:
            trend_direction = 'BULLISH'
            di_spread = plus_di - minus_di
        else:
            trend_direction = 'BEARISH'
            di_spread = minus_di - plus_di

        # DI 차이의 의미
        if di_spread > 10:
            direction_strength = 'STRONG'
        elif di_spread > 5:
            direction_strength = 'MODERATE'
        else:
            direction_strength = 'WEAK'

        # 종합 시그널
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
    On-Balance Volume (OBV) 지표 - 볼륨과 가격 관계 분석
    """

    def __init__(self):
        super().__init__(
            name="OBV",
            description="On-Balance Volume (볼륨 기반 가격 모멘텀 지표)"
        )
        self.column_obv = "OBV"
        self.column_obv_ema = "OBV_EMA_20"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """OBV 계산"""
        df_copy = df.copy()

        df_copy[self.column_obv] = talib.OBV(
            df_copy['Close'].values,
            df_copy['Volume'].values
        )

        # OBV의 EMA를 추가로 계산 (추세 확인용)
        df_copy[self.column_obv_ema] = talib.EMA(
            df_copy[self.column_obv].values,
            timeperiod=20
        )

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """OBV 값 해석"""
        obv = row[self.column_obv]
        obv_ema = row[self.column_obv_ema]

        if pd.isna(obv) or pd.isna(obv_ema):
            return {
                'obv': None,
                'obv_ema': None,
                'signal': 'UNKNOWN'
            }

        # OBV와 OBV EMA 비교
        if obv > obv_ema:
            signal = 'BULLISH'  # 볼륨 기반 모멘텀 상승
        elif obv < obv_ema:
            signal = 'BEARISH'  # 볼륨 기반 모멘텀 하락
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
    Awesome Oscillator (AO) 지표 - 시장 모멘텀 측정
    """

    def __init__(self, fast_period: int = 5, slow_period: int = 34):
        super().__init__(
            name=f"AO_{fast_period}_{slow_period}",
            description=f"Awesome Oscillator (빠른 기간: {fast_period}, 느린 기간: {slow_period})"
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.column_ao = "AO"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Awesome Oscillator 계산"""
        df_copy = df.copy()

        # 중간 가격 (HL/2)
        median_price = (df_copy['High'] + df_copy['Low']) / 2

        # 빠른 SMA와 느린 SMA 계산
        fast_sma = median_price.rolling(window=self.fast_period).mean()
        slow_sma = median_price.rolling(window=self.slow_period).mean()

        # AO 계산
        df_copy[self.column_ao] = fast_sma - slow_sma

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """AO 값 해석"""
        ao = row[self.column_ao]

        if pd.isna(ao):
            return {
                'value': None,
                'signal': 'UNKNOWN'
            }

        # AO 값에 따른 시그널
        if ao > 0:
            if ao > 0.5:  # 임의의 임계값
                signal = 'STRONG_BULLISH'
            else:
                signal = 'BULLISH'
        elif ao < 0:
            if ao < -0.5:  # 임의의 임계값
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
    Money Flow Index (MFI) 지표 - 가격과 볼륨을 결합한 모멘텀 지표
    """

    def __init__(self, period: int = 14):
        super().__init__(
            name=f"MFI_{period}",
            description=f"Money Flow Index (기간: {period})"
        )
        self.period = period
        self.column_mfi = f"MFI_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """MFI 계산"""
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
        """MFI 값 해석"""
        mfi = row[self.column_mfi]

        if pd.isna(mfi):
            return {
                'value': None,
                'signal': 'UNKNOWN'
            }

        # MFI 값에 따른 시그널
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

        # 강도 계산 (50에서 멀수록 강한 신호)
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
    윌리엄스 %R 지표 - 과매수/과매도 상태 측정
    """

    def __init__(self, period: int = 14):
        super().__init__(
            name=f"WilliamsR_{period}",
            description=f"윌리엄스 %R (기간: {period})"
        )
        self.period = period
        self.column_willr = f"WilliamsR_{period}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """윌리엄스 %R 계산"""
        df_copy = df.copy()

        df_copy[self.column_willr] = talib.WILLR(
            df_copy['High'].values,
            df_copy['Low'].values,
            df_copy['Close'].values,
            timeperiod=self.period
        )

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """윌리엄스 %R 값 해석"""
        willr = row[self.column_willr]

        if pd.isna(willr):
            return {
                'value': None,
                'signal': 'UNKNOWN'
            }

        # 윌리엄스 %R 값에 따른 시그널 (-100 ~ 0 범위)
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
    샹들리에 엑시트(Chandelier Exit) 지표 - 변동성 기반 트레일링 스탑
    """

    def __init__(self, period: int = 22, multiplier: float = 3.0):
        super().__init__(
            name=f"ChandelierExit_{period}_{multiplier}",
            description=f"샹들리에 엑시트 (기간: {period}, 승수: {multiplier})"
        )
        self.period = period
        self.multiplier = multiplier
        self.column_ce_long = "CE_Long"
        self.column_ce_short = "CE_Short"
        self.column_signal = "CE_Signal"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """샹들리에 엑시트 계산"""
        df_copy = df.copy()

        # ATR 계산
        atr = talib.ATR(
            df_copy['High'].values,
            df_copy['Low'].values,
            df_copy['Close'].values,
            timeperiod=self.period
        )

        # 최고가/최저가 계산
        high_max = df_copy['High'].rolling(window=self.period).max()
        low_min = df_copy['Low'].rolling(window=self.period).min()

        # 롱 포지션 엑시트 (최고가 - ATR * 승수)
        df_copy[self.column_ce_long] = high_max - (atr * self.multiplier)

        # 숏 포지션 엑시트 (최저가 + ATR * 승수)
        df_copy[self.column_ce_short] = low_min + (atr * self.multiplier)

        # 시그널 계산 (현재 종가가 롱/숏 엑시트를 기준으로 위치)
        signals = []
        for i in range(len(df_copy)):
            if pd.isna(df_copy[self.column_ce_long].iloc[i]) or pd.isna(df_copy[self.column_ce_short].iloc[i]):
                signals.append(np.nan)
            elif df_copy['Close'].iloc[i] > df_copy[self.column_ce_long].iloc[i]:
                signals.append(1)  # 롱 포지션 유지
            elif df_copy['Close'].iloc[i] < df_copy[self.column_ce_short].iloc[i]:
                signals.append(-1)  # 숏 포지션 유지
            else:
                signals.append(0)  # 중립

        df_copy[self.column_signal] = signals

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """샹들리에 엑시트 값 해석"""
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

        # 신호 해석
        if ce_signal == 1:
            signal = 'BULLISH'
            stop_loss = ce_long
        elif ce_signal == -1:
            signal = 'BEARISH'
            stop_loss = ce_short
        else:
            signal = 'NEUTRAL'
            stop_loss = None

        # 현재 가격과 엑시트 레벨 간의 거리
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
    슈퍼트렌드 지표
    """

    def __init__(self, period: int = 10, multiplier: float = 3.0):
        super().__init__(
            name=f"Supertrend_{period}_{multiplier}",
            description=f"슈퍼트렌드 (기간: {period}, 승수: {multiplier})"
        )
        self.period = period
        self.multiplier = multiplier
        self.column_supertrend = "Supertrend"
        self.column_direction = "Supertrend_Direction"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """슈퍼트렌드 계산"""
        df_copy = df.copy()

        # ATR 계산
        atr = talib.ATR(
            df_copy['High'].values,
            df_copy['Low'].values,
            df_copy['Close'].values,
            timeperiod=self.period
        )

        # 기본 밴드 계산
        hl2 = (df_copy['High'] + df_copy['Low']) / 2
        upper_band = hl2 + (self.multiplier * atr)
        lower_band = hl2 - (self.multiplier * atr)

        # 슈퍼트렌드 계산
        supertrend = pd.Series(np.nan, index=df_copy.index)
        direction = pd.Series(np.nan, index=df_copy.index)

        # 첫 번째 유효한 값 찾기
        start_idx = self.period

        # 첫 번째 값 초기화
        if start_idx < len(df_copy):
            supertrend.iloc[start_idx] = upper_band.iloc[start_idx]
            direction.iloc[start_idx] = 1 if df_copy['Close'].iloc[start_idx] <= upper_band.iloc[start_idx] else -1

        # 나머지 값 계산
        for i in range(start_idx + 1, len(df_copy)):
            curr_close = df_copy['Close'].iloc[i]
            curr_upper = upper_band.iloc[i]
            curr_lower = lower_band.iloc[i]
            prev_supertrend = supertrend.iloc[i - 1]
            prev_direction = direction.iloc[i - 1]

            # 이전 방향이 상승(1)인 경우
            if prev_direction == 1:
                # 슈퍼트렌드 값 계산
                curr_supertrend = max(curr_lower, prev_supertrend)
                # 방향 결정
                curr_direction = 1 if curr_close <= curr_supertrend else -1
            # 이전 방향이 하락(-1)인 경우
            else:
                # 슈퍼트렌드 값 계산
                curr_supertrend = min(curr_upper, prev_supertrend)
                # 방향 결정
                curr_direction = -1 if curr_close >= curr_supertrend else 1

            supertrend.iloc[i] = curr_supertrend
            direction.iloc[i] = curr_direction

        df_copy[self.column_supertrend] = supertrend
        df_copy[self.column_direction] = direction

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """슈퍼트렌드 값 해석"""
        close = row['Close']
        supertrend = row[self.column_supertrend]
        direction = row[self.column_direction]

        if pd.isna(supertrend) or pd.isna(direction):
            return {
                'value': None,
                'direction': None,
                'signal': 'UNKNOWN'
            }

        # 방향에 따른 시그널
        if direction == -1:  # 가격이 슈퍼트렌드 위에 있음
            signal = 'BULLISH'
        else:  # 가격이 슈퍼트렌드 아래에 있음
            signal = 'BEARISH'

        # 가격과 슈퍼트렌드 라인의 거리
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
    지수이동평균(EMA) 지표
    """

    def __init__(self, periods: List[int] = [9, 20, 50, 100, 200]):
        name_parts = [str(p) for p in periods]
        super().__init__(
            name=f"EMA_{'_'.join(name_parts)}",
            description=f"지수이동평균 (기간: {', '.join(name_parts)})"
        )
        self.periods = periods
        self.columns = [f"EMA_{period}" for period in self.periods]

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """지수이동평균 계산"""
        df_copy = df.copy()

        for period, column in zip(self.periods, self.columns):
            df_copy[column] = talib.EMA(df_copy['Close'].values, timeperiod=period)

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """지수이동평균 값 해석"""
        close = row['Close']
        ema_values = {column: row[column] for column in self.columns}

        if any(pd.isna(value) for value in ema_values.values()):
            return {
                'values': ema_values,
                'signal': 'UNKNOWN'
            }

        # 단기 EMA와 장기 EMA 비교
        short_term = ema_values[self.columns[0]]  # 첫 번째 (가장 짧은) 기간
        mid_term = ema_values[self.columns[len(self.columns) // 2]]  # 중간 기간
        long_term = ema_values[self.columns[-1]]  # 마지막 (가장 긴) 기간

        # 가격이 모든 EMA 위에 있고, 단기>중기>장기 순서로 있으면 강한 상승세
        if (close > short_term > mid_term > long_term):
            trend = 'STRONG_BULLISH'
        # 가격이 모든 EMA 위에 있으면 상승세
        elif (close > short_term and close > long_term):
            trend = 'BULLISH'
        # 가격이 모든 EMA 아래에 있고, 단기<중기<장기 순서로 있으면 강한 하락세
        elif (close < short_term < mid_term < long_term):
            trend = 'STRONG_BEARISH'
        # 가격이 모든 EMA 아래에 있으면 하락세
        elif (close < short_term and close < long_term):
            trend = 'BEARISH'
        # 단기 EMA가 장기 EMA 위에 있으면 약한 상승세
        elif (short_term > long_term):
            trend = 'WEAK_BULLISH'
        # 단기 EMA가 장기 EMA 아래에 있으면 약한 하락세
        elif (short_term < long_term):
            trend = 'WEAK_BEARISH'
        # 그 외의 경우
        else:
            trend = 'NEUTRAL'

        # 이동평균 배열 패턴 확인
        aligned = True
        for i in range(len(self.columns) - 1):
            if ema_values[self.columns[i]] < ema_values[self.columns[i + 1]]:
                aligned = False
                break

        alignment = 'ALIGNED_BULLISH' if aligned else 'NOT_ALIGNED'

        # 골든 크로스/데드 크로스 확인 (현재 데이터만으로는 완벽히 확인 불가능하나 근사값 제공)
        cross_signal = 'NO_CROSS'
        if abs(short_term - long_term) / long_term < 0.002:  # 0.2% 이내로 가까우면 크로스 가능성
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
    스퀴즈 모멘텀(Squeeze Momentum) 지표 - 볼린저 밴드와 켈트너 채널을 결합하여 시장의 압축 상태와 모멘텀을 분석
    """

    def __init__(self, bb_length: int = 20, kc_length: int = 20, kc_mult: float = 1.5, bb_mult: float = 2.0,
                 mom_length: int = 12):
        super().__init__(
            name=f"SqueezeMomentum_{bb_length}_{kc_length}_{kc_mult}_{mom_length}",
            description=f"스퀴즈 모멘텀 (볼린저 기간: {bb_length}, 켈트너 기간: {kc_length}, 켈트너 승수: {kc_mult}, 모멘텀 기간: {mom_length})"
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
        """스퀴즈 모멘텀 계산"""
        df_copy = df.copy()

        # Bollinger Bands 계산
        basis = df_copy['Close'].rolling(window=self.bb_length).mean()
        dev = df_copy['Close'].rolling(window=self.bb_length).std()

        bb_upper = basis + (dev * self.bb_mult)
        bb_lower = basis - (dev * self.bb_mult)

        # Keltner Channel 계산
        tp = (df_copy['High'] + df_copy['Low'] + df_copy['Close']) / 3
        atr = talib.ATR(df_copy['High'].values, df_copy['Low'].values, df_copy['Close'].values,
                        timeperiod=self.kc_length)

        kc_upper = basis + (atr * self.kc_mult)
        kc_lower = basis - (atr * self.kc_mult)

        # 스퀴즈 상태 계산 (Bollinger Bands가 Keltner Channel 내에 있는 경우)
        df_copy[self.column_squeeze] = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(int)

        # 스퀴즈 상태 문자열 계산
        df_copy[self.column_squeeze_state] = np.where(df_copy[self.column_squeeze] == 1, "ON", "OFF")

        # 모멘텀 계산 (Donchian Channel의 중간값 기반 선형 회귀 경사)
        highest_high = df_copy['High'].rolling(window=self.mom_length).max()
        lowest_low = df_copy['Low'].rolling(window=self.mom_length).min()

        mid = (highest_high + lowest_low) / 2

        # 선형 회귀 경사 계산을 위한 함수
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

        # 모멘텀(선형 회귀 경사) 계산
        momentum = np.zeros(len(df_copy))

        for i in range(self.mom_length, len(df_copy)):
            momentum[i] = linreg(
                list(range(self.mom_length)),
                df_copy['Close'].iloc[i - self.mom_length + 1:i + 1].values - mid.iloc[
                                                                              i - self.mom_length + 1:i + 1].values
            )

        df_copy[self.column_momentum] = momentum

        # 모멘텀 히스토그램 상태 (양수/음수)
        df_copy[self.column_momentum_hist] = np.where(
            df_copy[self.column_momentum] >= 0,
            np.where(df_copy[self.column_momentum] > df_copy[self.column_momentum].shift(1), "INC_POS", "DEC_POS"),
            np.where(df_copy[self.column_momentum] < df_copy[self.column_momentum].shift(1), "DEC_NEG", "INC_NEG")
        )

        return df_copy

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """스퀴즈 모멘텀 값 해석"""
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

        # 시장 상태 해석
        if squeeze_state == "ON":
            if momentum > 0 and momentum_hist == "INC_POS":
                signal = "POTENTIAL_BULLISH_BREAKOUT"
                description = "스퀴즈 상태에서 강한 상승 모멘텀. 상승 돌파 준비 중."
            elif momentum > 0:
                signal = "SQUEEZE_BULLISH"
                description = "스퀴즈 상태에서 상승 모멘텀. 에너지 축적 중."
            elif momentum < 0 and momentum_hist == "DEC_NEG":
                signal = "POTENTIAL_BEARISH_BREAKOUT"
                description = "스퀴즈 상태에서 강한 하락 모멘텀. 하락 돌파 준비 중."
            elif momentum < 0:
                signal = "SQUEEZE_BEARISH"
                description = "스퀴즈 상태에서 하락 모멘텀. 에너지 축적 중."
            else:
                signal = "SQUEEZE_NEUTRAL"
                description = "스퀴즈 상태이나 모멘텀이 중립적."
        else:  # 스퀴즈 해제 상태
            if momentum > 0 and momentum_hist == "INC_POS":
                signal = "STRONG_BULLISH"
                description = "스퀴즈 해제 상태에서 강한 상승 모멘텀. 상승 추세 진행 중."
            elif momentum > 0 and momentum_hist == "DEC_POS":
                signal = "BULLISH_WEAKENING"
                description = "스퀴즈 해제 상태에서 상승 모멘텀이 약화되고 있음."
            elif momentum < 0 and momentum_hist == "DEC_NEG":
                signal = "STRONG_BEARISH"
                description = "스퀴즈 해제 상태에서 강한 하락 모멘텀. 하락 추세 진행 중."
            elif momentum < 0 and momentum_hist == "INC_NEG":
                signal = "BEARISH_WEAKENING"
                description = "스퀴즈 해제 상태에서 하락 모멘텀이 약화되고 있음."
            else:
                signal = "NEUTRAL"
                description = "스퀴즈 해제 상태에서 모멘텀이 약함."

        # 모멘텀 강도 계산
        momentum_strength = abs(momentum)

        # 강도 카테고리 정의
        if momentum_strength > 0.5:  # 임계값 조정 가능
            strength = "STRONG"
        elif momentum_strength > 0.2:  # 임계값 조정 가능
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