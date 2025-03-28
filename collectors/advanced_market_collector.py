import os
import pandas as pd
import numpy as np
import talib
import requests
from typing import Dict, Any, List

class AdvancedMarketCollector:
    def __init__(self, symbol: str = 'BTCUSDT', interval: str = '1h'):
        """
        고급 시장 데이터 수집기 초기화
        
        Args:
            symbol (str): 수집할 거래 심볼
            interval (str): 캔들 간격 (1h, 4h, 1d 등)
        """
        self.symbol = symbol
        self.interval = interval
        self.base_url = "https://api.binance.com/api/v3/klines"
    
    def _fetch_historical_data(self, limit: int = 500) -> pd.DataFrame:
        """
        바이낸스 API에서 역사적 가격 데이터 가져오기
        
        Args:
            limit (int): 가져올 최근 캔들 수
        
        Returns:
            pandas.DataFrame: OHLCV 데이터
        """
        params = {
            'symbol': self.symbol,
            'interval': self.interval,
            'limit': limit
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 
            'volume', 'close_time', 'quote_volume', 
            'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        
        # 숫자형 변환
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        return df
    
    def calculate_rsi(self, close_prices: np.ndarray, period: int = 14) -> float:
        """
        RSI (Relative Strength Index) 계산
        
        Args:
            close_prices (np.ndarray): 종가 배열
            period (int): RSI 계산 기간
        
        Returns:
            float: 최근 RSI 값
        """
        rsi = talib.RSI(close_prices, timeperiod=period)
        return rsi[-1]
    
    def calculate_bollinger_bands(
        self, 
        close_prices: np.ndarray, 
        period: int = 20, 
        std_dev: int = 2
    ) -> Dict[str, float]:
        """
        볼린저 밴드 계산
        
        Args:
            close_prices (np.ndarray): 종가 배열
            period (int): 이동평균선 기간
            std_dev (int): 표준편차 배수
        
        Returns:
            Dict: 볼린저 밴드 정보
        """
        upper, middle, lower = talib.BBANDS(
            close_prices, 
            timeperiod=period, 
            nbdevup=std_dev, 
            nbdevdn=std_dev, 
            matype=0
        )
        
        return {
            'upper_band': upper[-1],
            'middle_band': middle[-1],
            'lower_band': lower[-1],
            'current_price': close_prices[-1],
            'band_width_percentage': (upper[-1] - lower[-1]) / middle[-1] * 100
        }
    
    def calculate_atr(self, close_prices: np.ndarray, high_prices: np.ndarray, 
                      low_prices: np.ndarray, period: int = 14) -> float:
        """
        ATR (Average True Range) 계산
        
        Args:
            close_prices (np.ndarray): 종가 배열
            high_prices (np.ndarray): 고가 배열
            low_prices (np.ndarray): 저가 배열
            period (int): ATR 계산 기간
        
        Returns:
            float: 최근 ATR 값
        """
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)
        return atr[-1]
    
    def calculate_macd(self, close_prices: np.ndarray) -> Dict[str, float]:
        """
        MACD (Moving Average Convergence Divergence) 계산
        
        Args:
            close_prices (np.ndarray): 종가 배열
        
        Returns:
            Dict: MACD 지표
        """
        macd, signal, hist = talib.MACD(
            close_prices, 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        
        return {
            'macd': macd[-1],
            'signal': signal[-1],
            'histogram': hist[-1]
        }
    
    def get_market_indicators(self) -> Dict[str, Any]:
        """
        종합 시장 지표 수집
        
        Returns:
            Dict: 다양한 기술적 지표
        """
        # 데이터 가져오기
        df = self._fetch_historical_data()
        
        # NumPy 배열로 변환
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        
        # 지표 계산
        indicators = {
            'price': close_prices[-1],
            'rsi_14': self.calculate_rsi(close_prices),
            'bollinger_bands': self.calculate_bollinger_bands(close_prices),
            'atr_14': self.calculate_atr(close_prices, high_prices, low_prices),
            'macd': self.calculate_macd(close_prices)
        }
        
        # 추가 시장 컨텍스트
        indicators['market_context'] = self._analyze_market_context(indicators)
        
        return indicators
    
    def _analyze_market_context(self, indicators: Dict[str, Any]) -> str:
        """
        지표를 기반으로 시장 컨텍스트 분석
        
        Args:
            indicators (Dict): 계산된 시장 지표
        
        Returns:
            str: 시장 상황 설명
        """
        context_description = []
        
        # RSI 분석
        if indicators['rsi_14'] > 70:
            context_description.append("과매수 상태")
        elif indicators['rsi_14'] < 30:
            context_description.append("과매도 상태")
        
        # 볼린저 밴드 분석
        bb = indicators['bollinger_bands']
        price = bb['current_price']
        if price >= bb['upper_band']:
            context_description.append("상단 볼린저밴드 근접")
        elif price <= bb['lower_band']:
            context_description.append("하단 볼린저밴드 근접")
        
        # MACD 분석
        macd = indicators['macd']
        if macd['macd'] > macd['signal']:
            context_description.append("강세 신호")
        else:
            context_description.append("약세 신호")
        
        return ", ".join(context_description)

# 사용 예시
if __name__ == "__main__":
    collector = AdvancedMarketCollector()
    market_indicators = collector.get_market_indicators()
    
    print("시장 지표:")
    for key, value in market_indicators.items():
        print(f"{key}: {value}")
