from typing import Optional
from binance.client import Client
import os

class BinanceCollector:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        Binance API 클라이언트를 사용한 데이터 수집기 초기화
        
        Args:
            api_key (Optional[str]): Binance API 키 (없으면 환경변수에서 로드)
            api_secret (Optional[str]): Binance API 시크릿 (없으면 환경변수에서 로드)
        """
        # API 키가 제공되지 않으면 환경변수에서 가져오기
        self.api_key = api_key or os.getenv('BINANCE_TEST_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_TEST_API_SECRET')
        
        # API 키 없이도 기본 기능 (시세 조회 등) 사용 가능
        self.client = Client(self.api_key, self.api_secret)
    
    def get_btc_price(self) -> Optional[float]:
        """
        Binance API를 사용하여 BTCUSDT 현재 가격을 가져옵니다.
        
        Returns:
            float: BTCUSDT 현재 가격
            None: API 호출 실패 시
        """
        try:
            # 티커 정보를 가져오는 메서드 사용
            ticker = self.client.get_symbol_ticker(symbol="BTCUSDT")
            return float(ticker['price'])
        
        except Exception as e:
            print(f"Binance API 호출 중 오류 발생: {e}")
            return None
    
    def get_all_btc_pairs_prices(self) -> dict:
        """
        모든 BTC 페어의 가격 정보를 가져옵니다.
        
        Returns:
            dict: 심볼별 현재 가격 정보
        """
        try:
            all_tickers = self.client.get_all_tickers()
            btc_pairs = {ticker['symbol']: float(ticker['price']) 
                        for ticker in all_tickers if 'BTC' in ticker['symbol']}
            return btc_pairs
        
        except Exception as e:
            print(f"모든 BTC 페어 가격 조회 중 오류 발생: {e}")
            return {}
    
    def get_recent_trades(self, symbol: str = "BTCUSDT", limit: int = 10) -> list:
        """
        최근 거래 내역을 가져옵니다.
        
        Args:
            symbol (str): 거래 심볼
            limit (int): 가져올 거래 수
        
        Returns:
            list: 최근 거래 내역 목록
        """
        try:
            trades = self.client.get_recent_trades(symbol=symbol, limit=limit)
            return trades
        
        except Exception as e:
            print(f"최근 거래 내역 조회 중 오류 발생: {e}")
            return []
    
    def get_historical_klines(
        self, 
        symbol: str = "BTCUSDT", 
        interval: str = "1h", 
        limit: int = 24
    ) -> list:
        """
        과거 캔들(K-Line) 데이터를 가져옵니다.
        
        Args:
            symbol (str): 거래 심볼
            interval (str): 캔들 간격 (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit (int): 가져올 캔들 수
        
        Returns:
            list: 캔들 데이터 목록
        """
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # 캔들 데이터 포맷팅
            formatted_klines = []
            for k in klines:
                formatted_klines.append({
                    'open_time': k[0],
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    'close_time': k[6],
                    'quote_volume': float(k[7]),
                    'trades': k[8],
                    'taker_buy_volume': float(k[9]),
                    'taker_buy_quote_volume': float(k[10])
                })
            
            return formatted_klines
        
        except Exception as e:
            print(f"과거 캔들 데이터 조회 중 오류 발생: {e}")
            return []

# 사용 예시
if __name__ == "__main__":
    # python-binance 패키지 설치 필요: pip install python-binance
    
    collector = BinanceCollector()
    
    # 현재 BTCUSDT 가격 조회
    current_price = collector.get_btc_price()
    print(f"현재 BTCUSDT 가격: {current_price}")
    
    # 최근 BTCUSDT 거래 내역 조회
    recent_trades = collector.get_recent_trades(limit=5)
    print("\n최근 거래 내역:")
    for trade in recent_trades[:3]:  # 첫 3개만 출력
        print(f"가격: {trade['price']}, 수량: {trade['qty']}, 시간: {trade['time']}")
    
    # 과거 1시간 캔들 데이터 조회
    hourly_klines = collector.get_historical_klines(interval="1h", limit=5)
    print("\n최근 시간별 캔들 데이터:")
    for candle in hourly_klines:
        print(f"시간: {candle['open_time']}, 시가: {candle['open']}, 종가: {candle['close']}, 고가: {candle['high']}, 저가: {candle['low']}")
