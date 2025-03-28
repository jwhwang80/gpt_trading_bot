import pandas as pd
from typing import Dict, Any, List, Optional
from binance_collector import BinanceCollector
from advanced_market_collector import AdvancedMarketCollector
from crypto_sentiment_collector import CryptoSentimentCollector
import os
import json
from datetime import datetime

class MarketDataIntegrator:
    def __init__(self, symbol: str = 'BTCUSDT', interval: str = '1h'):
        """
        여러 데이터 소스에서 시장 데이터를 수집하고 통합하는 클래스
        
        Args:
            symbol (str): 수집할 거래 심볼
            interval (str): 캔들 간격 (1h, 4h, 1d 등)
        """
        self.symbol = symbol
        self.interval = interval
        self.binance_collector = BinanceCollector()
        self.advanced_collector = AdvancedMarketCollector(symbol, interval)
        self.sentiment_collector = CryptoSentimentCollector('bitcoin')
        self.history_file = 'market_data_history.json'
        
        # 히스토리 파일 초기화
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w') as f:
                json.dump([], f)
    
    def collect_comprehensive_data(self) -> Dict[str, Any]:
        """
        다양한 소스에서 종합적인 시장 데이터 수집 및 통합
        
        Returns:
            Dict[str, Any]: 통합된 시장 데이터
        """
        # 기본 가격 데이터
        current_price = self.binance_collector.get_btc_price()
        if not current_price:
            print("가격 데이터를 가져오는데 실패했습니다. 대체 값 사용")
            current_price = 0
        
        try:
            # 고급 시장 지표
            market_indicators = self.advanced_collector.get_market_indicators()
        except Exception as e:
            print(f"고급 시장 지표 수집 실패: {e}")
            market_indicators = {
                'rsi_14': 50,
                'market_context': '데이터 수집 실패'
            }
        
        try:
            # 시장 감성 데이터
            sentiment_data = self.sentiment_collector.get_comprehensive_market_sentiment()
            fear_greed = sentiment_data.get('fear_greed_index', {}).get('value', 50)
            market_sentiment = sentiment_data.get('market_sentiment', '중립')
        except Exception as e:
            print(f"시장 감성 데이터 수집 실패: {e}")
            fear_greed = 50
            market_sentiment = '중립'
        
        # 통합 데이터
        integrated_data = {
            "timestamp": datetime.now().isoformat(),
            "price": current_price,
            "fear_greed_index": fear_greed,
            "rsi": market_indicators.get('rsi_14', 50),
            "news_sentiment": market_sentiment,
            "market_context": market_indicators.get('market_context', ''),
            "technical_indicators": {
                "macd": market_indicators.get('macd', {}),
                "bollinger_bands": market_indicators.get('bollinger_bands', {})
            }
        }
        
        # 히스토리에 저장
        self._save_to_history(integrated_data)
        
        return integrated_data
    
    def _save_to_history(self, data: Dict[str, Any]) -> None:
        """
        수집된 데이터를 히스토리에 저장
        
        Args:
            data (Dict[str, Any]): 저장할 시장 데이터
        """
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
            
            # 최대 100개 항목 유지
            history.append(data)
            if len(history) > 100:
                history = history[-100:]
            
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"히스토리 저장 실패: {e}")
    
    def get_recent_market_trends(self, days: int = 7) -> Dict[str, Any]:
        """
        최근 시장 추세 데이터 가져오기
        
        Args:
            days (int): 가져올 일수
        
        Returns:
            Dict[str, Any]: 시장 추세 정보
        """
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
            
            if not history:
                return {"trend": "데이터 없음"}
            
            # 최근 가격 추세 계산
            recent_prices = [entry.get('price', 0) for entry in history[-days*24:] if entry.get('price', 0) > 0]
            
            if not recent_prices or len(recent_prices) < 2:
                return {"trend": "추세 계산 불가"}
            
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100
            
            # RSI 추세
            recent_rsi = [entry.get('rsi', 50) for entry in history[-days*24:]]
            rsi_change = recent_rsi[-1] - recent_rsi[0] if recent_rsi else 0
            
            return {
                "price_change_percentage": price_change,
                "starting_price": recent_prices[0],
                "current_price": recent_prices[-1],
                "rsi_trend": "상승" if rsi_change > 5 else "하락" if rsi_change < -5 else "횡보",
                "overall_trend": "상승세" if price_change > 3 else "하락세" if price_change < -3 else "횡보세"
            }
        except Exception as e:
            print(f"추세 데이터 계산 실패: {e}")
            return {"trend": "계산 오류", "error": str(e)}
