import requests
import json
import pandas as pd
from typing import Dict, Any
import os
from datetime import datetime, timedelta

class CryptoSentimentCollector:
    def __init__(self, symbol: str = 'bitcoin'):
        """
        암호화폐 감성 및 온체인 지표 수집기
        
        Args:
            symbol (str): 수집할 암호화폐 심볼
        """
        self.symbol = symbol
        self.fear_greed_api = "https://api.alternative.me/fng/"
        self.glassnode_api = "https://api.glassnode.com/v1/metrics"
        self.glassnode_api_key = os.getenv('GLASSNODE_API_KEY')
    
    def get_fear_and_greed_index(self) -> Dict[str, Any]:
        """
        현재 암호화폐 공포-탐욕 지수 수집
        
        Returns:
            Dict: 공포-탐욕 지수 정보
        """
        try:
            response = requests.get(f"{self.fear_greed_api}?limit=1")
            data = response.json()['data'][0]
            
            return {
                'value': int(data['value']),
                'value_classification': data['value_classification'],
                'timestamp': datetime.fromtimestamp(int(data['timestamp']))
            }
        except Exception as e:
            print(f"공포-탐욕 지수 수집 오류: {e}")
            return {
                'value': None,
                'value_classification': 'ERROR',
                'timestamp': datetime.now()
            }
    
    def _get_glassnode_data(self, metric: str, resolution: str = '1d') -> float:
        """
        Glassnode API에서 온체인 지표 수집
        
        Args:
            metric (str): 수집할 온체인 지표
            resolution (str): 데이터 해상도 (1d, 1h 등)
        
        Returns:
            float: 해당 지표의 최근 값
        """
        if not self.glassnode_api_key:
            print("Glassnode API 키가 설정되지 않았습니다.")
            return None
        
        params = {
            'a': self.symbol,  # asset
            'm': metric,        # metric
            'r': resolution,    # resolution
            's': (datetime.now() - timedelta(days=30)).timestamp(),  # start time
            'i': resolution,    # interval
            'api_key': self.glassnode_api_key
        }
        
        try:
            response = requests.get(self.glassnode_api, params=params)
            data = response.json()
            
            # 데이터가 있으면 마지막 값 반환
            if data and len(data) > 0:
                return float(data[-1][1])
            return None
        
        except Exception as e:
            print(f"{metric} 수집 오류: {e}")
            return None
    
    def get_on_chain_indicators(self) -> Dict[str, Any]:
        """
        주요 온체인 지표 수집
        
        Returns:
            Dict: 다양한 온체인 지표
        """
        indicators = {}
        
        # 주요 온체인 지표 목록
        on_chain_metrics = [
            'blockchain_net_realized_profit_loss',  # 순 실현 손익
            'blockchain_transaction_volume_usd',    # 일일 거래량 (USD)
            'market_hash_rate',                     # 해시 레이트 (네트워크 보안)
            'market_nvts',                          # NVT 비율 (네트워크 가치 대 거래량)
            'addresses_active_count'                # 활성 주소 수
        ]
        
        for metric in on_chain_metrics:
            try:
                value = self._get_glassnode_data(metric)
                if value is not None:
                    indicators[metric] = value
            except Exception as e:
                print(f"{metric} 수집 중 오류: {e}")
        
        return indicators
    
    def get_comprehensive_market_sentiment(self) -> Dict[str, Any]:
        """
        종합 시장 감성 및 온체인 지표 수집
        
        Returns:
            Dict: 종합 시장 감성 정보
        """
        # 공포-탐욕 지수 수집
        fear_greed = self.get_fear_and_greed_index()
        
        # 온체인 지표 수집
        on_chain_indicators = self.get_on_chain_indicators()
        
        # 감성 분석
        sentiment_analysis = self._analyze_market_sentiment(
            fear_greed, 
            on_chain_indicators
        )
        
        return {
            'fear_greed_index': fear_greed,
            'on_chain_indicators': on_chain_indicators,
            'market_sentiment': sentiment_analysis
        }
    
    def _analyze_market_sentiment(
        self, 
        fear_greed: Dict[str, Any], 
        on_chain_indicators: Dict[str, Any]
    ) -> str:
        """
        수집된 지표를 바탕으로 시장 감성 분석
        
        Args:
            fear_greed (Dict): 공포-탐욕 지수
            on_chain_indicators (Dict): 온체인 지표
        
        Returns:
            str: 시장 감성 설명
        """
        sentiment_description = []
        
        # 공포-탐욕 지수 기반 분석
        if fear_greed['value'] is not None:
            if fear_greed['value'] <= 20:
                sentiment_description.append("극심한 공포")
            elif fear_greed['value'] <= 40:
                sentiment_description.append("시장 공포")
            elif fear_greed['value'] >= 80:
                sentiment_description.append("극심한 탐욕")
            elif fear_greed['value'] >= 60:
                sentiment_description.append("시장 탐욕")
        
        # 온체인 지표 분석
        if 'blockchain_net_realized_profit_loss' in on_chain_indicators:
            net_profit_loss = on_chain_indicators['blockchain_net_realized_profit_loss']
            if net_profit_loss > 0:
                sentiment_description.append("순 실현 이익")
            else:
                sentiment_description.append("순 실현 손실")
        
        if 'addresses_active_count' in on_chain_indicators:
            active_addresses = on_chain_indicators['addresses_active_count']
            if active_addresses > 500000:  # 예시 임계값
                sentiment_description.append("높은 네트워크 활동")
            else:
                sentiment_description.append("낮은 네트워크 활동")
        
        return ", ".join(sentiment_description) if sentiment_description else "중립"

# 사용 예시
if __name__ == "__main__":
    # Glassnode API 키를 환경변수로 설정 필요
    # export GLASSNODE_API_KEY='your_glassnode_api_key'
    
    collector = CryptoSentimentCollector(symbol='bitcoin')
    market_sentiment = collector.get_comprehensive_market_sentiment()
    
    print("시장 감성 정보:")
    for key, value in market_sentiment.items():
        print(f"{key}: {value}")
