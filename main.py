from typing import Dict, Any

from collectors.binance_collector import BinanceCollector
from strategy.prompt_generator import PromptGenerator
from strategy.gpt_interface import GPTInterface
from execution.excel_tracker import ExcelTracker
from utils.log_saver import LogSaver
from strategy.reflection_prompt_generator import ReflectionPromptGenerator

class TradingPipeline:
    def __init__(self):
        self.binance_collector = BinanceCollector()
        self.gpt_interface = GPTInterface()
        self.excel_tracker = ExcelTracker()
    
    def collect_market_data(self) -> Dict[str, Any]:
        """시장 데이터 수집"""
        current_price = self.binance_collector.get_btc_price()
        
        # TODO: 실제 구현 시 공포탐욕지수, RSI, 뉴스감정 등 추가
        return {
            "price": current_price,
            "fear_greed_index": 60,  # 예시
            "rsi": 55,  # 예시
            "news_sentiment": "보통"  # 예시
        }
    
    def request_trading_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """GPT에 트레이딩 전략 요청"""
        prompt = PromptGenerator.generate_strategy_prompt(market_data)
        gpt_response = self.gpt_interface.call_gpt_with_prompt(prompt)
        return PromptGenerator.parse_gpt_response(gpt_response)
    
    def execute_trade(self, strategy: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """전략 실행 및 기록"""
        if strategy['action'] == 'BUY' and strategy.get('confidence', 0) > 50:
            position_size = min(0.05, strategy['confidence'] / 2000)  # 최대 5% 포지션
            self.excel_tracker.log_entry('BTCUSDT', 'strategy_001', current_price, position_size, 1)
            return {
                "symbol": 'BTCUSDT',
                "action": 'BUY',
                "entry_price": current_price,
                "position_size": position_size
            }
        return {"action": "NO_TRADE"}
    
    def request_reflection(self, strategy: Dict[str, Any], trade_result: Dict[str, Any]) -> Dict[str, Any]:
        """전략 복기 요청"""
        if trade_result['action'] == 'BUY':
            profit_percentage = ((trade_result['exit_price'] - trade_result['entry_price']) / trade_result['entry_price']) * 100
            reflection_prompt = ReflectionPromptGenerator.build_reflection_prompt(
                strategy, 
                trade_result['entry_price'], 
                trade_result['exit_price'], 
                profit_percentage
            )
            reflection_response = self.gpt_interface.call_gpt_with_prompt(reflection_prompt)
            return ReflectionPromptGenerator.parse_reflection_response(reflection_response)
        return {}
    
    def run_trading_pipeline(self):
        """전체 트레이딩 파이프라인 실행"""
        try:
            # 1. 시장 데이터 수집
            market_data = self.collect_market_data()
            
            # 2. GPT에 전략 요청
            strategy = self.request_trading_strategy(market_data)
            
            # 3. 전략 실행
            trade_result = self.execute_trade(strategy, market_data['price'])
            
            # 4. 복기 요청 (실제 시뮬레이션에서는 exit price 필요)
            trade_result['exit_price'] = market_data['price'] * 1.02  # 임시로 2% 수익 가정
            reflection = self.request_reflection(strategy, trade_result)
            
            # 5. 로그 저장
            LogSaver.save_log_with_result(
                market_data, 
                PromptGenerator.generate_strategy_prompt(market_data), 
                strategy, 
                trade_result, 
                reflection
            )
        
        except Exception as e:
            print(f"트레이딩 파이프라인 실행 중 오류: {e}")


if __name__ == "__main__":
    # 1시간마다 자동 실행
    pipeline = TradingPipeline()
    pipeline.run_trading_pipeline()