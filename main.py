from typing import Dict, Any
import os
import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

from collectors.technical_indicator_calculator import TechnicalIndicatorCalculator
from strategy.prompt_generator import PromptGenerator
from strategy.gpt_interface import GPTInterface
from execution.excel_tracker import ExcelTracker
from execution.position_manager import PositionManager
from utils.log_saver import LogSaver
from strategy.reflection_prompt_generator import ReflectionPromptGenerator


class TradingPipeline:
    def __init__(self, symbol: str = 'BTCUSDT', interval: str = '4H'):
        """
        트레이딩 파이프라인 초기화

        Args:
            symbol (str): 거래 심볼 (기본값: 'BTCUSDT')
            interval (str): 기술적 지표 분석 간격 (기본값: '4H')
        """
        self.symbol = symbol
        self.interval = interval
        self.indicator_calculator = TechnicalIndicatorCalculator(f"{symbol}_1m_3000day_data.csv", symbol)
        self.gpt_interface = GPTInterface()
        self.excel_tracker = ExcelTracker()
        self.position_manager = PositionManager(self.excel_tracker)

    async def update_historical_data(self):
        """
        최신 역사적 데이터를 다운로드하여 CSV 파일 업데이트
        """
        try:
            print("최신 역사적 데이터 다운로드 중...")

            # download_historical_chart 모듈을 동적으로 임포트하고 실행
            import importlib.util
            import sys

            # 모듈 경로 설정
            module_path = os.path.join(os.path.dirname(__file__), 'collectors/download_historical_chart.py')

            # 모듈 스펙 생성 및 로드
            spec = importlib.util.spec_from_file_location("download_historical_chart", module_path)
            download_module = importlib.util.module_from_spec(spec)
            sys.modules["download_historical_chart"] = download_module
            spec.loader.exec_module(download_module)

            # 모듈의 main 함수 실행 (비동기 함수)
            await download_module.main()

            print("역사적 데이터 업데이트 완료")
            return True

        except Exception as e:
            print(f"역사적 데이터 업데이트 중 오류 발생: {e}")
            return False

    def collect_market_data(self) -> Dict[str, Any]:
        """
        기술적 지표 계산기를 사용하여 시장 데이터 수집

        Returns:
            Dict[str, Any]: 수집된 시장 데이터
        """
        try:
            print("데이터 로드 시작...")
            self.indicator_calculator.load_data(days=120)
            print("데이터 로드 완료")

            print("기술적 지표 계산 시작...")
            results = self.indicator_calculator.calculate_all_timeframes()
            print("기술적 지표 계산 완료")

            print(f"시간프레임 데이터 추출: {self.interval}")
            timeframe_data = results[self.interval]['interpretation']
            print("시간프레임 데이터 추출 완료")

            print("현재 가격 추출...")
            close_price = timeframe_data.get('close_price')
            print(f"추출된 close_price 타입: {type(close_price)}, 값: {close_price}")

            # 안전하게 float로 변환
            try:
                current_price = float(close_price)
                print(f"변환된 current_price: {current_price}")
            except (TypeError, ValueError) as e:
                print(f"가격 변환 오류: {e}")
                current_price = 0.0

            # 지표 데이터를 안전하게 처리
            indicators = {}
            if 'indicators' in timeframe_data and isinstance(timeframe_data['indicators'], dict):
                indicators = self._sanitize_data(timeframe_data['indicators'])

            # 타임스탬프 안전하게 추출
            timestamp = timeframe_data.get('timestamp')
            if timestamp is not None:
                timestamp = str(timestamp)

            # 통합 데이터 반환 (간소화된 버전)
            return {
                "price": current_price,
                "indicators": indicators,
                "timestamp": timestamp,
                "timeframe": self.interval
            }

        except Exception as e:
            print(f"시장 데이터 수집 중 오류 발생: {e}")
            traceback.print_exc()  # 상세 오류 정보 출력
            # 기본 데이터 반환
            return {
                "price": 0.0,
                "indicators": {},
                "error": str(e)
            }

    def request_trading_strategy(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        GPT에 트레이딩 전략 요청

        Args:
            market_data (Dict[str, Any]): 시장 데이터

        Returns:
            Dict[str, Any]: GPT의 전략 응답
        """
        try:
            # 과거 거래 내역 로드
            historical_trades = PromptGenerator.load_historical_trades(max_trades=3)

            # 전략 프롬프트 생성
            prompt = PromptGenerator.generate_strategy_prompt(
                market_data,
                historical_trades
            )

            # GPT 호출
            gpt_response = self.gpt_interface.call_gpt_with_prompt(prompt)

            # 응답 파싱
            return PromptGenerator.parse_gpt_response(gpt_response)
        except Exception as e:
            print(f"트레이딩 전략 요청 중 오류 발생: {e}")
            return {
                "action": "HOLD",
                "confidence": 0,
                "reasoning": f"전략 요청 중 오류: {str(e)}"
            }

    def execute_trade(self, strategy: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """
        전략 실행 및 기록

        Args:
            strategy (Dict[str, Any]): GPT가 제안한 전략
            current_price (float): 현재 가격

        Returns:
            Dict[str, Any]: 거래 실행 결과
        """
        try:
            # 포지션 매니저를 통한 거래 실행
            return self.position_manager.execute_trade(strategy, current_price, self.symbol)
        except Exception as e:
            print(f"거래 실행 중 오류 발생: {e}")
            return {
                "action": "ERROR",
                "reason": f"거래 실행 오류: {str(e)}"
            }

    def request_reflection(self, strategy: Dict[str, Any], trade_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        전략 복기 요청

        Args:
            strategy (Dict[str, Any]): 원래 전략
            trade_result (Dict[str, Any]): 거래 결과

        Returns:
            Dict[str, Any]: 복기 결과
        """
        try:
            if isinstance(trade_result, dict) and trade_result.get('action') in ['BUY', 'SELL']:
                # 임시로 가정한 청산 가격 (실제로는 나중에 결정)
                if 'exit_price' not in trade_result:
                    trade_result['exit_price'] = trade_result.get('entry_price', 0) * 1.02  # 임시로 2% 수익 가정

                # 수익률 계산
                entry_price = float(trade_result.get('entry_price', 0))
                exit_price = float(trade_result.get('exit_price', 0))

                profit_percentage = 0
                if entry_price > 0:
                    if trade_result['action'] == 'BUY':
                        profit_percentage = ((exit_price - entry_price) / entry_price) * 100
                    else:  # 'SELL'
                        profit_percentage = ((entry_price - exit_price) / entry_price) * 100

                # 복기 프롬프트 생성
                reflection_prompt = ReflectionPromptGenerator.build_reflection_prompt(
                    strategy,
                    entry_price,
                    exit_price,
                    profit_percentage
                )

                # GPT 호출
                reflection_response = self.gpt_interface.call_gpt_with_prompt(reflection_prompt)

                # 응답 파싱
                return ReflectionPromptGenerator.parse_reflection_response(reflection_response)

            return {
                "overall_assessment": "분석 불필요",
                "key_insights": ["HOLD 전략으로 거래가 실행되지 않았습니다."],
                "reflection_date": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"복기 요청 중 오류 발생: {e}")
            traceback.print_exc()  # 오류 추적 정보 출력
            return {
                "overall_assessment": "오류",
                "key_insights": [f"복기 처리 중 오류: {str(e)}"],
                "reflection_date": datetime.now().isoformat()
            }

    async def run_trading_pipeline(self):
        """
        전체 트레이딩 파이프라인 실행
        """
        market_data = None
        strategy = None
        trade_result = None
        reflection = None
        prompt = None

        try:
            # 1. 역사적 데이터 업데이트
            if False:
                try:
                    await self.update_historical_data()
                except Exception as e:
                    print(f"역사적 데이터 업데이트 실패, 기존 데이터로 계속 진행: {e}")

            # 2. 시장 데이터 수집
            market_data = self.collect_market_data()
            price = market_data.get('price', 0.0)
            if not isinstance(price, (int, float)) or price <= 0:
                print(f"유효한 가격 데이터를 수집하지 못했습니다. 가격: {price} 파이프라인 중단.")
                return

            # 3. GPT에 전략 요청
            strategy = self.request_trading_strategy(market_data)

            # 4. 전략 실행
            trade_result = self.execute_trade(strategy, float(price))

            # 5. 복기 요청
            reflection = self.request_reflection(strategy, trade_result)

            # 6. 로그 저장
            prompt = PromptGenerator.generate_strategy_prompt(market_data)

            # 모든 데이터를 JSON으로 직렬화 가능한 상태로 변환
            market_data_clean = self._sanitize_data(market_data)
            strategy_clean = self._sanitize_data(strategy)
            trade_result_clean = self._sanitize_data(trade_result)
            reflection_clean = self._sanitize_data(reflection)

            # 직접 변환된 데이터로 JSON 문자열 생성 테스트 (오류 발생시 조기 감지)
            try:
                json.dumps(market_data_clean)
                json.dumps(strategy_clean)
                json.dumps(trade_result_clean)
                json.dumps(reflection_clean)
            except Exception as e:
                print(f"JSON 직렬화 테스트 실패: {e}")
                traceback.print_exc()
                raise

            LogSaver.save_log_with_result(
                market_data_clean,
                prompt,
                strategy_clean,
                trade_result_clean,
                reflection_clean
            )

            print(f"트레이딩 파이프라인 실행 완료: {trade_result.get('action', 'NO_ACTION')}")

        except Exception as e:
            print(f"트레이딩 파이프라인 실행 중 오류: {e}")
            traceback.print_exc()

            # 오류가 발생해도 로그 저장 시도
            try:
                if market_data and strategy:
                    prompt = prompt or PromptGenerator.generate_strategy_prompt(market_data)

                    # 간소화된 데이터로 로그 저장
                    simplified_data = {
                        "price": float(market_data.get("price", 0)) if isinstance(market_data, dict) else 0,
                        "error": "데이터 정리 오류"
                    }

                    simplified_result = {"error": str(e), "action": "ERROR"}
                    simplified_reflection = {"error": "복기 생성 실패"}

                    LogSaver.save_log_with_result(
                        simplified_data,
                        prompt,
                        self._sanitize_data(strategy),
                        simplified_result,
                        simplified_reflection
                    )
            except Exception as log_error:
                print(f"오류 로그 저장 실패: {log_error}")
                traceback.print_exc()

    def _sanitize_data(self, data):
        """
        NumPy, pandas 객체를 포함한 데이터를 JSON 직렬화 가능한 형식으로 변환

        Args:
            data: 변환할 데이터 (딕셔너리, 리스트 등)

        Returns:
            데이터의 Python 네이티브 타입 버전
        """
        if data is None:
            return None

        # 기본 타입이면 그대로 반환
        if isinstance(data, (str, int, float, bool)) or data is None:
            return data

        # NumPy 타입 변환
        if isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.bool_):
            return bool(data)

        # Pandas 타입 변환
        if isinstance(data, pd.Timestamp):
            return data.isoformat()
        if hasattr(data, '__module__') and data.__module__ == 'numpy':
            # 기타 NumPy 타입 처리
            return float(data) if np.issubdtype(data.dtype, np.number) else str(data)

        # 스칼라 값에 대해서만 pd.isna 사용
        try:
            # pd.isna는 배열 전체에 적용될 수 있으므로,
            # 배열이 아닌 단일 값에 대해서만 호출
            if not hasattr(data, '__len__') or isinstance(data, (str, bytes)):
                if pd.isna(data):  # NaN, NaT 등 처리
                    return None
        except:
            pass  # isna 호출에 실패하면 무시

        # 컬렉션 타입 재귀적 처리
        if isinstance(data, dict):
            return {k: self._sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._sanitize_data(item) for item in data]

        # 기타 객체는 문자열로 변환 시도
        try:
            return str(data)
        except:
            return "변환 불가 객체"


if __name__ == "__main__":
    # 비동기 실행을 위한 이벤트 루프 설정
    import asyncio

    # 파이프라인 인스턴스 생성 및 실행
    pipeline = TradingPipeline()
    asyncio.run(pipeline.run_trading_pipeline())