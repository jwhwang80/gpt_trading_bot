import json
from typing import Dict, Any, List, Optional
import pandas as pd
import os
from datetime import datetime


class PromptGenerator:
    @staticmethod
    def generate_strategy_prompt(
            market_data: Dict[str, Any],
            historical_trades: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        시장 데이터와 과거 거래를 기반으로 GPT 전략 프롬프트 생성

        Args:
            market_data (Dict): 현재 시장 데이터
            historical_trades (Optional[List]): 과거 거래 내역

        Returns:
            str: GPT에 전달할 프롬프트 문자열
        """
        # 기본 시장 데이터 포맷팅
        prompt = f"""
현재 시장 상황을 분석하고 {market_data.get('timeframe', '4H')} 시간프레임의 BTCUSDT에 대한 트레이딩 전략을 제안해주세요.

시장 데이터:
- 현재 가격: {market_data.get('price', 0)}
- 타임스탬프: {market_data.get('timestamp', '정보 없음')}
- 시간프레임: {market_data.get('timeframe', '4H')}
"""

        # 지표 데이터 추가
        indicators = market_data.get('indicators', {})
        if indicators:
            prompt += "\n기술적 지표:\n"
            # 주요 지표만 선택적으로 추가
            for indicator_name, indicator_data in indicators.items():
                # 지표 이름과 값만 간략하게 표시
                if isinstance(indicator_data, dict) and 'signal' in indicator_data:
                    prompt += f"- {indicator_name}: 신호 = {indicator_data['signal']}"
                    if 'value' in indicator_data:
                        prompt += f", 값 = {indicator_data['value']}"
                    prompt += "\n"

        # 과거 거래 내역 정보 추가
        if historical_trades and len(historical_trades) > 0:
            prompt += "\n최근 거래 내역:\n"
            for i, trade in enumerate(historical_trades[-3:]):  # 최근 3개 거래만 표시
                action = trade.get('action', 'UNKNOWN')
                entry_price = trade.get('entry_price', 0)
                result = trade.get('result', '정보 없음')

                prompt += f"#{i + 1}: {action} @ {entry_price:.2f}, 결과: {result}\n"

        # 응답 요구사항 추가
        prompt += """
요구사항:
1. JSON 형식으로 응답 (예: {"action": "BUY/SELL/HOLD", "confidence": 0-100, "reasoning": "설명", "stop_loss": 가격, "take_profit": 가격})
2. action은 반드시 BUY, SELL, HOLD 중 하나여야 합니다.
3. confidence는 0-100 사이의 숫자로 표현하세요.
4. 전략의 근거를 reasoning 필드에 명확히 제시하세요.
5. 진입을 제안하는 경우 (BUY 또는 SELL) 반드시 stop_loss와 take_profit 가격을 제안하세요.
6. 포지션 사이즈는 confidence에 비례하며, 최대 5%를 초과하지 않습니다.
7. 현재 시장 상황, 기술적 지표, 추세를 종합적으로 고려하세요.

JSON 형식으로만 답변해주세요.
"""

        return prompt

    @staticmethod
    def parse_gpt_response(gpt_response: str) -> Dict[str, Any]:
        """
        GPT 응답을 JSON으로 파싱합니다.

        Args:
            gpt_response (str): GPT의 응답 문자열

        Returns:
            Dict: 파싱된 전략 JSON
        """
        try:
            # JSON 부분만 추출하기 위한 간단한 파싱 로직
            if '{' in gpt_response and '}' in gpt_response:
                start = gpt_response.find('{')
                end = gpt_response.rfind('}') + 1
                json_str = gpt_response[start:end]
                parsed_data = json.loads(json_str)
            else:
                parsed_data = json.loads(gpt_response)

            # 필수 필드 검증
            if 'action' not in parsed_data:
                parsed_data['action'] = 'HOLD'
                parsed_data['reasoning'] = parsed_data.get('reasoning', '') + " (기본 HOLD 액션 적용)"

            if 'confidence' not in parsed_data:
                parsed_data['confidence'] = 0

            # action 정규화
            if parsed_data['action'].upper() not in ['BUY', 'SELL', 'HOLD']:
                parsed_data['action'] = 'HOLD'
                parsed_data['reasoning'] = parsed_data.get('reasoning', '') + " (유효하지 않은 액션, HOLD 적용)"

            parsed_data['action'] = parsed_data['action'].upper()

            return parsed_data

        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}. 원본 응답: {gpt_response[:100]}...")
            return {
                "action": "HOLD",
                "confidence": 0,
                "reasoning": "JSON 파싱 실패"
            }

    @staticmethod
    def load_historical_trades(max_trades: int = 5) -> List[Dict[str, Any]]:
        """
        과거 거래 내역을 로드합니다.

        Args:
            max_trades (int): 로드할 최대 거래 수

        Returns:
            List[Dict[str, Any]]: 과거 거래 목록
        """
        try:
            if not os.path.exists('trades.xlsx'):
                return []

            df = pd.read_excel('trades.xlsx')

            # 거래 타입별로 필터링
            entry_trades = df[df['type'] == 'ENTRY'].sort_values('timestamp', ascending=False).head(max_trades)
            exit_trades = df[df['type'] == 'EXIT']

            historical_trades = []

            for _, entry in entry_trades.iterrows():
                # 해당 진입에 대한 청산 찾기
                matching_exit = exit_trades[
                    (exit_trades['round'] == entry['round']) &
                    (exit_trades['strategy_id'] == entry['strategy_id'])
                    ]

                trade_result = "진행 중"
                profit_loss = 0

                if not matching_exit.empty:
                    exit_price = matching_exit.iloc[0]['exit_price']
                    if entry['symbol'].endswith('USDT'):  # USDT 페어인 경우
                        if entry['action'] == 'BUY':
                            profit_loss = (exit_price - entry['entry_price']) / entry['entry_price'] * 100
                        else:  # SELL인 경우
                            profit_loss = (entry['entry_price'] - exit_price) / entry['entry_price'] * 100

                    trade_result = f"{'이익' if profit_loss > 0 else '손실'} ({profit_loss:.2f}%)"

                historical_trades.append({
                    "action": entry['action'] if 'action' in entry else 'BUY',
                    "entry_price": entry['entry_price'],
                    "timestamp": entry['timestamp'],
                    "confidence": 0,  # 이 정보는 Excel에 없을 수 있음
                    "symbol": entry['symbol'],
                    "strategy_id": entry['strategy_id'],
                    "result": trade_result,
                    "profit_loss": profit_loss
                })

            return historical_trades

        except Exception as e:
            print(f"과거 거래 내역을 로드하는 중 오류 발생: {e}")
            return []