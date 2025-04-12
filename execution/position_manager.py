import pandas as pd
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from execution.excel_tracker import ExcelTracker  # 수정: 올바른 임포트 경로 사용


class PositionManager:
    def __init__(self, excel_tracker: ExcelTracker = None):
        """
        포지션 관리 클래스 초기화

        Args:
            excel_tracker (ExcelTracker, optional): 엑셀 트래커 인스턴스
        """
        self.excel_tracker = excel_tracker or ExcelTracker()
        self.max_position_size = 0.05  # 최대 포지션 크기 (5%)
        self.risk_per_trade = 0.01  # 거래당 리스크 (1%)

    def calculate_position_size(
            self,
            action: str,
            confidence: float,
            current_price: float,
            stop_loss: Optional[float] = None
    ) -> float:
        """
        리스크 관리를 고려한 포지션 크기 계산

        Args:
            action (str): 거래 액션 (BUY/SELL)
            confidence (float): 전략 신뢰도 (0-100)
            current_price (float): 현재 가격
            stop_loss (Optional[float]): 손절가

        Returns:
            float: 계산된 포지션 크기 (0-1 사이 비율)
        """
        if action == 'HOLD' or confidence <= 0:
            return 0

        # 1. 신뢰도 기반 기본 포지션 계산
        confidence_based_size = min(confidence / 200, self.max_position_size)

        # 2. 손절가 기반 리스크 관리 (손절가가 있는 경우)
        if stop_loss and stop_loss > 0:
            risk_distance = 0
            if action == 'BUY' and stop_loss < current_price:
                # 롱 포지션의 리스크 거리
                risk_distance = abs((current_price - stop_loss) / current_price)
            elif action == 'SELL' and stop_loss > current_price:
                # 숏 포지션의 리스크 거리
                risk_distance = abs((stop_loss - current_price) / current_price)

            if risk_distance > 0:
                # 리스크 거리 기반 포지션 크기 계산
                risk_based_size = self.risk_per_trade / risk_distance
                # 두 방식 중 더 작은 크기 선택
                return min(confidence_based_size, risk_based_size, self.max_position_size)

        # 손절가가 없거나 유효하지 않은 경우 신뢰도 기반 크기 사용
        return confidence_based_size

    def get_open_positions(self, symbol: str) -> Dict[str, Any]:
        """
        현재 열린 포지션 정보 조회

        Args:
            symbol (str): 거래 심볼

        Returns:
            Dict[str, Any]: 열린 포지션 정보
        """
        try:
            if not os.path.exists('trades.xlsx'):
                return {"open_positions": [], "total_size": 0}

            df = pd.read_excel('trades.xlsx')

            # 해당 심볼의 거래만 필터링
            symbol_trades = df[df['symbol'] == symbol]

            # 진입, 청산 거래 분리
            entries = symbol_trades[symbol_trades['type'] == 'ENTRY']
            exits = symbol_trades[symbol_trades['type'] == 'EXIT']

            open_positions = []
            total_position_size = 0

            # 각 진입 거래에 대해 청산되지 않은 포지션 찾기
            for _, entry in entries.iterrows():
                # 해당 라운드에 청산 내역이 있는지 확인
                matching_exit = exits[exits['round'] == entry['round']]

                if matching_exit.empty:  # 청산 내역이 없으면 열린 포지션
                    position = {
                        "entry_price": entry['entry_price'],
                        "position_size": entry['position_size'],
                        "entry_time": entry['timestamp'],
                        "round": entry['round'],
                        "strategy_id": entry['strategy_id']
                    }

                    # 액션 정보가 있으면 추가
                    if 'action' in entry:
                        position['action'] = entry['action']

                    open_positions.append(position)
                    total_position_size += entry['position_size']

            return {
                "open_positions": open_positions,
                "total_size": total_position_size
            }

        except Exception as e:
            print(f"열린 포지션 조회 중 오류 발생: {e}")
            return {"open_positions": [], "total_size": 0, "error": str(e)}

    def execute_trade(
            self,
            strategy: Dict[str, Any],
            current_price: float,
            symbol: str = 'BTCUSDT'
    ) -> Dict[str, Any]:
        """
        전략 실행 및 기록

        Args:
            strategy (Dict[str, Any]): 전략 정보
            current_price (float): 현재 가격
            symbol (str): 거래 심볼

        Returns:
            Dict[str, Any]: 거래 실행 결과
        """
        action = strategy.get('action', 'HOLD')
        confidence = strategy.get('confidence', 0)
        stop_loss = strategy.get('stop_loss', 0)

        # HOLD 액션이면 거래 없음
        if action == 'HOLD' or confidence <= 0:
            return {"action": "NO_TRADE", "reason": "HOLD 전략 또는 낮은 신뢰도"}

        # 현재 열린 포지션 확인
        open_positions = self.get_open_positions(symbol)
        current_position_size = open_positions['total_size']

        # 포지션 크기 계산
        position_size = self.calculate_position_size(action, confidence, current_price, stop_loss)

        # 최대 포지션 크기 제한 검사
        if current_position_size + position_size > self.max_position_size:
            adjusted_size = max(0, self.max_position_size - current_position_size)
            if adjusted_size <= 0:
                return {
                    "action": "NO_TRADE",
                    "reason": f"최대 포지션 크기 초과 (현재: {current_position_size:.2%})"
                }
            position_size = adjusted_size

        # 거래 라운드 번호 결정
        next_round = len(open_positions['open_positions']) + 1

        # 엑셀에 거래 기록
        self.excel_tracker.log_entry(
            symbol,
            strategy.get('strategy_id', 'strategy_001'),
            current_price,
            position_size,
            next_round
        )

        # 거래 결과 반환
        return {
            "symbol": symbol,
            "action": action,
            "entry_price": current_price,
            "position_size": position_size,
            "confidence": confidence,
            "stop_loss": stop_loss,
            "take_profit": strategy.get('take_profit', 0),
            "round": next_round
        }

    def close_position(
            self,
            symbol: str,
            strategy_id: str,
            round_num: int,
            exit_price: float
    ) -> Dict[str, Any]:
        """
        포지션 청산 및 기록

        Args:
            symbol (str): 거래 심볼
            strategy_id (str): 전략 ID
            round_num (int): 거래 라운드 번호
            exit_price (float): 청산 가격

        Returns:
            Dict[str, Any]: 청산 결과
        """
        try:
            # 해당 라운드의 포지션 정보 확인
            open_positions = self.get_open_positions(symbol)

            target_position = None
            for position in open_positions['open_positions']:
                if position['round'] == round_num and position['strategy_id'] == strategy_id:
                    target_position = position
                    break

            if not target_position:
                return {"error": "청산할 포지션을 찾을 수 없습니다."}

            # 엑셀에 청산 기록
            self.excel_tracker.log_exit(
                symbol,
                strategy_id,
                exit_price,
                round_num,
                target_position.get('position_size', 0)
            )

            # PnL 계산
            entry_price = target_position['entry_price']
            position_size = target_position.get('position_size', 0)
            action = target_position.get('action', 'BUY')

            if action == 'BUY':
                profit_percentage = ((exit_price - entry_price) / entry_price) * 100
            else:  # SELL
                profit_percentage = ((entry_price - exit_price) / entry_price) * 100

            return {
                "symbol": symbol,
                "strategy_id": strategy_id,
                "round": round_num,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "position_size": position_size,
                "profit_percentage": profit_percentage,
                "status": "SUCCESS"
            }

        except Exception as e:
            print(f"포지션 청산 중 오류 발생: {e}")
            return {"error": str(e)}