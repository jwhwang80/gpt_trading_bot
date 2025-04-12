import pandas as pd
import os
from datetime import datetime


class ExcelTracker:
    def __init__(self, filename='trades.xlsx'):
        self.filename = filename
        self._init_excel()

    def _init_excel(self):
        """
        Excel 파일이 존재하지 않을 경우 초기화합니다.
        """
        if not os.path.exists(self.filename):
            columns = [
                'timestamp', 'symbol', 'strategy_id',
                'entry_price', 'exit_price',
                'position_size', 'round', 'type'
            ]
            df = pd.DataFrame(columns=columns)
            df.to_excel(self.filename, index=False)

    def log_entry(self, symbol, strategy_id, entry_price, position_size, round_num):
        """
        진입 내역을 Excel에 기록합니다.

        Args:
            symbol (str): 거래 심볼
            strategy_id (str): 전략 ID
            entry_price (float): 진입 가격
            position_size (float): 포지션 크기
            round_num (int): 거래 라운드 번호
        """
        try:
            df = pd.read_excel(self.filename)
            new_entry = pd.DataFrame({
                'timestamp': [datetime.now()],
                'symbol': [symbol],
                'strategy_id': [strategy_id],
                'entry_price': [entry_price],
                'position_size': [position_size],
                'round': [round_num],
                'type': ['ENTRY']
            })

            # 수정: concat 경고 해결
            if df.empty:
                df = new_entry.copy()
            else:
                df = pd.concat([df, new_entry], ignore_index=True)

            df.to_excel(self.filename, index=False)
        except Exception as e:
            print(f"진입 기록 중 오류 발생: {e}")

    def log_exit(self, symbol, strategy_id, exit_price, round_num, position_size=None):
        """
        청산 내역을 Excel에 기록합니다.

        Args:
            symbol (str): 거래 심볼
            strategy_id (str): 전략 ID
            exit_price (float): 청산 가격
            round_num (int): 거래 라운드 번호
            position_size (float, optional): 청산하는 포지션 크기
        """
        try:
            df = pd.read_excel(self.filename)
            new_exit = pd.DataFrame({
                'timestamp': [datetime.now()],
                'symbol': [symbol],
                'strategy_id': [strategy_id],
                'exit_price': [exit_price],
                'round': [round_num],
                'position_size': [position_size],
                'type': ['EXIT']
            })

            # 수정: concat 경고 해결
            if df.empty:
                df = new_exit.copy()
            else:
                df = pd.concat([df, new_exit], ignore_index=True)

            df.to_excel(self.filename, index=False)
        except Exception as e:
            print(f"청산 기록 중 오류 발생: {e}")