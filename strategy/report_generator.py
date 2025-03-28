import pandas as pd
from typing import Dict, Any

class ReportGenerator:
    @staticmethod
    def generate_report(symbol: str, strategy_id: str) -> Dict[str, Any]:
        """
        특정 심볼과 전략 ID에 대한 트레이딩 리포트를 생성합니다.
        
        Args:
            symbol (str): 거래 심볼 (예: BTCUSDT)
            strategy_id (str): 전략 식별자
        
        Returns:
            Dict: 전략 리포트 정보
        """
        try:
            # trades.xlsx 파일에서 해당 심볼과 전략 ID의 데이터 로드
            df = pd.read_excel('trades.xlsx')
            strategy_trades = df[
                (df['symbol'] == symbol) & 
                (df['strategy_id'] == strategy_id)
            ]
            
            # 진입 및 청산 트레이드 분리
            entry_trades = strategy_trades[strategy_trades['type'] == 'ENTRY']
            exit_trades = strategy_trades[strategy_trades['type'] == 'EXIT']
            
            # 가중평균 진입가 계산
            weighted_avg_entry = (
                entry_trades['entry_price'] * entry_trades['position_size']
            ).sum() / entry_trades['position_size'].sum()
            
            # 실현 손익 계산
            realized_pnl = 0
            for entry_row in entry_trades.itertuples():
                matching_exit = exit_trades[
                    (exit_trades['round'] == entry_row.round)
                ]
                if not matching_exit.empty:
                    exit_price = matching_exit.iloc[0]['exit_price']
                    realized_pnl += (
                        exit_price - entry_row.entry_price
                    ) * entry_row.position_size
            
            # 현재 잔여 포지션 비중 계산
            total_entry_size = entry_trades['position_size'].sum()
            total_exit_size = exit_trades['position_size'].sum() if 'position_size' in exit_trades.columns else 0
            remaining_position = total_entry_size - total_exit_size
            
            return {
                "symbol": symbol,
                "strategy_id": strategy_id,
                "weighted_avg_entry_price": weighted_avg_entry,
                "realized_pnl": realized_pnl,
                "remaining_position_size": remaining_position,
                "total_trades": len(entry_trades),
                "profitable_trades": len(exit_trades[exit_trades['exit_price'] > weighted_avg_entry])
            }
        
        except Exception as e:
            print(f"리포트 생성 중 오류 발생: {e}")
            return {
                "symbol": symbol,
                "strategy_id": strategy_id,
                "error": str(e)
            }

# 사용 예시
if __name__ == "__main__":
    report = ReportGenerator.generate_report('BTCUSDT', 'strategy_001')
    print(report)
