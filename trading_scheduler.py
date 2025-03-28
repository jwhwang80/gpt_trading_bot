import schedule
import time
from main import TradingPipeline

class TradingScheduler:
    def __init__(self, interval_hours: int = 1):
        """
        트레이딩 파이프라인 스케줄러 초기화
        
        Args:
            interval_hours (int): 파이프라인 실행 간격 (시간)
        """
        self.pipeline = TradingPipeline()
        self.interval_hours = interval_hours
    
    def start_scheduling(self):
        """
        스케줄링 시작 및 주기적 파이프라인 실행
        """
        schedule.every(self.interval_hours).hours.do(self.pipeline.run_trading_pipeline)
        
        print(f"{self.interval_hours}시간마다 트레이딩 파이프라인 실행")
        
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    @staticmethod
    def run_once():
        """
        한 번만 파이프라인 실행
        """
        pipeline = TradingPipeline()
        pipeline.run_trading_pipeline()

# 사용 예시
if __name__ == "__main__":
    # 1시간마다 자동 실행
    scheduler = TradingScheduler(interval_hours=1)
    scheduler.start_scheduling()
    
    # 또는 한 번만 실행
    # TradingScheduler.run_once()
