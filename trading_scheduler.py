import schedule
import time
import asyncio
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

    def _run_pipeline(self):
        """
        비동기 파이프라인을 동기 환경에서 실행하기 위한 래퍼 함수
        """
        asyncio.run(self.pipeline.run_trading_pipeline())

    def start_scheduling(self):
        """
        스케줄링 시작 및 주기적 파이프라인 실행
        """
        # 비동기 파이프라인을 동기적으로 실행하는 래퍼 함수 등록
        schedule.every(self.interval_hours).hours.do(self._run_pipeline)

        print(f"{self.interval_hours}시간마다 트레이딩 파이프라인 실행")

        # 바로 첫 실행
        print("첫 번째 실행 시작...")
        self._run_pipeline()

        # 이후 스케줄에 따라 실행
        while True:
            schedule.run_pending()
            time.sleep(60)  # 1분마다 스케줄 확인 (CPU 부하 감소)

    @staticmethod
    def run_once():
        """
        한 번만 파이프라인 실행
        """
        pipeline = TradingPipeline()
        asyncio.run(pipeline.run_trading_pipeline())


# 사용 예시
if __name__ == "__main__":
    # 1시간마다 자동 실행
    scheduler = TradingScheduler(interval_hours=1)
    scheduler.start_scheduling()

    # 또는 한 번만 실행
    # TradingScheduler.run_once()