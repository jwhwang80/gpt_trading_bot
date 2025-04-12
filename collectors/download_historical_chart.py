import os
import asyncio
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta, timezone
import time
import csv
from io import StringIO

api_key = os.environ.get('BINANCE_TEST_API_KEY')
api_secret = os.environ.get('BINANCE_TEST_API_SECRET')

SYMBOL = "BTCUSDT"
DAYS = 3000
CSV_FILE = f"{SYMBOL}_1m_{DAYS}day_data.csv"  # 1분봉으로 변경
MAX_RETRIES = 3


async def fetch_historical_chart(client, symbol, start_time, end_time, interval):
    """Binance에서 1분봉 데이터를 가져오는 함수 (예외 처리 포함)"""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            klines = client.get_historical_klines(
                symbol,
                interval,
                start_time.strftime("%Y-%m-%d %H:%M:%S"),
                end_time.strftime("%Y-%m-%d %H:%M:%S")
            )

            if not klines:
                print(f"⚠️ 데이터 없음: {start_time} ~ {end_time}")
                return []

            data = [
                [
                    datetime.fromtimestamp(kline[0] / 1000, tz=timezone.utc),  # timestamp
                    float(kline[1]),  # Open
                    float(kline[2]),  # High
                    float(kline[3]),  # Low
                    float(kline[4]),  # Close
                    float(kline[5])  # Volume
                ]
                for kline in klines
            ]

            # 🔥 최신 시간이 첫 번째가 되도록 정렬 (내림차순)
            data.sort(reverse=True, key=lambda x: x[0])

            return data

        except Exception as e:
            print(f"⚠️ 오류 발생 (시도 {attempt}/{MAX_RETRIES}): {e}")
            if "Too many requests" in str(e):
                print("⏳ API 제한으로 10초 대기 후 재시도...")
                time.sleep(10)  # Binance 속도 제한 시 대기
            else:
                time.sleep(2)  # 일반적인 네트워크 오류 시 2초 대기

    print(f"❌ {start_time} ~ {end_time} 데이터 가져오기 실패. 스킵함.")
    return []


async def main():
    client = Client(api_key, api_secret, {"timeout": 5})

    # CSV 파일이 존재하는지 확인
    file_exists = os.path.exists(CSV_FILE)

    # 현재 시간
    current_time = datetime.now(timezone.utc)

    if file_exists:
        try:
            # 파일의 첫 번째 행만 읽어서 최신 데이터 시간 확인 (시간 역순으로 저장되므로)
            with open(CSV_FILE, 'r') as f:
                # 헤더 읽기
                header = f.readline().strip()
                # 첫 번째 데이터 행 읽기
                first_line = f.readline().strip()
                if first_line:
                    # CSV 형식으로 파싱
                    first_row = next(csv.reader(StringIO(first_line)))
                    # 'Time' 열의 인덱스 찾기
                    columns = header.split(',')
                    time_index = columns.index('Time')
                    # 최신 데이터 시간 파싱
                    latest_time = pd.to_datetime(first_row[time_index])

                    # 시작 시간을 가장 최근 데이터 이후로 설정 (1분 추가)
                    start_time = pd.to_datetime(latest_time).to_pydatetime().replace(tzinfo=timezone.utc) + timedelta(
                        minutes=1)

                    print(f"📈 기존 데이터의 마지막 시간: {latest_time}")
                    print(f"🔄 {start_time}부터 현재까지 데이터를 가져옵니다.")
                else:
                    print("⚠️ 기존 파일이 비어 있습니다. 전체 기간 데이터를 다운로드합니다.")
                    start_time = current_time - timedelta(days=DAYS)
        except Exception as e:
            print(f"⚠️ 기존 파일 읽기 오류: {e}")
            start_time = current_time - timedelta(days=DAYS)
    else:
        print("⚠️ 기존 파일이 없습니다. 전체 기간 데이터를 다운로드합니다.")
        start_time = current_time - timedelta(days=DAYS)

    # 만약 시작 시간이 현재 시간보다 이후라면, 이미 최신 상태
    if start_time >= current_time:
        print("✅ 데이터가 이미 최신 상태입니다.")
        return

    # API 호출 시간 제한을 고려하여 하루 단위로 데이터 가져오기
    # 1분봉은 5분봉보다 데이터 양이 많으므로 더 짧은 기간으로 나눕니다
    delta = timedelta(hours=12)  # 12시간 단위로 변경

    # 시작 시간부터 현재 시간까지 데이터 가져오기
    end_time = current_time
    chunk_start_time = start_time

    # 새 데이터를 담을 리스트
    all_data = []

    while chunk_start_time < end_time:
        # 청크 종료 시간 계산
        chunk_end_time = chunk_start_time + delta
        if chunk_end_time > end_time:
            chunk_end_time = end_time

        print(f"📡 Fetching data from {chunk_start_time} to {chunk_end_time}...")

        data = await fetch_historical_chart(client, SYMBOL, chunk_start_time, chunk_end_time,
                                            Client.KLINE_INTERVAL_1MINUTE)

        if data:
            all_data.extend(data)  # 정렬된 데이터 추가
            print(f"✅ {len(data)}개 데이터 저장 완료.")
        else:
            print(f"⚠️ {chunk_start_time} ~ {chunk_end_time} 데이터 없음. 다음으로 진행.")

        # 다음 청크의 시작 시간 업데이트
        chunk_start_time = chunk_end_time

        # API 제한 방지를 위해 대기
        await asyncio.sleep(1)

    if all_data:
        # 새 데이터를 DataFrame으로 변환
        df_new = pd.DataFrame(all_data, columns=["Time", "Open", "High", "Low", "Close", "Volume"])

        if file_exists:
            # 파일이 이미 존재하면 새 데이터만 추가
            # 파일 시작 부분에 새 데이터 추가 (역순 정렬 유지)
            df_new = df_new.sort_values(by="Time", ascending=False)

            # 중복 방지를 위해 새 데이터와 기존 파일의 첫 몇 줄을 비교
            with open(CSV_FILE, 'r') as f:
                header = f.readline().strip()  # 헤더 읽기

            # 새 데이터를 임시 파일에 쓰기
            temp_file = f"{CSV_FILE}.temp"
            df_new.to_csv(temp_file, index=False, mode='w')

            # 기존 파일 내용을 임시 파일에 추가
            with open(CSV_FILE, 'r') as src, open(temp_file, 'a') as dst:
                # 헤더 건너뛰기
                next(src)
                # 나머지 모든 줄 복사
                for line in src:
                    dst.write(line)

            # 임시 파일을 원래 파일로 이동
            os.replace(temp_file, CSV_FILE)

            print(f"📂 새로운 데이터 {len(df_new)}개를 기존 파일 앞부분에 추가했습니다.")
        else:
            # 새 파일 생성
            df_new = df_new.sort_values(by="Time", ascending=False)  # 내림차순 정렬
            df_new.to_csv(CSV_FILE, index=False)
            print(f"📂 총 {len(df_new)}개 데이터를 {CSV_FILE}에 저장 완료.")
    else:
        print("⚠️ 저장할 새로운 데이터가 없습니다.")


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())