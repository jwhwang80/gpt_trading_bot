import os
import asyncio
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta, timezone
import time

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
    all_data = []

    # 현재 시간
    current_time = datetime.now(timezone.utc)

    # 파일이 존재하면 가장 최근 데이터의 시간을 가져와서 시작 시간으로 설정
    if file_exists:
        try:
            existing_df = pd.read_csv(CSV_FILE)
            if not existing_df.empty:
                # 'Time' 열을 datetime으로 변환
                existing_df['Time'] = pd.to_datetime(existing_df['Time'])

                # 가장 최근 시간 찾기 (내림차순으로 정렬된 경우 첫 번째 행)
                latest_time = existing_df['Time'].max()

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
        df = pd.DataFrame(all_data, columns=["Time", "Open", "High", "Low", "Close", "Volume"])

        # CSV로 저장 (기존 파일이 있다면 추가)
        if file_exists:
            # 중복 방지를 위해 기존 데이터와 새 데이터를 합치고 중복 제거
            existing_df = pd.read_csv(CSV_FILE)
            existing_df['Time'] = pd.to_datetime(existing_df['Time'])

            # 새 데이터프레임 합치기
            combined_df = pd.concat([existing_df, df])

            # 중복 제거하고 시간 기준으로 내림차순 정렬
            combined_df = combined_df.drop_duplicates(subset=['Time']).sort_values(by="Time", ascending=False)

            # 최종 데이터프레임 저장
            combined_df.to_csv(CSV_FILE, index=False)
            print(f"📂 총 {len(combined_df)}개 데이터를 {CSV_FILE}에 저장 완료 (새로운 데이터 {len(df)}개).")
        else:
            # 새 파일 생성
            df = df.sort_values(by="Time", ascending=False)  # 내림차순 정렬
            df.to_csv(CSV_FILE, index=False)
            print(f"📂 총 {len(df)}개 데이터를 {CSV_FILE}에 저장 완료.")

        #convert_csv_to_excel()
    else:
        print("⚠️ 저장할 새로운 데이터가 없습니다.")


def convert_csv_to_excel():
    """CSV 파일을 엑셀로 변환하는 함수"""
    df = pd.read_csv(CSV_FILE)

    # 'Time' 열을 datetime으로 변환하고 내림차순 정렬
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values(by="Time", ascending=False)

    excel_filename = CSV_FILE.replace(".csv", ".xlsx")
    df.to_excel(excel_filename, index=False)
    print(f"📊 CSV 데이터를 {excel_filename}로 변환 완료.")


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())