import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, Any, List, Optional, Tuple


class MVRVZScoreCalculator:
    """
    Bitcoin MVRV Z-Score 계산기

    MVRV Z-Score = (Market Value - Realized Value) / Standard Deviation of (Market Value)
    시장 가치(Market Value)와 실현 가치(Realized Value)의 차이를
    시장 가치의 표준편차로 나눈 값입니다.
    """

    def __init__(self, csv_file: str = "mvrv_data.csv"):
        """
        MVRV Z-Score 계산기 초기화

        Args:
            csv_file (str): MVRV 데이터를 저장할 CSV 파일 경로
        """
        # 커뮤니티 API URL 사용 (API 키 필요 없음)
        self.base_url = "https://community-api.coinmetrics.io/v4"
        self.asset = "btc"  # 비트코인
        self.csv_file = csv_file

        # 여러 윈도우 기간 설정
        self.short_window = 365  # 1년
        self.long_window = 1460  # 4년 (365*4)

        # 초기화 시 CSV 파일 확인 및 로드
        self.df = self._load_csv_data()

    def _load_csv_data(self) -> pd.DataFrame:
        """
        CSV 파일에서 기존 MVRV 데이터를 로드합니다.
        파일이 없으면 빈 데이터프레임을 반환합니다.

        Returns:
            pd.DataFrame: MVRV 데이터가 포함된 데이터프레임
        """
        try:
            if os.path.exists(self.csv_file):
                print(f"기존 CSV 파일 로드: {self.csv_file}")
                df = pd.read_csv(self.csv_file)

                # 날짜 타입 변환
                df["date"] = pd.to_datetime(df["date"])

                # 타임존 정보 통일 (naive datetime으로 변환)
                if not df.empty:
                    df["date"] = df["date"].dt.tz_localize(None)

                # 날짜 기준으로 내림차순 정렬 (최신 날짜가 먼저 오도록)
                df = df.sort_values("date", ascending=False)

                # 최신 데이터 확인
                if not df.empty:
                    latest_date = df["date"].max()
                    print(f"최신 데이터 날짜: {latest_date}")

                return df
            else:
                print(f"CSV 파일 없음: {self.csv_file}. 새 데이터를 다운로드합니다.")
                return pd.DataFrame(columns=["date", "market_cap", "realized_cap", "mvrv_ratio",
                                             "market_cap_std_1y", "market_cap_std_4y",
                                             "mvrv_z_score_1y", "mvrv_z_score_4y", "mvrv_z_score_historical"])
        except Exception as e:
            print(f"CSV 파일 로드 중 오류 발생: {e}")
            return pd.DataFrame(columns=["date", "market_cap", "realized_cap", "mvrv_ratio",
                                         "market_cap_std_1y", "market_cap_std_4y",
                                         "mvrv_z_score_1y", "mvrv_z_score_4y", "mvrv_z_score_historical"])

    def _get_latest_date_in_csv(self) -> Optional[datetime]:
        """
        CSV 파일에서 가장 최근 날짜를 가져옵니다.

        Returns:
            Optional[datetime]: 가장 최근 날짜 또는 파일이 비어있으면 None
        """
        if not self.df.empty:
            latest_date = self.df["date"].max()
            # 타임존 정보 제거 (naive datetime으로 변환)
            if latest_date.tzinfo is not None:
                latest_date = latest_date.replace(tzinfo=None)
            return latest_date
        return None

    def _make_api_request(self, endpoint: str, params: Dict) -> Dict[str, Any]:
        """
        CoinMetrics API에 요청을 보내는 내부 메서드

        Args:
            endpoint (str): API 엔드포인트
            params (Dict): API 요청 파라미터

        Returns:
            Dict[str, Any]: API 응답 데이터
        """
        url = f"{self.base_url}/{endpoint}"

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # HTTP 오류 확인
            return response.json()
        except requests.RequestException as e:
            print(f"API 요청 오류: {e}")
            # API 요청이 실패한 경우 빈 데이터 반환
            return {"data": []}

    def update_market_and_realized_cap(self) -> pd.DataFrame:
        """
        최신 비트코인 시장 가치와 실현 가치 데이터를 가져와 CSV 파일에 추가합니다.
        이미 저장된 데이터는 다시 가져오지 않고, 최신 데이터만 업데이트합니다.

        Returns:
            pd.DataFrame: 업데이트된 MVRV 데이터가 포함된 데이터프레임
        """
        # 최신 날짜 확인
        latest_date = self._get_latest_date_in_csv()

        # API에서 가져올 시작 날짜 설정
        if latest_date:
            # 이미 데이터가 있다면 최신 날짜 다음 날부터 데이터 요청
            start_date = latest_date + timedelta(days=1)
            # 모든 날짜를 타임존이 없는 형태로 통일
            start_date = start_date.replace(tzinfo=None)
        else:
            # 데이터가 없다면 기본 기간(4년) 동안의 데이터 요청
            start_date = datetime.now() - timedelta(days=self.long_window)
            # 타임존 정보 제거
            start_date = start_date.replace(tzinfo=None)

        # 현재 날짜 기준 종료 날짜 설정 (타임존 없음)
        end_date = datetime.now().replace(tzinfo=None)

        # 시작 날짜가 현재 날짜보다 이후라면 업데이트 필요 없음
        if start_date >= end_date:
            print("이미 최신 데이터가 저장되어 있습니다.")
            return self.df

        # 날짜 형식을 API에 맞게 변환
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        print(f"데이터 업데이트: {start_str} ~ {end_str}")

        # 지표 정의
        metrics = "CapMrktCurUSD,CapRealUSD"

        # 데이터를 저장할 리스트
        all_data = []

        # 날짜 범위를 분할하여 데이터 요청
        current_start = start_date

        while current_start < end_date:
            # 100일 단위로 나누어 요청 (API 제한)
            current_end = min(current_start + timedelta(days=100), end_date)

            # 날짜 형식 변환
            current_start_str = current_start.strftime('%Y-%m-%d')
            current_end_str = current_end.strftime('%Y-%m-%d')

            print(f"데이터 요청: {current_start_str} ~ {current_end_str}")

            # API 요청 파라미터
            params = {
                "metrics": metrics,
                "assets": self.asset,
                "start_time": current_start_str,
                "end_time": current_end_str,
                "page_size": 100,
                "pretty": False
            }

            # API 요청
            response = self._make_api_request("timeseries/asset-metrics", params)

            # 응답 데이터 처리
            if "data" in response and response["data"]:
                all_data.extend(response["data"])

            # 다음 시작일 설정
            current_start = current_end

            # API 제한 방지를 위한 짧은 대기
            time.sleep(1)

        # 결과가 없으면 기존 데이터프레임 반환
        if not all_data:
            print("새로운 데이터가 없습니다.")
            if self.df.empty:
                print("기존 데이터도 없습니다. API 요청 또는 네트워크 연결을 확인하세요.")
            return self.df

        # 새 데이터프레임 생성
        new_df = pd.DataFrame(all_data)

        # 열 이름 변경
        new_df = new_df.rename(columns={
            "time": "date",
            "CapMrktCurUSD": "market_cap",
            "CapRealUSD": "realized_cap"
        })

        # 날짜 타입 변환 및 타임존 제거
        new_df["date"] = pd.to_datetime(new_df["date"])
        new_df["date"] = new_df["date"].dt.tz_localize(None)  # 타임존 제거

        # 숫자 타입 변환
        new_df["market_cap"] = pd.to_numeric(new_df["market_cap"])
        new_df["realized_cap"] = pd.to_numeric(new_df["realized_cap"])

        # 자산 열 제거
        if "asset" in new_df.columns:
            new_df = new_df.drop("asset", axis=1)

        # 새 데이터와 기존 데이터 합치기
        if self.df.empty:
            combined_df = new_df
        else:
            # 날짜 중복 제거 (최신 데이터 유지)
            combined_df = pd.concat([self.df, new_df])
            combined_df = combined_df.drop_duplicates(subset=["date"], keep="first")

        # 날짜 기준으로 내림차순 정렬
        combined_df = combined_df.sort_values("date", ascending=False)

        # 데이터프레임 업데이트
        self.df = combined_df

        print(f"데이터 업데이트 완료: 총 {len(self.df)}개 항목")

        return self.df

    def calculate_mvrv_data(self) -> pd.DataFrame:
        """
        MVRV 관련 데이터 계산

        Returns:
            pd.DataFrame: MVRV 데이터가 계산된 데이터프레임
        """
        if self.df.empty:
            print("데이터가 없습니다. 먼저 데이터를 가져오세요.")
            return self.df

        # 정렬 방향 임시 변경 (오름차순으로 변경하여 계산)
        temp_df = self.df.sort_values("date", ascending=True).copy()

        # 1. MVRV 비율 계산 = 시장 가치 / 실현 가치
        temp_df["mvrv_ratio"] = temp_df["market_cap"] / temp_df["realized_cap"]

        # 2. 시장 가치와 실현 가치의 차이
        temp_df["market_realized_delta"] = temp_df["market_cap"] - temp_df["realized_cap"]

        # 이동 윈도우 계산 (오름차순 정렬 상태에서)

        # 1. 1년 이동 윈도우 기반 계산
        temp_df["market_cap_std_1y"] = temp_df["market_cap"].rolling(window=self.short_window).std()
        temp_df["mvrv_z_score_1y"] = temp_df["market_realized_delta"] / temp_df["market_cap_std_1y"]

        # 2. 4년 이동 윈도우 기반 계산
        temp_df["market_cap_std_4y"] = temp_df["market_cap"].rolling(window=self.long_window).std()
        temp_df["mvrv_z_score_4y"] = temp_df["market_realized_delta"] / temp_df["market_cap_std_4y"]

        # 3. 전체 기간 기반 계산
        entire_period_market_cap_std = temp_df["market_cap"].std()
        temp_df["mvrv_z_score_historical"] = temp_df["market_realized_delta"] / entire_period_market_cap_std

        # 다시 내림차순 정렬
        self.df = temp_df.sort_values("date", ascending=False)

        return self.df

    def save_to_csv(self) -> None:
        """
        현재 데이터프레임을 CSV 파일로 저장합니다.
        """
        if self.df.empty:
            print("저장할 데이터가 없습니다.")
            return

        try:
            # 파일이 있는 디렉토리 확인
            output_dir = os.path.dirname(self.csv_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # CSV 파일로 저장
            self.df.to_csv(self.csv_file, index=False)
            print(f"MVRV 데이터를 {self.csv_file}에 저장했습니다. (총 {len(self.df)}개 항목)")

        except Exception as e:
            print(f"CSV 파일 저장 중 오류 발생: {e}")

    def get_latest_mvrv_z_scores(self, recalculate: bool = True) -> Dict[str, Any]:
        """
        최신 MVRV Z-Score 값들을 가져옵니다.

        Args:
            recalculate (bool): 최신 데이터를 다시 계산할지 여부

        Returns:
            Dict[str, Any]: 최신 MVRV Z-Score 정보
        """
        try:
            if recalculate:
                # 최신 데이터 업데이트
                self.update_market_and_realized_cap()

                # MVRV 데이터 계산
                self.calculate_mvrv_data()

                # CSV 파일 저장
                self.save_to_csv()

            # 데이터프레임이 비어있는지 확인
            if self.df.empty:
                print("경고: 데이터프레임이 비어 있습니다. API 요청 또는 네트워크 연결을 확인하세요.")
                return {
                    "date": None,
                    "market_cap": None,
                    "realized_cap": None,
                    "mvrv_ratio": None,
                    "market_realized_delta": None,
                    "z_scores": {
                        "1y": None,
                        "4y": None,
                        "historical": None
                    },
                    "signals": {
                        "1y": "DATA_ERROR",
                        "4y": "DATA_ERROR",
                        "historical": "DATA_ERROR"
                    },
                    "error": "데이터를 가져오지 못했습니다."
                }

            # 최신 데이터가 필요한 모든 열을 가지고 있는지 확인
            latest = self.df.iloc[0]
            required_columns = ["market_cap", "realized_cap", "mvrv_ratio",
                                "mvrv_z_score_1y", "mvrv_z_score_4y", "mvrv_z_score_historical"]

            missing_columns = [col for col in required_columns if col not in latest or pd.isna(latest[col])]
            if missing_columns:
                print(f"경고: 다음 열의 데이터가 없습니다: {missing_columns}")
                print("데이터 계산을 다시 시도합니다.")
                # 데이터 다시 계산 시도
                self.calculate_mvrv_data()
                latest = self.df.iloc[0]

            # Z-Score 해석
            signal_1y = self._interpret_z_score(latest["mvrv_z_score_1y"] if "mvrv_z_score_1y" in latest else None)
            signal_4y = self._interpret_z_score(latest["mvrv_z_score_4y"] if "mvrv_z_score_4y" in latest else None)
            signal_historical = self._interpret_z_score(
                latest["mvrv_z_score_historical"] if "mvrv_z_score_historical" in latest else None)

            return {
                "date": latest["date"],
                "market_cap": latest["market_cap"],
                "realized_cap": latest["realized_cap"],
                "mvrv_ratio": latest["mvrv_ratio"],
                "market_realized_delta": latest["market_realized_delta"] if "market_realized_delta" in latest else None,
                "z_scores": {
                    "1y": latest["mvrv_z_score_1y"] if "mvrv_z_score_1y" in latest else None,
                    "4y": latest["mvrv_z_score_4y"] if "mvrv_z_score_4y" in latest else None,
                    "historical": latest["mvrv_z_score_historical"] if "mvrv_z_score_historical" in latest else None
                },
                "signals": {
                    "1y": signal_1y,
                    "4y": signal_4y,
                    "historical": signal_historical
                }
            }
        except Exception as e:
            print(f"MVRV Z-Score 계산 중 오류 발생: {e}")
            return {
                "date": None,
                "market_cap": None,
                "realized_cap": None,
                "mvrv_ratio": None,
                "market_realized_delta": None,
                "z_scores": {
                    "1y": None,
                    "4y": None,
                    "historical": None
                },
                "signals": {
                    "1y": "ERROR",
                    "4y": "ERROR",
                    "historical": "ERROR"
                },
                "error": str(e)
            }

    def _interpret_z_score(self, z_score: float) -> str:
        """
        MVRV Z-Score 값을 해석합니다.

        Args:
            z_score (float): MVRV Z-Score 값

        Returns:
            str: 해석 신호
        """
        if pd.isna(z_score):
            return "UNKNOWN"

        if z_score > 7:
            return "EXTREME_OVERVALUED"
        elif z_score > 3:
            return "OVERVALUED"
        elif z_score > 1:
            return "SLIGHTLY_OVERVALUED"
        elif z_score < -0.5:
            return "UNDERVALUED"
        elif z_score < -1:
            return "OPPORTUNITY_TO_BUY"
        else:
            return "NEUTRAL"


# 사용 예시
if __name__ == "__main__":
    calculator = MVRVZScoreCalculator("output/mvrv_data.csv")

    # 최신 데이터 업데이트 및 계산
    result = calculator.get_latest_mvrv_z_scores(recalculate=True)

    print(f"날짜: {result['date']}")

    # None 체크 추가
    if result['market_cap'] is not None:
        print(f"시장 가치: ${result['market_cap'] / 1e9:.2f}B")
    else:
        print("시장 가치: 데이터 없음")

    if result['realized_cap'] is not None:
        print(f"실현 가치: ${result['realized_cap'] / 1e9:.2f}B")
    else:
        print("실현 가치: 데이터 없음")

    if result['mvrv_ratio'] is not None:
        print(f"MVRV 비율: {result['mvrv_ratio']:.2f}")
    else:
        print("MVRV 비율: 데이터 없음")

    print("\nMVRV Z-Score:")
    if result['z_scores']['1y'] is not None:
        print(f"1년 기준: {result['z_scores']['1y']:.2f} ({result['signals']['1y']})")
    else:
        print("1년 기준: 데이터 없음")

    if result['z_scores']['4y'] is not None:
        print(f"4년 기준: {result['z_scores']['4y']:.2f} ({result['signals']['4y']})")
    else:
        print("4년 기준: 데이터 없음")

    if result['z_scores']['historical'] is not None:
        print(f"역사적 기준: {result['z_scores']['historical']:.2f} ({result['signals']['historical']})")
    else:
        print("역사적 기준: 데이터 없음")