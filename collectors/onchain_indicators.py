import pandas as pd
import requests
import json
import os
from typing import Dict, Any, List, Optional
import time
from datetime import datetime

from collectors.base_indicator import BaseIndicator


class BitcoinDataAPI:
    """
    bitcoin-data.com API와 통신하기 위한 유틸리티 클래스
    """
    BASE_URL = "https://bitcoin-data.com/v1"

    @staticmethod
    def get_indicator(endpoint: str) -> Dict[str, Any]:
        """
        지정된 엔드포인트에서 최신 지표 데이터를 가져옵니다.

        Args:
            endpoint (str): API 엔드포인트 (예: 'sopr/last')

        Returns:
            Dict[str, Any]: API 응답 데이터
        """
        try:
            url = f"{BitcoinDataAPI.BASE_URL}/{endpoint}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # HTTP 오류 확인
            return response.json()
        except requests.RequestException as e:
            print(f"API 요청 오류 ({endpoint}): {e}")
            return {"error": str(e)}


class OnchainIndicatorBase(BaseIndicator):
    """
    bitcoin-data.com 기반 온체인 지표의 기본 클래스
    """

    def __init__(self, name: str, description: str, endpoint: str, csv_dir: str = "output/onchain_data"):
        """
        온체인 지표 초기화

        Args:
            name (str): 지표 이름
            description (str): 지표 설명
            endpoint (str): API 엔드포인트
            csv_dir (str): CSV 파일 저장 디렉토리
        """
        super().__init__(name=name, description=description)
        self.endpoint = endpoint
        self.column_name = name
        self.latest_data = None
        self.csv_dir = csv_dir
        self.csv_path = os.path.join(csv_dir, f"{name.lower()}_data.csv")

        # CSV 디렉토리가 없으면 생성
        os.makedirs(csv_dir, exist_ok=True)

    def fetch_latest_data(self) -> Dict[str, Any]:
        """
        CSV 파일을 먼저 확인하고, 필요하면 API에서 최신 지표 데이터를 가져옵니다.

        Returns:
            Dict[str, Any]: 최신 지표 데이터
        """
        try:
            # 오늘 날짜 확인
            today = datetime.now().date()

            # CSV 파일 존재 여부 확인
            if os.path.exists(self.csv_path):
                # CSV 파일 로드
                df = pd.read_csv(self.csv_path)

                # 날짜 열 확인 및 변환
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])

                    # 내림차순 정렬 (최신 데이터가 맨 위)
                    df = df.sort_values('date', ascending=False)

                    # 최신 데이터의 날짜 확인
                    if not df.empty:
                        latest_date = df['date'].iloc[0].date()

                        # 오늘 데이터가 이미 있으면 API 호출 생략
                        if latest_date == today:
                            print(f"{self.name}: 오늘 데이터가 이미 CSV에 있습니다. API 호출 생략.")
                            self.latest_data = df.iloc[0].to_dict()
                            return self.latest_data

            # CSV에 오늘 데이터가 없거나 파일이 없을 경우 API 호출
            print(f"{self.name}: API에서 최신 데이터 요청 중...")
            self.latest_data = BitcoinDataAPI.get_indicator(self.endpoint)

            # API 응답 저장
            if "error" not in self.latest_data:
                # 새 데이터 준비
                new_data = {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    self.column_name: self._extract_indicator_value(self.latest_data)
                }

                # API 응답의 다른 필드도 포함
                for key, value in self.latest_data.items():
                    if key not in new_data and key != "time":
                        new_data[key] = value

                # 새 데이터프레임 생성
                new_df = pd.DataFrame([new_data])

                # CSV 파일이 존재하면 기존 데이터와 병합
                if os.path.exists(self.csv_path):
                    df = pd.read_csv(self.csv_path)

                    # 날짜 열 변환
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])

                    # 중복 제거 (같은 날짜가 있으면 새 데이터 우선)
                    if 'date' in df.columns:
                        df['date_only'] = df['date'].dt.date
                        df = df[df['date_only'] != today]
                        df = df.drop('date_only', axis=1)

                    # 새 데이터를 위에 추가
                    df = pd.concat([new_df, df], ignore_index=True)
                else:
                    df = new_df

                # 날짜 열 변환
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])

                # 날짜 내림차순 정렬 (최신 데이터가 맨 위)
                df = df.sort_values('date', ascending=False)

                # CSV 파일로 저장
                df.to_csv(self.csv_path, index=False)
                print(f"{self.name}: 데이터를 {self.csv_path}에 저장했습니다.")

            return self.latest_data

        except Exception as e:
            print(f"{self.name} 데이터 가져오기 중 오류: {e}")
            self.latest_data = {"error": str(e)}
            return self.latest_data

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        CSV 파일에서 데이터를 로드하거나 API에서 최신 데이터를 가져와 데이터프레임에 추가합니다.

        Args:
            df (pd.DataFrame): 원본 데이터프레임

        Returns:
            pd.DataFrame: 지표가 추가된 데이터프레임
        """
        df_copy = df.copy()

        # 최신 데이터 가져오기 (CSV 파일 우선, 필요시 API 호출)
        data = self.fetch_latest_data()

        # 모든 행에 동일한 값 추가
        if "error" not in data:
            # API 응답에서 지표 값 추출
            indicator_value = self._extract_indicator_value(data)
            df_copy[self.column_name] = indicator_value
            print(f"Latest {self.name} = {indicator_value}")
        else:
            print(f"Error fetching {self.name}: {data['error']}")
            df_copy[self.column_name] = float('nan')

        return df_copy

    def _extract_indicator_value(self, data: Dict[str, Any]) -> float:
        """
        API 응답에서 지표 값을 추출합니다. 하위 클래스에서 오버라이드해야 합니다.

        Args:
            data (Dict[str, Any]): API 응답 데이터

        Returns:
            float: 추출된 지표 값
        """
        # 기본 구현은 일반적인 패턴을 따릅니다. 키가 다른 경우 하위 클래스에서 오버라이드합니다.
        # 예: data["sopr"] 또는 data["value"] 등
        key = self.name.lower().replace('-', '_')

        if key in data:
            return float(data[key])

        # 키를 찾을 수 없는 경우, 첫 번째 숫자 값을 반환합니다.
        for k, v in data.items():
            try:
                return float(v)
            except (ValueError, TypeError):
                continue

        return float('nan')  # 지표를 찾을 수 없는 경우

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        """
        지표 값을 해석합니다. 하위 클래스에서 오버라이드해야 합니다.

        Args:
            row (pd.Series): 데이터 시리즈

        Returns:
            Dict[str, Any]: 지표 해석 결과
        """
        value = row.get(self.column_name)

        if pd.isna(value):
            return {
                'value': None,
                'signal': 'UNKNOWN',
                'description': 'No data available'
            }

        # 기본 구현 - 하위 클래스에서 오버라이드하여 특정 지표 해석 로직 추가
        return {
            'value': value,
            'signal': 'NEUTRAL',
            'description': f'Current {self.name} value is {value}'
        }

    def get_columns(self) -> List[str]:
        """
        이 지표가 생성하는 열 이름 목록을 반환합니다.

        Returns:
            List[str]: 열 이름 목록
        """
        return [self.column_name]

    def load_historical_data(self) -> pd.DataFrame:
        """
        CSV 파일에서 역사적 데이터를 로드합니다.

        Returns:
            pd.DataFrame: 역사적 데이터가 포함된 데이터프레임
        """
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)

            # 날짜 열 변환
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])

            # 날짜 내림차순 정렬 (최신 데이터가 맨 위)
            df = df.sort_values('date', ascending=False)

            return df
        else:
            # CSV 파일이 없으면 빈 데이터프레임 반환
            return pd.DataFrame(columns=['date', self.column_name])


class SOPRIndicator(OnchainIndicatorBase):
    """
    SOPR (Spent Output Profit Ratio) 지표
    """

    def __init__(self):
        super().__init__(
            name="SOPR",
            description="Spent Output Profit Ratio - 온체인 수익/손실 지표",
            endpoint="sopr/last"
        )

    def _extract_indicator_value(self, data: Dict[str, Any]) -> float:
        try:
            return float(data.get("sopr", 0))
        except (ValueError, TypeError):
            return float('nan')

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        value = row.get(self.column_name)

        if pd.isna(value):
            return {
                'value': None,
                'signal': 'UNKNOWN',
                'description': 'SOPR 데이터를 가져올 수 없습니다.'
            }

        if value > 1.05:
            signal = 'STRONG_PROFIT_TAKING'
            description = 'SOPR > 1.05: 상당한 이익 실현이 발생하고 있습니다. 단기 하락 압력 가능성이 있습니다.'
        elif value > 1.02:
            signal = 'PROFIT_TAKING'
            description = 'SOPR > 1.02: 이익 실현이 일어나고 있습니다.'
        elif value > 1.0:
            signal = 'SLIGHT_PROFIT'
            description = 'SOPR > 1.0: 약간의 이익 상태입니다.'
        elif value < 0.95:
            signal = 'STRONG_LOSS_SELLING'
            description = 'SOPR < 0.95: 상당한 손실 판매가 발생하고 있습니다. 투자자들이 손실을 감수하고 매도 중입니다.'
        elif value < 0.98:
            signal = 'LOSS_SELLING'
            description = 'SOPR < 0.98: 손실 판매가 일어나고 있습니다.'
        elif value < 1.0:
            signal = 'SLIGHT_LOSS'
            description = 'SOPR < 1.0: 약간의 손실 상태입니다.'
        else:  # value == 1.0
            signal = 'BREAKEVEN'
            description = 'SOPR = 1.0: 수익도 손실도 없는 균형점입니다.'

        return {
            'value': value,
            'signal': signal,
            'description': description
        }

class LTHSOPRIndicator(OnchainIndicatorBase):
    """
    장기 보유자 SOPR (Long-Term Holder SOPR) 지표
    """

    def __init__(self):
        super().__init__(
            name="LTH_SOPR",
            description="Long-Term Holder SOPR - 장기 보유자의 지출 출력 수익 비율",
            endpoint="lth-sopr/last"
        )

    def _extract_indicator_value(self, data: Dict[str, Any]) -> float:
        try:
            return float(data.get("lthSopr", 0))
        except (ValueError, TypeError):
            return float('nan')

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        value = row.get(self.column_name)

        if pd.isna(value):
            return {
                'value': None,
                'signal': 'UNKNOWN',
                'description': 'LTH SOPR 데이터를 가져올 수 없습니다.'
            }

        if value > 5.0:
            signal = 'LTH_EXTREME_PROFIT_TAKING'
            description = 'LTH SOPR > 5.0: 장기 보유자들이 상당한 이익을 실현하고 있습니다. 주요 배분이 진행 중일 수 있습니다.'
        elif value > 3.0:
            signal = 'LTH_PROFIT_TAKING'
            description = 'LTH SOPR > 3.0: 장기 보유자들이 이익을 실현하고 있습니다. 상단 형성 신호일 수 있습니다.'
        elif value > 1.5:
            signal = 'LTH_MODERATE_PROFIT'
            description = 'LTH SOPR > 1.5: 장기 보유자들이 중간 수준의 이익 상태입니다.'
        elif value > 1.0:
            signal = 'LTH_SLIGHT_PROFIT'
            description = 'LTH SOPR > 1.0: 장기 보유자들이 약간의 이익 상태입니다.'
        elif value < 0.8:
            signal = 'LTH_CAPITULATION'
            description = 'LTH SOPR < 0.8: 장기 보유자들이 상당한 손실을 감수하고 있습니다. 투매 국면일 수 있습니다.'
        elif value < 1.0:
            signal = 'LTH_SLIGHT_LOSS'
            description = 'LTH SOPR < 1.0: 장기 보유자들이 약간의 손실 상태입니다. 거의 발생하지 않는 현상입니다.'
        else:
            signal = 'LTH_NEUTRAL'
            description = 'LTH SOPR = 1.0: 장기 보유자들은 손익분기점 상태입니다.'

        # 추가 분석: 시장 사이클 감지
        if value > 3.0 and value < 5.0:
            description += " 이는 시장 사이클 중반~후반부를 나타낼 수 있습니다."
        elif value > 5.0:
            description += " 이는 시장 사이클 정점 부근을 나타낼 수 있습니다."

        return {
            'value': value,
            'signal': signal,
            'description': description
        }


class LTHMVRVIndicator(OnchainIndicatorBase):
    """
    장기 보유자 MVRV (Long-Term Holder MVRV) 지표
    """

    def __init__(self):
        super().__init__(
            name="LTH_MVRV",
            description="Long-Term Holder MVRV - 장기 보유자의 시장가치 대비 실현가치 비율",
            endpoint="lth-mvrv/last"
        )

    def _extract_indicator_value(self, data: Dict[str, Any]) -> float:
        try:
            return float(data.get("lthMvrv", 0))
        except (ValueError, TypeError):
            return float('nan')

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        value = row.get(self.column_name)

        if pd.isna(value):
            return {
                'value': None,
                'signal': 'UNKNOWN',
                'description': 'LTH MVRV 데이터를 가져올 수 없습니다.'
            }

        if value > 3.0:
            signal = 'LTH_EXTREME_PROFIT'
            description = 'LTH MVRV > 3.0: 장기 보유자들이 상당한 이익 상태입니다. 역사적으로 시장 사이클 상단 근처입니다.'
        elif value > 2.0:
            signal = 'LTH_OVERVALUED'
            description = 'LTH MVRV > 2.0: 장기 보유자들이 과대평가된 상태입니다. 이익 실현 발생 가능성이 높습니다.'
        elif value > 1.5:
            signal = 'LTH_PROFIT'
            description = 'LTH MVRV > 1.5: 장기 보유자들이 이익 상태입니다. 완만한 상승세를 나타냅니다.'
        elif value > 1.0:
            signal = 'LTH_SLIGHT_PROFIT'
            description = 'LTH MVRV > 1.0: 장기 보유자들이 약간의 이익 상태입니다.'
        elif value < 0.75:
            signal = 'LTH_STRONG_ACCUMULATION'
            description = 'LTH MVRV < 0.75: 장기 보유자들이 실현가치 대비 낮은 가격에 있습니다. 유리한 매수 기회입니다.'
        elif value < 1.0:
            signal = 'LTH_ACCUMULATION'
            description = 'LTH MVRV < 1.0: 장기 보유자들이 손실 상태입니다. 비교적 드문 상황으로, 장기 투자자에게 기회일 수 있습니다.'
        else:
            signal = 'LTH_NEUTRAL'
            description = 'LTH MVRV = 1.0: 장기 보유자들은 손익분기점 상태입니다.'

        # 추가 분석: 시장 사이클 위치
        cycle_position = ""
        if value > 2.5:
            cycle_position = "시장 사이클 후반부"
        elif value > 1.5:
            cycle_position = "시장 사이클 중반부"
        elif value < 1.0:
            cycle_position = "시장 사이클 초반부"
        else:
            cycle_position = "시장 사이클 초중반부"

        return {
            'value': value,
            'signal': signal,
            'cycle_position': cycle_position,
            'description': description
        }


class MVRVIndicator(OnchainIndicatorBase):
    """
    MVRV (Market Value to Realized Value) 지표
    """

    def __init__(self):
        super().__init__(
            name="MVRV",
            description="Market Value to Realized Value - 시장 가치와 실현 가치의 비율",
            endpoint="mvrv/last"
        )

    def _extract_indicator_value(self, data: Dict[str, Any]) -> float:
        try:
            return float(data.get("mvrv", 0))
        except (ValueError, TypeError):
            return float('nan')

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        value = row.get(self.column_name)

        if pd.isna(value):
            return {
                'value': None,
                'signal': 'UNKNOWN',
                'market_phase': 'UNKNOWN',
                'description': 'MVRV 데이터를 가져올 수 없습니다.'
            }

        if value > 3.5:
            signal = 'EXTREME_OVERVALUED'
            market_phase = 'BUBBLE_TOP'
            description = 'MVRV > 3.5: 시장이 극도로 과대평가되어 있습니다. 역사적으로 시장 상단에 근접한 상태입니다.'
        elif value > 2.5:
            signal = 'OVERVALUED'
            market_phase = 'EUPHORIA'
            description = 'MVRV > 2.5: 시장이 과대평가되어 있습니다. 고위험 영역입니다.'
        elif value > 1.5:
            signal = 'SLIGHTLY_OVERVALUED'
            market_phase = 'OPTIMISM'
            description = 'MVRV > 1.5: 시장이 약간 과대평가되어 있습니다. 단기 조정 가능성이 있습니다.'
        elif value < 0.8:
            signal = 'OPPORTUNITY_TO_BUY'
            market_phase = 'CAPITULATION'
            description = 'MVRV < 0.8: 시장이 상당히 저평가되어 있습니다. 역사적으로 좋은 매수 기회입니다.'
        elif value < 1.0:
            signal = 'UNDERVALUED'
            market_phase = 'FEAR'
            description = 'MVRV < 1.0: 시장이 저평가되어 있습니다. 장기 투자에 유리한 상태입니다.'
        else:
            signal = 'NEUTRAL'
            market_phase = 'NEUTRAL'
            description = 'MVRV 1.0-1.5: 시장이 공정 가치에 가깝습니다.'

        # 추가 정보: 시장 사이클 위치 추정
        if value > 3.0:
            cycle_position = "Top 10% (후기 사이클)"
        elif value > 2.0:
            cycle_position = "Top 25% (중후기 사이클)"
        elif value < 1.0:
            cycle_position = "Bottom 25% (초기 사이클)"
        else:
            cycle_position = "Mid-range (중기 사이클)"

        return {
            'value': value,
            'signal': signal,
            'market_phase': market_phase,
            'cycle_position': cycle_position,
            'description': description
        }


class MVRVZScoreIndicator(OnchainIndicatorBase):
    """
    MVRV Z-Score 지표
    """

    def __init__(self):
        super().__init__(
            name="MVRV_Z_Score",
            description="MVRV Z-Score - 표준편차로 정규화된 MVRV 지표",
            endpoint="mvrv-zscore/last"
        )

    def _extract_indicator_value(self, data: Dict[str, Any]) -> float:
        try:
            return float(data.get("mvrvZscore", 0))
        except (ValueError, TypeError):
            return float('nan')

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        value = row.get(self.column_name)

        if pd.isna(value):
            return {
                'value': None,
                'signal': 'UNKNOWN',
                'market_phase': 'UNKNOWN',
                'description': 'MVRV Z-Score 데이터를 가져올 수 없습니다.'
            }

        if value > 7:
            signal = 'EXTREME_OVERVALUED'
            market_phase = 'BUBBLE_TOP'
            description = 'Z-Score > 7: 극도로 과대평가된 상태. 비트코인 시장 사이클의 상단에 접근 중입니다.'
        elif value > 3:
            signal = 'OVERVALUED'
            market_phase = 'EUPHORIA'
            description = 'Z-Score > 3: 과대평가된 상태. 시장이 과열되었습니다. 고위험 영역입니다.'
        elif value > 1:
            signal = 'SLIGHTLY_OVERVALUED'
            market_phase = 'OPTIMISM'
            description = 'Z-Score > 1: 약간 과대평가된 상태. 단기 조정 가능성이 있습니다.'
        elif value < -0.5:
            signal = 'UNDERVALUED'
            market_phase = 'FEAR'
            description = 'Z-Score < -0.5: 저평가된 상태. 장기 투자에 유리합니다.'
        elif value < -1:
            signal = 'OPPORTUNITY_TO_BUY'
            market_phase = 'CAPITULATION'
            description = 'Z-Score < -1: 상당히 저평가된 상태. 역사적으로 좋은 매수 기회입니다.'
        else:
            signal = 'NEUTRAL'
            market_phase = 'NEUTRAL'
            description = 'Z-Score -0.5 ~ 1: 공정 가치에 가깝습니다.'

        return {
            'value': value,
            'signal': signal,
            'market_phase': market_phase,
            'description': description
        }


class NUPLIndicator(OnchainIndicatorBase):
    """
    NUPL (Net Unrealized Profit/Loss) 지표
    """

    def __init__(self):
        super().__init__(
            name="NUPL",
            description="Net Unrealized Profit/Loss - 미실현 수익/손실 지표",
            endpoint="nupl/last"
        )

    def _extract_indicator_value(self, data: Dict[str, Any]) -> float:
        try:
            return float(data.get("nupl", 0))
        except (ValueError, TypeError):
            return float('nan')

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        value = row.get(self.column_name)

        if pd.isna(value):
            return {
                'value': None,
                'signal': 'UNKNOWN',
                'market_phase': 'UNKNOWN',
                'description': 'NUPL 데이터를 가져올 수 없습니다.'
            }

        if value > 0.75:
            signal = 'EXTREME_GREED'
            market_phase = 'EUPHORIA'
            description = 'NUPL > 0.75: 대부분의 시장 참여자들이 큰 이익을 보고 있습니다. 역사적으로 시장 상단에 가까운 상태입니다.'
        elif value > 0.5:
            signal = 'GREED'
            market_phase = 'BELIEF'
            description = 'NUPL > 0.5: 시장에 상당한 미실현 이익이 있습니다. 투자자들의 이익 실현 가능성이 있습니다.'
        elif value > 0.25:
            signal = 'OPTIMISM'
            market_phase = 'OPTIMISM'
            description = 'NUPL > 0.25: 시장 참여자들의 낙관적인 심리가 반영된 상태입니다.'
        elif value < 0:
            signal = 'FEAR'
            market_phase = 'CAPITULATION'
            description = 'NUPL < 0: 시장 전체가 손실 상태입니다. 역사적으로 좋은 매수 기회일 수 있습니다.'
        elif value < 0.25:
            signal = 'HOPE_ANXIETY'
            market_phase = 'HOPE_ANXIETY'
            description = 'NUPL < 0.25: 시장이 불확실성을 경험하고 있습니다. 방향성 전환의 가능성이 있습니다.'
        else:
            signal = 'NEUTRAL'
            market_phase = 'NEUTRAL'
            description = 'NUPL 0.25-0.5: 비교적 중립적인 상태입니다.'

        return {
            'value': value,
            'signal': signal,
            'market_phase': market_phase,
            'description': description
        }

class ETFFlowIndicator(OnchainIndicatorBase):
    """
    ETF Flow 지표
    """

    def __init__(self):
        super().__init__(
            name="ETF_Flow",
            description="Bitcoin ETF Flow - ETF 현금 유입/유출 지표",
            endpoint="etf-flow-btc/last"
        )

    def _extract_indicator_value(self, data: Dict[str, Any]) -> float:
        try:
            return float(data.get("etfFlow", 0))
        except (ValueError, TypeError):
            return float('nan')

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        value = row.get(self.column_name)

        if pd.isna(value):
            return {
                'value': None,
                'signal': 'UNKNOWN',
                'description': 'ETF Flow 데이터를 가져올 수 없습니다.'
            }

        # 단위가 BTC인지 또는 달러인지에 따라 임계값을 조정해야 할 수 있습니다
        if value > 1000:
            signal = 'STRONG_INFLOW'
            description = f'ETF Flow > 1000: 상당한 현금 유입이 발생하고 있습니다. 매수 압력이 증가하고 있을 수 있습니다.'
        elif value > 100:
            signal = 'INFLOW'
            description = f'ETF Flow > 100: 현금 유입이 발생하고 있습니다.'
        elif value < -1000:
            signal = 'STRONG_OUTFLOW'
            description = f'ETF Flow < -1000: 상당한 현금 유출이 발생하고 있습니다. 매도 압력이 증가하고 있을 수 있습니다.'
        elif value < -100:
            signal = 'OUTFLOW'
            description = f'ETF Flow < -100: 현금 유출이 발생하고 있습니다.'
        else:
            signal = 'NEUTRAL'
            description = f'ETF Flow {value}: 큰 현금 흐름이 없는 중립적인 상태입니다.'

        return {
            'value': value,
            'signal': signal,
            'description': description
        }

class RealizedPriceIndicator(OnchainIndicatorBase):
    """
    Realized Price 지표
    """

    def __init__(self):
        super().__init__(
            name="Realized_Price",
            description="Realized Price - 비트코인 실현 가격",
            endpoint="realized-price/last"
        )

    def _extract_indicator_value(self, data: Dict[str, Any]) -> float:
        try:
            return float(data.get("realizedPrice", 0))
        except (ValueError, TypeError):
            return float('nan')

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        realized_price = row.get(self.column_name)

        if pd.isna(realized_price):
            return {
                'value': None,
                'signal': 'UNKNOWN',
                'description': 'Realized Price 데이터를 가져올 수 없습니다.'
            }

        # 현재 시장 가격과 실현 가격의 비교가 필요합니다
        # 현재 가격을 얻기 위해 입력 데이터에 의존할 수 있습니다
        # 이것은 예시이므로 현재 가격이 있다고 가정합니다
        current_price = row.get('Close', 0)  # 데이터프레임에 Close 열이 있다고 가정

        if current_price == 0 or pd.isna(current_price):
            ratio = float('nan')
            signal = 'UNKNOWN'
            description = '현재 가격 데이터를 사용할 수 없어 상대적 위치를 판단할 수 없습니다.'
        else:
            ratio = current_price / realized_price

            if ratio > 1.5:
                signal = 'SIGNIFICANTLY_ABOVE'
                description = f'현재 가격이 실현 가격보다 {(ratio - 1) * 100:.1f}% 높습니다. 시장이 과열 상태일 수 있습니다.'
            elif ratio > 1.1:
                signal = 'ABOVE'
                description = f'현재 가격이 실현 가격보다 {(ratio - 1) * 100:.1f}% 높습니다. 시장이 상승세입니다.'
            elif ratio > 0.95:
                signal = 'AROUND'
                description = f'현재 가격이 실현 가격 근처에 있습니다. 상대적으로 중립적인 상태입니다.'
            elif ratio > 0.8:
                signal = 'BELOW'
                description = f'현재 가격이 실현 가격보다 {(1 - ratio) * 100:.1f}% 낮습니다. 잠재적 매수 기회일 수 있습니다.'
            else:
                signal = 'SIGNIFICANTLY_BELOW'
                description = f'현재 가격이 실현 가격보다 {(1 - ratio) * 100:.1f}% 낮습니다. 역사적으로 강력한 매수 신호입니다.'

        return {
            'value': realized_price,
            'price_to_realized_ratio': ratio if not pd.isna(ratio) else None,
            'signal': signal,
            'description': description
        }

class STHSOPRIndicator(OnchainIndicatorBase):
    """
    단기 보유자 SOPR (Short-Term Holder SOPR) 지표
    """

    def __init__(self):
        super().__init__(
            name="STH_SOPR",
            description="Short-Term Holder SOPR - 단기 보유자의 지출 출력 수익 비율",
            endpoint="sth-sopr/last"
        )

    def _extract_indicator_value(self, data: Dict[str, Any]) -> float:
        try:
            return float(data.get("sthSopr", 0))
        except (ValueError, TypeError):
            return float('nan')

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        value = row.get(self.column_name)

        if pd.isna(value):
            return {
                'value': None,
                'signal': 'UNKNOWN',
                'description': 'STH SOPR 데이터를 가져올 수 없습니다.'
            }

        if value > 1.05:
            signal = 'STH_PROFIT_TAKING'
            description = 'STH SOPR > 1.05: 단기 보유자들이 이익을 실현하고 있습니다. 단기적 하락 압력이 있을 수 있습니다.'
        elif value > 1.0:
            signal = 'STH_SLIGHT_PROFIT'
            description = 'STH SOPR > 1.0: 단기 보유자들이 소폭 이익 상태입니다.'
        elif value < 0.95:
            signal = 'STH_LOSS_SELLING'
            description = 'STH SOPR < 0.95: 단기 보유자들이 손실을 감수하고 매도하고 있습니다. 투기적 포지션의 청산이 진행 중일 수 있습니다.'
        elif value < 1.0:
            signal = 'STH_SLIGHT_LOSS'
            description = 'STH SOPR < 1.0: 단기 보유자들이 소폭 손실 상태입니다.'
        else:
            signal = 'STH_NEUTRAL'
            description = 'STH SOPR = 1.0: 단기 보유자들은 손익분기점 상태입니다.'

        # 추가 분석: 시장 전환점 감지
        signal_suffix = ""
        if value < 1.0 and value > 0.95:
            signal_suffix = "_POTENTIAL_BOTTOM"
            description += " 이 레벨에서 1.0 위로 상승하면 강세장 초입을 나타낼 수 있습니다."
        elif value > 1.0 and value < 1.05:
            signal_suffix = "_MOMENTUM_CHECK"
            description += " 1.0 아래로 하락하면 상승 모멘텀이 약화될 수 있습니다."

        return {
            'value': value,
            'signal': signal + signal_suffix if signal_suffix else signal,
            'description': description
        }

class STHMVRVIndicator(OnchainIndicatorBase):
    """
    단기 보유자 MVRV (Short-Term Holder MVRV) 지표
    """

    def __init__(self):
        super().__init__(
            name="STH_MVRV",
            description="Short-Term Holder MVRV - 단기 보유자의 시장가치 대비 실현가치 비율",
            endpoint="sth-mvrv/last"
        )

    def _extract_indicator_value(self, data: Dict[str, Any]) -> float:
        try:
            return float(data.get("sthMvrv", 0))
        except (ValueError, TypeError):
            return float('nan')

    def interpret(self, row: pd.Series) -> Dict[str, Any]:
        value = row.get(self.column_name)

        if pd.isna(value):
            return {
                'value': None,
                'signal': 'UNKNOWN',
                'description': 'STH MVRV 데이터를 가져올 수 없습니다.'
            }

        if value > 2.0:
            signal = 'STH_EXTREME_PROFIT'
            description = 'STH MVRV > 2.0: 단기 보유자들이 상당한 이익 상태입니다. 과열된 시장 상태일 수 있습니다.'
        elif value > 1.5:
            signal = 'STH_OVERVALUED'
            description = 'STH MVRV > 1.5: 단기 보유자들이 과대평가된 상태입니다. 이익 실현 압력이 있을 수 있습니다.'
        elif value > 1.0:
            signal = 'STH_PROFIT'
            description = 'STH MVRV > 1.0: 단기 보유자들이 이익 상태입니다.'
        elif value < 0.8:
            signal = 'STH_UNDERVALUED'
            description = 'STH MVRV < 0.8: 단기 보유자들이 상당한 손실 상태입니다. 매수 기회일 수 있습니다.'
        elif value < 1.0:
            signal = 'STH_LOSS'
            description = 'STH MVRV < 1.0: 단기 보유자들이 손실 상태입니다.'
        else:
            signal = 'STH_NEUTRAL'
            description = 'STH MVRV = 1.0: 단기 보유자들은 손익분기점 상태입니다.'

        return {
            'value': value,
            'signal': signal,
            'description': description
        }