import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional


class TechnicalIndicatorCalculator:
    """
    다양한 기술적 지표를 계산하고 관리하는 계산기
    """
    def __init__(self, csv_file_path: Optional[str] = None, symbol: str = "BTCUSDT"):
        """
        기술적 지표 계산기 초기화
        
        Args:
            csv_file_path (Optional[str]): 1분봉 데이터가 저장된 CSV 파일 경로
            symbol (str): 암호화폐 심볼
        """
        self.csv_file_path = csv_file_path or f"{symbol}_1m_3000day_data.csv"
        self.symbol = symbol
        self.df_1m = None
        self.df_4h = None
        self.df_1d = None
        self.indicators = {}
        
        # 기본 지표 등록
        self.register_default_indicators()
    
    def register_indicator(self, indicator):
        """
        새로운 기술적 지표를 등록합니다.
        
        Args:
            indicator: 등록할 지표 객체 (BaseIndicator 클래스 상속)
        """
        self.indicators[indicator.name] = indicator
        print(f"지표 등록됨: {indicator.name} - {indicator.description}")

    def register_default_indicators(self) -> None:
        """모든 기술적 지표들을 자동으로 등록합니다."""
        # indicator_classes 모듈에서 모든 클래스 가져오기
        import inspect
        import indicator_classes

        # BaseIndicator를 상속받은 모든 클래스 찾기
        indicator_classes_list = []
        for name, obj in inspect.getmembers(indicator_classes):
            # 클래스이며, BaseIndicator의 서브클래스이고, BaseIndicator 자체가 아닌 경우
            if (inspect.isclass(obj) and
                    issubclass(obj, indicator_classes.BaseIndicator) and
                    obj is not indicator_classes.BaseIndicator):
                indicator_classes_list.append(obj)

        print(f"총 {len(indicator_classes_list)}개의 지표 클래스를 찾았습니다.")

        # 각 지표 클래스의 인스턴스 생성 및 등록
        for indicator_class in indicator_classes_list:
            try:
                # 클래스 이름 출력
                class_name = indicator_class.__name__
                print(f"등록 중: {class_name}")

                # 기본 파라미터로 인스턴스 생성
                indicator = indicator_class()

                # 인스턴스 등록
                self.register_indicator(indicator)

            except Exception as e:
                print(f"지표 {indicator_class.__name__} 등록 중 오류 발생: {e}")

        print(f"총 {len(self.indicators)}개의 지표가 성공적으로 등록되었습니다.")

    def load_data(self, days: int = 120) -> None:
        """
        1분봉 데이터를 로드하고 전처리합니다.
        최근 지정된 일수(days)의 데이터만 로드합니다.

        Args:
            days (int): 로드할 최근 일수 (기본값: 120일)
        """
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.csv_file_path}")

        # CSV 파일 로드
        print(f"1분봉 데이터 로드 중: {self.csv_file_path}")
        df = pd.read_csv(self.csv_file_path)

        # 'Time' 열을 datetime으로 변환
        df['Time'] = pd.to_datetime(df['Time'])

        # 시간 순으로 정렬 (오름차순)
        df = df.sort_values(by='Time', ascending=True)

        # 현재 시간 기준 최근 'days'일 데이터만 필터링
        end_date = df['Time'].max()
        start_date = end_date - pd.Timedelta(days=days)
        df = df[df['Time'] >= start_date]

        print(f"최근 {days}일 데이터 필터링: {start_date} ~ {end_date}")
        print(f"필터링된 데이터 포인트 수: {len(df)}")

        # 인덱스 설정
        df.set_index('Time', inplace=True)

        # 필터링된 데이터 저장
        self.df_1m = df

        print(f"1분봉 데이터 로드 완료: {len(self.df_1m)} 데이터 포인트")

        # 4시간봉과 1일봉으로 리샘플링
        self._resample_data()

    def _resample_data(self) -> None:
        """1분봉 데이터를 4시간봉과 1일봉으로 리샘플링합니다."""
        if self.df_1m is None:
            raise ValueError("데이터가 로드되지 않았습니다. 먼저 load_data()를 호출하세요.")
        
        # 4시간봉 리샘플링
        print("4시간봉으로 리샘플링 중...")
        self.df_4h = self.df_1m.resample('4H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        self.df_4h.dropna(inplace=True)
        print(f"4시간봉 리샘플링 완료: {len(self.df_4h)} 캔들")
        
        # 1일봉 리샘플링
        print("1일봉으로 리샘플링 중...")
        self.df_1d = self.df_1m.resample('1D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        self.df_1d.dropna(inplace=True)
        print(f"1일봉 리샘플링 완료: {len(self.df_1d)} 캔들")
    
    def calculate_indicator(self, df: pd.DataFrame, indicator_name: str) -> pd.DataFrame:
        """
        특정 지표를 계산합니다.
        
        Args:
            df (pd.DataFrame): OHLCV 데이터가 포함된 데이터프레임
            indicator_name (str): 계산할 지표 이름
            
        Returns:
            pd.DataFrame: 계산된 지표가 추가된 데이터프레임
        """
        if indicator_name not in self.indicators:
            raise ValueError(f"등록되지 않은 지표입니다: {indicator_name}")
        
        return self.indicators[indicator_name].calculate(df)
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 등록된 지표를 계산합니다.
        
        Args:
            df (pd.DataFrame): OHLCV 데이터가 포함된 데이터프레임
            
        Returns:
            pd.DataFrame: 모든 지표가 추가된 데이터프레임
        """
        result_df = df.copy()
        
        for indicator_name, indicator in self.indicators.items():
            try:
                print(f"{indicator_name} 계산 중...")
                result_df = indicator.calculate(result_df)
            except Exception as e:
                print(f"{indicator_name} 계산 중 오류 발생: {e}")
        
        return result_df
    
    def interpret_indicators(self, row: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        각 지표의 현재 값을 해석합니다.
        
        Args:
            row (pd.Series): 지표 값을 포함하는 데이터 시리즈
            
        Returns:
            Dict[str, Dict[str, Any]]: 각 지표별 해석 결과
        """
        interpretations = {}
        
        for indicator_name, indicator in self.indicators.items():
            try:
                interpretations[indicator_name] = indicator.interpret(row)
            except Exception as e:
                print(f"{indicator_name} 해석 중 오류 발생: {e}")
                interpretations[indicator_name] = {'error': str(e)}
        
        return interpretations
    
    def calculate_technical_indicators(self, timeframe: str = '4H', periods: int = 120) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        특정 시간프레임에 대한 기술적 지표를 계산합니다.
        
        Args:
            timeframe (str): 시간프레임 ('4H' 또는 '1D')
            periods (int): 사용할 최근 캔들 수
        
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: 계산된 지표ㅡ가 포함된 데이터프레임과 해석 결과
        """
        if timeframe == '4H':
            df = self.df_4h.copy()
        elif timeframe == '1D':
            df = self.df_1d.copy()
        else:
            raise ValueError("지원되지 않는 시간프레임입니다. '4H' 또는 '1D'만 가능합니다.")
        
        # 최근 캔들만 사용
        df = df.iloc[-periods:]
        
        if len(df) < periods:
            print(f"경고: 요청된 {periods} 캔들보다 적은 {len(df)} 캔들만 사용 가능합니다.")
        
        print(f"{timeframe} 시간프레임에 대한 기술적 지표 계산 중 (캔들 수: {len(df)})...")
        
        # 모든 지표 계산
        df_with_indicators = self.calculate_all_indicators(df)
        
        # 가장 최근 데이터에 대한 해석
        latest_row = df_with_indicators.iloc[-1]
        interpretations = self.interpret_indicators(latest_row)
        
        # 결과 형식화
        result = {
            'timeframe': timeframe,
            'timestamp': df_with_indicators.index[-1],
            'close_price': latest_row['Close'],
            'indicators': interpretations
        }
        
        print(f"{timeframe} 시간프레임에 대한 기술적 지표 계산 완료")
        
        return df_with_indicators, result
    
    def calculate_all_timeframes(self) -> Dict[str, Dict[str, Any]]:
        """4시간봉과 1일봉에 대한 모든 기술적 지표를 계산합니다."""
        if self.df_1m is None:
            self.load_data()
        
        # 4시간봉 지표 계산
        df_4h, interpretation_4h = self.calculate_technical_indicators('4H', 120)
        
        # 1일봉 지표 계산
        df_1d, interpretation_1d = self.calculate_technical_indicators('1D', 120)
        
        return {
            '4H': {
                'data': df_4h,
                'interpretation': interpretation_4h
            },
            '1D': {
                'data': df_1d,
                'interpretation': interpretation_1d
            }
        }
    
    def save_indicators_to_csv(self, results: Dict[str, Dict[str, Any]], output_dir: str = 'output') -> None:
        """
        계산된 지표를 CSV 파일로 저장합니다.
        
        Args:
            results (dict): calculate_all_timeframes()의 결과
            output_dir (str): 출력 디렉토리
        """
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 현재 시간을 파일명에 포함
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 각 시간프레임 데이터 저장
        for timeframe in results:
            df = results[timeframe]['data']
            filename = f"{output_dir}/{self.symbol}_{timeframe}_{timestamp}.csv"
            df.to_csv(filename)
            print(f"{timeframe} 지표를 저장했습니다: {filename}")
        
        # 해석 결과 저장
        with open(f"{output_dir}/interpretation_{timestamp}.txt", 'w') as f:
            f.write(f"기술적 지표 해석 ({timestamp})\n\n")
            
            for timeframe in results:
                f.write(f"==== {timeframe} 시간프레임 ====\n")
                interp = results[timeframe]['interpretation']
                f.write(f"시간: {interp['timestamp']}\n")
                f.write(f"종가: {interp['close_price']}\n\n")
                
                for indicator_name, indicator_data in interp['indicators'].items():
                    f.write(f"{indicator_name}:\n")
                    for key, value in indicator_data.items():
                        if isinstance(value, (float, np.float64)):
                            f.write(f"  {key}: {value:.2f}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                    f.write("\n")
                
                f.write("\n\n")
        
        print(f"해석 결과를 저장했습니다: {output_dir}/interpretation_{timestamp}.txt")
    
    def get_registered_indicators(self) -> Dict[str, Dict[str, str]]:
        """
        현재 등록된 모든 지표의 정보를 반환합니다.
        
        Returns:
            Dict[str, Dict[str, str]]: 지표 이름과 설명
        """
        return {name: indicator.get_info() for name, indicator in self.indicators.items()}


# 독립적인 모듈로 사용될 때의 예시
if __name__ == "__main__":
    # 계산기 초기화
    calculator = TechnicalIndicatorCalculator('../BTCUSDT_1m_3000day_data.csv')
    
    # 현재 등록된 지표 확인
    print("등록된 지표:")
    for name, info in calculator.get_registered_indicators().items():
        print(f"- {name}: {info['description']}")
    
    # 데이터 로드
    calculator.load_data()
    
    # 모든 시간프레임에 대한 지표 계산
    results = calculator.calculate_all_timeframes()
    print(results)
    # 결과 저장
    calculator.save_indicators_to_csv(results)
