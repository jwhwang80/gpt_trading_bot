import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any


# 추가: JSON 직렬화를 위한 사용자 정의 인코더
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)


class LogSaver:
    @staticmethod
    def save_log_with_result(
            input_data: Dict[str, Any],
            prompt: str,
            gpt_response: Dict[str, Any],
            trade_result: Dict[str, Any],
            reflection: Dict[str, Any]
    ) -> str:
        """
        각 트레이딩 판단의 전체 로그를 저장합니다.

        Args:
            input_data (Dict): 초기 입력 데이터
            prompt (str): GPT에 전달한 프롬프트
            gpt_response (Dict): GPT의 응답
            trade_result (Dict): 트레이드 결과
            reflection (Dict): GPT의 복기 결과

        Returns:
            str: 생성된 로그 파일 경로
        """
        # 로그 디렉토리 생성 (YYYY-MM-DD_HH-MM 형식)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        log_dir = os.path.join('logs', timestamp)
        os.makedirs(log_dir, exist_ok=True)

        # 로그 데이터 통합
        log_data = {
            "timestamp": timestamp,
            "input_data": input_data,
            "prompt": prompt,
            "gpt_response": gpt_response,
            "trade_result": trade_result,
            "reflection": reflection
        }

        # JSON 로그 파일 저장 - 수정: CustomJSONEncoder 사용
        log_filename = os.path.join(log_dir, 'trade_log.json')
        with open(log_filename, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4, cls=CustomJSONEncoder)

        # 텍스트 로그 파일 생성 (읽기 쉬운 형식)
        txt_filename = os.path.join(log_dir, 'trade_log.txt')
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(f"타임스탬프: {timestamp}\n\n")
            f.write("입력 데이터:\n")
            f.write(json.dumps(input_data, ensure_ascii=False, indent=2, cls=CustomJSONEncoder) + "\n\n")
            f.write("GPT 프롬프트:\n")
            f.write(prompt + "\n\n")
            f.write("GPT 응답:\n")
            f.write(json.dumps(gpt_response, ensure_ascii=False, indent=2, cls=CustomJSONEncoder) + "\n\n")
            f.write("트레이드 결과:\n")
            f.write(json.dumps(trade_result, ensure_ascii=False, indent=2, cls=CustomJSONEncoder) + "\n\n")
            f.write("복기 결과:\n")
            f.write(json.dumps(reflection, ensure_ascii=False, indent=2, cls=CustomJSONEncoder))

        return log_filename


# 사용 예시
if __name__ == "__main__":
    sample_input = {"price": 50000, "rsi": 62}
    sample_prompt = "트레이딩 전략 제안"
    sample_gpt_response = {"action": "BUY", "confidence": 75}
    sample_trade_result = {"entry_price": 50100, "exit_price": 51200}
    sample_reflection = {"overall_assessment": "성공"}

    log_file = LogSaver.save_log_with_result(
        sample_input, sample_prompt,
        sample_gpt_response, sample_trade_result,
        sample_reflection
    )
    print(f"로그 파일 생성: {log_file}")