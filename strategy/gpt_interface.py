import os
import json
from typing import Dict, Any, Optional


class GPTInterface:
    def __init__(self, api_key: Optional[str] = None):
        """
        GPT API 인터페이스 초기화

        Args:
            api_key (Optional[str]): OpenAI API 키 (환경변수 또는 직접 입력)
        """
        # 최신 OpenAI 클라이언트 라이브러리 사용
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))

            if not (api_key or os.getenv('OPENAI_API_KEY')):
                raise ValueError("OpenAI API 키가 필요합니다.")
        except ImportError:
            raise ImportError("최신 OpenAI 라이브러리를 설치하세요: pip install openai")

    def call_gpt_with_prompt(
            self,
            prompt: str,
            model: str = "gpt-3.5-turbo",
            max_tokens: int = 500
    ) -> str:
        """
        주어진 프롬프트로 GPT API 호출

        Args:
            prompt (str): GPT에 전달할 프롬프트
            model (str): 사용할 GPT 모델
            max_tokens (int): 최대 토큰 수

        Returns:
            str: GPT의 응답
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful trading strategy assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"GPT API 호출 중 오류 발생: {e}")
            return json.dumps({
                "error": str(e),
                "action": "HOLD",
                "confidence": 0
            })

    def validate_json_response(self, response: str) -> Dict[str, Any]:
        """
        GPT 응답의 JSON 유효성 검사

        Args:
            response (str): GPT 응답 문자열

        Returns:
            Dict: 검증된 JSON 객체
        """
        try:
            # JSON 부분만 추출하기 위한 간단한 파싱 로직 추가
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                return json.loads(json_str)
            return json.loads(response)
        except json.JSONDecodeError:
            print(f"GPT 응답을 JSON으로 파싱하는데 실패했습니다. 원본 응답: {response[:100]}...")
            return {
                "action": "HOLD",
                "confidence": 0,
                "reasoning": "JSON 파싱 실패"
            }