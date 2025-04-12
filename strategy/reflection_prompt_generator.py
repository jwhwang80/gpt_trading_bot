import json
from typing import Dict, Any, List
from datetime import datetime


class ReflectionPromptGenerator:
    @staticmethod
    def build_reflection_prompt(
            gpt_strategy: Dict[str, Any],
            entry_price: float = 0,
            exit_price: float = 0,
            profit_percentage: float = 0
    ) -> str:
        """
        GPT에게 전략에 대한 복기를 요청하는 프롬프트를 생성합니다.

        Args:
            gpt_strategy (Dict): 원래 GPT가 제안한 전략
            entry_price (float): 진입 가격
            exit_price (float): 청산 가격
            profit_percentage (float): 수익률

        Returns:
            str: 복기를 위한 프롬프트
        """
        # 간소화된 전략 정보 준비 (너무 복잡하지 않게)
        strategy_info = {
            "action": gpt_strategy.get("action", "UNKNOWN"),
            "confidence": gpt_strategy.get("confidence", 0),
            "reasoning": gpt_strategy.get("reasoning", "설명 없음"),
            "stop_loss": gpt_strategy.get("stop_loss", 0),
            "take_profit": gpt_strategy.get("take_profit", 0),
        }

        prompt_template = f"""
최근 실행한 트레이딩 전략에 대한 심층 복기를 수행해주세요. 이 복기는 향후 전략 개선에 중요합니다.

원래 전략:
{json.dumps(strategy_info, indent=2, ensure_ascii=False)}

거래 결과:
- 진입 가격: {entry_price}
- 청산 가격: {exit_price}
- 수익률: {profit_percentage:.2f}%
- 거래 타입: {"롱" if gpt_strategy.get("action") == "BUY" else "숏" if gpt_strategy.get("action") == "SELL" else "홀딩"}

분석 관점:
1. 전략의 성공/실패 요인 상세 분석
2. 결과와 초기 전략 사이의 차이점 식별
3. 향후 유사한 시장 상황에서 개선할 점 제안
4. 리스크 관리 측면에서 개선점 제안
5. JSON 형식으로 응답 (아래 구조 준수)

응답 JSON 구조:
{{
    "overall_assessment": "성공/부분성공/실패",
    "key_insights": ["인사이트 1", "인사이트 2", "인사이트 3"],
    "improvement_recommendations": ["추천 1", "추천 2", "추천 3"],
    "confidence_in_original_strategy": 0-100,
    "risk_management_score": 0-100,
    "future_considerations": "향후 이러한 전략 실행시 고려할 점"
}}

반드시 JSON 형식으로 응답해 주세요.
"""

        return prompt_template

    @staticmethod
    def parse_reflection_response(reflection_response: str) -> Dict[str, Any]:
        """
        GPT의 복기 응답을 파싱합니다.

        Args:
            reflection_response (str): GPT의 복기 응답 문자열

        Returns:
            Dict: 파싱된 복기 결과
        """
        try:
            # 응답이 잘린 경우 처리 (추가)
            if '{' in reflection_response and '}' not in reflection_response:
                print("응답이 잘렸습니다. 부분 응답을 복구합니다.")
                # 부분 응답에 대한 기본 구조 반환
                return {
                    "overall_assessment": "부분 응답",
                    "key_insights": ["응답이 잘려서 완전히 파싱할 수 없습니다."],
                    "improvement_recommendations": ["전체 응답을 받기 위해 다시 시도하세요."],
                    "confidence_in_original_strategy": 50,
                    "reflection_date": datetime.now().isoformat()
                }

            # JSON 부분만 추출하기 위한 간단한 파싱 로직
            if '{' in reflection_response and '}' in reflection_response:
                start = reflection_response.find('{')
                end = reflection_response.rfind('}') + 1
                json_str = reflection_response[start:end]
                parsed_data = json.loads(json_str)
            else:
                parsed_data = json.loads(reflection_response)

            # 필수 필드 확인
            required_fields = [
                "overall_assessment",
                "key_insights",
                "improvement_recommendations",
                "confidence_in_original_strategy"
            ]

            for field in required_fields:
                if field not in parsed_data:
                    if field == "overall_assessment":
                        parsed_data[field] = "평가 실패"
                    elif field in ["key_insights", "improvement_recommendations"]:
                        parsed_data[field] = ["파싱 오류"]
                    else:
                        parsed_data[field] = 0

            # 날짜 추가
            parsed_data["reflection_date"] = datetime.now().isoformat()

            return parsed_data

        except json.JSONDecodeError as e:
            print(f"복기 응답 JSON 파싱 실패: {e}. 원본 응답: {reflection_response[:100]}...")
            return {
                "overall_assessment": "분석 실패",
                "key_insights": ["JSON 파싱 오류"],
                "improvement_recommendations": [],
                "confidence_in_original_strategy": 0,
                "reflection_date": datetime.now().isoformat()
            }