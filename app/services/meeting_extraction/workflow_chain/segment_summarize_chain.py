import os
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from app.config.config import settings


# 단락 요약 스키마
class SegmentSummaryOutput(BaseModel):
    title: str
    bullets: List[str]


SEGMENT_SUMMARY_SYSTEM_PROMPT = """
당신은 회의록 일부(세그먼트)를 보고, 시간 구간 단락 요약을 만듭니다.

규칙:
- 새 정보를 만들지 마세요. 입력에 있는 내용만 요약하세요.
- title은 이 세그먼트의 핵심 주제를 한 문장으로 요약하세요, 
- bullets는 주요 논의/결정/액션아이템 위주로 정리해 주세요.
"""

SEGMENT_SUMMARY_HUMAN_PROMPT = """
segment_index: {segment_index}
transcript:
{transcript}
출력은 반드시 JSON 형식의 구조화 결과를 출력하라
"""


def build_segment_chain(llm: Any):
    """
    시간대별 단락 요약 chain 생성 함수
    """

    segment_prompt = ChatPromptTemplate.from_messages([
        ("system", SEGMENT_SUMMARY_SYSTEM_PROMPT),
        ("human", SEGMENT_SUMMARY_HUMAN_PROMPT),
    ])

    chain = segment_prompt | llm.with_structured_output(SegmentSummaryOutput)
    return chain


