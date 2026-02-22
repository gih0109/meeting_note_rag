import os
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from app.config.config import settings


# 추출 결과 스키마
class AgendaItem(BaseModel):
    title: str


class Decision(BaseModel):
    timescope: Literal["past", "present", "future"]
    title: str
    content: str
    tags: List[str]


class ActionItem(BaseModel):
    timescope: Literal["past", "present", "future"]
    objective: Optional[str] = None
    task: Optional[str] = None
    due_date: Optional[str] = None
    priority: Optional[str] = None
    background: Optional[str] = None


class KeyDiscussionPoint(BaseModel):
    timescope: Literal["past", "present", "future"]
    title: str
    discussions: List[str]


class MeetingSummarizationOutput(BaseModel):
    agenda: List[AgendaItem] = Field(default_factory=list)
    key_discussion_points: List[KeyDiscussionPoint] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)


# 오타 교정 스키마
class TranscriptLineModel(BaseModel):
    timestamp: str
    text: str

class CorrectedChunk(BaseModel):
    chunk_id: str
    lines: List[TranscriptLineModel]


# 단락 요약 스키마
class SegmentSummaryOutput(BaseModel):
    title: str
    bullets: List[str]


# 프롬프트
MEETING_SUMMARIZATION_PROMPT = """
당신은 회의록(전사 텍스트)에서 핵심 정보를 추출하여 구조화하는 AI 어시스턴트입니다.

아래 Transcript를 분석하여, 다음 4가지 정보를 출력하세요.
- agenda
- key_discussion_points
- decisions
- action_items

[항목별 추출 가이드]
- agenda: 회의에서 다룬 "주제/안건" 목록입니다. 문장 요약이 아니라 짧은 토픽(명사구)으로 작성하세요. (중복 제거)
- key_discussion_points: 의논/토론의 핵심 쟁점입니다. 단순 주제(agenda)보다 한 단계 구체적으로, 무엇이 문제였고 어떤 관점/대안이 논의되었는지 중심으로 작성하세요.
- decisions: 최종 결론/합의/승인처럼 “결정되었다”는 근거가 있는 경우만 포함하세요. 검토/제안/아이디어는 decision이 아닙니다.
- action_items: 누군가가 “하기로 한 후속 작업/해야 할 일”만 포함하세요. 단순 의견/희망/가능성은 action_item이 아닙니다.

[timescope 분류 기준]
timescope는 "Meeting Date(현재 회의 시점)"을 기준으로, 논의사항/결정사항/액션아이템을 세 가지 시간 구간으로 분류합니다.

- past: 과거에 결정된 결정사항들, 과거에 의논한 의논사항들, 과거 상황(현재 회의 이전에 이미 발생/결정/완료된 내용)
  예) "지난주에 A로 결정했다", "이전 회의에서 합의", "이미 배포 완료"

- present: 현재 결정된 결정사항들, 현재 상황에 따른 의논사항들, 현재 상황(현재 회의에서 논의/결정/확인된 내용 또는 현재 진행 중인 상태)
  예) "오늘 회의에서 A로 결정", "현재 진행 중", "지금 기준 현황 공유"

- future: 미래로 결정을 미룬 결정사항들, 미래에 의논하자고 약속한 의논사항들, 미래에 예측한 상황(현재 회의 이후에 하기로 한 계획/약속/보류/후속 조치)
  예) "다음 회의에서 다시 논의", "추후 결정", "이번 주 내 작업 예정", "기한(deadline)이 미래"


[출력 규칙]
- 출력은 반드시 스키마에 맞는 JSON만 제공하세요(설명 문장 금지).
- 모든 항목(agenda/key_discussion_points/decisions/action_items)의 각 아이템에는 timescope를 반드시 포함하세요: past/present/future
- timestamp/due_date 등은 Transcript에 등장하는 형식을 그대로 사용하세요.
- priority는 High/Medium/Low 중 하나로 채우되, 근거가 없으면 null로 두세요.
- Transcript에 없는 정보는 절대 추측하지 마세요.
"""

MEETING_SUMMARIZATION_HUMAN_PROMPT = """
Meeting Date: {meeting_date}
transcript:
{transcript}
출력은 반드시 JSON 형식의 구조화 결과를 출력하라
"""

MEETING_SUMMARY_NORMALIZE_PROMPT = """
You are a meeting summary normalizer.

Rules:
1) Do NOT invent new information.
2) Do NOT create new timestamps.
3) When merging, start_timestamp=min, end_timestamp=max among existing.
4) Deduplicate conservatively.

Return ONLY MeetingSummarizationResponse JSON.

meeting_date: {meeting_date}

INPUT_JSON:
{input_json}
"""

CHUNK_CORRECTION_PROMPT = """
당신은 회의 전사 텍스트의 오타/띄어쓰기/끊긴 문장만 최소한으로 교정합니다.

규칙:
1) timestamp 문자열은 절대 수정/추가/삭제하지 마세요. 순서도 바꾸지 마세요.
2) 각 timestamp에 매핑된 text만 교정하세요.
3) 의미를 바꾸거나 내용을 요약/확장/재구성하지 마세요.
4) 새 정보(인물/기능/결정 등)를 절대 추가하지 마세요.
5) 출력은 반드시 JSON이며, 입력과 동일한 timestamp 목록을 가져야 합니다.

입력:
chunk_id: {chunk_id}
lines(JSON):
{lines_json}
"""

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

def build_llm(model_name: str):
    if "gemini-2" in model_name:
        return ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL_NAME,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            api_key=settings.GOOGLE_API_KEY,
        )
    elif "gemini-3" in model_name:
        return ChatGoogleGenerativeAI(
            model_name=settings.LLM_MODEL_NAME,
            temperature=1.0,
            api_key=settings.GOOGLE_API_KEY,
        )

def build_segment_chain(model_name: str):
    """
    시간대별 단락 요약 chain 생성 함수
    """

    llm = build_llm(model_name=model_name)

    segment_prompt = ChatPromptTemplate.from_messages([
        ("system", SEGMENT_SUMMARY_SYSTEM_PROMPT),
        ("human", SEGMENT_SUMMARY_HUMAN_PROMPT),
    ])

    chain = segment_prompt | llm.with_structured_output(SegmentSummaryOutput)
    return chain

def build_extract_info_chain(model_name: str):
    """
    회의에서 논의사항, 결정사항, 액션아이템 추출 chain 생성 함수
    """
    llm = build_llm(model_name=model_name)

    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", MEETING_SUMMARIZATION_PROMPT),
        ("human", MEETING_SUMMARIZATION_HUMAN_PROMPT),
    ])

    chain = extract_prompt | llm.with_structured_output(MeetingSummarizationOutput)
    return chain

