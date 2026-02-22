import os
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from langchain_core.prompts import ChatPromptTemplate


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


def build_extract_info_chain(llm: Any):
    """
    회의에서 논의사항, 결정사항, 액션아이템 추출 chain 생성 함수
    """

    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", MEETING_SUMMARIZATION_PROMPT),
        ("human", MEETING_SUMMARIZATION_HUMAN_PROMPT),
    ])

    chain = extract_prompt | llm.with_structured_output(MeetingSummarizationOutput)
    return chain


def build_normalize_info_chain(llm: Any):
    """
    여러 chunk 에서 추출된 정보들을 종합하고 중복을 제거하여 정리하는 chain 생성 함수
    """
    normalize_prompt = ChatPromptTemplate.from_messages([
        ("system", MEETING_SUMMARY_NORMALIZE_PROMPT),
    ])

    chain = normalize_prompt | llm.with_structured_output(MeetingSummarizationOutput)
    return chain
