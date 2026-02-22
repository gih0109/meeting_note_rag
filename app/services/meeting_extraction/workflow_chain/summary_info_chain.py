from typing import Any, List, Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class AgendaItem(BaseModel):
    title: str


class Decision(BaseModel):
    timescope: Literal["past", "present", "future"]
    title: str
    content: str
    tags: List[str] = Field(default_factory=list)


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
    discussions: List[str] = Field(default_factory=list)


class MeetingSummarizationOutput(BaseModel):
    agenda: List[AgendaItem] = Field(default_factory=list)
    key_discussion_points: List[KeyDiscussionPoint] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)
    action_items: List[ActionItem] = Field(default_factory=list)


MEETING_SUMMARIZATION_PROMPT = """
당신은 회의 대화 텍스트에서 핵심 정보를 추출해 구조화하는 AI 어시스턴트입니다.

아래 Transcript를 분석하여 다음 4개 항목을 추출하세요.
- agenda
- key_discussion_points
- decisions
- action_items

[항목별 추출 가이드]
- agenda: 회의에서 다룬 "주제/안건" 목록입니다. 문장 요약이 아니라 짧은 토픽(명사구)으로 작성하세요. 중복은 제거하세요.
- key_discussion_points: 의논/토론의 핵심 쟁점입니다. 단순 주제보다 한 단계 구체적으로, 무엇이 문제였고 어떤 관점/대안이 논의되었는지 중심으로 작성하세요.
- decisions: 최종 결론/합의/승인처럼 "결정되었다"는 근거가 있는 경우만 포함하세요. 검토/제안/아이디어는 decision이 아닙니다.
- action_items: 누군가가 하기로 한 후속 작업/해야 할 일만 포함하세요. 단순 의견/희망/가능성은 action_item이 아닙니다.

[timescope 분류 기준]
Meeting Date를 기준으로 key_discussion_points/decisions/action_items의 각 아이템에 timescope를 부여하세요.
- past: 현재 회의 이전에 발생/결정/완료된 내용
- present: 현재 회의 시점에서 논의/결정/확인 중인 내용
- future: 현재 회의 이후의 계획/후속조치/예정된 내용

[출력 규칙]
- 출력은 반드시 스키마에 맞는 JSON만 반환하세요.
- key_discussion_points/decisions/action_items의 각 아이템에는 timescope를 반드시 포함하세요.
- due_date는 transcript 근거가 있을 때만 채우고, 없으면 null로 두세요.
- priority는 High/Medium/Low 중 하나 또는 null만 사용하세요.
- Transcript에 없는 정보를 추측해서 추가하지 마세요.
"""


MEETING_SUMMARIZATION_HUMAN_PROMPT = """
Meeting Date: {meeting_date}
transcript:
{transcript}
출력은 반드시 JSON 형식의 구조화된 결과만 반환하세요.
"""


MEETING_SUMMARY_NORMALIZE_PROMPT = """
당신은 회의 요약 정규화를 하는 AI 어시스턴트 입니다.

입력은 여러 chunk에서 추출된 MeetingSummarizationOutput JSON 배열입니다.
아래 규칙을 지키며 하나의 MeetingSummarizationOutput으로 통합하세요.

규칙:
1) 새로운 사실/근거를 만들지 마세요.
2) 의미가 실질적으로 같은 항목만 보수적으로 병합하세요.
3) 중복 제거 시 더 구체적이고 정보량이 많은 표현을 우선하세요.
4) timescope는 원문 근거가 있을 때만 유지/수정하세요.
5) due_date/priority는 근거 없는 값으로 채우지 마세요.
6) 출력은 MeetingSummarizationOutput 스키마 JSON만 반환하세요.

meeting_date: {meeting_date}

input_json:
{input_json}
"""


def build_extract_info_chain(llm: Any):
    """회의 transcript에서 구조화 정보를 추출하는 chain 생성"""

    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", MEETING_SUMMARIZATION_PROMPT),
        ("human", MEETING_SUMMARIZATION_HUMAN_PROMPT),
    ])

    return extract_prompt | llm.with_structured_output(MeetingSummarizationOutput)


def build_normalize_info_chain(llm: Any):
    """chunk별 추출 결과를 정규화하는 chain 생성"""

    normalize_prompt = ChatPromptTemplate.from_messages([
        ("system", MEETING_SUMMARY_NORMALIZE_PROMPT),
    ])

    return normalize_prompt | llm.with_structured_output(MeetingSummarizationOutput)
