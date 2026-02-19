import os
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from app.config.config import settings


# 추출 결과 스키마
class Decision(BaseModel):
    timescope: Literal["past", "present", "future"]
    title: str
    content: str
    tags: List[str]
    start_timestamp: str
    end_timestamp: str


class ActionItem(BaseModel):
    timescope: Literal["past", "present", "future"]
    objective: Optional[str] = None
    task: Optional[str] = None
    due_date: Optional[str] = None
    priority: Optional[str] = None
    background: Optional[str] = None
    start_timestamp: str
    end_timestamp: str


class KeyDiscussionPoint(BaseModel):
    timescope: Literal["past", "present", "future"]
    title: str
    discussions: List[str]
    start_timestamp: str
    end_timestamp: str


class Bucket(BaseModel):
    agenda: List[str] = []
    key_discussion_points: List[KeyDiscussionPoint] = []
    decisions: List[Decision] = []
    action_items: List[ActionItem] = []


class MeetingSummarizationResponse(BaseModel):
    past: Bucket = Bucket()
    present: Bucket = Bucket()
    future: Bucket = Bucket()


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
당신은 회의록(전사 텍스트)에서 핵심 정보를 추출하여 '과거/현재/미래' 시간대별로 구조화하는 AI 어시스턴트입니다.

아래 Transcript를 분석하여, 다음 4가지 정보를 반드시 시간대별로 분류해서 출력하세요.
- agenda
- key_discussion_points
- decisions
- action_items

[출력 규칙]
- 출력은 반드시 스키마에 맞는 JSON만 제공하세요(설명 문장 금지).
- 각 버킷(past/present/future) 내부의 agenda, key_discussion_points, decisions, action_items는 없으면 빈 리스트([])로 두세요.
- timestamp는 Transcript에 등장하는 형식을 그대로 사용하세요.
- priority는 High/Medium/Low 중 하나로 채우되, 근거가 없으면 null로 두세요.
- Transcript에 없는 정보는 절대 추측하지 마세요. 특히 청크 범위 밖 내용은 생성하지 마세요.

Meeting Date: {meeting_date}
Transcript:
{transcript}
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
start_timestamp: {start_timestamp}
end_timestamp: {end_timestamp}
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

    llm = build_llm(model_name=model_name)

    segment_prompt = ChatPromptTemplate.from_messages([
        ("system", SEGMENT_SUMMARY_SYSTEM_PROMPT),
        ("human", SEGMENT_SUMMARY_HUMAN_PROMPT),
    ])

    chain = segment_prompt | llm.with_structured_output(SegmentSummaryOutput)
    return chain
