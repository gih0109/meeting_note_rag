import enum
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    String, Text, DateTime, ForeignKey, Integer, Enum, JSON, Boolean
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class RunStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class Timescope(str, enum.Enum):
    past = "past"
    present = "present"
    future = "future"


class Meeting(Base):
    """회의록 원문 및 메타데이터 원본"""
    __tablename__ = "meetings"

    meeting_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    meeting_date: Mapped[str] = mapped_column(String(32), default="")

    raw_transcript: Mapped[str] = mapped_column(Text) # 원문은 그대로 저장(감사/재현/디버깅)
    transcript_sha256: Mapped[str] = mapped_column(String(64), index=True) # 동일 텍스트 중복 업로드 방지/식별

    # 파싱 메타
    start_timestamp: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    end_timestamp: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    line_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    runs: Mapped[List["SummaryRun"]] = relationship(back_populates="meeting")


class SummaryRun(Base):
    """비동기 라우터 처리 + 버전관리 + 재실행 이력"""
    __tablename__ = "summary_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.meeting_id"), index=True)

    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus), default=RunStatus.queued)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    pipeline_version: Mapped[str] = mapped_column(String(32), default="v1") # 재실행 버전
    llm_model: Mapped[str] = mapped_column(String(64), default="")
    emb_model: Mapped[str] = mapped_column(String(128), default="")

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    meeting: Mapped["Meeting"] = relationship(back_populates="runs")


class DecisionRow(Base):
    """회의록에서 추출된 결정사항"""
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.meeting_id"), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("summary_runs.run_id"), index=True)

    timescope: Mapped[Timescope] = mapped_column(Enum(Timescope))
    title: Mapped[str] = mapped_column(String(256))
    content: Mapped[str] = mapped_column(Text)
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)

    start_timestamp: Mapped[str] = mapped_column(String(16))
    end_timestamp: Mapped[str] = mapped_column(String(16))

    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1024), nullable=True)  # bge-m3는 1024 차원인 경우가 흔함(환경에 따라 확인 필요)


class ActionItemRow(Base):
    """회의록에서 추출된 액션아이템"""
    __tablename__ = "action_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.meeting_id"), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("summary_runs.run_id"), index=True)

    timescope: Mapped[Timescope] = mapped_column(Enum(Timescope))
    objective: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    task: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    due_date: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    priority: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    background: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    start_timestamp: Mapped[str] = mapped_column(String(16))
    end_timestamp: Mapped[str] = mapped_column(String(16))

    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1024), nullable=True)


class KeyDiscussionRow(Base):
    """회의록에서 추출된 의논사항"""
    __tablename__ = "key_discussion_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.meeting_id"), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("summary_runs.run_id"), index=True)

    timescope: Mapped[Timescope] = mapped_column(Enum(Timescope))
    title: Mapped[str] = mapped_column(String(256))
    discussions: Mapped[List[str]] = mapped_column(ARRAY(Text), default=list)

    start_timestamp: Mapped[str] = mapped_column(String(16))
    end_timestamp: Mapped[str] = mapped_column(String(16))

    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1024), nullable=True)


class SegmentSummaryRow(Base):
    __tablename__ = "segment_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.meeting_id"), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("summary_runs.run_id"), index=True)

    segment_index: Mapped[int] = mapped_column(Integer)
    start_timestamp: Mapped[str] = mapped_column(String(16))
    end_timestamp: Mapped[str] = mapped_column(String(16))

    title: Mapped[str] = mapped_column(String(256))
    bullets: Mapped[List[str]] = mapped_column(ARRAY(Text), default=list)
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)

    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1024), nullable=True)


class RagNodeType(str, enum.Enum):
    decision = "decision"
    action_item = "action_item"
    discussion = "discussion"
    segment = "segment"


class RagNode(Base):
    """rag 검색용"""
    __tablename__ = "rag_nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meeting_id: Mapped[str] = mapped_column(ForeignKey("meetings.meeting_id"), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("summary_runs.run_id"), index=True)

    node_type: Mapped[RagNodeType] = mapped_column(Enum(RagNodeType), index=True)
    source_table: Mapped[str] = mapped_column(String(64))
    source_pk: Mapped[str] = mapped_column(String(64))  # 원본 row id를 문자열로 저장(범용)

    start_timestamp: Mapped[str] = mapped_column(String(16))
    end_timestamp: Mapped[str] = mapped_column(String(16))

    text: Mapped[str] = mapped_column(Text)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1024), nullable=True)

