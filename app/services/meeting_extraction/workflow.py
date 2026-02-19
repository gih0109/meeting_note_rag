
from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import Annotated, TypedDict

import anyio
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.config import settings
from app.services.meeting_extraction.func.parser import parse_transcript, TranscriptLine, parse_ts_to_seconds
from app.services.meeting_extraction.func.chunk import TranscriptChunk, make_chunks

# 체인/스키마/임베딩
from app.services.meeting_extraction.chains import (
    CorrectedChunk,
    MeetingSummarizationResponse,
    SegmentSummary,
    build_correct_chain,
    build_extract_chain,
    build_normalize_chain,
    build_segment_chain,
)
from app.services.meeting_extraction.embeddings import EmbeddingService

# DB 모델
from app.db.models import (
    Meeting,
    SummaryRun,
    DecisionRow,
    ActionItemRow,
    KeyDiscussionRow,
    SegmentSummaryRow,
    RagNode,
)


def _as_enum(enum_cls: Any, value: str) -> Any:
    """
    If enum_cls is a Python Enum class, convert string to enum; otherwise return string.
    Works for both DB-Enum schema and TEXT+CHECK schema.
    """
    if enum_cls is None:
        return value
    try:
        return enum_cls(value)
    except Exception:
        return value


_ALLOWED_TIMESCOPE = {"past", "present", "future"}

def ensure_timescope(ts: str) -> str:
    if ts not in _ALLOWED_TIMESCOPE:
        raise ValueError(f"invalid timescope: {ts}")
    return ts


# workflow state 정의


class WorkflowState(TypedDict, total=False):
    # inputs
    session: AsyncSession
    run_id: str

    # loaded from DB
    meeting_id: str
    meeting_date: str
    raw_transcript: str

    # parsed / chunked
    parsed_lines: List[Dict[str, str]]  # [{"timestamp":..., "text":...}]
    chunks: List[Dict[str, Any]]        # chunk dicts (see _chunk_to_dict)

    # map outputs (reduce via list concatenation)
    chunk_outputs: Annotated[List[Dict[str, Any]], operator.add]
    segment_outputs: Annotated[List[Dict[str, Any]], operator.add]

    # combined / normalized
    combined: Dict[str, Any]
    normalized: Dict[str, Any]

    # status/error
    error_message: str