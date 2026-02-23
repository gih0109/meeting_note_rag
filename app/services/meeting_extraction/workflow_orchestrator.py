import asyncio
import hashlib

from datetime import date, datetime
from typing import Any, Dict, List, Optional, Union

from app.services.meeting_extraction.func.parser import parse_speech_from_meeting_text
from app.services.meeting_extraction.workflow_chain.segment_summarize_chain import build_segment_chain
from app.services.meeting_extraction.workflow_chain.summary_info_chain import (
    build_extract_info_chain,
    build_normalize_info_chain,
)
from app.services.meeting_extraction.workflow_chain.agenda_cls_chain import build_agenda_cls_chain

from app.services.meeting_extraction.workflow_sub_node.summary_info_workflow import build_summary_info_graph
from app.services.meeting_extraction.workflow_sub_node.topic_segment_workflow import build_topic_segment_graph
from app.services.meeting_extraction.workflow_sub_node.decision_tracking_workflow import (
    Decision as TrackingDecision,
    build_decision_tracking_graph,
)


def _to_date(value: Union[str, date, datetime]) -> date:
    if isinstance(value, date):
        return value
    elif isinstance(value, datetime):
        return value.date()
    
    return date.fromisoformat(value)


def _default_normalized_info():
    pass


def _make_decision_id(meeting_id: str):

    pass


class MeetingExtractionOrchestrator:
    """
    3개 그래프 `topic_segment_graph`, `summary_info_graph`, `decision_graph` 를 하나의 usecase 로 묶어 실행
    """
    def __init__(
        self, 
        topic_segment_graph: Any, 
        summary_info_graph: Any, 
        decision_tracking_graph: Any
    ):
        self.topic_graph = topic_segment_graph
        self.summary_graph = summary_info_graph
        self.decision_graph = decision_tracking_graph

    
    def _to_tracking_decision(
        self,
        meeting_id: str,
        meeting_date: date,
        idx: int,
        item: Dict[str, Any],
    ) -> Optional[TrackingDecision]:
        """
        summary_info_graph 의 state["decisions"] 을 decision_tracking_graph_kwargs 그래프의 입력으로 변환
        """
        title = item.get("title", "").strip()
        content = item.get("content", "").strip()
        if not title and not content:
            return None
        
        decision_id = _make_decision_id(meeting_id=meeting_id) # TODO _make_decision_id 완성시키기

        tag_list = item.get("tags", [])
        
        return TrackingDecision(
            decision_id=decision_id,
            track_id=decision_id,   # 신규 입력 시 임시값, 그래프 내에서 재조정됨
            version=1,
            change_type="created",
            meeting_id=meeting_id,
            meeting_date=meeting_date,
            decision_title=title or f"decision-{idx}",
            raw_text=content,
            tags=tag_list,
            related_decision_ids=[],
        )
    
    async def run(
        self,
        meeting_id,
        meeting_date,
        meeting_text: str,
    ) -> Dict[str, Any]:
        """
        전체 파이프라인 실행 
        """
        pass


def build_meeting_extraction_orchestrator(
    embedding_model: Any,
    llm: Any,
    related_store: Any,
    topic_segment_graph_kwargs: Dict[str, Any] | None = None,
    summary_info_graph_kwargs: Dict[str, Any] | None = None,
    decision_tracking_graph_kwargs: Dict[str, Any] | None = None,
) -> MeetingExtractionOrchestrator:
    """
    그래프 조립
    """
    # kwargs 처리
    topic_segment_graph_kwargs = topic_segment_graph_kwargs if topic_segment_graph_kwargs is not None else {}
    summary_info_graph_kwargs = summary_info_graph_kwargs if summary_info_graph_kwargs is not None else {}
    decision_tracking_graph_kwargs = decision_tracking_graph_kwargs if decision_tracking_graph_kwargs is not None else {}

    # build chain
    segment_chain = build_segment_chain(llm)
    extract_info_chain = build_extract_info_chain(llm)
    normalize_info_chain = build_normalize_info_chain(llm)
    agenda_cls_chain = build_agenda_cls_chain(llm)

    # build graph
    topic_segment_graph = build_topic_segment_graph(
        embedding_model=embedding_model,
        segment_summarize_chain=segment_chain,
        **topic_segment_graph_kwargs
    )
    summary_info_graph = build_summary_info_graph(
        embedding_model=embedding_model,
        normalize_info_chain=normalize_info_chain,
        **summary_info_graph_kwargs,
    )
    decision_tracking_graph = build_decision_tracking_graph(
        related_store=related_store,
        agenda_cls_chain=agenda_cls_chain,
        **decision_tracking_graph_kwargs,
    )

    # build orchestrator
    return MeetingExtractionOrchestrator(
        topic_segment_graph=topic_segment_graph,
        summary_info_graph=summary_info_graph,
        decision_tracking_graph=decision_tracking_graph,
    )
