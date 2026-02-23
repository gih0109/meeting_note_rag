from datetime import date, datetime, timezone
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_neo4j import Neo4jVector
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


class Decision(BaseModel):
    """그래프에 적재되는 정규화된 의사결정 엔티티."""
    decision_id: str
    track_id: str
    version: int = 1
    change_type: Literal["created", "updated", "changed", "canceled", "unchanged"] = "created"
    meeting_id: str
    meeting_date: date
    decision_title: str
    raw_text: str
    tags: List[str] = Field(default_factory=list)
    related_decision_ids: List[str] = Field(default_factory=list)


class AgendaClsOutput(BaseModel):
    """신규 결정과 후보 결정의 관계 판정 결과."""
    is_same_agenda: bool
    relation_strength_score: Optional[float] = Field(default=None, ge=0, le=1)
    directness_score: Optional[float] = Field(default=None, ge=0, le=1)
    change_type: Literal["created", "updated", "changed", "canceled", "unchanged"]
    base_decision_id: Optional[str]
    reason: str


class DecisionGraphState(TypedDict, total=False):
    """LangGraph 노드 간에 전달되는 상태 스키마."""
    new_decision: Decision
    candidates: List[Tuple[Document, float]]
    related_rows: List[Dict[str, Any]]
    rep_rows: List[Dict[str, Any]]
    new_node_id: str


def decision_node_id(decision: Decision) -> str:
    """Neo4j 노드 ID 규칙(decision_id + version)을 만드는 함수"""
    return f"{decision.decision_id}:v{decision.version}"


def decision_to_page_content(decision: Decision) -> str:
    """벡터 검색/색인에 사용할 텍스트 본문을 구성하는 함수"""
    return f"[Title] {decision.decision_title}\n[Text] {decision.raw_text}\n"


def decision_to_document(decision: Decision) -> Document:
    """Decision을 벡터스토어 적재용 Document로 변환"""
    return Document(
        page_content=decision_to_page_content(decision),
        metadata={
            "id": decision_node_id(decision),
            "decision_id": decision.decision_id,
            "track_id": decision.track_id,
            "version": decision.version,
            "change_type": decision.change_type,
            "meeting_id": decision.meeting_id,
            "meeting_date": decision.meeting_date.isoformat(),
            "decision_title": decision.decision_title,
            "raw_text": decision.raw_text,
            "tags": decision.tags,
            "related_decision_ids": decision.related_decision_ids,
        },
    )


def format_candidate_for_prompt(doc: Document) -> str:
    """후보 결정을 LLM 프롬프트 입력 문자열로 포맷팅"""
    m = doc.metadata
    return (
        f"decision_id: {m.get('decision_id')}\n"
        f"track_id: {m.get('track_id')}\n"
        f"version: {m.get('version')}\n"
        f"meeting_id: {m.get('meeting_id')}\n"
        f"meeting_date: {m.get('meeting_date')}\n"
        f"title: {m.get('decision_title')}\n"
        f"text: {m.get('raw_text')}\n"
    )


class DecisionIndexerNodes:
    """
    의사결정 적재 파이프라인을 Langgraph 노드로 구성

    """
    def __init__(
        self,
        related_store: Any,
        agenda_cls_chain: Any,
        similarity_threshold: float= 0.5,
        directness_threshold: float= 0.5,
        related_top_k: int = 20,
        llm_max_worker: int = 1,
        node_label: str = "Decision",
        relationship_type: str = "PRECEDES",
    ):
        self.related_store = related_store
        self.agenda_cls_chain = agenda_cls_chain
        self.similarity_threshold = similarity_threshold
        self.directness_threshold = directness_threshold
        self.related_top_k = related_top_k
        self.runnable_config = RunnableConfig(max_concurrency=llm_max_worker) # batch 연산을 위한 runnable config
        self.node_label = node_label
        self.relationship_type = relationship_type


    def search_candidate(self, state: DecisionGraphState):
        """유사도 검색으로 후보 결정을 찾는 노드"""
        new_decision = state["new_decision"]
        query = decision_to_page_content(new_decision)

        # 유사도 검색
        raw = self.related_store.similarity_search_with_score(query=query, k=self.related_top_k)
        
        # similarity_threshold 및 같은 회의는 제외하는 필터링
        filtered = [
            (doc, score) for doc, score in raw
            if doc.metadata.get("meeting_id") != new_decision.meeting_id and score >= self.similarity_threshold
        ]
        return {"candidates": filtered}
    

    def classfiy_candidate(self, state: DecisionGraphState):
        """후보별로 LLM chain 이 연관성을 판별 노드"""
        new_decision = state["new_decision"]
        candidates = state["candidates"]
        if not candidates:
            return {"related_rows": []}
        
        payload = []
        new_text = f"title: {new_decision.decision_title}\ntext: {new_decision.raw_text}\n"
        for doc, _ in candidates:
            payload.append(
                {
                    "new_decision": new_text,
                    "candidate_decision": format_candidate_for_prompt(doc),
                }
            )
        
        output = self.agenda_cls_chain.batch(
            payload,
            config=self.runnable_config,
            return_exceptions=True,
        )

        row_list = []
        error_list = []
        for (doc, score), out in zip(candidates, output):
            if isinstance(out, Exception):
                error_list.append({
                    "candidate_decision_id": str(doc.metadata.get("decision_id")),
                    "error": repr(out),
                })
                continue
            row_list.append(
                {
                    "candidate_node_id": doc.metadata.get("id"),
                    "candidate_decision_id": doc.metadata.get("decision_id"),
                    "candidate_track_id": doc.metadata.get("track_id") or doc.metadata.get("decision_id"),
                    "candidate_version": int(doc.metadata.get("version") or 1),
                    "meeting_date": doc.metadata.get("meeting_date") or "",
                    "relation_score": out.relation_strength_score,
                    "directness_score": out.directness_score,
                    "is_same_agenda": out.is_same_agenda,
                    "change_type": out.change_type,
                    "base_decision_id": out.base_decision_id,
                    "reason": out.reason,
                    "rag_sim_score": score,
                }
            )
        
        if candidates and len(row_list) == 0: # 전부 실패 시
            raise RuntimeError(f"all llm classification chain failed. {error_list[:3]}")
        if error_list: # 부분 실패 시
            print(f"[Warning] llm classification chain partial failed: {len(error_list)}")

        return {"related_rows": row_list}
    

    def pick_representatives(self, state: DecisionGraphState):
        """
        그래프 그룹 내 대표 후보를 고르고 신규 결정 연결을 결정 노드
        """

        def _sort_key_fn(x: Dict[str, Any]):
            """정렬용 내부 함수"""
            # 우선순위: direct_precedes -> relation -> rag_sim -> version -> meeting_date
            direct = x.get("directness_score")
            rel = x.get("relation_score")
            rag = x.get("rag_sim_score")
            ver = x.get("candidate_version") or 0
            mdate = str(x.get("meeting_date") or "")
            return (
                -(float(direct) if direct is not None else -1.0),
                -(float(rel) if rel is not None else -1.0),
                -(float(rag) if rag is not None else -1.0),
                -int(ver),
                -int(mdate.replace("-", "") or 0),
            )
        
        def _base_max(rep: Dict[str, Any]):
            # 단일 트랙일 시 그룹 후보 중 하나를 선택하기 위한 정렬 함수
            primary = rep.get("directness_score")
            if primary is None:
                primary = rep.get("relation_score")
            return (float(primary) if primary is not None else -1.0, int(rep.get("candidate_version") or 0))

        new_decision = state["new_decision"]
        related_row_list = state.get("related_rows", [])
        
        same_agenda = [r for r in related_row_list if r.get("is_same_agenda") if True and r.get("candidate_node_id")]

        if not same_agenda:
            new_decision.track_id = new_decision.decision_id
            new_decision.version = 1
            new_decision.change_type = "created"
            return {"new_decision": new_decision, "rep_rows": []}
        
        tracked_dict = defaultdict(list)
        for a in same_agenda:
            track_id = a.get("candidate_track_id") or a.get("candidate_decision_id")
            tracked_dict[track_id].append(a)

        # 대표
        rep_list = []
        for _, row_list in tracked_dict.items():
            # 정렬
            sorted_row_list = sorted(row_list, key=_sort_key_fn)
            # 가장 높은 직접성/관련성 선택
            rep = sorted_row_list[0]
            primary = rep.get("directness_score")
            if primary is None: # 직접성 점수가 없으면 관련성 점수로 대체
                primary = rep.get("relation_score")
            # 대표 점수가 임계값 미만이면 해당 트랙 연결은 생략
            if primary is None or float(primary) < self.directness_threshold:
                continue
            
            # 선택된 후보
            rep["supporting_candidate_node_ids"] = [
                r.get("candidate_node_id") for r in sorted_row_list[1:] if r.get("candidate_node_id")
            ]
            rep_list.append(rep)
        
        # 모든 트랙이 임계값 이하라면, 신규 트랙 시작으로 처리
        if len(rep_list) == 0:
            new_decision.track_id = new_decision.decision_id
            new_decision.version = 1
            new_decision.change_type = "created"
            return {"new_decision": new_decision, "rep_rows": []}
        
        # 그룹 개수 확인
        rep_track_group_set = {r.get("candidate_track_id") for r in rep_list if r.get("candidate_track_id")}
        is_merge = True if len(rep_track_group_set) >= 2 else False # 기존 단일 트랙에 귀속 여부

        if is_merge:
            # 여러 그룹과 연결로 판별될 경우 신규로 처리
            new_decision.track_id = new_decision.decision_id
            new_decision.version = 1
            new_decision.change_type = "created"
        else:
            # 단일 그룹 -> 최종 기준 후보를 선택
            base = max(rep_list, key=_base_max)
            new_decision.track_id = base.get("candidate_track_id") or new_decision.decision_id
            new_decision.version = int(base.get("candidate_version") or 1) + 1
            new_decision.change_type = base.get("change_type") or "updated"

        return {"new_decision": new_decision, "rep_rows": rep_list}
    

    def write_new_node(self, state: DecisionGraphState):
        """확정된 신규 결정을 벡터스토어(Neo4j)에 노드로 적재 노드"""
        decision = state["new_decision"]
        doc = decision_to_document(decision)
        node_id = doc.metadata["id"]
        self.related_store.add_documents([doc], ids=[node_id])
        return {"new_node_id": node_id}
    

    def upsert_edges(self, state: DecisionGraphState):
        """대표 후보들과 신규 노드 간 엣지를 neo4j store에 upsert 노드"""
        new_node_id = state["new_node_id"]
        rep_rows = state["rep_rows"]
        self._upsert_edges_to_neo4jvector(new_node_id, rep_rows)
        return {}
    

    def _upsert_edges_to_neo4jvector(self, new_node_id: str, rep_rows: List[Dict[str, Any]]):
        """neo4j vectorstore 에 엣지를 upsert 메서드"""

        row_list = []
        for r in rep_rows:
            src = r.get("candidate_node_id")
            if not src:
                continue
            props = {
                "directness_score": r.get("directness_score"),
                "relation_score": r.get("relation_score"),
                "rag_sim_score": r.get("rag_sim_score"),
                "change_type": r.get("change_type"),
                "base_decision_id": r.get("base_decision_id"),
                "reason": r.get("reason"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            supporting = r.get("supporting_candidate_node_ids", [])
            if supporting:
                props["supporting_candidate_node_ids"] = supporting
            row_list.append({"src": src, "dst": new_node_id, "props": props})

        if len(row_list) == 0:
            return

        cypher = f"""
        UNWIND $rows AS row
        MATCH (a:`{self.node_label}` {{id: row.src}})
        MATCH (b:`{self.node_label}` {{id: row.dst}})
        // 방향 고정: existing(a) -> new(b)
        MERGE (a)-[r:{self.relationship_type}]->(b)
        SET r += row.props
        """
        try:
            self.related_store.query(cypher, params={"rows": row_list})
        except TypeError:
            self.related_store.query(cypher, {"rows": row_list})


def build_decision_tracking_graph(related_store: Any, agenda_cls_chain: Any, **node_kargs):
    """decision_tracking_graph 조립 함수"""

    nodes = DecisionIndexerNodes(
        related_store=related_store,
        agenda_cls_chain=agenda_cls_chain,
        **node_kargs,
    )

    g = StateGraph(DecisionGraphState)

    g.add_node("search_candidate", nodes.search_candidate)
    g.add_node("classfiy_candidate", nodes.classfiy_candidate)
    g.add_node("pick_representatives", nodes.pick_representatives)
    g.add_node("write_new_node", nodes.write_new_node)
    g.add_node("upsert_edges", nodes.upsert_edges)

    g.add_edge(START, "search_candidate")
    g.add_edge("search_candidate", "classfiy_candidate")
    g.add_edge("classfiy_candidate", "pick_representatives")
    g.add_edge("pick_representatives", "write_new_node")
    g.add_edge("write_new_node", "upsert_edges")
    g.add_edge("upsert_edges", END)

    return g.compile()
