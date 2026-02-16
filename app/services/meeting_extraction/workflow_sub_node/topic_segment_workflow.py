import math
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable, RunnableConfig

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict


@dataclass
class Unit:
    """텍스트 조각 단위"""
    idx: int
    text: str


@dataclass
class Segment:
    """주제 구분 단락"""
    seg_idx: int
    text: str
    summary: Optional[str] = None
    unit_start: int
    unit_end: int


class SegmentState(TypedDict, total=False):
    meeting_text: str # 전체 텍스트
    units: List[Unit] # 텍스트 조각 리스트
    unit_embeddings: List[List[float]]
    Segments: List[Segment]
    Segment_summaries: List[str]


def _l2_normalize(vector: List[float]) -> List[float]:
    """코사인 유사도 계산을 위한 벡터 L2 정규화"""
    s = 0.0
    for v in vector:
        s += v * v
    n = math.sqrt(s) if s > 0 else 1.0
    return [x / n for x in vector]


def _consine_dot_product(a: List[float], b: List[float]) -> float:
    """코사인 유사도 계산을 위한 dot product 계산"""
    s = 0.0
    for i, j in zip(a, b):
        s += i * j
    return s


def _avg_vectors(vectors: List[List[float]]) -> List[float]:
    """
    여러 임베딩을 평균 계산
    세그먼트의 대표 의미 벡터
    """
    if not vectors:
        return []
    
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i, val in enumerate(v):
            acc[i] += val
    n = float(len(vectors))
    return _l2_normalize([x / n for x in acc]) # L2 정규화


### langgraph node
class TopicSegmentNodes:
    """
    
    """
    def __init__(
        self,
        embedding_model: Any,
        segment_summarize_chain: Any,
        llm_max_worker: int = 1,
        unit_max_chars: int = 500,
        unit_overlap_chars: int = 80,
        similarity_threshold: float = 0.7,
        max_units_per_segment: int = 20,
        postprocess_min_chars: int = 300,
    ):
        self.embedding_model = embedding_model
        self.segment_chain = segment_summarize_chain
        self.runnable_config = RunnableConfig(max_concurrency=llm_max_worker)

        # 각 노드 설정값
        self.max_chars = unit_max_chars
        self.overlap_chars = unit_overlap_chars
        self.similarity_threshold = similarity_threshold
        self.max_units_per_segment = max_units_per_segment
        self.min_chars = postprocess_min_chars
        

    def meeting_text_to_dict(self, state: SegmentState) -> Dict[str, Any]:
        """텍스트 정규화 노드"""
        text = state["meeting_text"]
        text = text.replace("\r\n", "\n").strip()
        return {"meeting_text": text}


    def make_units(self, state: SegmentState) -> Dict[str, Any]:
        """텍스트를 유닛으로 분할하는 노드"""
        text = state["meeting_text"]
        chunks = []

        i = 0
        while i < len(text):
            j = min(i + self.max_chars, len(text))
            chunks = text[i:j].strip()

            if chunks:
                chunks.append(chunks)

            i = j - self.overlap_chars
            if i < 0:
                i = 0
            if j == len(text):
                break

        units = [Unit(idx=k, text=t) for k, t in enumerate(chunks)]
        return {"units": units}


    def embedding_units(self, state: SegmentState) -> Dict[str, Any]:
        """
        각 유닛 텍스트를 임베딩 벡터로 변환하는 노드

        Args:
            state (SegmentState): langgraph state
            embedding_model (HuggingFaceEmbeddings): embedding model instance
        
        """
        units = state["units"]
        texts = [unit.text for unit in units]
        embeds = self.embedding_model.embed_documents(texts)
        embeds = [_l2_normalize(list(map(float, embed))) for embed in embeds]
        return {"unit_embeddings": embeds}


    def make_segments(self, state: SegmentState) -> Dict[str, Any]:
        """
        임베딩 유사도를 이용해 유닛을 순차적으로 묶어 세그먼트 생성하는 노드 \n
        현재 세그먼트의 평균 임베딩값과 다음 유닛의 임베딩값의 코사인 유사도를 비교후 현재 세그먼트에 추가하거나 새 세그먼트 시작

        Args:
            state (SegmentState): langgraph state
            similarity_threshold (float): 유닛을 기존 세그먼트에 추가할지 결정하는 코사인 유사도 임계값. (0 ~ 1) default=0.7
            max_units_per_segment (int): 기존 세그먼트가 가질 수 있는 유닛의 최대값. default=20 

        """
        units = state["units"]
        embs = state["unit_embeddings"]

        segments = []
        if not units:
            return {"segments": segments}
        
        cur_start = 0
        cur_vectors = [embs[0]]
        seg_idx = 0
        for i in range(1, len(units)):
            segments_emb_avg = _avg_vectors(cur_vectors)
            sim = _consine_dot_product(segments_emb_avg, embs[i]) # 기존 세그먼트 임베딩과 새 유닛 임베딩의 코사인 유사도 계산

            long_flag = (i - cur_start) >= self.max_units_per_segment
            if sim >= self.similarity_threshold and not long_flag:
                cur_vectors.append(embs[i])
                continue
            
            seg_text = "\n".join(unit.text for unit in units[cur_start:i])
            segments.append(
                Segment(
                    seg_idx=seg_idx,
                    text=seg_text,
                    unit_start=cur_start,
                    unit_end=i-1
                )
            )
            seg_idx += 1
            cur_start += 1
            cur_vectors = [embs[i]]

        seg_text = "\n".join(u.text for u in units[cur_start:len(units)])
        segments.append(Segment(seg_idx=seg_idx, unit_start=cur_start, unit_end=len(units) - 1, text=seg_text))

        return {"segments": segments}


    def postprocess_segments(self, state: SegmentState) -> Dict[str, Any]:
        """
        너무 짧은 세그먼트는 병합하는 노드
        """
        segs = list(state["segments"])
        if not segs:
            return {"segments": segs}

        merged: List[Segment] = []
        i = 0
        while i < len(segs):
            s = segs[i]
            if len(s.text) >= self.min_chars or i == len(segs) - 1:
                merged.append(s)
                i += 1
                continue

            nxt = segs[i + 1]
            new_text = s.text + "\n" + nxt.text
            merged.append(
                Segment(
                    seg_idx=len(merged),
                    unit_start=s.unit_start,
                    unit_end=nxt.unit_end,
                    text=new_text,
                )
            )
            i += 2

        for k, s in enumerate(merged):
            s.seg_idx = k

        return {"segments": merged}


    def summarize_segments(self, state: SegmentState) -> Dict[str, Any]:
        segs = state["segments"]
        summaries = []


        payload = []
        for seg in segs:
            payload.append({
                
            })

        outputs = self.segment_chain.batch(
            payload, 
            config=self.runnable_config, 
            return_exception=True
        )

        return {"segments": segs, "segment_summaries": summaries}


def build_topic_segment_graph(embedding_model: Any, segment_summarize_chain: Any, **node_kwargs):
    """graph 조립 함수"""

    nodes = TopicSegmentNodes(
        embedding_model=embedding_model,
        segment_summarize_chain=segment_summarize_chain,
        **node_kwargs,
    )

    g = StateGraph(SegmentState)

    g.add_node("meeting_text_to_dict", nodes.meeting_text_to_dict)
    g.add_node("make_units", nodes.make_units)
    g.add_node("embedding_units", nodes.embedding_units)
    g.add_node("make_segments", nodes.make_segments)
    g.add_node("postprocess_segments", nodes.postprocess_segments)
    g.add_node("summarize_segments", nodes.summarize_segments)

    g.set_entry_point("meeting_text_to_dict")
    g.add_edge("meeting_text_to_dict", "make_units")
    g.add_edge("make_units", "embedding_units")
    g.add_edge("embedding_units", "make_segments")
    g.add_edge("make_segments", "postprocess_segments")
    g.add_edge("postprocess_segments", "summarize_segments")
    g.add_edge("summarize_segments", END)

    return g.compile()
