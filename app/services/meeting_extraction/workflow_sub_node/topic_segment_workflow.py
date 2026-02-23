import math
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable, RunnableConfig

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict
from app.services.meeting_extraction.func.parser import parse_speech_from_meeting_text as shared_parse_speech_from_meeting_text


@dataclass
class Speech:
    idx: int
    start_timestamp: str
    end_timestamp: str
    text: str
    speaker: Optional[str]


@dataclass
class Unit:
    """텍스트 조각 단위"""
    idx: int
    text: str
    start_timestamp: str
    end_timestamp: str
    speech_start_idx: int
    speech_end_idx: int


@dataclass
class Segment:
    """주제 구분 단락"""
    seg_idx: int
    text: str
    unit_start_idx: int
    unit_end_idx: int
    start_timestamp: str
    end_timestamp: str
    summary: Optional[str] = None


@dataclass
class SegmentSummaryOutput:
    """
    단락별 요약 최종 출력
    """
    segment_index: int
    start_timestamp: Any
    end_timestamp: Any
    title: str
    bullets: List[str]


class SegmentState(TypedDict, total=False):
    """
    segment graph 용 State
    """
    meeting_text: str # 전체 텍스트
    speech_list: List[Speech] # 대화 리스트
    units: List[Unit] # 텍스트 조각 리스트
    unit_embeddings: List[List[float]]
    segments: List[Segment]
    segment_summaries: List[SegmentSummaryOutput]


def _ts_to_seconds(ts: str) -> float:
    parts = [int(x) for x in ts.split(":")]
    if len(parts) == 2:  # mm:ss
        mm, ss = parts
        return mm * 60 + ss
    hh, mm, ss = parts  # hh:mm:ss
    return hh * 3600 + mm * 60 + ss


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


def parse_speech_from_meeting_text(meeting_text: str) -> List[Speech]:
    parsed = shared_parse_speech_from_meeting_text(meeting_text)
    return [
        Speech(
            idx=speech.idx,
            start_timestamp=speech.start_timestamp,
            end_timestamp=speech.end_timestamp,
            text=speech.text,
            speaker=speech.speaker,
        )
        for speech in parsed
    ]


### langgraph node
class TopicSegmentNodes:
    """
    각 단락의 요약을 만드는 sub workflow node 를 모아놓은 class
    Args:
        - embedding_model (HuggingFaceEmbeddings) : embedding model instance
        - segment_summarize_chain (Runnable) : segment chain
        - llm_max_worker (int) : llm api invoke 시 병렬로 작업할 batch 수
        - unit_max_chars (int) : 각 단락 기본 단위 unit의 글자 수 
        - unit_overlap_speech_num (int) : 각 단락 unit 당 겹치는 대화 수
        - similarity_threshold (float) : 유닛을 기존 세그먼트에 추가할지 결정하는 코사인 유사도 임계값. (0 ~ 1) default=0.7
        - max_units_per_segment (int) : 기존 세그먼트가 가질 수 있는 유닛의 최대값. default=20 
        - postprocess_min_chars (int) : 해당 값 미만의 unit 은 병합
    """
    def __init__(
        self,
        embedding_model: Any,
        segment_summarize_chain: Any,
        llm_max_worker: int = 1,
        unit_max_chars: int = 500,
        # unit_overlap_chars: int = 80,
        unit_overlap_speech_num: int = 2,
        similarity_threshold: float = 0.7,
        max_units_per_segment: int = 20,
        postprocess_min_chars: int = 300,
    ):
        self.embedding_model = embedding_model
        self.segment_chain = segment_summarize_chain
        self.runnable_config = RunnableConfig(max_concurrency=llm_max_worker) # batch 연산을 위한 runnable config

        # 각 노드 설정값
        self.max_chars = unit_max_chars
        # self.overlap_chars = unit_overlap_chars
        self.overlap_speech_num = unit_overlap_speech_num
        self.similarity_threshold = similarity_threshold
        self.max_units_per_segment = max_units_per_segment
        self.min_chars = postprocess_min_chars
        

    # def meeting_text_to_dict(self, state: SegmentState) -> Dict[str, Any]:
    #     """텍스트 정규화 노드"""
    #     text = state["meeting_text"]
    #     text = text.replace("\r\n", "\n").strip()
    #     return {"meeting_text": text}


    # def make_units(self, state: SegmentState) -> Dict[str, Any]:
    #     """텍스트를 유닛으로 분할하는 노드"""
    #     text = state["meeting_text"]
    #     chunks = []

    #     i = 0
    #     while i < len(text):
    #         j = min(i + self.max_chars, len(text))
    #         chunk = text[i:j].strip()

    #         if chunk:
    #             chunks.append(chunk)

    #         i = j - self.overlap_chars
    #         if i < 0:
    #             i = 0
    #         if j >= len(text):
    #             break

    #     units = [Unit(idx=k, text=t) for k, t in enumerate(chunks)]
    #     return {"units": units}


    def make_units(self, state: SegmentState) -> Dict[str, Any]:
        """
        speech_list 기반으로 Unit 생성
        """
        speech_list = state["speech_list"]
        if not speech_list:
            return {"units": []}
        
        units = []
        unit_idx = 0
        i = 0 # idx1
        while i < len(speech_list):
            # 초기화
            cur_len = 0
            j = i # idx2
            line_list = []

            start_ts = speech_list[i].start_timestamp
            end_ts = speech_list[i].end_timestamp

            # unit 을 만들기 위해 max_chars 만큼 합치기 
            while j < len(speech_list):
                speech = speech_list[j]
                line = speech.text.strip()
                len_line = len(line) + (1 if line else 0)

                if line_list and cur_len + len_line > self.max_chars:
                    break

                line_list.append(line)
                cur_len += len_line
                end_ts = speech.end_timestamp
                j += 1

            text = "\n".join(line_list).strip()
            if text:
                units.append(
                    Unit(
                        idx=unit_idx,
                        text=text,
                        start_timestamp=start_ts,
                        end_timestamp=end_ts,
                        speech_start_idx=i,
                        speech_end_idx=j-1
                    )
                )
                unit_idx += 1
            
            # while 탈출
            if j >= len(speech_list):
                break
            # overlap
            next_i = j - self.overlap_speech_num
            if next_i <= i:
                next_i = i + 1
            i = next_i

        return {"units": units}


    def embedding_units(self, state: SegmentState) -> Dict[str, Any]:
        """
        각 유닛 텍스트를 임베딩 벡터로 변환하는 노드
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
                    unit_start_idx=cur_start,
                    unit_end_idx=i-1,
                    start_timestamp=units[cur_start].start_timestamp,
                    end_timestamp=units[i - 1].end_timestamp,
                    summary=None,
                )
            )
            seg_idx += 1
            cur_start = i
            cur_vectors = [embs[i]]

        seg_text = "\n".join(u.text for u in units[cur_start:len(units)])
        segments.append(
            Segment(
                seg_idx=seg_idx,
                text=seg_text,
                unit_start_idx=cur_start,
                unit_end_idx=len(units) - 1,
                start_timestamp=units[cur_start].start_timestamp,
                end_timestamp=units[len(units) - 1].end_timestamp,
                summary=None,
            )    
        )

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
                    text=new_text,
                    unit_start_idx=s.unit_start_idx,
                    unit_end_idx=nxt.unit_end_idx,
                    start_timestamp=s.start_timestamp,
                    end_timestamp=nxt.end_timestamp,
                    summary=None,
                )
            )
            i += 2

        for k, s in enumerate(merged):
            s.seg_idx = k

        return {"segments": merged}


    def summarize_segments(self, state: SegmentState) -> Dict[str, Any]:
        seg_list = state["segments"]
        unit_list = state["units"]


        payload = []
        for seg in seg_list:
            payload.append({
                "segment_index": seg.seg_idx,
                "start_timestamp": seg.start_timestamp, 
                "end_timestamp": seg.end_timestamp,
                "transcript": seg.text,
            })

        llm_outputs = self.segment_chain.batch(
            payload, 
            config=self.runnable_config, 
            return_exception=True
        )

        summary_outputs = []
        for seg, out in zip(seg_list, llm_outputs):
            if isinstance(out, Exception):
                output = SegmentSummaryOutput(
                    segment_index=seg.seg_idx,
                    start_timestamp=seg.start_timestamp,
                    end_timestamp=seg.end_timestamp,
                    title="",
                    bullets=[]
                )
            else:
                output = SegmentSummaryOutput(
                    segment_index=seg.seg_idx,
                    start_timestamp=seg.start_timestamp,
                    end_timestamp=seg.end_timestamp,
                    title=out.title,
                    bullets=out.bullets,
                )
            summary_outputs.append(output)

        return {"segments": seg_list, "segment_summaries": summary_outputs}


def build_topic_segment_graph(embedding_model: Any, segment_summarize_chain: Any, **node_kwargs):
    """graph 조립 함수"""

    nodes = TopicSegmentNodes(
        embedding_model=embedding_model,
        segment_summarize_chain=segment_summarize_chain,
        **node_kwargs,
    )

    g = StateGraph(SegmentState)

    # g.add_node("meeting_text_to_dict", nodes.meeting_text_to_dict)
    g.add_node("make_units", nodes.make_units)
    g.add_node("embedding_units", nodes.embedding_units)
    g.add_node("make_segments", nodes.make_segments)
    g.add_node("postprocess_segments", nodes.postprocess_segments)
    g.add_node("summarize_segments", nodes.summarize_segments)

    # g.set_entry_point("meeting_text_to_dict")
    # g.add_edge("meeting_text_to_dict", "make_units")
    g.set_entry_point("make_units")
    g.add_edge("make_units", "embedding_units")
    g.add_edge("embedding_units", "make_segments")
    g.add_edge("make_segments", "postprocess_segments")
    g.add_edge("postprocess_segments", "summarize_segments")
    g.add_edge("summarize_segments", END)

    return g.compile()
