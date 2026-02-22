import re
import math
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable, RunnableConfig

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict


_TS_RE = re.compile(r"^\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*$")

@dataclass
class Speech:
    idx: int
    start_timestamp: str
    end_timestamp: str
    text: str
    speaker: Optional[str]


@dataclass
class Chunk:
    """텍스트 조각 단위"""
    idx: int
    text: str
    start_timestamp: str
    end_timestamp: str
    speech_start_idx: int
    speech_end_idx: int


class SummaryState(TypedDict, total=False):
    """
    summay graph 용 State
    """
    meeting_text: str # 전체 텍스트
    speech_list: List[Speech] # 대화 리스트
    chunk_list: List[Chunk]
    info_list: list


def _ts_to_seconds(ts: str) -> float:
    parts = [int(x) for x in ts.split(":")]
    if len(parts) == 2:  # mm:ss
        mm, ss = parts
        return mm * 60 + ss
    hh, mm, ss = parts  # hh:mm:ss
    return hh * 3600 + mm * 60 + ss


class SummaryInfoNodes:
    """
    회의록에서 의논사항, 결정사항, 액션아이템을 추출하는 노드를 모아놓은 class    
    """
    def __init__(
        self,
        embedding_model: Any,
        extract_info_chain: Any,
        integrate_info_chain: Any,
        llm_max_worker: int = 1,
        chunk_max_chars: int = 4096,
        chunk_overlap_speech_num: int = 8,
        postprocess_min_char: int = 400,
    ):
        self.embedding_model = embedding_model
        self.extract_info_chain = extract_info_chain
        self.integrate_info_chain = integrate_info_chain
        self.runnable_config = RunnableConfig(max_concurrency=llm_max_worker) # batch 연산을 위한 runnable config

        # 각 노드 설정값
        self.chunk_max_chars = chunk_max_chars
        self.chunk_overlap_speech_num = chunk_overlap_speech_num
        self.min_chars = postprocess_min_char

    
    def make_chunks(self, state: SummaryState) -> Dict[str, Any]:
        """
        speech_list 기반으로 chunk 생성
        """
        speech_list = state["speech_list"]
        if not speech_list:
            return {"chunks": []}
        
        chunk_list = []
        chunk_idx = 0
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

                if line_list and cur_len + len_line > self.chunk_max_chars:
                    break

                line_list.append(line)
                cur_len += len_line
                end_ts = speech.end_timestamp
                j += 1

            text = "\n".join(line_list).strip()
            if text:
                chunk_list.append(
                    Chunk(
                        idx=chunk_idx,
                        text=text,
                        start_timestamp=start_ts,
                        end_timestamp=end_ts,
                        speech_start_idx=i,
                        speech_end_idx=j-1,
                    )
                )
                chunk_idx += 1

            # while 탈출
            if j >= len(speech_list):
                break
            # overlap
            next_i = j - self.overlap_speech_num
            if next_i <= i:
                next_i = i + 1
            i = next_i

        return {"chunks": chunk_list}
    

    def extract_info(self, state: SummaryState) -> Dict[str, Any]:
        """
        각 chunk 에서 의논사항, 결정사항, 액션아이템을 추출하는 chain 을 실행하는 노드
        """
        
        pass


    def integrate_info(self, state: SummaryState) -> Dict[str, Any]:

        pass


def build_summary_info_graph(
    embedding_model: Any, 
    extract_info_chain: Any, 
    integrate_info_chain: Any, 
    **node_kwargs,
):
    nodes = SummaryInfoNodes(
        embedding_model=embedding_model,
        extract_info_chain=extract_info_chain,
        integrate_info_chain=integrate_info_chain,
        **node_kwargs,
    )

    g = StateGraph(SummaryState)

    